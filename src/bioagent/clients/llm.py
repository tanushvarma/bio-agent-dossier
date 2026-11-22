"""Thin LLM wrappers for structured tool calling: Gemini (default) and Anthropic.

Each provider exposes a *session* that keeps its own native message history, so
provider-specific details (Gemini thought signatures, Anthropic tool_use blocks)
are echoed back exactly. The agent only sees LLMTurn / ToolResult objects, which
also makes the whole loop easy to mock in tests.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import ConfigError, Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A provider call failed after retries, or returned an unusable response."""


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema, refs inlined


class ToolRequest(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    id: str
    name: str
    content: dict[str, Any]


class LLMTurn(BaseModel):
    text: str = ""
    tool_calls: list[ToolRequest] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMSession(Protocol):
    def next(self) -> LLMTurn: ...
    def add_tool_results(self, results: list[ToolResult]) -> None: ...
    def finalize(self, schema: type[T], instruction: str,
                 json_schema: dict[str, Any] | None = None) -> tuple[T, int, int]: ...


class LLM(Protocol):
    provider: str
    model: str

    def session(self, system: str, user: str, tools: list[ToolSpec]) -> LLMSession: ...


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every {"$ref": "#/$defs/X"} with the definition and drop $defs/titles.
    Both providers accept the result; neither has to resolve references."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def drop_properties(schema: dict[str, Any], names: set[str]) -> dict[str, Any]:
    """Remove properties (e.g. verifier-only fields) everywhere in a schema."""
    if isinstance(schema, dict):
        out = {k: drop_properties(v, names) for k, v in schema.items()}  # type: ignore[arg-type]
        if isinstance(out.get("properties"), dict):
            out["properties"] = {k: v for k, v in out["properties"].items() if k not in names}
            if "required" in out:
                out["required"] = [r for r in out["required"] if r not in names]
        return out
    if isinstance(schema, list):
        return [drop_properties(v, names) for v in schema]  # type: ignore[return-value]
    return schema


def _append_user_text(history: list, role_of: Callable[[Any], str], make_text: Callable[[str], Any],
                      append_to_last: Callable[[Any, Any], None], new_user: Callable[[Any], Any],
                      text: str) -> None:
    """Add a user instruction without creating two consecutive user turns."""
    if history and role_of(history[-1]) == "user":
        append_to_last(history[-1], make_text(text))
    else:
        history.append(new_user(make_text(text)))


def _gemini_retryable(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    return code in (429, 500, 502, 503, 504)


class GeminiLLM:
    provider = "gemini"

    def __init__(self, api_key: str | None, model: str, client: Any = None, max_attempts: int = 4,
                 sleep: Callable[[float], None] | None = None):
        if client is None:
            if not api_key:
                raise ConfigError("GEMINI_API_KEY is not set.")
            from google import genai
            client = genai.Client(api_key=api_key)
        self.client, self.model = client, model
        kw: dict[str, Any] = {"sleep": sleep} if sleep else {}
        self._retry = Retrying(stop=stop_after_attempt(max_attempts), wait=wait_exponential(multiplier=2, max=40),
                               retry=retry_if_exception(_gemini_retryable), reraise=True, **kw)

    def session(self, system: str, user: str, tools: list[ToolSpec]) -> _GeminiSession:
        return _GeminiSession(self, system, user, tools)


class _GeminiSession:
    def __init__(self, llm: GeminiLLM, system: str, user: str, tools: list[ToolSpec]):
        from google.genai import types

        self.llm, self.types, self.system = llm, types, system
        self.contents: list[Any] = [types.Content(role="user", parts=[types.Part.from_text(text=user)])]
        self.tool = types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
            for t in tools
        ])
        self._native_ids: dict[str, str | None] = {}

    def _generate(self, config: Any) -> Any:
        from google.genai import errors

        try:
            return self.llm._retry(self.llm.client.models.generate_content, model=self.llm.model,
                                   contents=self.contents, config=config)
        except errors.APIError as exc:
            raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc

    @staticmethod
    def _usage(resp: Any) -> tuple[int, int]:
        u = getattr(resp, "usage_metadata", None)
        if not u:
            return 0, 0
        out = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        return u.prompt_token_count or 0, out

    def next(self) -> LLMTurn:
        t = self.types
        resp = self._generate(t.GenerateContentConfig(
            system_instruction=self.system, tools=[self.tool],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True)))
        content = resp.candidates[0].content if resp.candidates else None
        if content is not None:
            self.contents.append(content)       # keeps thought signatures intact
        calls = []
        for i, fc in enumerate(resp.function_calls or []):
            cid = fc.id or f"gemini-{len(self.contents)}-{i}"
            self._native_ids[cid] = fc.id
            calls.append(ToolRequest(id=cid, name=fc.name, args=dict(fc.args or {})))
        text = "".join(p.text for p in (content.parts or []) if getattr(p, "text", None)
                       and not getattr(p, "thought", False)) if content else ""
        tin, tout = self._usage(resp)
        return LLMTurn(text=text, tool_calls=calls, input_tokens=tin, output_tokens=tout)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        t = self.types
        parts = [t.Part(function_response=t.FunctionResponse(
            id=self._native_ids.get(r.id), name=r.name, response={"result": r.content})) for r in results]
        self.contents.append(t.Content(role="user", parts=parts))

    def finalize(self, schema: type[T], instruction: str,
                 json_schema: dict[str, Any] | None = None) -> tuple[T, int, int]:
        t = self.types
        _append_user_text(self.contents, lambda c: c.role, lambda s: t.Part.from_text(text=s),
                          lambda c, p: c.parts.append(p), lambda p: t.Content(role="user", parts=[p]),
                          instruction)
        resp = self._generate(t.GenerateContentConfig(
            system_instruction=self.system, tools=[self.tool],
            tool_config=t.ToolConfig(function_calling_config=t.FunctionCallingConfig(mode="NONE")),
            response_mime_type="application/json",
            response_json_schema=json_schema or inline_refs(schema.model_json_schema())))
        tin, tout = self._usage(resp)
        return schema.model_validate_json(resp.text or ""), tin, tout


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, api_key: str | None, model: str, client: Any = None, max_tokens: int = 4096):
        if client is None:
            if not api_key:
                raise ConfigError("ANTHROPIC_API_KEY is not set.")
            try:
                import anthropic
            except ImportError as exc:
                raise ConfigError("LLM_PROVIDER=anthropic needs: pip install 'bioagent[anthropic]'") from exc
            client = anthropic.Anthropic(api_key=api_key, max_retries=4)   # SDK retries 429/5xx itself
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def session(self, system: str, user: str, tools: list[ToolSpec]) -> _AnthropicSession:
        return _AnthropicSession(self, system, user, tools)


class _AnthropicSession:
    def __init__(self, llm: AnthropicLLM, system: str, user: str, tools: list[ToolSpec]):
        self.llm, self.system = llm, system
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": user}]}]
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]

    def _create(self, **kw: Any) -> Any:
        import anthropic

        try:
            return self.llm.client.messages.create(model=self.llm.model, max_tokens=self.llm.max_tokens,
                                                   system=self.system, messages=self.messages, **kw)
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

    @staticmethod
    def _usage(resp: Any) -> tuple[int, int]:
        u = getattr(resp, "usage", None)
        return (getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0) if u else (0, 0)

    def next(self) -> LLMTurn:
        resp = self._create(tools=self.tools)
        self.messages.append({"role": "assistant", "content": resp.content})
        calls = [ToolRequest(id=b.id, name=b.name, args=dict(b.input or {}))
                 for b in resp.content if getattr(b, "type", "") == "tool_use"]
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        tin, tout = self._usage(resp)
        return LLMTurn(text=text, tool_calls=calls, input_tokens=tin, output_tokens=tout)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.id, "content": json.dumps(r.content)} for r in results]})

    def finalize(self, schema: type[T], instruction: str,
                 json_schema: dict[str, Any] | None = None) -> tuple[T, int, int]:
        _append_user_text(self.messages, lambda m: m["role"], lambda s: {"type": "text", "text": s},
                          lambda m, b: m["content"].append(b), lambda b: {"role": "user", "content": [b]},
                          instruction)
        submit = {"name": "submit_dossier", "description": "Submit the final structured dossier.",
                  "input_schema": json_schema or inline_refs(schema.model_json_schema())}
        resp = self._create(tools=[*self.tools, submit], tool_choice={"type": "tool", "name": "submit_dossier"})
        block = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
        if block is None:
            raise LLMError("model did not return the submit_dossier tool call")
        tin, tout = self._usage(resp)
        return schema.model_validate(block.input), tin, tout


def make_llm(settings: Settings) -> LLM:
    """Provider is chosen by LLM_PROVIDER (default gemini, the key this repo already uses)."""
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
