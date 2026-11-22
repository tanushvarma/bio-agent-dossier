"""Provider adapters against fake SDK clients that return real SDK response objects (no network)."""
from __future__ import annotations

import pytest

from bioagent.clients.llm import AnthropicLLM, GeminiLLM, ToolResult, ToolSpec
from bioagent.schemas import DossierDraft

SPEC = [ToolSpec(name="search_literature", description="d",
                 parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
FINAL = ('{"summary":"ok","drug_claims":[],"mechanism_claims":'
         '[{"statement":"MET drives resistance.","evidence_ids":["30000002"]}]}')


def test_gemini_session_round_trip():
    types = pytest.importorskip("google.genai.types")

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append(([c.role for c in contents], config))
            if len(self.calls) == 1:
                d = {"candidates": [{"content": {"role": "model", "parts": [
                        {"function_call": {"name": "search_literature", "args": {"query": "EGFR"}},
                         "thought_signature": "c2ln"}]}}],
                     "usage_metadata": {"prompt_token_count": 120, "candidates_token_count": 15,
                                        "thoughts_token_count": 5}}
            else:
                d = {"candidates": [{"content": {"role": "model", "parts": [{"text": FINAL}]}}],
                     "usage_metadata": {"prompt_token_count": 300, "candidates_token_count": 40}}
            return types.GenerateContentResponse.model_validate(d)

    client = type("C", (), {"models": Models()})()
    s = GeminiLLM(None, "gemini-3.5-flash-lite", client=client).session("sys", "user", SPEC)
    turn = s.next()
    assert [c.name for c in turn.tool_calls] == ["search_literature"]
    assert (turn.input_tokens, turn.output_tokens) == (120, 20)          # thinking tokens count as output
    s.add_tool_results([ToolResult(id=turn.tool_calls[0].id, name="search_literature", content={"n": 1})])
    draft, tin, tout = s.finalize(DossierDraft, "final")
    assert draft.mechanism_claims[0].evidence_ids == ["30000002"] and (tin, tout) == (300, 40)
    roles, config = client.models.calls[-1]
    assert roles == ["user", "model", "user"]                           # no back-to-back user turns
    assert s.contents[1].parts[0].thought_signature                     # echoed back intact
    assert config.response_json_schema and config.tool_config.function_calling_config.mode == "NONE"


def test_anthropic_session_round_trip():
    anthropic_types = pytest.importorskip("anthropic.types")

    class Messages:
        def __init__(self):
            self.kwargs = []

        def create(self, **kw):
            self.kwargs.append(kw)
            base = {"id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                    "stop_reason": "tool_use", "usage": {"input_tokens": 200, "output_tokens": 30}}
            if len(self.kwargs) == 1:
                base["content"] = [{"type": "text", "text": "Searching."},
                                   {"type": "tool_use", "id": "tu_1", "name": "search_literature",
                                    "input": {"query": "EGFR"}}]
            else:
                base["content"] = [{"type": "tool_use", "id": "tu_2", "name": "submit_dossier",
                                    "input": {"summary": "ok", "drug_claims": [], "mechanism_claims": []}}]
            return anthropic_types.Message.model_validate(base)

    client = type("C", (), {"messages": Messages()})()
    s = AnthropicLLM(None, "claude-haiku-5-5", client=client).session("sys", "user", SPEC)
    turn = s.next()
    assert turn.text == "Searching." and turn.tool_calls[0].id == "tu_1"
    s.add_tool_results([ToolResult(id="tu_1", name="search_literature", content={"status": "success"})])
    draft, _, _ = s.finalize(DossierDraft, "final")
    last = client.messages.kwargs[-1]
    assert draft.summary == "ok"
    assert last["tool_choice"] == {"type": "tool", "name": "submit_dossier"}
    assert [m["role"] for m in last["messages"]] == ["user", "assistant", "user"]
    assert [b["type"] for b in last["messages"][-1]["content"]] == ["tool_result", "text"]
