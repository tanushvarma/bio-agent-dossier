"""Conversational bio-research agent with Gemini function calling."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from google.genai import types

from gemini_client import DEFAULT_MODEL, get_client
from state import AgentState
from tools import execute_tool

EventCallback = Callable[[dict[str, Any]], None]

SYSTEM_PROMPT = """You are Bio-Agent, an open-source conversational biomedical research assistant.

Your mission: help users discover biological targets, mine literature, extract experimental
methodology, and produce executive research dossiers with protocol conflict analysis.

Behavior:
- Be conversational, clear, and scientifically precise.
- When the user asks about a gene/target or wants a dossier, use your tools proactively.
- Typical dossier workflow: fetch_gene_profile (if gene-like) -> search_literature ->
  extract_methodologies -> synthesize_dossier.
- If NCBI lookup fails, continue with literature search using the user's topic.
- If literature search returns few papers, broaden the query and try again once.
- After tools complete, summarize findings for the user in plain language.
- When a dossier is generated, highlight key insights and protocol conflicts.
- For general biomedical questions without needing live data, answer directly without tools.
- Never invent paper titles, DOIs, or experimental parameters not returned by tools.
"""

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="fetch_gene_profile",
        description="Fetch NCBI gene metadata (symbol, aliases, map location, summary) for a human gene symbol.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "gene_symbol": types.Schema(
                    type=types.Type.STRING,
                    description="Gene symbol such as VEGFA, TP53, EGFR",
                ),
            },
            required=["gene_symbol"],
        ),
    ),
    types.FunctionDeclaration(
        name="search_literature",
        description="Search Semantic Scholar for recent open-access papers on a target or topic.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="Search query, e.g. VEGFA or VEGFA hydrogel angiogenesis",
                ),
                "limit": types.Schema(
                    type=types.Type.INTEGER,
                    description="Number of papers to retrieve (3-5)",
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="extract_methodologies",
        description="Extract structured experimental parameters from papers already in agent state.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="synthesize_dossier",
        description="Generate an executive literature dossier from gathered gene data and papers.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "target": types.Schema(
                    type=types.Type.STRING,
                    description="Target name for the dossier heading (optional if already set)",
                ),
            },
        ),
    ),
]


def _emit(callback: EventCallback | None, event: dict[str, Any]) -> None:
    if callback:
        callback(event)


def _history_to_contents(history: list[dict[str, str]]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in history[-12:]:
        role = "user" if message["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=message["content"])])
        )
    return contents


class BioResearchAgent:
    """Gemini-powered conversational agent with tool use."""

    def __init__(self, *, model: str | None = None, max_steps: int = 8) -> None:
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.client = get_client()

    def run(
        self,
        user_message: str,
        *,
        state: AgentState,
        history: list[dict[str, str]] | None = None,
        on_event: EventCallback | None = None,
    ) -> str:
        history = history or []
        contents = _history_to_contents(history)
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        state_context = json.dumps(state.to_context())
        system_instruction = (
            f"{SYSTEM_PROMPT}\n\nCurrent agent state: {state_context}"
        )

        _emit(on_event, {"type": "status", "message": "Thinking..."})

        for step in range(self.max_steps):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
                    temperature=0.3,
                ),
            )

            if not response.candidates:
                break

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                break

            function_calls = [
                part.function_call
                for part in candidate.content.parts
                if part.function_call
            ]

            if not function_calls:
                text = (response.text or "").strip()
                if text:
                    _emit(on_event, {"type": "complete", "message": text})
                    return text
                break

            contents.append(candidate.content)

            response_parts: list[types.Part] = []
            for call in function_calls:
                name = call.name or "unknown"
                args = dict(call.args) if call.args else {}
                _emit(on_event, {"type": "tool_start", "tool": name, "args": args})

                result = execute_tool(state, name, args)
                _emit(on_event, {"type": "tool_done", "tool": name, "result": result})

                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response=result,
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=response_parts))
            system_instruction = (
                f"{SYSTEM_PROMPT}\n\nCurrent agent state: {json.dumps(state.to_context())}"
            )
            _emit(on_event, {"type": "status", "message": f"Planning step {step + 2}..."})

        fallback = (
            "I gathered some information but couldn't finish a full response. "
            "Please try rephrasing your request or ask me to generate the dossier."
        )
        if state.dossier:
            fallback = state.dossier["dossier_markdown"]
        _emit(on_event, {"type": "complete", "message": fallback})
        return fallback
