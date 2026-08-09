"""Conversational Streamlit UI for the Bio-Agent research assistant."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent import BioResearchAgent
from state import AgentState

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

REQUIRED_KEYS = ("GEMINI_API_KEY",)
OPTIONAL_KEYS = (
    "NCBI_EMAIL",
    "NCBI_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_SYNTHESIS_MODEL",
)

EXAMPLE_PROMPTS = [
    "Build a dossier for VEGFA",
    "What does NCBI say about TP53?",
    "Find recent open-access papers on collagen hydrogel angiogenesis",
    "Compare experimental methods for VEGFA in hydrogel studies",
]


def _load_runtime_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return

    for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
        if key in secrets and not os.getenv(key):
            os.environ[key] = str(secrets[key])


def _missing_keys() -> list[str]:
    return [key for key in REQUIRED_KEYS if not os.getenv(key)]


def _gemini_key_problem() -> str | None:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        return "GEMINI_API_KEY is not set."
    lowered = key.lower()
    if "your_" in lowered or "paste" in lowered or lowered.endswith("_here"):
        return "GEMINI_API_KEY is still placeholder text."
    if len(key) < 20:
        return "GEMINI_API_KEY looks too short."
    return None


def _init_session() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict[str, str]] = []
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = AgentState()
    if "tool_events" not in st.session_state:
        st.session_state.tool_events: list[dict[str, Any]] = []


def _render_sidebar() -> None:
    st.header("Bio-Agent")
    st.caption("Open-source conversational research assistant")

    st.markdown("**API Status**")
    st.write(f"Gemini: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
    st.write(f"NCBI key: {'✅' if os.getenv('NCBI_API_KEY') else 'email only'}")

    st.divider()
    st.markdown("**Session State**")
    ctx = st.session_state.agent_state.to_context()
    st.json(ctx)

    if st.session_state.agent_state.dossier:
        dossier = st.session_state.agent_state.dossier
        st.download_button(
            "Download latest dossier",
            data=dossier["dossier_markdown"],
            file_name=f"{dossier['target'].replace(' ', '_')}_dossier.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_state = AgentState()
        st.session_state.tool_events = []
        st.rerun()

    st.divider()
    st.markdown("**Example prompts**")
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, use_container_width=True, key=f"example_{prompt[:20]}"):
            st.session_state.pending_prompt = prompt


def _render_tool_events(events: list[dict[str, Any]]) -> None:
    tool_events = [e for e in events if e.get("type") in {"tool_start", "tool_done"}]
    if not tool_events:
        return

    with st.expander("Agent tool activity", expanded=False):
        for event in tool_events:
            if event["type"] == "tool_start":
                st.markdown(f"**→ {event['tool']}** `{json.dumps(event.get('args', {}))}`")
            elif event["type"] == "tool_done":
                result = event.get("result", {})
                status = result.get("status", "unknown")
                summary = result.get("summary") or result.get("error") or status
                st.caption(f"✓ {event['tool']}: {summary}")


def _render_artifacts(state: AgentState) -> None:
    if state.gene_data:
        with st.expander("NCBI gene profile", expanded=False):
            st.json(state.gene_data)

    if state.papers:
        with st.expander(f"Mined papers ({len(state.papers)})", expanded=False):
            for index, paper in enumerate(state.papers, start=1):
                st.markdown(f"**{index}. {paper.get('title', 'Untitled')}**")
                if paper.get("methodology"):
                    st.code(json.dumps(paper["methodology"], indent=2), language="json")

    if state.dossier:
        with st.expander("Full executive dossier", expanded=True):
            st.markdown(state.dossier["dossier_markdown"])


def main() -> None:
    st.set_page_config(
        page_title="Bio-Agent",
        page_icon="🧬",
        layout="wide",
    )
    _load_runtime_secrets()
    _init_session()

    missing = _missing_keys()
    gemini_problem = _gemini_key_problem()

    with st.sidebar:
        _render_sidebar()

    st.title("Bio-Agent")
    st.markdown(
        "Open-source conversational agent for **target discovery**, "
        "**literature mining**, and **executive dossier synthesis**."
    )

    if missing:
        st.warning("Set `GEMINI_API_KEY` in `.env` to use the agent.")
    elif gemini_problem:
        st.error(gemini_problem)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool_events"):
                _render_tool_events(message["tool_events"])
            if message.get("show_artifacts"):
                _render_artifacts(st.session_state.agent_state)

    pending = st.session_state.pop("pending_prompt", None)
    user_input = st.chat_input("Ask about a gene, target, or research topic...")
    prompt = pending or user_input

    if prompt and not missing and not gemini_problem:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        events: list[dict[str, Any]] = []
        status_box = st.empty()

        def on_event(event: dict[str, Any]) -> None:
            events.append(event)
            if event.get("type") == "status":
                status_box.info(event.get("message", "Working..."))
            elif event.get("type") == "tool_start":
                status_box.info(f"Using tool: **{event.get('tool')}**")

        with st.chat_message("assistant"):
            with st.spinner("Agent working..."):
                agent = BioResearchAgent()
                reply = agent.run(
                    prompt,
                    state=st.session_state.agent_state,
                    history=st.session_state.chat_history[:-1],
                    on_event=on_event,
                )
            status_box.empty()
            st.markdown(reply)
            _render_tool_events(events)
            _render_artifacts(st.session_state.agent_state)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": reply,
                "tool_events": events,
                "show_artifacts": bool(st.session_state.agent_state.dossier),
            }
        )


if __name__ == "__main__":
    main()
