"""Tool implementations for the conversational bio-research agent."""

from __future__ import annotations

from typing import Any

from dossier_synthesizer import synthesize_dossier
from ncbi_extractor import fetch_gene_data
from paper_miner import extract_methodology, search_papers
from state import AgentState


def _ok(data: Any = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "success", "data": data}
    payload.update(extra)
    return payload


def _err(message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


def tool_fetch_gene_profile(state: AgentState, gene_symbol: str) -> dict[str, Any]:
    symbol = gene_symbol.strip().upper()
    try:
        gene_data = fetch_gene_data(symbol)
        state.gene_data = gene_data
        state.target = state.target or symbol
        return _ok(
            gene_data,
            summary=(
                f"Found {gene_data.get('gene_symbol')} (ID {gene_data.get('gene_id')}) "
                f"at {gene_data.get('map_location') or 'unknown location'}."
            ),
        )
    except Exception as exc:
        return _err(str(exc), suggestion="Try treating the input as a research topic instead.")


def tool_search_literature(
    state: AgentState,
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    q = query.strip()
    if not q:
        return _err("Query must not be empty.")

    state.queries_tried.append(q)
    state.target = state.target or q

    try:
        papers = search_papers(q, limit=max(3, min(limit, 5)))
        state.papers = papers
        titles = [p.get("title", "Untitled") for p in papers[:3]]
        return _ok(
            {"count": len(papers), "titles": titles},
            summary=f"Retrieved {len(papers)} paper(s).",
        )
    except Exception as exc:
        return _err(
            str(exc),
            suggestion="Wait briefly and retry, or add SEMANTIC_SCHOLAR_API_KEY to .env.",
        )


def tool_extract_methodologies(state: AgentState) -> dict[str, Any]:
    if not state.papers:
        return _err("No papers in state. Call search_literature first.")

    extracted = 0
    for paper in state.papers:
        try:
            paper["methodology"] = extract_methodology(paper)
            if paper["methodology"]:
                extracted += 1
        except Exception:
            paper["methodology"] = None

    return _ok(
        {"papers_processed": len(state.papers), "methodologies_extracted": extracted},
        summary=f"Extracted methodology from {extracted}/{len(state.papers)} papers.",
    )


def tool_synthesize_dossier(state: AgentState, target: str | None = None) -> dict[str, Any]:
    dossier_target = (target or state.target or "").strip()
    if not dossier_target:
        return _err("No target set. Ask the user which gene or topic to analyze.")

    try:
        dossier = synthesize_dossier(state.gene_data, state.papers, target=dossier_target)
        state.dossier = dossier
        state.target = dossier_target
        preview = dossier["dossier_markdown"][:500]
        return _ok(
            {
                "target": dossier_target,
                "model": dossier["model"],
                "paper_count": dossier["paper_count"],
                "preview": preview,
            },
            summary="Executive dossier generated.",
            dossier_markdown=dossier["dossier_markdown"],
        )
    except Exception as exc:
        return _err(str(exc))


def execute_tool(state: AgentState, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "fetch_gene_profile":
        result = tool_fetch_gene_profile(state, args.get("gene_symbol", ""))
    elif name == "search_literature":
        result = tool_search_literature(
            state,
            args.get("query", ""),
            int(args.get("limit", 5)),
        )
    elif name == "extract_methodologies":
        result = tool_extract_methodologies(state)
    elif name == "synthesize_dossier":
        result = tool_synthesize_dossier(state, args.get("target"))
    else:
        result = _err(f"Unknown tool: {name}")

    state.log_tool(name, args, result)
    return result
