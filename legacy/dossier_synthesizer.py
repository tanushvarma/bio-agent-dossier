"""Module 3: Gemini dossier synthesis and protocol conflict detection."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from gemini_client import generate_text

load_dotenv(Path(__file__).resolve().parent / ".env")

SYNTHESIS_MODEL = os.getenv("GEMINI_SYNTHESIS_MODEL", os.getenv("GEMINI_MODEL", "gemini-flash-latest"))


def _paper_label(paper: dict[str, Any], index: int) -> str:
    year = paper.get("year") or "n/a"
    title = paper.get("title") or "Untitled"
    return f"Paper {index}: {title} ({year})"


def _format_methodology(methodology: dict[str, Any] | None) -> str:
    if not methodology:
        return "No structured methodology extracted."
    return json.dumps(methodology, indent=2)


def _build_synthesis_prompt(
    gene_data: dict[str, Any] | None,
    papers: list[dict[str, Any]],
    target: str,
) -> str:
    gene_block = json.dumps(gene_data, indent=2) if gene_data else "No NCBI gene profile available."

    paper_blocks: list[str] = []
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.get("authors") or []) or "Unknown"
        paper_blocks.append(
            f"""{_paper_label(paper, index)}
Authors: {authors}
URL: {paper.get("url") or "N/A"}
Abstract: {paper.get("abstract") or "No abstract available."}
Extracted methodology JSON:
{_format_methodology(paper.get("methodology"))}
"""
        )

    papers_block = "\n\n".join(paper_blocks) if paper_blocks else "No papers were retrieved."

    return f"""You are a senior biomedical research analyst writing an Executive Literature Review dossier.

Target query: {target}

Use ONLY the evidence provided below. Do not invent citations, parameters, or experimental details.

=== NCBI GENE BASELINE ===
{gene_block}

=== MINED LITERATURE + EXTRACTED METHODOLOGIES ===
{papers_block}

Write a ~2-page executive dossier in Markdown using EXACTLY these section headings:

# Executive Literature Review: {target}

## 1. Executive Summary
Provide a concise overview of the target and the state of recent literature.

## 2. Biological Target Profile
Summarize NCBI baseline biology (function, aliases, map location, disease relevance).

## 3. Literature Landscape
Summarize the retrieved papers, their focus, and key findings.

## 4. Consolidated Methodology Overview
Synthesize experimental approaches across papers (models, biomaterials, key parameters, outcomes).
Cross-reference parameters by paper label (e.g., Paper 1, Paper 2).

## 5. Protocol Conflicts
Explicitly compare experimental parameters across papers.
Flag discrepancies in cell lines, biomaterials, concentrations, speeds, curing times, temperatures, or outcomes.
If no conflicts are supported by the data, state: "No material protocol conflicts detected in the available abstracts."

## 6. Research Gaps and Recommendations
Identify gaps and propose next experimental steps.

Formatting rules:
- Be precise and evidence-grounded.
- Use bullet lists where helpful.
- Keep the total length near two printed pages.
- Do not include a references section beyond inline paper labels and URLs already provided.
""".strip()


def synthesize_dossier(
    gene_data: dict[str, Any] | None,
    papers: list[dict[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    """
    Combine NCBI baseline data and mined paper methodologies into an executive dossier.

    Returns a dict with target, dossier_markdown, model, and generated_at keys.
    """
    if not target or not target.strip():
        raise ValueError("target must be a non-empty string.")

    prompt = _build_synthesis_prompt(gene_data, papers, target.strip())

    try:
        dossier_markdown = generate_text(
            prompt,
            model=SYNTHESIS_MODEL,
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini dossier synthesis failed: {exc}") from exc

    return {
        "target": target.strip(),
        "dossier_markdown": dossier_markdown,
        "model": SYNTHESIS_MODEL,
        "paper_count": len(papers),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_target_dossier(
    target: str,
    *,
    paper_limit: int = 5,
    include_gene_profile: bool = True,
) -> dict[str, Any]:
    """
    End-to-end helper: fetch gene data, mine papers, and synthesize the dossier.

    Individual upstream failures are handled defensively where possible.
    """
    from ncbi_extractor import fetch_gene_data
    from paper_miner import mine_papers

    gene_data: dict[str, Any] | None = None
    if include_gene_profile:
        try:
            gene_data = fetch_gene_data(target)
        except Exception:
            gene_data = None

    try:
        papers = mine_papers(target, limit=paper_limit)
    except Exception:
        papers = []

    return synthesize_dossier(gene_data, papers, target=target)


def _sample_gene_data() -> dict[str, Any]:
    return {
        "gene_symbol": "VEGFA",
        "gene_id": "7422",
        "description": "vascular endothelial growth factor A",
        "aliases": ["VEGF", "VPF"],
        "map_location": "6p21.1",
        "summary": (
            "Angiogenesis-related growth factor implicated in vascular development "
            "and tumor biology."
        ),
    }


def _sample_papers() -> list[dict[str, Any]]:
    return [
        {
            "title": "Hydrogel delivery of VEGFA in endothelial culture",
            "year": 2024,
            "authors": ["A. Researcher", "B. Scientist"],
            "url": "https://example.org/paper-1",
            "abstract": (
                "Human umbilical vein endothelial cells (HUVEC) were cultured in "
                "2% alginate hydrogel and treated with 50 ng/mL VEGFA for 72 h."
            ),
            "methodology": {
                "cell_lines": ["HUVEC"],
                "biomaterials": ["2% alginate hydrogel"],
                "centrifuge_speeds": [],
                "curing_times": ["72 h"],
                "concentrations": ["50 ng/mL VEGFA"],
                "temperatures": ["37 C"],
                "outcomes": ["Increased tube formation"],
                "other_parameters": [],
            },
        },
        {
            "title": "VEGFA stimulation in microvascular endothelial models",
            "year": 2023,
            "authors": ["C. Analyst"],
            "url": "https://example.org/paper-2",
            "abstract": (
                "Human microvascular endothelial cells (HMEC-1) were embedded in "
                "collagen hydrogel and exposed to 20 ng/mL VEGFA for 48 h at 37 C."
            ),
            "methodology": {
                "cell_lines": ["HMEC-1"],
                "biomaterials": ["collagen hydrogel"],
                "centrifuge_speeds": [],
                "curing_times": ["48 h"],
                "concentrations": ["20 ng/mL VEGFA"],
                "temperatures": ["37 C"],
                "outcomes": ["Enhanced migration"],
                "other_parameters": [],
            },
        },
    ]


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "VEGFA"
    print(f"Synthesizing dossier for: {query}\n")

    if os.getenv("GEMINI_API_KEY"):
        result = synthesize_dossier(_sample_gene_data(), _sample_papers(), target=query)
        print(result["dossier_markdown"])
        print(f"\n--- Generated with {result['model']} ({result['paper_count']} papers) ---")
    else:
        print("GEMINI_API_KEY not set. Showing sample prompt preview instead:\n")
        preview = _build_synthesis_prompt(_sample_gene_data(), _sample_papers(), query)
        print(preview[:1500])
        print("\n... [truncated] ...\n")
        print("Add GEMINI_API_KEY to .env to generate the full dossier.")
