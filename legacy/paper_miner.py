"""Module 2: Semantic Scholar literature mining + Gemini methodology extraction."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from gemini_client import generate_text

load_dotenv(Path(__file__).resolve().parent / ".env")

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PAPER_FIELDS = (
    "title,abstract,year,authors,url,isOpenAccess,openAccessPdf,externalIds,citationCount"
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

METHODOLOGY_SCHEMA = {
    "cell_lines": [],
    "biomaterials": [],
    "centrifuge_speeds": [],
    "curing_times": [],
    "concentrations": [],
    "temperatures": [],
    "outcomes": [],
    "other_parameters": [],
}


def _get_gemini_api_key() -> str:
    from gemini_client import get_gemini_api_key

    return get_gemini_api_key()


def _semantic_scholar_headers() -> dict[str, str]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        return {"x-api-key": api_key}
    return {}


def _get_with_retry(
    url: str,
    *,
    params: dict[str, Any],
    timeout: int = 30,
    max_retries: int = 5,
) -> requests.Response:
    headers = _semantic_scholar_headers()
    last_response: requests.Response | None = None

    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        last_response = response

        if response.status_code == 429:
            if attempt < max_retries - 1:
                time.sleep(2**attempt * 10)
                continue
            break

        try:
            response.raise_for_status()
        except requests.RequestException:
            if attempt >= max_retries - 1:
                raise
            time.sleep(2**attempt * 2)
            continue

        return response

    if last_response is not None and last_response.status_code == 429:
        raise RuntimeError(
            "Semantic Scholar rate limit reached. Wait a minute and retry, or set "
            "SEMANTIC_SCHOLAR_API_KEY in .env for higher limits."
        )

    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("Semantic Scholar request failed after retries.")


def _paper_authors(authors: list[dict[str, Any]] | None) -> list[str]:
    if not authors:
        return []
    return [author.get("name", "") for author in authors if author.get("name")]


def _normalize_paper(raw: dict[str, Any]) -> dict[str, Any]:
    open_access_pdf = raw.get("openAccessPdf") or {}
    external_ids = raw.get("externalIds") or {}

    return {
        "paper_id": raw.get("paperId"),
        "title": raw.get("title") or "Untitled",
        "abstract": raw.get("abstract") or "",
        "year": raw.get("year"),
        "authors": _paper_authors(raw.get("authors")),
        "url": raw.get("url"),
        "is_open_access": bool(raw.get("isOpenAccess")),
        "open_access_pdf_url": open_access_pdf.get("url"),
        "doi": external_ids.get("DOI"),
        "pmid": external_ids.get("PubMed"),
        "citation_count": raw.get("citationCount"),
    }


def search_papers(target: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search Semantic Scholar for recent open-access papers related to a target.

    Returns normalized paper metadata dictionaries.
    """
    if not target or not target.strip():
        raise ValueError("target must be a non-empty string.")

    query = target.strip()
    params = {
        "query": query,
        "limit": min(max(limit, 1), 10),
        "fields": PAPER_FIELDS,
    }

    try:
        response = _get_with_retry(SEMANTIC_SCHOLAR_SEARCH_URL, params=params)
        payload = response.json()
    except (requests.RequestException, RuntimeError) as exc:
        raise RuntimeError(f"Semantic Scholar search failed for '{query}': {exc}") from exc

    papers = payload.get("data") or []
    normalized = [_normalize_paper(paper) for paper in papers]

    open_access = [paper for paper in normalized if paper["is_open_access"]]
    if open_access:
        return open_access[:limit]

    return normalized[:limit]


def _extract_json_block(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating fenced code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response JSON must be an object.")
    return parsed


def _build_extraction_prompt(paper: dict[str, Any]) -> str:
    return f"""
You are a biomedical methods extraction assistant.

Extract experimental methodology details from the paper metadata below.
Return ONLY valid JSON matching this exact schema (use empty lists when unknown):

{json.dumps(METHODOLOGY_SCHEMA, indent=2)}

Rules:
- Use concise string entries in each list.
- Do not invent values that are not supported by the text.
- If a category has no evidence, return an empty list for that key.
- Do not include markdown or commentary outside the JSON object.

Paper title: {paper.get("title", "")}
Year: {paper.get("year", "unknown")}
Abstract:
{paper.get("abstract") or "No abstract available."}
""".strip()


def extract_methodology(paper: dict[str, Any]) -> dict[str, Any] | None:
    """
    Use Gemini to extract structured experimental parameters from a paper abstract.

    Returns methodology JSON or None if extraction fails.
    """
    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        return None

    try:
        text = generate_text(
            _build_extraction_prompt(paper),
            model=GEMINI_MODEL,
            temperature=0.1,
            json_mode=True,
        )
        return _extract_json_block(text)
    except Exception:
        return None


def mine_papers(target: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search for papers and enrich each with Gemini-extracted methodology.

    Failed extractions are skipped cleanly; the paper is still returned with
    methodology set to None.
    """
    papers = search_papers(target, limit=limit)
    enriched: list[dict[str, Any]] = []

    for paper in papers:
        record = dict(paper)
        try:
            record["methodology"] = extract_methodology(paper)
        except Exception:
            record["methodology"] = None
        enriched.append(record)

    return enriched


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "VEGFA hydrogel"
    print(f"Mining papers for: {query}\n")

    try:
        results = search_papers(query, limit=3)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1) from exc

    print(f"Found {len(results)} paper(s) from Semantic Scholar:\n")
    for idx, paper in enumerate(results, start=1):
        print(f"{idx}. {paper['title']} ({paper.get('year', 'n/a')})")
        print(f"   Open access: {paper['is_open_access']}")
        print(f"   URL: {paper.get('url')}\n")

    if os.getenv("GEMINI_API_KEY"):
        print("Running Gemini methodology extraction on first paper...\n")
        if results:
            methodology = extract_methodology(results[0])
            print(json.dumps(methodology, indent=2))
    else:
        print("Set GEMINI_API_KEY in .env to test methodology extraction.")
