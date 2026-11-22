"""Module 1: NCBI Entrez gene data fetcher via BioPython."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from Bio import Entrez
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# NCBI requires a contact email for Entrez requests.
Entrez.email = os.getenv("NCBI_EMAIL", "bio-agent-dossier@example.com")

_ncbi_api_key = os.getenv("NCBI_API_KEY")
if _ncbi_api_key:
    Entrez.api_key = _ncbi_api_key


def _text(element: ET.Element | None, default: str = "") -> str:
    if element is None or element.text is None:
        return default
    return element.text.strip()


def _parse_gene_xml(xml_text: str, gene_symbol: str) -> dict[str, Any]:
    """Parse NCBI Gene XML into a structured dictionary."""
    root = ET.fromstring(xml_text)
    # Avoid XPath paths containing hyphens (e.g. Entrezgene-Set), which ElementTree
    # interprets as subtraction rather than a literal tag name.
    gene_node = root.find("Entrezgene")
    if gene_node is None:
        for child in root.iter():
            if child.tag == "Entrezgene":
                gene_node = child
                break
    if gene_node is None:
        raise ValueError(f"No Entrezgene record found for '{gene_symbol}'.")

    gene_id = _text(gene_node.find("Entrezgene_track-info/Gene-track/Gene-track_geneid"))
    description = _text(gene_node.find("Entrezgene_gene/Gene-ref/Gene-ref_desc"))
    official_symbol = _text(
        gene_node.find("Entrezgene_gene/Gene-ref/Gene-ref_locus")
    ) or gene_symbol

    aliases: list[str] = []
    for synonym in gene_node.findall(
        "Entrezgene_gene/Gene-ref/Gene-ref_syn/Gene-ref_syn_E"
    ):
        alias = _text(synonym)
        if alias and alias not in aliases:
            aliases.append(alias)

    map_location = ""
    location_node = gene_node.find("Entrezgene_location")
    if location_node is not None:
        for element in location_node.iter():
            if element.tag in {"Maps_display-str", "Gene-ref_maploc"}:
                map_location = _text(element)
                if map_location:
                    break

    if not map_location:
        for element in gene_node.iter():
            if element.tag == "Gene-ref_maploc":
                map_location = _text(element)
                break

    summary = ""
    summary_node = gene_node.find("Entrezgene_summary")
    if summary_node is not None:
        summary = _text(summary_node)

    if not summary:
        summary_parts: list[str] = []
        for comment in gene_node.findall("Entrezgene_summary/Gene-commentary"):
            label = _text(comment.find("Gene-commentary_label"))
            text = _text(comment.find("Gene-commentary_text"))
            if text:
                if label:
                    summary_parts.append(f"{label}: {text}")
                else:
                    summary_parts.append(text)
        summary = " ".join(summary_parts).strip()

    if not summary:
        summary = description

    return {
        "gene_symbol": official_symbol,
        "gene_id": gene_id,
        "description": description,
        "aliases": aliases,
        "map_location": map_location,
        "summary": summary,
    }


def fetch_gene_data(gene_symbol: str) -> dict[str, Any]:
    """
    Fetch gene metadata from NCBI Entrez for a given gene symbol.

    Returns a dict with keys: gene_symbol, gene_id, description, aliases,
    map_location, and summary.
    """
    if not gene_symbol or not gene_symbol.strip():
        raise ValueError("gene_symbol must be a non-empty string.")

    query = gene_symbol.strip()

    try:
        search_handle = Entrez.esearch(
            db="gene",
            term=f"{query}[Gene Name] AND Homo sapiens[Organism]",
            retmax=1,
        )
        search_record = Entrez.read(search_handle)
        search_handle.close()
    except Exception as exc:
        raise RuntimeError(f"NCBI esearch failed for '{query}': {exc}") from exc

    id_list = search_record.get("IdList", [])
    if not id_list:
        raise ValueError(f"No NCBI Gene record found for symbol '{query}'.")

    gene_id = id_list[0]

    try:
        fetch_handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
        xml_text = fetch_handle.read()
        fetch_handle.close()
    except Exception as exc:
        raise RuntimeError(f"NCBI efetch failed for gene ID '{gene_id}': {exc}") from exc

    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8")

    return _parse_gene_xml(xml_text, query)


if __name__ == "__main__":
    import json
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "VEGFA"
    print(f"Fetching NCBI data for: {symbol}\n")
    data = fetch_gene_data(symbol)
    print(json.dumps(data, indent=2))
