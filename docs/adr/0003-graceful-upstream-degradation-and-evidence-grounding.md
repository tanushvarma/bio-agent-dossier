---
status: accepted
---

# Graceful Upstream Degradation and Evidence-Only Grounding

When either NCBI Entrez or Semantic Scholar fails (rate limits, missing symbol, network timeout) or per-paper extraction returns `None`, the tool surfaces a structured error and allows dossier synthesis to proceed on whatever evidence was gathered, while the synthesizer prompt strictly forbids fabricating numbers, citations, or parameters. Biomedical research APIs frequently throttle unauthenticated clients; halting the entire workflow on a partial failure wastes already-fetched data, whereas allowing ungrounded fill-in would produce scientifically dangerous hallucinations.

## Consequences

- Every Executive Dossier must still render all six canonical sections and explicitly flag unavailable sources as Research Gaps.
- Semantic Scholar requests enforce exponential backoff on HTTP 429 and advise setting `SEMANTIC_SCHOLAR_API_KEY` on exhaustion.
