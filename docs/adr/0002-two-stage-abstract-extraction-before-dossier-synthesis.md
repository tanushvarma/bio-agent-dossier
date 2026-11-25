---
status: accepted
---

# Two-Stage Methodology Extraction Before Dossier Synthesis

We run a dedicated low-temperature (`0.1`) JSON extraction pass per Mined Paper to populate a fixed 8-field `Extracted Methodology` schema before compiling the six-section Executive Dossier, rather than feeding raw abstracts straight into one synthesis prompt. Separating structured parameter extraction from narrative synthesis produces consistent cross-paper comparison tables and makes per-paper extraction failures isolatable without corrupting the rest of the literature set.

## Consequences

- Extraction relies on paper abstracts (not full-text PDFs), so parameters unreported in abstracts legitimately stay empty lists and surface as Research Gaps in the Executive Dossier.
- Changing the methodology categories requires updating `METHODOLOGY_SCHEMA` in `paper_miner.py` and the comparison instructions in `dossier_synthesizer.py` together.
