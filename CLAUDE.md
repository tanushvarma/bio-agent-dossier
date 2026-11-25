# Bio-Agent Dossier (`bio-agent-dossier`)

Conversational biomedical research agent (Python 3.11+, Streamlit, Gemini function calling) that pulls human gene profiles from NCBI Entrez, mines open-access literature from Semantic Scholar, extracts structured experimental methodology, and synthesizes evidence-grounded Markdown Executive Dossiers.

## Key Commands

- Launch app: `streamlit run app.py`
- Smoke-test modules directly:
  - `python ncbi_extractor.py VEGFA`
  - `python paper_miner.py "VEGFA hydrogel"`
  - `python dossier_synthesizer.py VEGFA`
- Lint & test (when configured): `ruff check .` and `pytest -q`

## Repo Rules

### Style (Non-Negotiable)
- Write idiomatic, flat Python 3.11+. Calibrate complexity to the problem: prefer plain functions and Pydantic/dataclass models over ABCs, factories, or multi-layer wrapper classes.
- Comments: near-zero. NEVER write narrating comments (`# Step 1:`, `# Validate input`, `# Call API`) or redundant docstrings that restate the function signature. Only comment non-obvious invariants, rate limits, or biological/API quirks.
- No `# TODO`, `# FIXME`, or placeholder stubs. Implement it or omit it.
- No broad `except Exception: pass` or silent fallbacks. Catch specific exceptions and surface structured errors.
- No emojis anywhere in code, logs, CLI output, or README. Use standard `logging`.
- Never disable SSL verification (`verify=False`).

### Token & Edit Discipline
- Never read full files when `rg` or targeted line ranges suffice.
- Keep diffs surgical; do not rewrite working code outside the active phase.
- Run `ruff check .` and `pytest -q` before declaring any phase done when available. Fix all failures yourself.

### External APIs & Secrets
- Environment variables live in gitignored `.env`: `GEMINI_API_KEY` (required), `GEMINI_MODEL` (default `gemini-flash-latest`), `NCBI_EMAIL`, `NCBI_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`.
- NCBI E-utilities: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/` (respect 3 req/s limit without `NCBI_API_KEY`; scoped to `Homo sapiens[Organism]`).
- Semantic Scholar Graph API: `https://api.semanticscholar.org/graph/v1/paper/search` (exponential backoff on HTTP 429).
- Planned / upstream-referenced APIs (see `origin/cursor/conversational-bio-agent`): Open Targets GraphQL (`https://api.platform.opentargets.org/api/v4/graphql`) and Europe PMC REST (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`).

## Agent skills

### Issue tracker

Follow `docs/agents/issue-tracker.md` for how issues and specs are managed in this repo (local markdown under `.scratch/`; swap to GitHub Issues after running `gh auth login`).

### Triage labels

Follow `docs/agents/triage-labels.md` for the mapping between skill triage roles and this repo's tracker labels.

### Domain docs

Follow `docs/agents/domain.md` for how to read and use `GLOSSARY.md` and `docs/adr/` when exploring or changing this codebase.
