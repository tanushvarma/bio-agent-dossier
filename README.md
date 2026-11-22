# bioagent

A bounded tool-calling agent that builds a short dossier for a human drug target (gene symbol in,
structured JSON out) from Open Targets, NCBI Gene and Europe PMC, and then checks every claim it
made against the identifiers those tools returned in the same run.

## Problem statement

Language models asked to summarise a drug target produce fluent text with drug names, mechanisms
and citations that are often wrong in ways that are hard to see: a drug that acts on a paralogue,
a PMID that does not exist, an alias silently mapped to the wrong gene. For target triage the
useful output is narrower than a free-text summary:

- the canonical identity of the target (Ensembl and NCBI Gene IDs), or an explicit refusal;
- drugs at late clinical stage with their mechanism, each tied to a ChEMBL ID;
- mechanistic statements, each tied to a PMID;
- a record of which API calls produced which IDs, and what the run cost.

bioagent constrains the model to four typed tools, caps the run at six model calls, forces the
final answer into a Pydantic schema, and runs a deterministic verifier that removes (or flags)
any claim whose evidence IDs or drug names did not come back from a tool in that run. It is
measured against a frozen benchmark of 50 target-drug pairs and 10 negative controls.

## Architecture

```
  gene symbol
      |
      v
 +----------------------------+   invalid, alias or ambiguous symbol
 | guardrails.preflight       |------------------------------------> status="abstained"
 | regex + resolve_target     |   (no LLM call is made)
 +----------------------------+
      | Ensembl ID + NCBI Gene ID
      v
 +----------------------------+  tool calls  +----------------------------------------+
 | agent.BioAgent             |------------->| tools.Toolbox                          |
 | <= 5 tool steps            |<-------------|  resolve_target                        |
 | +  1 structured finalize   |   results    |  fetch_opentargets_drugs -> OT GraphQL |
 | clients.llm: Gemini        |              |  fetch_ncbi_gene         -> E-utils    |
 |  (default) or Anthropic    |              |  search_literature       -> EuropePMC  |
 +----------------------------+              |  every call -> ToolTrace               |
      | DossierDraft (validated JSON)        +----------------------------------------+
      v                                        clients/_http.py: httpx timeouts, tenacity
 +----------------------------+                retry on 429/5xx, NCBI throttle,
 | guardrails.verify_citations|                SQLite response cache
 | + TokenBudget              |
 +----------------------------+
      |
      v
 RunResult = TargetDossier + Verification + ToolTrace + tokens + latency
      |
      +--> cli.py (bioagent run)    +--> app/streamlit_app.py    +--> evals/run_eval.py
```

| Path | Contents |
|---|---|
| `src/bioagent/schemas.py` | Pydantic v2 models: tool trace, claims, dossier, verification, run result |
| `src/bioagent/clients/` | Open Targets, NCBI, Europe PMC and LLM clients; shared HTTP layer and cache |
| `src/bioagent/tools.py` | The four tools, their argument schemas, and the executor that records the trace |
| `src/bioagent/agent.py` | The bounded loop |
| `src/bioagent/guardrails.py` | Pre-flight resolution, citation verifier, token budget |
| `evals/` | Dataset builder, frozen dataset, scorer, results |
| `app/streamlit_app.py` | UI only; imports `bioagent` |
| `legacy/` | The previous conversational prototype, kept for reference. Not installed, linted or tested |

Determinism boundaries: identity resolution, tool execution, verification and scoring are
deterministic; with the cache enabled, tool outputs are replayed byte-for-byte. The model's
choice of tool calls and its wording are not deterministic.

## Benchmark results

<!-- benchmark:begin -->
No results are committed yet: the dataset has to be built against the live APIs first (see
Evaluation below). A full run with `--update-readme` replaces this block with the generated table.
<!-- benchmark:end -->

Metric definitions (all computed in `evals/metrics.py`, no LLM-as-judge):

| Metric | Definition |
|---|---|
| Target resolution accuracy | Resolver's Ensembl ID (trace step 0) equals the expected ID |
| Drug precision / recall | Drugs named in the final dossier vs. Open Targets drugs at Phase 3 or approved for the target; macro mean over positive cases |
| Hallucination rate | Claims that failed verification / all claims the model emitted |
| Citation validity | Cited ChEMBL IDs and PMIDs returned by a tool in the same run / all cited IDs |
| Negative-control FP rate | Fake symbol not refused, or forbidden drug present in the final dossier, / negative controls |
| Latency p50 / p95, tokens, cost | Wall-clock per case; tokens from provider usage metadata; cost from `src/bioagent/pricing.py` |

## Quickstart

Requires Python 3.10+, a Gemini API key (or an Anthropic key with `LLM_PROVIDER=anthropic`) and a
contact email for NCBI E-utilities.

```bash
git clone https://github.com/tanushvarma/bio-agent-dossier.git
cd bio-agent-dossier
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,app]"
cp .env.example .env                                    # set GEMINI_API_KEY and NCBI_EMAIL
```

CLI:

```bash
bioagent run EGFR                  # RunResult JSON without the trace
bioagent run EGFR --with-trace     # include every tool and LLM call
bioagent run HER2                  # alias: abstains; suggests ERBB2 if Open Targets returns it
```

Exit codes: 0 ok, 1 error or configuration problem, 2 abstained, 3 token budget exceeded.

Streamlit:

```bash
streamlit run app/streamlit_app.py
```

Docker (secrets are passed at run time, never baked into the image):

```bash
docker build -t bioagent .
docker run --rm -p 8501:8501 --env-file .env -v bioagent-cache:/app/.cache bioagent
```

### Evaluation

```bash
python evals/build_dataset.py              # once; writes evals/dataset_50.jsonl (+ .meta.json), commit both
python evals/run_eval.py --limit 5         # smoke run, mixes positive and negative cases
python evals/run_eval.py --update-readme   # full run; writes evals/results/ and updates this README
```

`build_dataset.py` refuses to overwrite an existing dataset unless given `--force`.

### Tests

```bash
ruff check .
pytest -q              # all tests, no network
pytest -q -m smoke     # metric calculations on synthetic fixtures only
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | -, `gemini-3.5-flash-lite` | Gemini credentials and model |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | -, `claude-haiku-5-5` | Requires `pip install -e ".[anthropic]"` |
| `NCBI_EMAIL`, `NCBI_API_KEY` | required, optional | E-utilities policy; the key raises the limit from 3 to 10 req/s |
| `BIOAGENT_MAX_STEPS` | `6` | Hard cap on LLM calls per run (tool steps + finalize) |
| `BIOAGENT_MAX_TOKENS_PER_RUN` | `60000` | Input + output tokens; the run aborts cleanly above this |
| `BIOAGENT_CITATION_MODE` | `strip` | `strip` removes unverified claims, `flag` keeps them marked |
| `BIOAGENT_CACHE_ENABLED`, `BIOAGENT_CACHE_PATH` | `true`, `.cache/bioagent.sqlite` | HTTP response cache |

## Design tradeoffs and known limitations

- **Verification is referential, not semantic.** The verifier proves that every cited ID was
  returned by a tool in this run and that drug names match the ChEMBL record. It does not prove
  that a cited abstract supports the sentence citing it. A mechanism claim can cite a real,
  retrieved, irrelevant PMID and pass.
- **Exact-symbol resolution.** Aliases (`HER2`) and ambiguous symbols are refused with candidate
  suggestions instead of being mapped. This trades convenience for never producing a dossier
  about the wrong gene.
- **Fixed budget.** Six model calls and four tool calls per step bound cost and latency. Targets
  with large literatures get a shallow search.
- **Drug coverage.** The drug tool returns at most 25 drugs per target, while recall is measured
  against every Phase 3+ or approved drug. Recall is understated for targets with many drugs
  (EGFR, VEGFA).
- **Upstream schema drift.** Open Targets removed `Target.knownDrugs` in release 26.03; drugs now
  come from `drugAndClinicalCandidates`. Clinical-stage labels are mapped to Phase 3+/approved by
  `stage_rank` in `evals/build_dataset.py`; the labels actually seen are recorded in
  `evals/dataset_50.meta.json` and should be checked after each rebuild.
- **Cache has no expiry.** Repeat runs and evals replay identical API responses, which is the
  point, but data goes stale until `.cache/` is deleted or `BIOAGENT_CACHE_ENABLED=false`.
- **Single-run numbers.** Benchmark figures come from one run. Tool outputs are cached, but model
  output varies between runs.
- **Peer review is approximated** by Europe PMC records from MEDLINE (`SRC:MED`) with an abstract,
  excluding retraction notices.
- **Negative pair controls** pass a real symbol whose dossier must not name a drug that Open
  Targets does not link to the target. Because the verifier removes unreturned drugs, the
  headline FP rate is expected to be near zero; the harness also reports how often the model
  attempted such a claim before verification.
- **Cost estimates** exist only for models listed in `src/bioagent/pricing.py`; other models report
  `n/a` unless `--price-in` and `--price-out` are given.
- **Provider coverage.** The Anthropic adapter is tested against SDK response objects with a mocked
  client, not against the live API.
- **Docker.** CI lints and tests the package but does not build the image.

## License

MIT. See `LICENSE`.
