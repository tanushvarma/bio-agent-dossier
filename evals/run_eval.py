"""Run the frozen benchmark through the full bioagent pipeline and score it deterministically.

    python evals/run_eval.py --limit 5            # quick smoke run (mixes positives and negatives)
    python evals/run_eval.py --concurrency 3      # full run

Writes per-case results with full traces to --out (default evals/results/latest.json) and a
Markdown table to summary.md next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from bioagent.config import ConfigError, get_settings  # noqa: E402
from bioagent.pricing import PRICES  # noqa: E402
from bioagent.schemas import RunResult  # noqa: E402
from evals.metrics import aggregate, score_case, to_markdown  # noqa: E402

log = logging.getLogger("bioagent.eval")
DATASET = ROOT / "evals" / "dataset_50.jsonl"
README = ROOT / "README.md"
BEGIN, END = "<!-- benchmark:begin -->", "<!-- benchmark:end -->"
DEFAULT_OUT = ROOT / "evals" / "results" / "latest.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(cases: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """First `limit` cases taken round-robin across case types, so a smoke run covers negatives too."""
    if not limit or limit >= len(cases):
        return cases
    by_type: dict[str, list[dict]] = {}
    for c in cases:
        by_type.setdefault(c["type"], []).append(c)
    queues = [by_type[t] for t in ("positive", "negative_symbol", "negative_pair") if t in by_type]
    out: list[dict] = []
    i = 0
    while len(out) < limit:
        q = queues[i % len(queues)]
        idx = i // len(queues)
        if idx < len(q):
            out.append(q[idx])
        i += 1
    return out


def run_cases(cases: list[dict[str, Any]], agent_factory: Callable[[], Any], concurrency: int = 3,
              progress: Callable[[str], None] = log.info) -> list[tuple[dict, RunResult]]:
    """One agent per worker thread (separate HTTP clients and LLM sessions)."""
    local = threading.local()

    def one(case: dict[str, Any]) -> tuple[dict, RunResult]:
        if not hasattr(local, "agent"):
            local.agent = agent_factory()
        t0 = time.perf_counter()
        try:
            res = local.agent.run(case["gene_symbol"])
        except (RuntimeError, OSError, httpx.HTTPError) as exc:   # agent.run reports expected failures itself
            res = RunResult(status="error", query=case["gene_symbol"], message=f"{type(exc).__name__}: {exc}",
                            total_latency_ms=(time.perf_counter() - t0) * 1000)
        progress(f"  {case['case_id']:12s} {case['gene_symbol']:11s} -> {res.status:14s} "
                 f"{res.total_latency_ms / 1000:6.1f}s {res.input_tokens + res.output_tokens:>7,} tok")
        return case, res

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, cases))        # map keeps dataset order


def evaluate(cases: list[dict[str, Any]], agent_factory: Callable[[], Any], out: Path, *,
             concurrency: int = 3, meta: dict[str, Any] | None = None,
             prices: dict | None = None, progress: Callable[[str], None] = log.info) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    pairs = run_cases(cases, agent_factory, concurrency, progress)
    scores = [score_case(c, r, prices or PRICES) for c, r in pairs]
    metrics = aggregate(scores)
    llm = next((call for _, r in pairs for call in r.trace.llm_calls), None)
    meta = {"started_at": started, "finished_at": datetime.now(UTC).isoformat(),
            "provider": llm.provider if llm else None, "model": llm.model if llm else None,
            "concurrency": concurrency, **(meta or {})}
    report = {"meta": meta, "metrics": metrics,
              "cases": [{"case": c, "score": s, "result": r.model_dump(mode="json")}
                        for (c, r), s in zip(pairs, scores, strict=True)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out.parent / "summary.md").write_text(to_markdown(metrics, meta, scores), encoding="utf-8")
    return report


def update_readme(readme: Path, summary_md: str) -> bool:
    """Replace the text between the benchmark markers in the README with the generated summary."""
    text = readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        return False
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    body = summary_md.split("\n", 1)[1].strip()          # drop the summary's own H1
    readme.write_text(f"{head}{BEGIN}\n{body}\n{END}{tail}", encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run and score the bioagent benchmark.")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--limit", type=int, default=None, help="score only N cases (round-robin across types)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--price-in", type=float, help="USD per 1M input tokens (overrides the built-in table)")
    ap.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    ap.add_argument("--update-readme", action="store_true",
                    help="paste the summary table into README.md between the benchmark markers")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.dataset.exists():
        log.error("%s not found. Build it first: python evals/build_dataset.py", args.dataset)
        return 1
    from bioagent.agent import BioAgent

    settings = get_settings()
    try:
        BioAgent.from_settings(settings)
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        return 1
    prices = dict(PRICES)
    if args.price_in is not None and args.price_out is not None:
        model = settings.gemini_model if settings.llm_provider == "gemini" else settings.anthropic_model
        prices[(settings.llm_provider, model)] = (args.price_in, args.price_out)

    raw = args.dataset.read_bytes()
    cases = select(load_cases(args.dataset), args.limit)
    log.info("Running %d cases with concurrency %d", len(cases), args.concurrency)
    report = evaluate(cases, lambda: BioAgent.from_settings(settings), args.out, concurrency=args.concurrency,
                      prices=prices, meta={"dataset": str(args.dataset.relative_to(ROOT))
                                           if args.dataset.is_relative_to(ROOT) else str(args.dataset),
                                           "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                                           "limit": args.limit})
    print((args.out.parent / "summary.md").read_text(encoding="utf-8"))
    log.info("Full results: %s", args.out)
    if args.update_readme:
        if args.limit:
            log.warning("Not updating README from a --limit run; run the full benchmark.")
        elif update_readme(README, (args.out.parent / "summary.md").read_text(encoding="utf-8")):
            log.info("README.md benchmark section updated.")
        else:
            log.warning("README.md has no %s / %s markers; not updated.", BEGIN, END)
    return 0 if report["metrics"]["n_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
