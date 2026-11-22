"""Command line: `bioagent run EGFR` prints the RunResult as JSON. `bioagent eval` arrives with evals/."""
from __future__ import annotations

import argparse
import sys

from .agent import BioAgent
from .config import ConfigError

EXIT = {"ok": 0, "abstained": 2, "aborted_budget": 3, "error": 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioagent", description="Grounded target-dossier agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="build a dossier for one gene symbol")
    run.add_argument("gene_symbol")
    run.add_argument("--with-trace", action="store_true", help="include the full tool trace")
    sub.add_parser("eval", help="run the benchmark (not built yet)")
    args = parser.parse_args(argv)

    if args.cmd == "eval":
        print("bioagent eval: the benchmark is built in a later phase (evals/run_eval.py).", file=sys.stderr)
        return 1
    try:
        agent = BioAgent.from_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    res = agent.run(args.gene_symbol)
    exclude = None if args.with_trace else {"trace"}
    print(res.model_dump_json(indent=2, exclude=exclude))
    return EXIT[res.status]


if __name__ == "__main__":
    raise SystemExit(main())
