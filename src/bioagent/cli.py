"""Command line.

    bioagent run EGFR [--with-trace]   RunResult as JSON on stdout
    bioagent eval [run_eval options]   benchmark (needs a source checkout with evals/)
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys

from .agent import BioAgent
from .config import PROJECT_ROOT, ConfigError

log = logging.getLogger("bioagent")
EXIT = {"ok": 0, "abstained": 2, "aborted_budget": 3, "error": 1}


def _eval(argv: list[str]) -> int:
    script = PROJECT_ROOT / "evals" / "run_eval.py"
    if not script.exists():
        log.error("bioagent eval needs a source checkout: %s not found", script)
        return 1
    spec = importlib.util.spec_from_file_location("bioagent_run_eval", script)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.main(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if argv[:1] == ["eval"]:                 # forwarded verbatim; argparse REMAINDER drops leading options
        return _eval(argv[1:])
    parser = argparse.ArgumentParser(prog="bioagent", description="Grounded target-dossier agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="build a dossier for one gene symbol")
    run.add_argument("gene_symbol")
    run.add_argument("--with-trace", action="store_true", help="include the full tool trace")
    sub.add_parser("eval", help="run the benchmark; options are passed to evals/run_eval.py")
    args = parser.parse_args(argv)
    try:
        agent = BioAgent.from_settings()
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        return 1
    res = agent.run(args.gene_symbol)
    exclude = None if args.with_trace else {"trace"}
    print(res.model_dump_json(indent=2, exclude=exclude))
    return EXIT[res.status]


if __name__ == "__main__":
    raise SystemExit(main())
