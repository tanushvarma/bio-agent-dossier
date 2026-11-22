"""Command-line entry points, without network or API keys."""
from __future__ import annotations

from bioagent import cli
from bioagent.agent import BioAgent
from bioagent.schemas import Abstention, RunResult


def test_run_prints_json_and_maps_exit_code(monkeypatch, capsys):
    class Agent:
        def run(self, symbol):
            return RunResult(status="abstained", query=symbol,
                             abstention=Abstention(query=symbol, reason="not_found", message="no"))

    monkeypatch.setattr(BioAgent, "from_settings", classmethod(lambda cls, s=None: Agent()))
    assert cli.main(["run", "FAKEG1"]) == 2
    out = capsys.readouterr().out
    assert '"status": "abstained"' in out and '"trace"' not in out


def test_eval_dispatches_to_run_eval(tmp_path):
    assert cli.main(["eval", "--dataset", str(tmp_path / "missing.jsonl")]) == 1
