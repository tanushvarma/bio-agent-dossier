"""Render the Streamlit app headlessly with a fake agent and a fake results file."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from bioagent.schemas import (
    DrugClaim,
    LLMCall,
    RunResult,
    TargetDossier,
    ToolCall,
    ToolTrace,
    UnverifiedClaim,
    Verification,
)

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
EGFR = "ENSG00000146648"


def _result(symbol: str) -> RunResult:
    if symbol.upper() == "HER2":
        from bioagent.schemas import Abstention
        return RunResult(status="abstained", query=symbol, message="m",
                         abstention=Abstention(query=symbol, reason="not_found", message="Did you mean: ERBB2?"))
    trace = ToolTrace(tool_calls=[ToolCall(step=0, tool="resolve_target", args={}, ok=True, latency_ms=3)],
                      llm_calls=[LLMCall(step=1, kind="finalize", provider="gemini", model="gemini-3.5-flash-lite",
                                         input_tokens=1000, output_tokens=200)])
    d = TargetDossier(query=symbol, approved_symbol="EGFR", ensembl_id=EGFR, entrez_id="1956", summary="EGFR summary.",
                      drug_claims=[DrugClaim(drug_name="GEFITINIB", chembl_id="CHEMBL939", evidence_ids=["CHEMBL939"],
                                             verified=True)])
    v = Verification(mode="strip", n_claims=2, n_verified=1, grounding_score=0.5, n_cited_ids=2,
                     unverified_claims=[UnverifiedClaim(kind="drug", text="Cetuximab", unknown_ids=["CHEMBL1201577"],
                                                        reasons=["not returned"])],
                     ungrounded_ids=["CHEMBL1201577"])
    return RunResult(status="ok", query=symbol, dossier=d, verification=v, trace=trace, steps_used=1,
                     input_tokens=1000, output_tokens=200, total_latency_ms=5300)


class FakeAgent:
    def run(self, symbol: str) -> RunResult:
        return _result(symbol)


@pytest.fixture
def app(monkeypatch, tmp_path):
    from bioagent.agent import BioAgent
    monkeypatch.setattr(BioAgent, "from_settings", classmethod(lambda cls, s=None: FakeAgent()))
    results = tmp_path / "latest.json"
    score = {"case_id": "pos-001", "type": "positive", "area": "oncology", "gene_symbol": "EGFR", "status": "ok",
             "expected_status": "ok", "target_resolved": True, "drug_precision": 1.0, "drug_recall": 0.5,
             "n_claims": 2, "n_unverified": 1, "latency_s": 5.3, "input_tokens": 1000, "output_tokens": 200}
    metrics = {"n_cases": 1, "n_positive": 1, "n_negative": 0, "n_errors": 0, "target_resolution_acc": 1.0,
               "drug_precision": 1.0, "drug_recall": 0.5, "hallucination_rate": 0.5, "citation_validity": 0.5,
               "negative_control_fp_rate": None, "latency_p50_s": 5.3, "latency_p95_s": 5.3, "mean_tokens": 1200,
               "est_cost_per_query_usd": 0.0008}
    results.write_text(json.dumps({"meta": {"provider": "gemini", "model": "m"}, "metrics": metrics,
                                   "cases": [{"case": {"case_id": "pos-001"}, "score": score,
                                              "result": _result("EGFR").model_dump(mode="json")}]}))
    monkeypatch.setenv("BIOAGENT_EVAL_RESULTS", str(results))
    return AppTest.from_file(APP, default_timeout=30)


def test_dossier_tab_renders_result_and_badge(app):
    at = app.run()
    assert not at.exception
    at.text_input[0].input("egfr")
    at.button[0].click().run()
    assert not at.exception
    assert any(s.value == "EGFR" for s in at.subheader)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Grounding score"] == "50.0%" and metrics["Latency"] == "5.3 s"
    assert metrics["Est. cost"] == "$0.0008"
    assert any("Flagged: 1/2 claims removed" in m.value for m in at.markdown)          # st.badge renders as markdown


def test_dossier_tab_shows_abstention(app):
    at = app.run()
    at.text_input[0].input("HER2")
    at.button[0].click().run()
    assert any("Did you mean: ERBB2?" in w.value for w in at.warning)


def test_eval_tab_reads_results(app):
    at = app.run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Target resolution"] == "100.0%" and labels["Drug recall"] == "50.0%"
    assert labels["Negative-control FP rate"] == "n/a"
    assert len(at.dataframe) >= 1 and at.selectbox[0].value == "pos-001"
