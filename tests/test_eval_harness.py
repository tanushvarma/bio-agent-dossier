"""Eval harness on synthetic fixtures: every metric checked against hand-computed values.

Run just these with:  pytest -m smoke
"""
from __future__ import annotations

import json

import pytest

from bioagent.clients import NcbiClient, OpenTargetsClient
from bioagent.schemas import (
    DrugClaim,
    LLMCall,
    MechanismClaim,
    ReturnedIds,
    RunResult,
    TargetDossier,
    ToolCall,
    ToolTrace,
    UnverifiedClaim,
    Verification,
)
from conftest import EGFR, make_http
from evals.build_dataset import build, stage_rank
from evals.metrics import aggregate, cost_usd, percentile, score_case, to_markdown
from evals.run_eval import evaluate, select

pytestmark = pytest.mark.smoke

ACHE = "ENSG00000087085"


def _trace(ensembl: str | None, tokens: tuple[int, int] = (0, 0)) -> ToolTrace:
    t = ToolTrace(tool_calls=[ToolCall(step=0, tool="resolve_target", args={}, ok=True, latency_ms=1,
                                       returned_ids=ReturnedIds(ensembl_ids=[ensembl] if ensembl else []))])
    if sum(tokens):
        t.llm_calls.append(LLMCall(step=1, kind="tool_step", provider="gemini", model="gemini-3.5-flash-lite",
                                   input_tokens=tokens[0], output_tokens=tokens[1]))
    return t


def _ok(symbol: str, ensembl: str, drugs: list[DrugClaim], n_mech: int, verification: Verification,
        latency_ms: float, tokens: tuple[int, int]) -> RunResult:
    mechs = [MechanismClaim(statement=f"Mechanism statement number {i}.", evidence_ids=["1"]) for i in range(n_mech)]
    d = TargetDossier(query=symbol, approved_symbol=symbol, ensembl_id=ensembl, summary="s",
                      drug_claims=drugs, mechanism_claims=mechs)
    return RunResult(status="ok", query=symbol, dossier=d, verification=verification, trace=_trace(ensembl, tokens),
                     input_tokens=tokens[0], output_tokens=tokens[1], total_latency_ms=latency_ms)


EXPECTED = [{"chembl_id": "CHEMBL939", "name": "GEFITINIB"},
            {"chembl_id": "CHEMBL3353410", "name": "OSIMERTINIB"},
            {"chembl_id": "CHEMBL553", "name": "ERLOTINIB"}]
POS = {"case_id": "pos-001", "type": "positive", "area": "oncology", "gene_symbol": "EGFR",
       "expected_status": "ok", "ensembl_id": EGFR, "drug": EXPECTED[1], "expected_drugs": EXPECTED}
POS2 = {**POS, "case_id": "pos-002", "gene_symbol": "ACHE", "area": "neurology", "ensembl_id": ACHE,
        "drug": {"chembl_id": "CHEMBL502", "name": "DONEPEZIL"},
        "expected_drugs": [{"chembl_id": "CHEMBL502", "name": "DONEPEZIL"}]}
NEG_SYM = {"case_id": "neg-sym-001", "type": "negative_symbol", "gene_symbol": "FAKEG1", "expected_status": "abstain"}
NEG_PAIR = {"case_id": "neg-pair-001", "type": "negative_pair", "gene_symbol": "EGFR", "expected_status": "abstain",
            "ensembl_id": EGFR, "forbidden_drug": {"chembl_id": "CHEMBL502", "name": "DONEPEZIL"}}


def _fixture_results() -> dict[str, RunResult]:
    # EGFR: 3 drug claims + 1 mechanism = 4 claims, 1 unverified; 5 cited IDs, 1 not in trace
    egfr = _ok("EGFR", EGFR, [DrugClaim(drug_name="GEFITINIB", chembl_id="CHEMBL939", evidence_ids=["CHEMBL939"]),
                              DrugClaim(drug_name="osimertinib mesylate", evidence_ids=["CHEMBL3353410"]),
                              DrugClaim(drug_name="AFATINIB", evidence_ids=["CHEMBL1173655"])], 1,
               Verification(mode="strip", n_claims=4, n_verified=3, grounding_score=0.75, n_cited_ids=5,
                            unverified_claims=[UnverifiedClaim(kind="mechanism", text="x", reasons=["r"])],
                            ungrounded_ids=["99"]),
               latency_ms=4000, tokens=(1000, 200))
    # ACHE resolves to the wrong Ensembl ID and names no drugs: 2 claims, both verified, 2 cited IDs
    ache = _ok("ACHE", "ENSG00000000009", [], 2,
               Verification(mode="strip", n_claims=2, n_verified=2, grounding_score=1.0, n_cited_ids=2),
               latency_ms=2000, tokens=(500, 100))
    fake = RunResult(status="abstained", query="FAKEG1", trace=ToolTrace(), total_latency_ms=100)
    # negative pair: the model tried DONEPEZIL, the verifier removed it -> attempted but not a false positive
    pair = _ok("EGFR", EGFR, [DrugClaim(drug_name="GEFITINIB", evidence_ids=["CHEMBL939"])], 0,
               Verification(mode="strip", n_claims=2, n_verified=1, grounding_score=0.5, n_cited_ids=2,
                            unverified_claims=[UnverifiedClaim(kind="drug", text="Donepezil", reasons=["r"])],
                            ungrounded_ids=["CHEMBL502"]),
               latency_ms=6000, tokens=(1500, 300))
    return {"pos-001": egfr, "pos-002": ache, "neg-sym-001": fake, "neg-pair-001": pair}


# ------------------------------------------------------------------ unit metrics
def test_percentile_matches_numpy_linear():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([4, 1, 3, 2], 95) == pytest.approx(3.85)
    assert percentile([7], 95) == 7 and percentile([], 50) is None


def test_cost():
    assert cost_usd("gemini", "gemini-3.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(2.80)
    assert cost_usd("anthropic", "unknown-model", 10, 10) is None


def test_positive_case_scoring():
    s = score_case(POS, _fixture_results()["pos-001"])
    assert s["target_resolved"] is True
    assert (s["n_predicted_drugs"], s["n_expected_drugs"]) == (3, 3)
    assert s["drug_precision"] == pytest.approx(2 / 3)       # AFATINIB is not in ground truth
    assert s["drug_recall"] == pytest.approx(2 / 3)          # ERLOTINIB missed; salt form still matches
    assert s["anchor_drug_found"] is True
    assert s["cost_usd"] == pytest.approx((1000 * 0.30 + 200 * 2.50) / 1e6)


def test_wrong_target_and_no_drugs():
    s = score_case(POS2, _fixture_results()["pos-002"])
    assert s["target_resolved"] is False
    assert s["drug_precision"] is None and s["drug_recall"] == 0.0 and s["anchor_drug_found"] is False


def test_negative_symbol_scoring():
    assert score_case(NEG_SYM, _fixture_results()["neg-sym-001"])["false_positive"] is False
    leaked = RunResult(status="ok", query="FAKEG1", trace=_trace(None))
    assert score_case(NEG_SYM, leaked)["false_positive"] is True


def test_negative_pair_scoring():
    s = score_case(NEG_PAIR, _fixture_results()["neg-pair-001"])
    assert s["false_positive"] is False and s["fp_attempted"] is True and s["target_resolved"] is True
    bad = _ok("EGFR", EGFR, [DrugClaim(drug_name="Donepezil", evidence_ids=["CHEMBL502"])], 0,
              Verification(mode="flag", n_claims=1, n_verified=0, grounding_score=0.0, n_cited_ids=1),
              latency_ms=1, tokens=(1, 1))
    assert score_case(NEG_PAIR, bad)["false_positive"] is True


def test_aggregate_on_fixture_set():
    res = _fixture_results()
    scores = [score_case(c, res[c["case_id"]]) for c in (POS, POS2, NEG_SYM, NEG_PAIR)]
    m = aggregate(scores)
    assert (m["n_cases"], m["n_positive"], m["n_negative"], m["n_errors"]) == (4, 2, 2, 0)
    assert m["target_resolution_acc"] == pytest.approx(2 / 3)       # EGFR ok, ACHE wrong, pair ok
    assert m["drug_precision"] == pytest.approx(2 / 3)              # ACHE precision undefined -> excluded
    assert m["drug_recall"] == pytest.approx((2 / 3 + 0) / 2)
    assert m["anchor_drug_recall"] == 0.5
    assert m["hallucination_rate"] == pytest.approx(2 / 8)          # (1 + 0 + 1) / (4 + 2 + 2)
    assert m["citation_validity"] == pytest.approx((9 - 2) / 9)     # cited 5+2+2, ungrounded 1+0+1
    assert m["negative_control_fp_rate"] == 0.0 and m["negative_control_fp_attempted"] == 1
    assert m["status_match_rate"] == 1.0
    assert m["latency_p50_s"] == pytest.approx(3.0)                 # [0.1, 2, 4, 6]
    assert m["latency_p95_s"] == pytest.approx(5.7)
    assert m["mean_tokens"] == pytest.approx((1200 + 600 + 1800) / 3)   # abstention excluded
    total = (1000 * 0.3 + 200 * 2.5 + 500 * 0.3 + 100 * 2.5 + 1500 * 0.3 + 300 * 2.5) / 1e6
    assert m["est_cost_usd"] == pytest.approx(total)
    assert m["est_cost_per_query_usd"] == pytest.approx(total / 3, abs=1e-6)
    md = to_markdown(m, {"model": "m", "provider": "p"}, scores)
    assert "| Target resolution accuracy | 66.7% |" in md and "| Hallucination rate" in md
    assert "`pos-002` ACHE" in md and "| neurology | 1 |" in md


# ------------------------------------------------------------------ harness end to end

def test_evaluate_writes_results_and_summary(tmp_path):
    res = _fixture_results()
    agents = {"EGFR": res["pos-001"], "ACHE": res["pos-002"], "FAKEG1": res["neg-sym-001"]}
    cases = [POS, POS2, NEG_SYM]
    out = tmp_path / "results" / "latest.json"
    report = evaluate(cases, lambda: type("A", (), {"run": lambda self, s: agents[s]})(), out,
                      concurrency=2, meta={"dataset": "fixture"}, progress=lambda m: None)
    saved = json.loads(out.read_text())
    assert [c["case"]["case_id"] for c in saved["cases"]] == ["pos-001", "pos-002", "neg-sym-001"]  # order kept
    assert saved["cases"][0]["result"]["trace"]["tool_calls"][0]["tool"] == "resolve_target"         # full trace
    assert saved["meta"]["model"] == "gemini-3.5-flash-lite" and report["metrics"]["n_cases"] == 3
    summary = (out.parent / "summary.md").read_text()
    assert summary.startswith("# bioagent evaluation summary") and "| Metric | Value |" in summary


def test_agent_exceptions_become_error_rows(tmp_path):
    class Boom:
        def run(self, s):
            raise RuntimeError("provider down")

    report = evaluate([POS], Boom, tmp_path / "r.json", concurrency=1, progress=lambda m: None)
    assert report["metrics"]["n_errors"] == 1 and report["cases"][0]["result"]["status"] == "error"


def test_limit_mixes_case_types():
    cases = [{**POS, "case_id": f"pos-{i}"} for i in range(6)] + [NEG_SYM, NEG_PAIR]
    assert [c["type"] for c in select(cases, 5)] == ["positive", "negative_symbol", "negative_pair",
                                                      "positive", "positive"]
    assert select(cases, None) == cases


# ------------------------------------------------------------------ dataset builder
@pytest.mark.parametrize("stage,rank", [("APPROVAL", 4), ("Approved", 4), ("PHASE_4", 4), ("PHASE_3", 3),
                                        ("Phase III", 3), ("PHASE_2_3", 2.5), ("PHASE_2", 2), ("EARLY_PHASE_1", 1),
                                        (None, 0), ("UNKNOWN", 0)])
def test_stage_rank(stage, rank):
    assert stage_rank(stage) == rank


def test_builder_verifies_every_record(world):
    from conftest import SEARCH
    SEARCH["DONEPEZIL"] = [{"id": "CHEMBL502", "name": "DONEPEZIL", "entity": "drug", "score": 5.0, "object": {}}]
    SEARCH["GEFITINIB"] = [{"id": "CHEMBL939", "name": "GEFITINIB", "entity": "drug", "score": 5.0, "object": {}}]
    try:
        ot = OpenTargetsClient(make_http("opentargets", world))
        ncbi = NcbiClient(make_http("ncbi", world), "lab@asu.edu")
        cases, meta = build(ot, ncbi,
                            seeds=[("oncology", "EGFR", "OSIMERTINIB"), ("oncology", "EGFR", "DONEPEZIL")],
                            quotas={"oncology": 2}, fakes=["EGFR", "FAKEG1", "bad symbol!"],
                            pairs=[("EGFR", "GEFITINIB"), ("EGFR", "DONEPEZIL")], log=lambda m: None)
    finally:
        SEARCH.pop("DONEPEZIL"), SEARCH.pop("GEFITINIB")
    assert [(c["case_id"], c["gene_symbol"]) for c in cases] == [("pos-001", "EGFR"), ("neg-sym-001", "FAKEG1"),
                                                                ("neg-pair-001", "EGFR")]
    pos = cases[0]
    assert (pos["ensembl_id"], pos["entrez_id"], pos["drug"]["chembl_id"]) == (EGFR, "1956", "CHEMBL3353410")
    assert pos["drug"]["mechanism_of_action"] and pos["reference_pmids"] == ["111", "222"]
    assert {d["chembl_id"] for d in pos["expected_drugs"]} == {"CHEMBL939", "CHEMBL3353410"}
    assert cases[2]["forbidden_drug"] == {"chembl_id": "CHEMBL502", "name": "DONEPEZIL"}
    reasons = " | ".join(s["reason"] for s in meta["skipped"])
    assert "DONEPEZIL not listed" in reasons            # anchor not linked to target
    assert "unexpectedly resolves" in reasons           # a real gene offered as a fake
    assert "not format-valid" in reasons
    assert "IS linked to EGFR" in reasons               # GEFITINIB is a real EGFR drug, not a negative
    assert meta["clinical_stages_seen"] == {"APPROVAL": 4}             # 2 drugs x 2 EGFR seeds checked
