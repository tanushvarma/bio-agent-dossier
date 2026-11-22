"""The bounded agent loop, end to end with a scripted LLM and mocked HTTP."""
from __future__ import annotations

import json

from bioagent.agent import BioAgent
from bioagent.clients.llm import LLMTurn, ToolRequest
from conftest import EGFR, FakeLLM

GOOD_FINAL = {
    "summary": "EGFR is a receptor tyrosine kinase targeted by approved inhibitors.",
    "drug_claims": [
        {"drug_name": "GEFITINIB", "chembl_id": "CHEMBL939", "clinical_stage": "APPROVAL",
         "evidence_ids": ["CHEMBL939"]},
        {"drug_name": "Cetuximab", "chembl_id": "CHEMBL1201577", "evidence_ids": ["CHEMBL1201577"]},  # fabricated
    ],
    "mechanism_claims": [
        {"statement": "MET amplification drives resistance to gefitinib.", "evidence_ids": ["30000002"]},
        {"statement": "Osimertinib binds C797 covalently.", "evidence_ids": ["31111111"]},           # fabricated
    ],
}


def _turn(*calls: tuple[str, dict], tokens=(200, 40)) -> LLMTurn:
    return LLMTurn(tool_calls=[ToolRequest(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(calls)],
                   input_tokens=tokens[0], output_tokens=tokens[1])


def test_full_run_is_grounded_and_traced(toolbox):
    llm = FakeLLM(turns=[
        _turn(("fetch_opentargets_drugs", {"ensembl_id": EGFR}), ("fetch_ncbi_gene", {"entrez_id": "1956"})),
        _turn(("search_literature", {"query": "EGFR AND resistance", "max_results": 5})),
        LLMTurn(text="enough", input_tokens=300, output_tokens=10),
    ], final=GOOD_FINAL)
    res = BioAgent(llm, toolbox, max_steps=6).run("EGFR")

    assert res.status == "ok" and res.steps_used == 4 and llm.n_finalize == 1
    d = res.dossier
    assert (d.approved_symbol, d.ensembl_id, d.entrez_id) == ("EGFR", EGFR, "1956")
    assert [c.drug_name for c in d.drug_claims] == ["GEFITINIB"]
    assert [c.statement for c in d.mechanism_claims] == ["MET amplification drives resistance to gefitinib."]
    v = res.verification
    assert v.n_claims == 4 and v.n_verified == 2 and v.grounding_score == 0.5
    assert set(v.ungrounded_ids) == {"CHEMBL1201577", "31111111"}

    t = res.trace
    assert [c.tool for c in t.tool_calls] == ["resolve_target", "fetch_opentargets_drugs", "fetch_ncbi_gene",
                                              "search_literature"]
    assert [c.step for c in t.tool_calls] == [0, 1, 1, 2]
    assert all(c.ok and c.latency_ms >= 0 for c in t.tool_calls)
    r = t.returned()
    assert r.ensembl_ids == [EGFR] and r.entrez_ids == ["1956"]
    assert r.chembl_ids == ["CHEMBL939", "CHEMBL3353410"] and r.pmids == ["30000001", "30000002"]
    assert [c.kind for c in t.llm_calls] == ["tool_step"] * 3 + ["finalize"]
    assert (res.input_tokens, res.output_tokens) == (200 + 200 + 300 + 100, 40 + 40 + 10 + 50)
    # the model's final schema hides verifier-only fields
    assert '"verified"' not in json.dumps(llm.final_schema)
    # tool results were fed back with the right ids
    assert [r.id for r in llm.sessions[0].results[0]] == ["c0", "c1"]


def test_step_cap_is_hard(toolbox):
    llm = FakeLLM(repeat_turn=_turn(("search_literature", {"query": "EGFR"})), final={"summary": "s"})
    res = BioAgent(llm, toolbox, max_steps=6).run("EGFR")
    assert res.status == "ok"
    assert llm.n_next == 5 and llm.n_finalize == 1 and res.steps_used == 6
    assert len(res.trace.llm_calls) == 6


def test_budget_abort_is_clean(toolbox):
    llm = FakeLLM(repeat_turn=_turn(("search_literature", {"query": "EGFR"}), tokens=(30_000, 1_000)))
    res = BioAgent(llm, toolbox, max_tokens_per_run=50_000).run("EGFR")
    assert res.status == "aborted_budget" and res.dossier is None
    assert llm.n_next == 2 and llm.n_finalize == 0
    assert "62,000 used" in res.message


def test_abstention_makes_no_llm_calls(toolbox):
    llm = FakeLLM()
    res = BioAgent(llm, toolbox).run("HER2")
    assert res.status == "abstained" and res.abstention.reason == "not_found"
    assert llm.sessions == [] and res.input_tokens == 0


def test_bad_tool_call_is_reported_to_model_not_raised(toolbox):
    llm = FakeLLM(turns=[_turn(("fetch_opentargets_drugs", {"ensembl_id": "EGFR"}))], final={"summary": "s"})
    res = BioAgent(llm, toolbox).run("EGFR")
    assert res.status == "ok"
    [fed_back] = llm.sessions[0].results[0]
    assert fed_back.content["status"] == "error" and "ensembl_id" in fed_back.content["error"]


def test_per_step_call_limit_still_answers_every_call(toolbox):
    calls = [("search_literature", {"query": f"EGFR {i}"}) for i in range(6)]
    llm = FakeLLM(turns=[_turn(*calls)], final={"summary": "s"})
    BioAgent(llm, toolbox, max_calls_per_step=4).run("EGFR")
    fed = llm.sessions[0].results[0]
    assert len(fed) == 6 and [f.content["status"] for f in fed].count("error") == 2


def test_invalid_final_json_is_an_error_result(toolbox):
    llm = FakeLLM(final={"summary": "s", "drug_claims": [{"drug_name": "X", "evidence_ids": []}]})
    res = BioAgent(llm, toolbox).run("EGFR")
    assert res.status == "error" and "schema" in res.message
