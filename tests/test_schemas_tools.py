"""Schema validation and tool argument validation."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bioagent.schemas import DossierDraft, DrugClaim, MechanismClaim, TargetDossier, ToolTrace
from conftest import EGFR


def test_claims_require_evidence_ids():
    with pytest.raises(ValidationError):
        DrugClaim(drug_name="GEFITINIB", evidence_ids=[])
    with pytest.raises(ValidationError):
        MechanismClaim(statement="EGFR signals through the MAPK pathway.")      # evidence_ids missing
    with pytest.raises(ValidationError):
        DrugClaim(drug_name="X", evidence_ids=["CHEMBL1"], dosage="10mg")     # extra fields forbidden


def test_evidence_ids_are_normalized_and_deduplicated():
    c = DrugClaim(drug_name="gefitinib", chembl_id="chembl939",
                  evidence_ids=["chembl939", "PMID: 30000001", "30000001"])
    assert c.chembl_id == "CHEMBL939"
    assert c.evidence_ids == ["CHEMBL939", "30000001"]


def test_bad_chembl_id_rejected():
    with pytest.raises(ValidationError):
        DrugClaim(drug_name="X", chembl_id="DB00317", evidence_ids=["1"])


def test_dossier_round_trips_through_json():
    d = TargetDossier(query="egfr", approved_symbol="EGFR", ensembl_id=EGFR, summary="s",
                      drug_claims=[DrugClaim(drug_name="GEFITINIB", evidence_ids=["CHEMBL939"])])
    assert TargetDossier.model_validate_json(d.model_dump_json()) == d


def test_tool_specs_are_the_four_tools_with_inline_schemas(toolbox):
    specs = toolbox.specs()
    assert [s.name for s in specs] == ["resolve_target", "fetch_opentargets_drugs", "fetch_ncbi_gene",
                                       "search_literature"]
    for s in specs:
        assert s.parameters["type"] == "object"
        assert "$ref" not in json.dumps(s.parameters) and "$defs" not in s.parameters
        assert s.parameters.get("additionalProperties") is False


@pytest.mark.parametrize("name,args,needle", [
    ("fetch_opentargets_drugs", {"ensembl_id": "EGFR"}, "ensembl_id"),
    ("fetch_opentargets_drugs", {"ensembl_id": EGFR, "extra": 1}, "extra"),
    ("search_literature", {"query": "x"}, "query"),
    ("search_literature", {"query": "EGFR", "max_results": 500}, "max_results"),
    ("fetch_ncbi_gene", {}, "gene_symbol or entrez_id"),
    ("resolve_target", {"gene_symbol": "EGFR; DROP"}, "gene_symbol"),
    ("not_a_tool", {}, "unknown tool"),
])
def test_invalid_tool_args_are_rejected_before_any_http(toolbox, world, name, args, needle):
    trace = ToolTrace()
    ex = toolbox.execute(name, args, step=1, trace=trace)
    assert ex.output is None and not ex.record.ok and needle in ex.record.error
    assert ex.llm_content()["status"] == "error"
    assert world.calls == [] and trace.tool_calls == [ex.record]


def test_tool_records_ids_latency_and_cache(toolbox, world):
    trace = ToolTrace()
    ex = toolbox.execute("fetch_opentargets_drugs", {"ensembl_id": EGFR}, step=1, trace=trace)
    r = ex.record
    assert r.ok and r.latency_ms >= 0 and not r.cached
    assert r.returned_ids.chembl_ids == ["CHEMBL939", "CHEMBL3353410"]
    assert r.returned_ids.ensembl_ids == [EGFR]
    assert {e.id: e.kind for e in r.evidence} == {"CHEMBL939": "chembl", "CHEMBL3353410": "chembl"}
    lit = toolbox.execute("search_literature", {"query": "EGFR resistance"}, step=2, trace=trace)
    assert lit.record.returned_ids.pmids == ["30000001", "30000002"]
    assert trace.returned().pmids == ["30000001", "30000002"]


def test_draft_schema_is_valid_json_for_models():
    schema = DossierDraft.model_json_schema()
    assert set(schema["properties"]) == {"summary", "drug_claims", "mechanism_claims"}
