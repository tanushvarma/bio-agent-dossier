"""Pre-flight abstention, citation verifier, token budget."""
from __future__ import annotations

import pytest

from bioagent.clients import EuropePmcClient, NcbiClient, OpenTargetsClient
from bioagent.guardrails import TokenBudget, preflight, verify_citations
from bioagent.schemas import Abstention, DrugClaim, MechanismClaim, Resolution, TargetDossier, ToolTrace
from bioagent.tools import Toolbox
from conftest import EGFR, ERBB2, make_http


@pytest.mark.parametrize("bad", ["", "   ", "EGFR inhibitor", "rm -rf", "-EGFR", "A" * 25, "EGFR;"])
def test_invalid_symbol_abstains_without_network(toolbox, world, bad):
    trace = ToolTrace()
    out = preflight(bad, toolbox, trace)
    assert isinstance(out, Abstention) and out.reason == "invalid_symbol"
    assert world.calls == [] and trace.tool_calls == []


def test_alias_is_not_guessed(toolbox):
    out = preflight("HER2", toolbox, ToolTrace())
    assert isinstance(out, Abstention) and out.reason == "not_found"
    assert out.candidates == ["ERBB2"] and "Did you mean: ERBB2" in out.message


def test_unknown_symbol_abstains(toolbox):
    out = preflight("NOTAGENE1", toolbox, ToolTrace())
    assert isinstance(out, Abstention) and out.reason == "not_found" and out.candidates == []


def test_ambiguous_symbol_abstains(toolbox):
    out = preflight("AMBIG1", toolbox, ToolTrace())
    assert isinstance(out, Abstention) and out.reason == "ambiguous"


def test_valid_symbol_resolves_to_canonical_ids(toolbox):
    trace = ToolTrace()
    out = preflight(" egfr ", toolbox, trace)
    assert isinstance(out, Resolution)
    assert (out.approved_symbol, out.ensembl_id, out.entrez_id) == ("EGFR", EGFR, "1956")
    assert trace.tool_calls[0].step == 0 and trace.tool_calls[0].tool == "resolve_target"
    assert ERBB2 not in trace.tool_calls[0].returned_ids.ensembl_ids


def test_resolver_works_without_ncbi(world):
    tb = Toolbox(OpenTargetsClient(make_http("opentargets", world)), None,
                 EuropePmcClient(make_http("europepmc", world)))
    out = preflight("EGFR", tb, ToolTrace())
    assert isinstance(out, Resolution) and out.ensembl_id == EGFR and out.entrez_id is None


def test_resolver_outage_abstains(world):
    world.fail["opentargets"] = [503] * 10
    tb = Toolbox(OpenTargetsClient(make_http("opentargets", world, max_attempts=2)),
                 NcbiClient(make_http("ncbi", world), "lab@asu.edu"), EuropePmcClient(make_http("europepmc", world)))
    out = preflight("EGFR", tb, ToolTrace())
    assert isinstance(out, Abstention) and out.reason == "lookup_failed"


@pytest.fixture
def gathered(toolbox) -> ToolTrace:
    trace = ToolTrace()
    toolbox.execute("fetch_opentargets_drugs", {"ensembl_id": EGFR}, 1, trace)
    toolbox.execute("search_literature", {"query": "EGFR resistance"}, 2, trace)
    return trace


def _dossier(drugs, mechs) -> TargetDossier:
    return TargetDossier(query="EGFR", approved_symbol="EGFR", ensembl_id=EGFR, summary="s",
                         drug_claims=drugs, mechanism_claims=mechs)


def test_verifier_catches_fabricated_pmid_and_drug(gathered):
    d = _dossier(
        drugs=[
            DrugClaim(drug_name="Gefitinib", chembl_id="CHEMBL939", evidence_ids=["CHEMBL939", "30000002"]),  # real
            DrugClaim(drug_name="Erlotinib", chembl_id="CHEMBL553", evidence_ids=["CHEMBL553"]),  # not returned
            DrugClaim(drug_name="Lapatinib", chembl_id="CHEMBL939", evidence_ids=["CHEMBL939"]),  # wrong name for id
        ],
        mechs=[
            MechanismClaim(statement="MET amplification drives gefitinib resistance.",
                           evidence_ids=["30000002", "99999999"]),                    # one real, one fabricated
            MechanismClaim(statement="EGFR dimerizes with HER3 upon ligand binding.",
                           evidence_ids=["PMID:12345678"]),                          # only fabricated
        ],
    )
    out, rep = verify_citations(d, gathered, mode="strip")
    assert rep.n_claims == 5 and rep.n_verified == 2 and rep.grounding_score == 0.4
    assert [c.drug_name for c in out.drug_claims] == ["Gefitinib"]
    assert [c.evidence_ids for c in out.mechanism_claims] == [["30000002"]]        # fabricated PMID stripped
    assert all(c.verified for c in [*out.drug_claims, *out.mechanism_claims])
    assert set(rep.ungrounded_ids) == {"CHEMBL553", "99999999", "12345678"}
    texts = {u.text: u.reasons for u in rep.unverified_claims}
    assert any("not returned by a drug tool" in r for r in texts["Erlotinib"])
    assert any("is GEFITINIB, not Lapatinib" in r for r in texts["Lapatinib"])
    assert "no grounded evidence" in texts["EGFR dimerizes with HER3 upon ligand binding."]
    assert {e.id for e in out.evidence} == {"CHEMBL939", "30000002"}               # only cited, grounded items


def test_verifier_flag_mode_keeps_claims_but_marks_them(gathered):
    d = _dossier([DrugClaim(drug_name="Erlotinib", evidence_ids=["CHEMBL553"])],
                 [MechanismClaim(statement="Osimertinib targets T790M mutant EGFR.", evidence_ids=["30000001"])])
    out, rep = verify_citations(d, gathered, mode="flag")
    assert [c.verified for c in out.drug_claims] == [False]
    assert [c.verified for c in out.mechanism_claims] == [True]
    assert rep.grounding_score == 0.5 and len(rep.unverified_claims) == 1


def test_verifier_ignores_ids_from_failed_tool_calls(toolbox, world):
    trace = ToolTrace()
    world.fail["europepmc"] = [404]
    toolbox.execute("search_literature", {"query": "EGFR"}, 1, trace)
    d = _dossier([], [MechanismClaim(statement="Something supported by a paper.", evidence_ids=["30000001"])])
    _, rep = verify_citations(d, trace)
    assert rep.n_verified == 0


def test_drug_name_tolerates_case_and_salt_forms(gathered):
    d = _dossier([DrugClaim(drug_name="osimertinib mesylate", evidence_ids=["CHEMBL3353410"])], [])
    _, rep = verify_citations(d, gathered)
    assert rep.grounding_score == 1.0


def test_no_claims_gives_no_score(gathered):
    _, rep = verify_citations(_dossier([], []), gathered)
    assert rep.grounding_score is None and rep.n_claims == 0


def test_token_budget():
    b = TokenBudget(1000)
    b.add(600, 300)
    assert not b.exceeded
    b.add(50, 51)
    assert b.exceeded and "1,001" in b.message()
