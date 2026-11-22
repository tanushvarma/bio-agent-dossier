"""Shared fakes: an httpx MockTransport that imitates Open Targets, NCBI and Europe PMC, and a scripted LLM."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from bioagent.clients import DiskCache, EuropePmcClient, JsonHttp, NcbiClient, OpenTargetsClient
from bioagent.clients.llm import LLMTurn, ToolResult, ToolSpec, inline_refs
from bioagent.tools import Toolbox

EGFR = "ENSG00000146648"
ERBB2 = "ENSG00000141736"


def _target_hit(ens: str, sym: str, score: float = 10.0) -> dict:
    return {"id": ens, "name": sym, "entity": "target", "score": score, "object": {"id": ens, "approvedSymbol": sym}}


SEARCH = {
    "EGFR": [_target_hit(EGFR, "EGFR"), _target_hit(ERBB2, "ERBB2", 2.0)],
    "HER2": [_target_hit(ERBB2, "ERBB2")],                                        # alias, no exact hit
    "AMBIG1": [_target_hit("ENSG00000000001", "AMBIG1"), _target_hit("ENSG00000000002", "AMBIG1")],
}

TARGETS = {
    EGFR: {
        "id": EGFR, "approvedSymbol": "EGFR", "approvedName": "epidermal growth factor receptor",
        "biotype": "protein_coding",
        "associatedDiseases": {"count": 2, "rows": [
            {"score": 0.81, "disease": {"id": "EFO_0003060", "name": "non-small cell lung carcinoma"}},
            {"score": 0.55, "disease": {"id": "EFO_0000519", "name": "glioblastoma"}}]},
        "drugAndClinicalCandidates": {"count": 2, "rows": [
            {"maxClinicalStage": "APPROVAL",
             "drug": {"id": "CHEMBL939", "name": "GEFITINIB", "drugType": "Small molecule",
                      "mechanismsOfAction": {"rows": [
                          {"mechanismOfAction": "Epidermal growth factor receptor erbB1 inhibitor",
                           "targets": [{"id": EGFR}]},
                          {"mechanismOfAction": "Unrelated other-target action", "targets": [{"id": ERBB2}]}]}},
             "diseases": [{"disease": {"name": "non-small cell lung carcinoma"}, "diseaseFromSource": None}]},
            {"maxClinicalStage": "APPROVAL",
             "drug": {"id": "CHEMBL3353410", "name": "OSIMERTINIB", "drugType": "Small molecule",
                      "mechanismsOfAction": {"rows": [
                          {"mechanismOfAction": "Epidermal growth factor receptor erbB1 inhibitor",
                           "targets": [{"id": EGFR}]}]}},
             "diseases": []}]},
    },
}

GENE_SUMMARY = {"uid": "1956", "name": "EGFR", "description": "epidermal growth factor receptor",
                "chromosome": "7", "maplocation": "7p11.2", "otheraliases": "ERBB, ERBB1, HER1",
                "summary": "The protein encoded by this gene is a transmembrane glycoprotein.",
                "organism": {"taxid": 9606}}

EPMC = {"hitCount": 3, "resultList": {"result": [
    {"pmid": "30000001", "title": "Osimertinib in <i>EGFR</i>-mutant NSCLC", "pubYear": "2018",
     "abstractText": "<h4>Background</h4>Third-generation EGFR inhibitor &amp; T790M.", "doi": "10.1/x"},
    {"pmid": "30000002", "title": "Gefitinib resistance mechanisms", "pubYear": "2015",
     "abstractText": "MET amplification drives resistance."},
    {"id": "PPR1", "source": "PPR", "title": "A preprint without a PMID", "pubYear": "2024"}]}}


@dataclass
class World:
    """Records every request; `fail` maps a URL substring to a list of status codes to return first."""
    calls: list[httpx.Request] = field(default_factory=list)
    fail: dict[str, list[int]] = field(default_factory=dict)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url)
        for frag, codes in self.fail.items():
            if frag in url and codes:
                return httpx.Response(codes.pop(0), headers={"Retry-After": "0"}, json={"error": "x"})
        if "opentargets" in url:
            body = json.loads(request.content)
            v = body["variables"]
            if "search(" in body["query"]:
                return httpx.Response(200, json={"data": {"search": {"hits": SEARCH.get(v["q"].upper(), [])}}})
            return httpx.Response(200, json={"data": {"target": TARGETS.get(v["id"])}})
        if "eutils" in url:
            p = request.url.params
            if "esearch" in url and p["db"] == "gene":
                ids = ["1956"] if "EGFR[SYM]" in p["term"].upper() else []
                return httpx.Response(200, json={"esearchresult": {"idlist": ids}})
            if "esummary" in url and p["db"] == "gene":
                return httpx.Response(200, json={"result": {"uids": ["1956"], "1956": GENE_SUMMARY}})
            if "esearch" in url and p["db"] == "pubmed":
                return httpx.Response(200, json={"esearchresult": {"idlist": ["111", "222"]}})
            if "esummary" in url and p["db"] == "pubmed":
                return httpx.Response(200, json={"result": {"uids": ["111", "222"],
                    "111": {"uid": "111", "title": "Paper A", "pubdate": "2020 Jan", "source": "Nature"},
                    "222": {"uid": "222", "title": "Paper B", "pubdate": "", "source": "Cell"}}})
        if "europepmc" in url:
            return httpx.Response(200, json=EPMC)
        return httpx.Response(404, json={"error": "no route"})


@pytest.fixture
def world() -> World:
    return World()


def make_http(ns: str, world: World, cache: DiskCache | None = None, **kw: Any) -> JsonHttp:
    return JsonHttp(ns, transport=httpx.MockTransport(world.handler), cache=cache, sleep=lambda s: None, **kw)


@pytest.fixture
def toolbox(world: World) -> Toolbox:
    return Toolbox(OpenTargetsClient(make_http("opentargets", world)),
                   NcbiClient(make_http("ncbi", world), "lab@asu.edu"),
                   EuropePmcClient(make_http("europepmc", world)))


class FakeSession:
    def __init__(self, llm: FakeLLM, tools: list[ToolSpec]):
        self.llm, self.tools = llm, tools
        self.results: list[list[ToolResult]] = []

    def next(self) -> LLMTurn:
        self.llm.n_next += 1
        if self.llm.repeat_turn is not None:
            return self.llm.repeat_turn
        return self.llm.turns.pop(0) if self.llm.turns else LLMTurn(text="done", input_tokens=10, output_tokens=5)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.results.append(results)

    def finalize(self, schema, instruction, json_schema=None):
        self.llm.n_finalize += 1
        self.llm.final_schema = json_schema
        return schema.model_validate(self.llm.final), *self.llm.final_tokens


class FakeLLM:
    provider, model = "fake", "fake-1"

    def __init__(self, turns: list[LLMTurn] | None = None, final: dict | None = None,
                 repeat_turn: LLMTurn | None = None, final_tokens: tuple[int, int] = (100, 50)):
        self.turns, self.repeat_turn = list(turns or []), repeat_turn
        self.final = final or {"summary": "x"}
        self.final_tokens = final_tokens
        self.n_next = self.n_finalize = 0
        self.sessions: list[FakeSession] = []
        self.final_schema: dict | None = None

    def session(self, system: str, user: str, tools: list[ToolSpec]) -> FakeSession:
        s = FakeSession(self, tools)
        self.sessions.append(s)
        return s


__all__ = ["EGFR", "ERBB2", "FakeLLM", "World", "inline_refs", "make_http"]
