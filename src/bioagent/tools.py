"""The four typed tools the agent can call, plus the executor that records every call in the ToolTrace."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .clients import ApiError, DiskCache, EuropePmcClient, JsonHttp, NcbiClient, OpenTargetsClient
from .clients.llm import ToolSpec, inline_refs
from .clients.ncbi import ncbi_throttle
from .config import ConfigError, Settings
from .schemas import (
    ENSEMBL_GENE_PATTERN,
    SYMBOL_PATTERN,
    EvidenceItem,
    Resolution,
    ReturnedIds,
    ToolCall,
    ToolTrace,
)

ABSTRACT_CHARS = 1200


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResolveTargetArgs(_Args):
    gene_symbol: str = Field(pattern=SYMBOL_PATTERN, description="HGNC gene symbol, e.g. EGFR")


class FetchDrugsArgs(_Args):
    ensembl_id: str = Field(pattern=ENSEMBL_GENE_PATTERN, description="Ensembl gene ID, e.g. ENSG00000146648")


class FetchGeneArgs(_Args):
    gene_symbol: str | None = Field(None, pattern=SYMBOL_PATTERN)
    entrez_id: str | None = Field(None, pattern=r"^\d{1,10}$", description="NCBI Gene ID, preferred if known")

    @model_validator(mode="after")
    def _one(self) -> FetchGeneArgs:
        if not (self.gene_symbol or self.entrez_id):
            raise ValueError("give gene_symbol or entrez_id")
        return self


class SearchLiteratureArgs(_Args):
    query: str = Field(min_length=2, max_length=300,
                       description="Europe PMC query, e.g. 'EGFR AND (resistance OR mechanism)'")
    max_results: int = Field(8, ge=1, le=20)


@dataclass
class ToolOutput:
    payload: dict[str, Any]
    returned_ids: ReturnedIds = field(default_factory=ReturnedIds)
    evidence: list[EvidenceItem] = field(default_factory=list)
    result: Any = None


@dataclass
class ToolDef:
    name: str
    description: str
    args_model: type[_Args]
    fn: Callable[[Any], ToolOutput]


@dataclass
class Execution:
    record: ToolCall
    output: ToolOutput | None

    def llm_content(self) -> dict[str, Any]:
        if self.output is None:
            return {"status": "error", "error": self.record.error}
        return {"status": "success", **self.output.payload}


def _exact(a: str | None, b: str) -> bool:
    return bool(a) and a.upper() == b.upper()  # type: ignore[union-attr]


def resolve_symbol(symbol: str, ot: OpenTargetsClient, ncbi: NcbiClient | None) -> Resolution:
    """Map a symbol to canonical IDs. Exact approved-symbol matches only: an alias or a near miss
    returns not_found with candidates, so callers abstain instead of guessing."""
    notes: list[str] = []
    hits = [h for h in ot.search(symbol) if h.entity == "target"]
    exact = {h.id: (h.approved_symbol or h.name) for h in hits if _exact(h.approved_symbol or h.name, symbol)}
    candidates = list(dict.fromkeys((h.approved_symbol or h.name) for h in hits if h.id not in exact))[:5]

    entrez_ids: list[str] = []
    if ncbi is not None:
        try:
            entrez_ids = [g.entrez_id for g in ncbi.search_gene(symbol) if _exact(g.symbol, symbol)]
        except (ApiError, httpx.HTTPError) as exc:
            notes.append(f"NCBI lookup failed: {exc}")
    else:
        notes.append("NCBI not configured; Entrez ID not resolved")

    if len(set(exact)) > 1 or len(entrez_ids) > 1:
        return Resolution(status="ambiguous", query=symbol, candidates=sorted(set(exact.values())) or candidates,
                          notes=[*notes, f"{len(exact)} Ensembl and {len(entrez_ids)} Entrez exact matches"])
    ensembl_id = next(iter(exact), None)
    entrez_id = entrez_ids[0] if entrez_ids else None
    if not ensembl_id and not entrez_id:
        return Resolution(status="not_found", query=symbol, candidates=candidates, notes=notes)
    return Resolution(status="resolved", query=symbol, approved_symbol=exact.get(ensembl_id or "", symbol.upper()),
                      ensembl_id=ensembl_id, entrez_id=entrez_id, candidates=candidates, notes=notes)


class Toolbox:
    def __init__(self, ot: OpenTargetsClient, ncbi: NcbiClient | None, lit: EuropePmcClient,
                 clock: Callable[[], float] = time.perf_counter):
        self.ot, self.ncbi, self.lit = ot, ncbi, lit
        self._clock = clock
        self.tools: dict[str, ToolDef] = {t.name: t for t in (
            ToolDef("resolve_target", "Resolve a gene symbol to its canonical Ensembl and NCBI Gene IDs.",
                    ResolveTargetArgs, self._resolve_target),
            ToolDef("fetch_opentargets_drugs",
                    "Open Targets: approved symbol, biotype, top associated diseases, and drugs/clinical "
                    "candidates (ChEMBL IDs, clinical stage, mechanism of action) for an Ensembl gene ID.",
                    FetchDrugsArgs, self._fetch_drugs),
            ToolDef("fetch_ncbi_gene", "NCBI Gene record: official name, summary, aliases, chromosome.",
                    FetchGeneArgs, self._fetch_gene),
            ToolDef("search_literature",
                    "Search peer-reviewed PubMed-indexed abstracts in Europe PMC. Returns PMIDs, titles, years "
                    "and abstracts. Use focused queries.", SearchLiteratureArgs, self._search_literature),
        )}

    @classmethod
    def from_settings(cls, s: Settings, transport: httpx.BaseTransport | None = None) -> Toolbox:
        cache = DiskCache(s.cache_path, enabled=s.cache_enabled)

        def http(ns: str, throttle: Any = None) -> JsonHttp:
            return JsonHttp(ns, timeout_s=s.http_timeout_s, cache=cache, throttle=throttle, transport=transport)

        try:
            ncbi: NcbiClient | None = NcbiClient(http("ncbi", ncbi_throttle(s.ncbi_api_key)),
                                                 s.ncbi_email, s.ncbi_api_key)
        except ConfigError:
            ncbi = None
        return cls(OpenTargetsClient(http("opentargets")), ncbi, EuropePmcClient(http("europepmc")))

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name=t.name, description=t.description,
                         parameters=inline_refs(t.args_model.model_json_schema())) for t in self.tools.values()]

    def _http_counts(self) -> tuple[int, int]:
        clients = [c for c in (self.ot, self.ncbi, self.lit) if c is not None]
        return sum(c.http.n_requests for c in clients), sum(c.http.n_cache_hits for c in clients)

    def execute(self, name: str, args: dict[str, Any], step: int, trace: ToolTrace) -> Execution:
        t0 = self._clock()
        before = self._http_counts()
        tool = self.tools.get(name)
        output: ToolOutput | None = None
        error: str | None = None
        if tool is None:
            error = f"unknown tool '{name}'; available: {', '.join(self.tools)}"
        else:
            try:
                output = tool.fn(tool.args_model.model_validate(args or {}))
            except ValidationError as exc:
                error = "invalid arguments: " + "; ".join(
                    f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
            except (ApiError, httpx.HTTPError, ConfigError, LookupError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        after = self._http_counts()
        record = ToolCall(
            step=step, tool=name, args=dict(args or {}), ok=output is not None, error=error,
            latency_ms=round((self._clock() - t0) * 1000, 2),
            cached=after[0] == before[0] and after[1] > before[1],
            returned_ids=output.returned_ids if output else ReturnedIds(),
            evidence=output.evidence if output else [],
        )
        trace.tool_calls.append(record)
        return Execution(record, output)

    def _resolve_target(self, a: ResolveTargetArgs) -> ToolOutput:
        r = resolve_symbol(a.gene_symbol, self.ot, self.ncbi)
        ids = ReturnedIds(ensembl_ids=[r.ensembl_id] if r.ensembl_id else [],
                          entrez_ids=[r.entrez_id] if r.entrez_id else [])
        return ToolOutput(r.model_dump(), ids, result=r)

    def _fetch_drugs(self, a: FetchDrugsArgs) -> ToolOutput:
        p = self.ot.target(a.ensembl_id)
        if p is None:
            raise LookupError(f"Open Targets has no target {a.ensembl_id}")
        payload = {
            "ensembl_id": p.ensembl_id, "approved_symbol": p.approved_symbol, "approved_name": p.approved_name,
            "biotype": p.biotype, "disease_count": p.disease_count,
            "top_diseases": [{"name": d.name, "id": d.disease_id, "score": round(d.score, 3)} for d in p.diseases],
            "drug_count": p.drug_count,
            "drugs": [d.model_dump() for d in p.drugs],
        }
        ev = [EvidenceItem(id=d.chembl_id, kind="chembl", label=d.name) for d in p.drugs]
        ids = ReturnedIds(ensembl_ids=[p.ensembl_id], chembl_ids=[d.chembl_id for d in p.drugs])
        return ToolOutput(payload, ids, ev, result=p)

    def _fetch_gene(self, a: FetchGeneArgs) -> ToolOutput:
        if self.ncbi is None:
            raise ConfigError("NCBI is not configured (set NCBI_EMAIL)")
        if a.entrez_id:
            g = self.ncbi.gene(a.entrez_id)
        else:
            g = next((x for x in self.ncbi.search_gene(a.gene_symbol or "") if _exact(x.symbol, a.gene_symbol or "")),
                     None)
        if g is None:
            raise LookupError("no matching NCBI Gene record")
        payload = g.model_dump()
        payload["summary"] = g.summary[:1500]
        return ToolOutput(payload, ReturnedIds(entrez_ids=[g.entrez_id]), result=g)

    def _search_literature(self, a: SearchLiteratureArgs) -> ToolOutput:
        arts = self.lit.search(a.query, a.max_results)
        payload = {"query": a.query, "n_results": len(arts), "articles": [
            {"pmid": x.pmid, "title": x.title, "year": x.pub_year, "abstract": x.abstract_text[:ABSTRACT_CHARS]}
            for x in arts]}
        ev = [EvidenceItem(id=x.pmid, kind="pmid", label=x.title, year=x.pub_year) for x in arts]
        return ToolOutput(payload, ReturnedIds(pmids=[x.pmid for x in arts]), ev, result=arts)
