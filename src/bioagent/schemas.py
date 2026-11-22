"""Pydantic v2 models shared by clients, tools, agent, guardrails and evals."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# HGNC-style symbol: letters/digits/hyphen, 1-20 chars, no leading/trailing hyphen.
SYMBOL_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,18}[A-Za-z0-9])?$"
ENSEMBL_GENE_PATTERN = r"^ENSG\d{11}$"
CHEMBL_PATTERN = r"^CHEMBL\d+$"
PMID_PATTERN = r"^\d{1,9}$"


def normalize_evidence_id(raw: str) -> str:
    """'PMID: 123' -> '123', 'chembl939' -> 'CHEMBL939'; anything else is stripped and kept."""
    s = str(raw).strip()
    m = re.fullmatch(r"(?i)pmid\s*:?\s*(\d+)", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"(?i)chembl\d+", s):
        return s.upper()
    return s


def norm_name(name: str) -> str:
    """Case/punctuation-insensitive drug-name key."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


# ----------------------------------------------------------------- client results
class SearchHit(BaseModel):
    id: str
    name: str
    entity: str
    approved_symbol: str | None = None
    score: float = 0.0


class DiseaseAssociation(BaseModel):
    disease_id: str
    name: str
    score: float


class KnownDrug(BaseModel):
    chembl_id: str
    name: str
    drug_type: str | None = None
    clinical_stage: str | None = None
    mechanisms: list[str] = Field(default_factory=list)
    diseases: list[str] = Field(default_factory=list)


class TargetProfile(BaseModel):
    ensembl_id: str
    approved_symbol: str
    approved_name: str | None = None
    biotype: str | None = None
    diseases: list[DiseaseAssociation] = Field(default_factory=list)
    disease_count: int = 0
    drugs: list[KnownDrug] = Field(default_factory=list)
    drug_count: int = 0


class GeneRecord(BaseModel):
    entrez_id: str
    symbol: str
    description: str = ""
    chromosome: str = ""
    map_location: str = ""
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    taxid: int | None = None


class PubmedRecord(BaseModel):
    pmid: str
    title: str
    year: int | None = None
    journal: str = ""


class Article(BaseModel):
    pmid: str
    title: str
    pub_year: int | None = None
    abstract_text: str = ""
    doi: str | None = None


# ----------------------------------------------------------------- tool trace
class ReturnedIds(BaseModel):
    ensembl_ids: list[str] = Field(default_factory=list)
    entrez_ids: list[str] = Field(default_factory=list)
    chembl_ids: list[str] = Field(default_factory=list)
    pmids: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Something a claim can cite. Only items returned by a tool in this run are citable."""

    id: str
    kind: Literal["chembl", "pmid"]
    label: str
    year: int | None = None

    @property
    def url(self) -> str:
        if self.kind == "pmid":
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.id}/"
        return f"https://www.ebi.ac.uk/chembl/compound_report_card/{self.id}/"


class ToolCall(BaseModel):
    """One tool execution, as recorded in the trace (not what the model asked for)."""

    step: int                                   # 0 = pre-flight guardrail, 1.. = agent steps
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None
    latency_ms: float
    cached: bool = False
    returned_ids: ReturnedIds = Field(default_factory=ReturnedIds)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class LLMCall(BaseModel):
    step: int
    kind: Literal["tool_step", "finalize"]
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    n_tool_calls: int = 0


class ToolTrace(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    def evidence_index(self) -> dict[str, EvidenceItem]:
        """Every citable item returned by a successful tool call in this run, by id."""
        out: dict[str, EvidenceItem] = {}
        for call in self.tool_calls:
            if call.ok:
                for item in call.evidence:
                    out.setdefault(item.id, item)
        return out

    def returned(self) -> ReturnedIds:
        merged = ReturnedIds()
        for call in self.tool_calls:
            if not call.ok:
                continue
            for field in ReturnedIds.model_fields:
                bucket = getattr(merged, field)
                bucket.extend(i for i in getattr(call.returned_ids, field) if i not in bucket)
        return merged


# ----------------------------------------------------------------- resolution
class Resolution(BaseModel):
    status: Literal["resolved", "ambiguous", "not_found"]
    query: str
    approved_symbol: str | None = None
    ensembl_id: str | None = None
    entrez_id: str | None = None
    candidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Abstention(BaseModel):
    status: Literal["abstained"] = "abstained"
    query: str
    reason: Literal["invalid_symbol", "not_found", "ambiguous", "lookup_failed"]
    message: str
    candidates: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- claims + dossier
class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ids: list[str] = Field(
        min_length=1, max_length=10,
        description="ChEMBL IDs (e.g. CHEMBL939) or PMIDs (digits only) returned by a tool in this run.",
    )
    verified: bool | None = Field(None, description="Set by the citation verifier, never by the model.")

    @field_validator("evidence_ids")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for raw in v:
            s = normalize_evidence_id(raw)
            if s and s not in out:
                out.append(s)
        if not out:
            raise ValueError("at least one evidence id is required")
        return out


class DrugClaim(Claim):
    drug_name: str = Field(min_length=1, max_length=120)
    chembl_id: str | None = Field(None, pattern=CHEMBL_PATTERN)
    clinical_stage: str | None = Field(None, max_length=40)
    mechanism: str | None = Field(None, max_length=200)

    @field_validator("chembl_id", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


class MechanismClaim(Claim):
    statement: str = Field(min_length=10, max_length=600)


class DossierDraft(BaseModel):
    """The final structured answer: a short summary plus evidence-backed claims."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1500)
    drug_claims: list[DrugClaim] = Field(default_factory=list, max_length=25)
    mechanism_claims: list[MechanismClaim] = Field(default_factory=list, max_length=15)


class TargetDossier(BaseModel):
    query: str
    approved_symbol: str
    ensembl_id: str | None = None
    entrez_id: str | None = None
    summary: str
    drug_claims: list[DrugClaim] = Field(default_factory=list)
    mechanism_claims: list[MechanismClaim] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UnverifiedClaim(BaseModel):
    kind: Literal["drug", "mechanism"]
    text: str
    unknown_ids: list[str] = Field(default_factory=list)
    reasons: list[str]


class Verification(BaseModel):
    mode: Literal["strip", "flag"]
    n_claims: int
    n_verified: int
    grounding_score: float | None            # verified / total claims; None when there were no claims
    unverified_claims: list[UnverifiedClaim] = Field(default_factory=list)
    ungrounded_ids: list[str] = Field(default_factory=list)   # cited IDs no tool returned


class RunResult(BaseModel):
    status: Literal["ok", "abstained", "aborted_budget", "error"]
    query: str
    dossier: TargetDossier | None = None
    abstention: Abstention | None = None
    verification: Verification | None = None
    trace: ToolTrace = Field(default_factory=ToolTrace)
    message: str | None = None
    steps_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0


class EvalResult(BaseModel):
    """One benchmark row (filled by evals/run_eval.py in a later phase)."""

    gene_symbol: str
    expected_ids: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    precision: float | None = None
    recall: float | None = None
    n_claims: int = 0
    n_unverified: int = 0
    abstained: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
