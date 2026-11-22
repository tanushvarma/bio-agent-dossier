"""Deterministic guardrails (no LLM calls): pre-flight entity resolution, post-flight citation
verification against the run's ToolTrace, and a token budget."""
from __future__ import annotations

import re
from typing import Literal

from .schemas import (
    SYMBOL_PATTERN,
    Abstention,
    DrugClaim,
    MechanismClaim,
    Resolution,
    TargetDossier,
    ToolTrace,
    UnverifiedClaim,
    Verification,
    norm_name,
)
from .tools import Toolbox


def preflight(gene_symbol: str, toolbox: Toolbox, trace: ToolTrace) -> Resolution | Abstention:
    """Resolve the input to canonical IDs or abstain. Recorded in the trace as step 0."""
    symbol = (gene_symbol or "").strip()
    if not re.fullmatch(SYMBOL_PATTERN, symbol):
        return Abstention(query=symbol, reason="invalid_symbol",
                          message="Not a valid gene symbol (letters, digits and hyphens, up to 20 characters).")
    ex = toolbox.execute("resolve_target", {"gene_symbol": symbol}, step=0, trace=trace)
    if ex.output is None:
        return Abstention(query=symbol, reason="lookup_failed",
                          message=f"Could not reach the resolver: {ex.record.error}")
    res: Resolution = ex.output.result
    if res.status == "ambiguous":
        return Abstention(query=symbol, reason="ambiguous", candidates=res.candidates,
                          message=f"'{symbol}' matches more than one gene; give the exact HGNC symbol.")
    if res.status == "not_found":
        hint = f" Did you mean: {', '.join(res.candidates)}?" if res.candidates else ""
        return Abstention(query=symbol, reason="not_found", candidates=res.candidates,
                          message=f"'{symbol}' is not an approved human gene symbol.{hint}")
    return res


def drug_name_matches(claimed: str, returned: str) -> bool:
    a, b = norm_name(claimed), norm_name(returned)
    return bool(a) and (a == b or a.startswith(b + " ") or b.startswith(a + " "))   # tolerates salt forms


def verify_citations(dossier: TargetDossier, trace: ToolTrace,
                     mode: Literal["strip", "flag"] = "strip") -> tuple[TargetDossier, Verification]:
    """Check every claim against what tools actually returned in this run.

    A claim is verified when it keeps at least one evidence ID that a tool returned, and, for
    drug claims, the drug name (and ChEMBL ID, if given) match a drug a tool returned.
    Evidence IDs no tool returned are removed in 'strip' mode (and the claim dropped if
    unverified); 'flag' mode keeps everything and marks verified=False."""
    index = trace.evidence_index()
    drug_by_id = {i: e.label for i, e in index.items() if e.kind == "chembl"}
    unverified: list[UnverifiedClaim] = []
    ungrounded: list[str] = []
    kept_drugs: list[DrugClaim] = []
    kept_mech: list[MechanismClaim] = []

    def check(claim: DrugClaim | MechanismClaim) -> tuple[bool, list[str], list[str]]:
        unknown = [i for i in claim.evidence_ids if i not in index]
        reasons = [f"evidence IDs not returned by any tool: {', '.join(unknown)}"] if unknown else []
        if len(unknown) == len(claim.evidence_ids):
            reasons.append("no grounded evidence")
        if isinstance(claim, DrugClaim):
            if claim.chembl_id and claim.chembl_id not in drug_by_id:
                reasons.append(f"{claim.chembl_id} was not returned by a drug tool")
            elif claim.chembl_id and not drug_name_matches(claim.drug_name, drug_by_id[claim.chembl_id]):
                reasons.append(f"{claim.chembl_id} is {drug_by_id[claim.chembl_id]}, not {claim.drug_name}")
            if not any(drug_name_matches(claim.drug_name, n) for n in drug_by_id.values()):
                reasons.append(f"drug '{claim.drug_name}' was not returned by a drug tool")
        fatal = [r for r in reasons if not r.startswith("evidence IDs not returned")]
        return not fatal, unknown, reasons

    for claim in [*dossier.drug_claims, *dossier.mechanism_claims]:
        ok, unknown, reasons = check(claim)
        ungrounded.extend(i for i in unknown if i not in ungrounded)
        new = claim.model_copy(update={"verified": ok})
        if mode == "strip" and unknown and ok:
            new = new.model_copy(update={"evidence_ids": [i for i in claim.evidence_ids if i not in unknown]})
        if not ok:
            text = claim.drug_name if isinstance(claim, DrugClaim) else claim.statement
            unverified.append(UnverifiedClaim(kind="drug" if isinstance(claim, DrugClaim) else "mechanism",
                                              text=text, unknown_ids=unknown, reasons=reasons))
        if ok or mode == "flag":
            (kept_drugs if isinstance(new, DrugClaim) else kept_mech).append(new)  # type: ignore[arg-type]

    n = len(dossier.drug_claims) + len(dossier.mechanism_claims)
    n_ok = n - len(unverified)
    n_cited = len({i for c in [*dossier.drug_claims, *dossier.mechanism_claims] for i in c.evidence_ids})
    cited = {i for c in [*kept_drugs, *kept_mech] for i in c.evidence_ids if i in index}
    verified = dossier.model_copy(update={
        "drug_claims": kept_drugs, "mechanism_claims": kept_mech,
        "evidence": [index[i] for i in sorted(cited)],
    })
    report = Verification(mode=mode, n_claims=n, n_verified=n_ok, n_cited_ids=n_cited,
                          grounding_score=round(n_ok / n, 4) if n else None,
                          unverified_claims=unverified, ungrounded_ids=ungrounded)
    return verified, report


class TokenBudget:
    """Input + output tokens across every LLM call in one run."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.used += input_tokens + output_tokens

    @property
    def exceeded(self) -> bool:
        return self.used > self.max_tokens

    def message(self) -> str:
        return f"Token budget exceeded: {self.used:,} used, limit {self.max_tokens:,}. Run aborted."
