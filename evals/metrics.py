"""Deterministic scoring. No LLM is involved in judging: every number comes from the frozen
dataset, the run's final dossier, its verification report and its ToolTrace.

Per case
  target_resolved   resolver's Ensembl ID (trace step 0) == expected Ensembl ID
  drug precision    predicted drugs that are approved/Phase 3+ for the target / predicted drugs
  drug recall       expected drugs the dossier names / expected drugs
  false_positive    negative symbol: the run did not abstain
                    negative pair:   the forbidden drug appears in the final dossier

Aggregate (macro means over cases where a value is defined; sums for rates)
  hallucination_rate          unverified claims / all claims the model emitted
  citation_validity           cited IDs that a tool returned / all cited IDs
  negative_control_fp_rate    false positives / negative controls
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from bioagent.guardrails import drug_name_matches
from bioagent.schemas import RunResult

# USD per 1M tokens (input, output). Override with --price-in/--price-out.
PRICES: dict[tuple[str, str], tuple[float, float]] = {
    ("gemini", "gemini-3.5-flash-lite"): (0.30, 2.50),
}


def percentile(values: Iterable[float], q: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default), q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _mean(values: Iterable[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def cost_usd(provider: str, model: str, tin: int, tout: int,
             prices: dict[tuple[str, str], tuple[float, float]] | None = None) -> float | None:
    p = (prices or PRICES).get((provider, model))
    return None if p is None else (tin * p[0] + tout * p[1]) / 1e6


def _matches(name: str, chembl: str | None, ref: dict[str, Any]) -> bool:
    return (chembl is not None and chembl == ref.get("chembl_id")) or drug_name_matches(name, ref.get("name", ""))


def resolved_ensembl(result: RunResult) -> str | None:
    for call in result.trace.tool_calls:
        if call.step == 0 and call.tool == "resolve_target" and call.ok:
            ids = call.returned_ids.ensembl_ids
            return ids[0] if ids else None
    return None


def score_case(case: dict[str, Any], result: RunResult,
               prices: dict[tuple[str, str], tuple[float, float]] | None = None) -> dict[str, Any]:
    kind = case["type"]
    d, v = result.dossier, result.verification
    claims = d.drug_claims if d else []
    s: dict[str, Any] = {
        "case_id": case["case_id"], "type": kind, "area": case.get("area"), "gene_symbol": case["gene_symbol"],
        "status": result.status, "expected_status": case["expected_status"],
        "latency_s": round(result.total_latency_ms / 1000, 3),
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "steps_used": result.steps_used,
        "n_claims": v.n_claims if v else 0, "n_unverified": len(v.unverified_claims) if v else 0,
        "n_cited_ids": v.n_cited_ids if v else 0, "n_ungrounded_ids": len(v.ungrounded_ids) if v else 0,
    }
    provider = next((c.provider for c in result.trace.llm_calls), None)
    model = next((c.model for c in result.trace.llm_calls), None)
    s["cost_usd"] = (cost_usd(provider, model, result.input_tokens, result.output_tokens, prices)
                     if provider and model else 0.0)

    if kind in ("positive", "negative_pair"):
        s["target_resolved"] = resolved_ensembl(result) == case["ensembl_id"]

    if kind == "positive":
        expected = case.get("expected_drugs") or []
        predicted = [(c.drug_name, c.chembl_id) for c in claims]
        tp_pred = sum(any(_matches(n, ch, e) for e in expected) for n, ch in predicted)
        tp_exp = sum(any(_matches(n, ch, e) for n, ch in predicted) for e in expected)
        s["n_predicted_drugs"], s["n_expected_drugs"] = len(predicted), len(expected)
        s["drug_precision"] = _ratio(tp_pred, len(predicted))
        s["drug_recall"] = _ratio(tp_exp, len(expected))
        s["anchor_drug_found"] = any(_matches(n, ch, case["drug"]) for n, ch in predicted)
    elif kind == "negative_symbol":
        s["false_positive"] = result.status != "abstained"
    elif kind == "negative_pair":
        f = case["forbidden_drug"]
        s["false_positive"] = any(_matches(c.drug_name, c.chembl_id, f) for c in claims)
        # the model tried to assert it, but the verifier removed it (informational)
        s["fp_attempted"] = bool(v) and any(
            u.kind == "drug" and drug_name_matches(u.text, f["name"]) for u in v.unverified_claims)
    return s


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [s for s in scores if s["type"] == "positive"]
    neg = [s for s in scores if s["type"] in ("negative_symbol", "negative_pair")]
    resolvable = [s for s in scores if "target_resolved" in s]
    ran_llm = [s for s in scores if s["input_tokens"] + s["output_tokens"] > 0]
    costs = [s["cost_usd"] for s in scores]
    claims, unver = sum(s["n_claims"] for s in scores), sum(s["n_unverified"] for s in scores)
    cited, ungrounded = sum(s["n_cited_ids"] for s in scores), sum(s["n_ungrounded_ids"] for s in scores)
    lat = [s["latency_s"] for s in scores]
    return {
        "n_cases": len(scores), "n_positive": len(pos), "n_negative": len(neg),
        "n_errors": sum(s["status"] == "error" for s in scores),
        "status_match_rate": _ratio(sum(s["status"] == ("abstained" if s["expected_status"] == "abstain"
                                                         and s["type"] == "negative_symbol" else "ok")
                                        for s in scores), len(scores)),
        "target_resolution_acc": _ratio(sum(s["target_resolved"] for s in resolvable), len(resolvable)),
        "drug_precision": _mean(s.get("drug_precision") for s in pos),
        "drug_recall": _mean(s.get("drug_recall") for s in pos),
        "anchor_drug_recall": _ratio(sum(s["anchor_drug_found"] for s in pos), len(pos)),
        "hallucination_rate": _ratio(unver, claims),
        "negative_control_fp_rate": _ratio(sum(s["false_positive"] for s in neg), len(neg)),
        "negative_control_fp_attempted": sum(bool(s.get("fp_attempted")) for s in neg),
        "citation_validity": _ratio(cited - ungrounded, cited),
        "total_claims": claims, "total_cited_ids": cited,
        "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
        "mean_tokens": _mean(s["input_tokens"] + s["output_tokens"] for s in ran_llm),
        "est_cost_usd": None if any(c is None for c in costs) else round(sum(costs), 6),
        "est_cost_per_query_usd": None if any(c is None for c in costs) or not ran_llm
        else round(sum(costs) / len(ran_llm), 6),
    }


# ------------------------------------------------------------------ markdown
_ROWS = [
    ("target_resolution_acc", "Target resolution accuracy", "pct"),
    ("drug_precision", "Drug precision (macro)", "pct"),
    ("drug_recall", "Drug recall (macro)", "pct"),
    ("anchor_drug_recall", "Anchor drug named", "pct"),
    ("hallucination_rate", "Hallucination rate (unverified / emitted claims)", "pct"),
    ("citation_validity", "Citation validity (cited IDs found in trace)", "pct"),
    ("negative_control_fp_rate", "Negative-control false-positive rate", "pct"),
    ("negative_control_fp_attempted", "Negative pairs the model tried (verifier removed)", "int"),
    ("status_match_rate", "Expected status (ok / abstain)", "pct"),
    ("latency_p50_s", "Latency p50 (s)", "f2"),
    ("latency_p95_s", "Latency p95 (s)", "f2"),
    ("mean_tokens", "Mean tokens per LLM run", "f0"),
    ("est_cost_per_query_usd", "Est. cost per LLM run (USD)", "usd"),
    ("est_cost_usd", "Est. total cost (USD)", "usd"),
    ("n_errors", "Errors", "int"),
]


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "n/a"
    return {"pct": f"{v:.1%}", "f2": f"{v:.2f}", "f0": f"{v:,.0f}", "usd": f"${v:.4f}", "int": f"{v}"}[kind]


def to_markdown(metrics: dict[str, Any], meta: dict[str, Any], scores: list[dict[str, Any]]) -> str:
    lines = [
        "# bioagent evaluation summary", "",
        f"- Run: {meta.get('started_at', '')}  |  provider/model: {meta.get('provider')}/{meta.get('model')}",
        f"- Dataset: `{meta.get('dataset')}` (sha256 {str(meta.get('dataset_sha256', ''))[:12]}), "
        f"{metrics['n_cases']} cases ({metrics['n_positive']} positive, {metrics['n_negative']} negative)"
        + (f", limited to {meta['limit']}" if meta.get("limit") else ""),
        "", "| Metric | Value |", "|---|---|",
    ]
    lines += [f"| {label} | {_fmt(metrics.get(key), kind)} |" for key, label, kind in _ROWS]
    areas = sorted({s["area"] for s in scores if s.get("area")})
    if areas:
        lines += ["", "## By therapeutic area (positive cases)", "",
                  "| Area | n | Resolution | Drug precision | Drug recall |", "|---|---|---|---|---|"]
        for a in areas:
            sub = [s for s in scores if s.get("area") == a and s["type"] == "positive"]
            lines.append(f"| {a} | {len(sub)} | {_fmt(_ratio(sum(s['target_resolved'] for s in sub), len(sub)), 'pct')}"
                         f" | {_fmt(_mean(s.get('drug_precision') for s in sub), 'pct')}"
                         f" | {_fmt(_mean(s.get('drug_recall') for s in sub), 'pct')} |")
    bad = [s for s in scores if s["status"] == "error" or s.get("false_positive")
           or s.get("target_resolved") is False]
    if bad:
        lines += ["", "## Cases to inspect", ""]
        lines += [f"- `{s['case_id']}` {s['gene_symbol']}: status={s['status']}"
                  + (", false positive" if s.get("false_positive") else "")
                  + (", wrong/no target" if s.get("target_resolved") is False else "") for s in bad]
    lines += ["", "Precision/recall are macro means over positive cases where the value is defined "
              "(precision needs at least one predicted drug). Ground truth = Open Targets drugs at clinical "
              "stage Phase 3 or approved, frozen when the dataset was built.", ""]
    return "\n".join(lines)
