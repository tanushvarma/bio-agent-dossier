"""Streamlit front end. UI only: every computation comes from the `bioagent` package.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").is_dir():                       # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from bioagent.agent import BioAgent  # noqa: E402
from bioagent.config import ConfigError, get_settings  # noqa: E402
from bioagent.pricing import cost_usd  # noqa: E402
from bioagent.schemas import RunResult  # noqa: E402

RESULTS = Path(os.environ.get("BIOAGENT_EVAL_RESULTS", ROOT / "evals" / "results" / "latest.json"))


@st.cache_resource(show_spinner=False)
def get_agent() -> BioAgent:
    return BioAgent.from_settings(get_settings())


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _run_cost(res: RunResult) -> float | None:
    call = res.trace.llm_calls[0] if res.trace.llm_calls else None
    return cost_usd(call.provider, call.model, res.input_tokens, res.output_tokens) if call else 0.0


@st.cache_data(show_spinner=False)
def _load_results(path: str, mtime: float) -> dict[str, Any]:     # mtime busts the cache after a new run
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_verification(res: RunResult) -> None:
    v = res.verification
    if v is None:
        return
    flagged = len(v.unverified_claims)
    if v.n_claims == 0:
        st.badge("No claims emitted", color="gray")
    elif flagged == 0:
        st.badge(f"Verified: {v.n_verified}/{v.n_claims} claims grounded in tool output", color="green")
    else:
        verb = "removed" if v.mode == "strip" else "flagged"
        st.badge(f"Flagged: {flagged}/{v.n_claims} claims {verb}, not grounded in tool output", color="orange")

    c = st.columns(5)
    c[0].metric("Grounding score", _pct(v.grounding_score))
    c[1].metric("Ungrounded IDs", len(v.ungrounded_ids))
    c[2].metric("Latency", f"{res.total_latency_ms / 1000:.1f} s")
    c[3].metric("Tokens", f"{res.input_tokens + res.output_tokens:,}")
    cost = _run_cost(res)
    c[4].metric("Est. cost", "n/a" if cost is None else f"${cost:.4f}")


def render_dossier(res: RunResult) -> None:
    d = res.dossier
    if d is None:
        return
    links = []
    if d.ensembl_id:
        links.append(f"[{d.ensembl_id}](https://platform.opentargets.org/target/{d.ensembl_id})")
    if d.entrez_id:
        links.append(f"[NCBI Gene {d.entrez_id}](https://www.ncbi.nlm.nih.gov/gene/{d.entrez_id})")
    st.subheader(d.approved_symbol)
    st.caption(" | ".join([*links, f"{d.provider}/{d.model}", f"{res.steps_used} LLM steps"]))
    st.write(d.summary)

    st.markdown("**Drug claims**")
    if d.drug_claims:
        st.dataframe(pd.DataFrame([{
            "drug": c.drug_name, "ChEMBL ID": c.chembl_id, "clinical stage": c.clinical_stage,
            "mechanism": c.mechanism, "evidence": ", ".join(c.evidence_ids), "verified": c.verified,
        } for c in d.drug_claims]), hide_index=True, width="stretch")
    else:
        st.caption("None.")

    st.markdown("**Mechanism claims**")
    if d.mechanism_claims:
        for c in d.mechanism_claims:
            mark = "" if c.verified else " (not verified)"
            st.markdown(f"- {c.statement} [{', '.join(c.evidence_ids)}]{mark}")
    else:
        st.caption("None.")

    if d.evidence:
        st.markdown("**Cited evidence**")
        st.dataframe(pd.DataFrame([{"id": e.id, "type": e.kind, "label": e.label, "year": e.year, "link": e.url}
                                   for e in d.evidence]),
                     column_config={"link": st.column_config.LinkColumn("link")}, hide_index=True, width="stretch")

    v = res.verification
    if v and v.unverified_claims:
        with st.expander(f"Claims that failed verification ({len(v.unverified_claims)})"):
            st.dataframe(pd.DataFrame([{"kind": u.kind, "claim": u.text, "unknown IDs": ", ".join(u.unknown_ids),
                                        "reasons": "; ".join(u.reasons)} for u in v.unverified_claims]),
                         hide_index=True, width="stretch")


def render_trace(res: RunResult) -> None:
    with st.expander(f"Tool trace ({len(res.trace.tool_calls)} tool calls, {len(res.trace.llm_calls)} LLM calls)"):
        if res.trace.tool_calls:
            st.dataframe(pd.DataFrame([{
                "step": t.step, "tool": t.tool, "ok": t.ok, "latency ms": t.latency_ms, "cached": t.cached,
                "IDs returned": sum(len(getattr(t.returned_ids, f)) for f in type(t.returned_ids).model_fields),
                "error": t.error,
            } for t in res.trace.tool_calls]), hide_index=True, width="stretch")
        st.json(res.trace.model_dump(mode="json"), expanded=False)


def dossier_tab() -> None:
    with st.form("run"):
        symbol = st.text_input("Gene symbol (HGNC)", placeholder="EGFR", max_chars=20)
        submitted = st.form_submit_button("Build dossier", type="primary")
    if submitted and symbol.strip():
        try:
            agent = get_agent()
        except ConfigError as exc:
            st.error(f"Configuration error: {exc}")
            return
        with st.spinner(f"Running agent for {symbol.strip().upper()}..."):
            st.session_state["result"] = agent.run(symbol.strip())

    res: RunResult | None = st.session_state.get("result")
    if res is None:
        st.caption("Enter a symbol. Unknown, ambiguous or alias symbols are refused rather than guessed.")
        return
    if res.status == "abstained" and res.abstention:
        st.warning(f"Abstained ({res.abstention.reason}): {res.abstention.message}")
    elif res.status == "aborted_budget":
        st.error(res.message or "Token budget exceeded.")
    elif res.status == "error":
        st.error(f"Run failed: {res.message}")
    render_verification(res)
    render_dossier(res)
    render_trace(res)
    st.download_button("Download result JSON", res.model_dump_json(indent=2), f"{res.query}_dossier.json",
                       "application/json")


HEADLINE = [("target_resolution_acc", "Target resolution", "pct"), ("drug_precision", "Drug precision", "pct"),
            ("drug_recall", "Drug recall", "pct"), ("hallucination_rate", "Hallucination rate", "pct"),
            ("citation_validity", "Citation validity", "pct"),
            ("negative_control_fp_rate", "Negative-control FP rate", "pct"),
            ("latency_p50_s", "Latency p50 (s)", "f2"), ("latency_p95_s", "Latency p95 (s)", "f2"),
            ("mean_tokens", "Mean tokens", "f0"), ("est_cost_per_query_usd", "Cost / query (USD)", "usd")]


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "n/a"
    return {"pct": f"{v:.1%}", "f2": f"{v:.2f}", "f0": f"{v:,.0f}", "usd": f"${v:.4f}"}[kind]


def _is_failure(s: dict[str, Any]) -> bool:
    return (s["status"] == "error" or bool(s.get("false_positive")) or s.get("target_resolved") is False
            or (s["type"] == "negative_symbol" and s["status"] != "abstained")
            or (s["type"] != "negative_symbol" and s["status"] != "ok") or s.get("n_unverified", 0) > 0)


def eval_tab() -> None:
    if not RESULTS.exists():
        st.info(f"No results at `{RESULTS}`. Build the dataset and run the benchmark:\n\n"
                "`python evals/build_dataset.py` then `python evals/run_eval.py`")
        return
    report = _load_results(str(RESULTS), RESULTS.stat().st_mtime)
    meta, m = report["meta"], report["metrics"]
    st.caption(f"{meta.get('started_at', '')} | {meta.get('provider')}/{meta.get('model')} | "
               f"{m['n_cases']} cases ({m['n_positive']} positive, {m['n_negative']} negative)"
               + (f" | limited to {meta['limit']}" if meta.get("limit") else "") + f" | {m['n_errors']} errors")
    for row in (HEADLINE[:5], HEADLINE[5:]):
        cols = st.columns(5)
        for col, (key, label, kind) in zip(cols, row, strict=True):
            col.metric(label, _fmt(m.get(key), kind))

    scores = [c["score"] for c in report["cases"]]
    df = pd.DataFrame([{
        "case": s["case_id"], "type": s["type"], "area": s.get("area"), "gene": s["gene_symbol"],
        "status": s["status"], "target ok": s.get("target_resolved"), "precision": s.get("drug_precision"),
        "recall": s.get("drug_recall"), "claims": s["n_claims"], "unverified": s["n_unverified"],
        "false positive": s.get("false_positive"), "latency s": s["latency_s"],
        "tokens": s["input_tokens"] + s["output_tokens"], "failure": _is_failure(s),
    } for s in scores])
    only = st.toggle("Only cases needing inspection", value=True,
                     help="Errors, wrong target, unexpected status, false positives, or any unverified claim.")
    view = df[df["failure"]] if only else df
    st.dataframe(view.drop(columns="failure"), hide_index=True, width="stretch",
                 column_config={"precision": st.column_config.NumberColumn(format="percent"),
                                "recall": st.column_config.NumberColumn(format="percent")})
    if view.empty:
        return
    pick = st.selectbox("Inspect case", view["case"].tolist())
    item = next(c for c in report["cases"] if c["score"]["case_id"] == pick)
    res = RunResult.model_validate(item["result"])
    left, right = st.columns(2)
    left.markdown("**Expected**")
    left.json(item["case"], expanded=False)
    right.markdown("**Score**")
    right.json(item["score"], expanded=False)
    render_verification(res)
    render_dossier(res)
    render_trace(res)


def main() -> None:
    st.set_page_config(page_title="bioagent: target dossiers", layout="wide")
    st.title("bioagent")
    st.caption("Grounded target dossiers from Open Targets, NCBI and Europe PMC. Every claim is checked "
               "against the IDs the tools actually returned.")
    t1, t2 = st.tabs(["Target Dossier", "Eval Benchmark"])
    with t1:
        dossier_tab()
    with t2:
        eval_tab()


main()
