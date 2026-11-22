"""Build the frozen benchmark: evals/dataset_50.jsonl (+ dataset_50.meta.json).

Only gene symbols, drug names and therapeutic areas are written by hand (SEEDS below). Every
identifier and fact in the dataset is fetched live and checked:

  positive (50)          symbol resolves to exactly one Ensembl + NCBI Gene ID; the anchor drug is
                         listed by Open Targets for that target at Phase 3 or approved; it has a
                         mechanism of action; PubMed has >= 1 paper naming both drug and gene.
  negative_symbol (5)    format-valid symbols that resolve to nothing in Open Targets or NCBI.
  negative_pair (5)      a real target + a real drug that Open Targets does NOT link to that target.

Seeds that fail a check are skipped (spares fill the quota) and listed in the meta file.
Run once and commit the output; re-running refuses to overwrite unless --force.

Note: the spec asked for Open Targets `knownDrugs`; Open Targets removed it in release 26.03.
Ground truth uses its replacement, `drugAndClinicalCandidates`, through bioagent's client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from bioagent.clients import ApiError, DiskCache, JsonHttp, NcbiClient, OpenTargetsClient  # noqa: E402
from bioagent.clients.ncbi import ncbi_throttle  # noqa: E402
from bioagent.config import ConfigError, get_settings  # noqa: E402
from bioagent.guardrails import drug_name_matches  # noqa: E402
from bioagent.schemas import SYMBOL_PATTERN, KnownDrug  # noqa: E402
from bioagent.tools import resolve_symbol  # noqa: E402

OUT = ROOT / "evals" / "dataset_50.jsonl"
QUOTAS = {"oncology": 13, "immunology": 12, "cardiometabolic": 13, "neurology": 12}

# (area, HGNC symbol, anchor drug). More than the quota per area; failures fall through to spares.
SEEDS: list[tuple[str, str, str]] = [
    ("oncology", "EGFR", "OSIMERTINIB"), ("oncology", "ERBB2", "TRASTUZUMAB"),
    ("oncology", "BRAF", "VEMURAFENIB"), ("oncology", "ABL1", "IMATINIB"), ("oncology", "ALK", "CRIZOTINIB"),
    ("oncology", "PDCD1", "PEMBROLIZUMAB"), ("oncology", "CD274", "ATEZOLIZUMAB"), ("oncology", "BTK", "IBRUTINIB"),
    ("oncology", "PARP1", "OLAPARIB"), ("oncology", "CDK4", "PALBOCICLIB"), ("oncology", "AR", "ENZALUTAMIDE"),
    ("oncology", "MS4A1", "RITUXIMAB"), ("oncology", "BCL2", "VENETOCLAX"), ("oncology", "VEGFA", "BEVACIZUMAB"),
    ("oncology", "MET", "CAPMATINIB"), ("oncology", "FLT3", "MIDOSTAURIN"), ("oncology", "KRAS", "SOTORASIB"),
    ("oncology", "PIK3CA", "ALPELISIB"), ("oncology", "MTOR", "EVEROLIMUS"), ("oncology", "CD38", "DARATUMUMAB"),
    ("oncology", "ESR1", "FULVESTRANT"), ("oncology", "PSMB5", "BORTEZOMIB"),
    ("immunology", "TNF", "ADALIMUMAB"), ("immunology", "IL6R", "TOCILIZUMAB"), ("immunology", "JAK1", "UPADACITINIB"),
    ("immunology", "IL17A", "SECUKINUMAB"), ("immunology", "IL23A", "GUSELKUMAB"),
    ("immunology", "IL12B", "USTEKINUMAB"),
    ("immunology", "IL4R", "DUPILUMAB"), ("immunology", "ITGA4", "NATALIZUMAB"), ("immunology", "IL1B", "CANAKINUMAB"),
    ("immunology", "TYK2", "DEUCRAVACITINIB"), ("immunology", "IL5", "MEPOLIZUMAB"),
    ("immunology", "TNFSF13B", "BELIMUMAB"), ("immunology", "IL1R1", "ANAKINRA"), ("immunology", "JAK3", "TOFACITINIB"),
    ("immunology", "JAK2", "RUXOLITINIB"),
    ("cardiometabolic", "HMGCR", "ATORVASTATIN"), ("cardiometabolic", "PCSK9", "EVOLOCUMAB"),
    ("cardiometabolic", "SLC5A2", "EMPAGLIFLOZIN"), ("cardiometabolic", "GLP1R", "SEMAGLUTIDE"),
    ("cardiometabolic", "DPP4", "SITAGLIPTIN"), ("cardiometabolic", "PPARG", "PIOGLITAZONE"),
    ("cardiometabolic", "NPC1L1", "EZETIMIBE"), ("cardiometabolic", "ACE", "LISINOPRIL"),
    ("cardiometabolic", "AGTR1", "LOSARTAN"), ("cardiometabolic", "ADRB1", "METOPROLOL"),
    ("cardiometabolic", "F10", "RIVAROXABAN"), ("cardiometabolic", "P2RY12", "CLOPIDOGREL"),
    ("cardiometabolic", "NR3C2", "SPIRONOLACTONE"), ("cardiometabolic", "VKORC1", "WARFARIN"),
    ("cardiometabolic", "CACNA1C", "AMLODIPINE"), ("cardiometabolic", "ANGPTL3", "EVINACUMAB"),
    ("cardiometabolic", "MTTP", "LOMITAPIDE"),
    ("neurology", "ACHE", "DONEPEZIL"), ("neurology", "DRD2", "HALOPERIDOL"), ("neurology", "SLC6A4", "FLUOXETINE"),
    ("neurology", "HTR1B", "SUMATRIPTAN"), ("neurology", "CALCRL", "ERENUMAB"), ("neurology", "SV2A", "LEVETIRACETAM"),
    ("neurology", "MAOB", "RASAGILINE"), ("neurology", "COMT", "ENTACAPONE"), ("neurology", "OPRM1", "BUPRENORPHINE"),
    ("neurology", "HTR2A", "PIMAVANSERIN"), ("neurology", "CHRNA4", "VARENICLINE"),
    ("neurology", "SLC6A3", "METHYLPHENIDATE"), ("neurology", "S1PR1", "FINGOLIMOD"),
    ("neurology", "GABRA1", "ZOLPIDEM"), ("neurology", "APP", "LECANEMAB"),
]

# the first 5 that resolve to nothing are used
FAKE_SYMBOLS = ["FAKEG1", "XR999B", "QZX7KQ", "NOTAGENE42", "ZZQ9X1", "BLRGH77", "XQ12ZZK"]
INVALID_PAIRS = [("EGFR", "DONEPEZIL"), ("ACHE", "OSIMERTINIB"), ("HMGCR", "RITUXIMAB"),
                 ("TNF", "ATORVASTATIN"), ("DRD2", "TRASTUZUMAB"), ("PCSK9", "IMATINIB"), ("BRAF", "SEMAGLUTIDE")]


def stage_rank(stage: str | None) -> float:
    """Approved = 4, Phase 4 = 4, Phase 3 = 3, Phase 2/3 = 2.5, lower = 0-2. Unknown strings = 0."""
    s = (stage or "").upper()
    if "APPROV" in s:
        return 4.0
    if re.search(r"PHASE[\s_]*(II|2)[\s_/]*(III|3)\b", s):
        return 2.5
    for pat, rank in ((r"PHASE[\s_]*(IV|4)\b", 4.0), (r"PHASE[\s_]*(III|3)\b", 3.0),
                      (r"PHASE[\s_]*(II|2)\b", 2.0), (r"PHASE[\s_]*(I|1)\b", 1.0)):
        if re.search(pat, s):
            return rank
    return 0.0


def _drug_dict(d: KnownDrug) -> dict[str, Any]:
    return {"chembl_id": d.chembl_id, "name": d.name, "clinical_stage": d.clinical_stage}


def build_positive(area: str, symbol: str, drug: str, ot: OpenTargetsClient, ncbi: NcbiClient,
                   stages: Counter) -> tuple[dict[str, Any] | None, str | None]:
    res = resolve_symbol(symbol, ot, ncbi)
    if res.status != "resolved" or not (res.ensembl_id and res.entrez_id):
        return None, f"resolution {res.status} (ensembl={res.ensembl_id}, entrez={res.entrez_id})"
    profile = ot.target(res.ensembl_id, max_drugs=1000)
    if profile is None:
        return None, "Open Targets has no target record"
    stages.update(d.clinical_stage or "<none>" for d in profile.drugs)
    expected = [d for d in profile.drugs if stage_rank(d.clinical_stage) >= 3]
    anchor = next((d for d in expected if drug_name_matches(drug, d.name)), None)
    if anchor is None:
        return None, f"{drug} not listed at Phase 3+/approved for {symbol}"
    if not anchor.mechanisms:
        return None, f"{anchor.name} has no mechanism of action for {symbol}"
    pubs = ncbi.search_pubmed(f'"{anchor.name}"[tiab] AND "{symbol}"[tiab]', retmax=3)
    if not pubs:
        return None, f"no PubMed paper names both {anchor.name} and {symbol}"
    return {
        "type": "positive", "area": area, "gene_symbol": symbol, "expected_status": "ok",
        "ensembl_id": res.ensembl_id, "entrez_id": res.entrez_id, "approved_symbol": profile.approved_symbol,
        "drug": {**_drug_dict(anchor), "mechanism_of_action": anchor.mechanisms[0]},
        "reference_pmids": [p.pmid for p in pubs],
        "expected_drugs": [_drug_dict(d) for d in expected],
    }, None


def build_negative_symbol(symbol: str, ot: OpenTargetsClient, ncbi: NcbiClient) -> tuple[dict | None, str | None]:
    if not re.fullmatch(SYMBOL_PATTERN, symbol):
        return None, "not format-valid (would test the regex, not the resolver)"
    res = resolve_symbol(symbol, ot, ncbi)
    if res.status == "resolved":
        return None, f"unexpectedly resolves to {res.ensembl_id or res.entrez_id}"
    return {"type": "negative_symbol", "gene_symbol": symbol, "expected_status": "abstain"}, None


def build_negative_pair(symbol: str, drug: str, ot: OpenTargetsClient,
                        ncbi: NcbiClient) -> tuple[dict | None, str | None]:
    res = resolve_symbol(symbol, ot, ncbi)
    if res.status != "resolved" or not res.ensembl_id:
        return None, f"target {symbol} did not resolve"
    hit = next((h for h in ot.search(drug, ("drug",)) if h.entity == "drug" and drug_name_matches(drug, h.name)), None)
    if hit is None:
        return None, f"drug {drug} not found in Open Targets"
    profile = ot.target(res.ensembl_id, max_drugs=1000)
    linked = profile.drugs if profile else []
    if any(d.chembl_id == hit.id or drug_name_matches(drug, d.name) for d in linked):
        return None, f"{drug} IS linked to {symbol} in Open Targets; not a negative"
    return {"type": "negative_pair", "gene_symbol": symbol, "expected_status": "abstain",
            "ensembl_id": res.ensembl_id, "forbidden_drug": {"chembl_id": hit.id, "name": hit.name}}, None


def build(ot: OpenTargetsClient, ncbi: NcbiClient, seeds=SEEDS, quotas=QUOTAS, fakes=FAKE_SYMBOLS,
          pairs=INVALID_PAIRS, n_fake: int = 5, n_pairs: int = 5, log=print) -> tuple[list[dict], dict]:
    cases: list[dict] = []
    skipped: list[dict] = []
    stages: Counter = Counter()
    filled: Counter = Counter()

    def attempt(label: str, fn, *args) -> dict | None:
        try:
            rec, why = fn(*args)
        except (ApiError, LookupError, httpx.HTTPError) as exc:
            rec, why = None, f"{type(exc).__name__}: {exc}"
        if rec is None:
            skipped.append({"seed": label, "reason": why})
            log(f"  skip {label}: {why}")
        return rec

    for area, symbol, drug in seeds:
        if filled[area] >= quotas.get(area, 0):
            continue
        rec = attempt(f"{area}:{symbol}/{drug}", build_positive, area, symbol, drug, ot, ncbi, stages)
        if rec:
            filled[area] += 1
            cases.append(rec)
            log(f"  ok   {area:15s} {symbol:9s} {rec['drug']['name']:18s} {len(rec['expected_drugs'])} drugs")
    for symbol in fakes:
        if sum(c["type"] == "negative_symbol" for c in cases) >= n_fake:
            break
        rec = attempt(f"fake:{symbol}", build_negative_symbol, symbol, ot, ncbi)
        if rec:
            cases.append(rec)
    for symbol, drug in pairs:
        if sum(c["type"] == "negative_pair" for c in cases) >= n_pairs:
            break
        rec = attempt(f"pair:{symbol}/{drug}", build_negative_pair, symbol, drug, ot, ncbi)
        if rec:
            cases.append(rec)

    counters = Counter(c["type"] for c in cases)
    for i, c in enumerate(cases):
        prefix = {"positive": "pos", "negative_symbol": "neg-sym", "negative_pair": "neg-pair"}[c["type"]]
        c["case_id"] = f"{prefix}-{sum(x['type'] == c['type'] for x in cases[:i]) + 1:03d}"
    meta = {"counts": dict(counters), "per_area": dict(filled), "quotas": quotas, "skipped": skipped,
            "clinical_stages_seen": dict(stages.most_common()),
            "ground_truth_rule": "Open Targets drugAndClinicalCandidates at Phase 3 or approved (stage_rank >= 3)"}
    return [{"case_id": c.pop("case_id"), **c} for c in cases], meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen dataset")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"{args.out} already exists (frozen). Use --force to rebuild.", file=sys.stderr)
        return 1
    s = get_settings()
    nocache = DiskCache(None, enabled=False)          # always build from fresh API responses
    try:
        ncbi = NcbiClient(JsonHttp("ncbi", cache=nocache, throttle=ncbi_throttle(s.ncbi_api_key)),
                          s.ncbi_email, s.ncbi_api_key)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    ot = OpenTargetsClient(JsonHttp("opentargets", cache=nocache, max_attempts=2))
    try:                                               # fail fast if the APIs are unreachable
        ot.search("EGFR")
        ncbi.esearch("gene", "EGFR[sym] AND 9606[taxid]", retmax=1)
    except (ApiError, httpx.HTTPError) as exc:
        print(f"Cannot reach Open Targets / NCBI ({type(exc).__name__}: {exc}). Check your network.",
              file=sys.stderr)
        return 1
    ot.http.max_attempts = 4
    print("Building benchmark (live Open Targets + NCBI)...")
    cases, meta = build(ot, ncbi)
    n_pos = meta["counts"].get("positive", 0)
    if n_pos < sum(QUOTAS.values()):
        print(f"WARNING: only {n_pos}/{sum(QUOTAS.values())} positive cases passed checks; add seeds.",
              file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases)
    args.out.write_text(body, encoding="utf-8")
    meta.update(built_at=datetime.now(timezone.utc).isoformat(), n_cases=len(cases),
                sha256=hashlib.sha256(body.encode()).hexdigest(), file=args.out.name)
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.out} ({meta['counts']}); {len(meta['skipped'])} seeds skipped.")
    print("Clinical stage values seen:", meta["clinical_stages_seen"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
