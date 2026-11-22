"""Open Targets Platform GraphQL client.

Note: Open Targets removed `Target.knownDrugs` in release 26.03. Drugs now come from
`Target.drugAndClinicalCandidates`, which is mapped onto the same KnownDrug fields
(ChEMBL ID, name, clinical stage, mechanism of action, diseases).
"""
from __future__ import annotations

from typing import Any

from ..schemas import DiseaseAssociation, KnownDrug, SearchHit, TargetProfile
from ._http import ApiError, JsonHttp

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

SEARCH_QUERY = """
query Search($q: String!, $entities: [String!], $size: Int!) {
  search(queryString: $q, entityNames: $entities, page: {index: 0, size: $size}) {
    hits { id name entity score object { ... on Target { id approvedSymbol } } }
  }
}"""

TARGET_QUERY = """
query TargetProfile($id: String!) {
  target(ensemblId: $id) {
    id approvedSymbol approvedName biotype
    associatedDiseases(page: {index: 0, size: 10}) { count rows { score disease { id name } } }
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug { id name drugType mechanismsOfAction { rows { mechanismOfAction targets { id } } } }
        diseases { disease { name } diseaseFromSource }
      }
    }
  }
}"""


class OpenTargetsClient:
    def __init__(self, http: JsonHttp, url: str = OT_URL):
        self.http = http
        self.url = url

    def _query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = self.http.post(self.url, {"query": query, "variables": variables}).data
        if payload.get("errors"):
            msg = "; ".join(str(e.get("message", e)) for e in payload["errors"])[:300]
            raise ApiError(f"opentargets: GraphQL error: {msg}")
        return payload.get("data") or {}

    def search(self, query_string: str, entity_names: tuple[str, ...] = ("target",),
               size: int = 10) -> list[SearchHit]:
        data = self._query(SEARCH_QUERY, {"q": query_string, "entities": list(entity_names), "size": size})
        hits = []
        for h in (data.get("search") or {}).get("hits") or []:
            obj = h.get("object") or {}
            hits.append(SearchHit(id=h["id"], name=h.get("name") or "", entity=h.get("entity") or "",
                                  approved_symbol=obj.get("approvedSymbol"), score=h.get("score") or 0.0))
        return hits

    def target(self, ensembl_id: str, max_drugs: int = 25) -> TargetProfile | None:
        t = self._query(TARGET_QUERY, {"id": ensembl_id}).get("target")
        if not t:
            return None
        assoc = t.get("associatedDiseases") or {}
        diseases = [
            DiseaseAssociation(disease_id=r["disease"]["id"], name=r["disease"]["name"], score=r.get("score") or 0)
            for r in assoc.get("rows") or [] if r.get("disease")
        ]
        cands = t.get("drugAndClinicalCandidates") or {}
        drugs: list[KnownDrug] = []
        seen: set[str] = set()
        for row in cands.get("rows") or []:
            drug = row.get("drug") or {}
            chembl = drug.get("id")
            if not chembl or chembl in seen:
                continue
            seen.add(chembl)
            moa_rows = ((drug.get("mechanismsOfAction") or {}).get("rows")) or []
            # a drug's MoA rows cover all its targets; keep the ones about this target
            mechanisms = sorted({
                m["mechanismOfAction"] for m in moa_rows
                if m.get("mechanismOfAction") and any(x.get("id") == ensembl_id for x in m.get("targets") or [])
            })
            dis = []
            for d in row.get("diseases") or []:
                name = (d.get("disease") or {}).get("name") or d.get("diseaseFromSource")
                if name and name not in dis:
                    dis.append(name)
            drugs.append(KnownDrug(chembl_id=chembl, name=drug.get("name") or chembl,
                                   drug_type=drug.get("drugType"), clinical_stage=row.get("maxClinicalStage"),
                                   mechanisms=mechanisms, diseases=dis[:5]))
        return TargetProfile(
            ensembl_id=t["id"], approved_symbol=t["approvedSymbol"], approved_name=t.get("approvedName"),
            biotype=t.get("biotype"), diseases=diseases, disease_count=assoc.get("count") or len(diseases),
            drugs=drugs[:max_drugs], drug_count=cands.get("count") or len(drugs),
        )
