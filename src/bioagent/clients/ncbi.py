"""NCBI E-utilities client: Gene and PubMed (esearch + esummary), JSON mode, throttled."""
from __future__ import annotations

import re
import threading
from typing import Any

from ..config import ConfigError
from ..schemas import GeneRecord, PubmedRecord
from ._http import JsonHttp, Throttle

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


_SHARED: dict[float, Throttle] = {}
_SHARED_LOCK = threading.Lock()


def ncbi_throttle(api_key: str | None) -> Throttle:
    """NCBI policy: at most 3 requests/s without an API key, 10/s with one. The limit is per
    machine, so every client in the process shares one throttle (eval workers included)."""
    interval = 0.11 if api_key else 0.34
    with _SHARED_LOCK:
        return _SHARED.setdefault(interval, Throttle(interval))


class NcbiClient:
    def __init__(self, http: JsonHttp, email: str | None, api_key: str | None = None):
        if not email or "@" not in email or email.endswith("@example.com"):
            raise ConfigError("NCBI_EMAIL must be set to a real contact email (NCBI E-utilities policy).")
        self.http = http
        self._identity = {"tool": "bioagent", "email": email, **({"api_key": api_key} if api_key else {})}

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.http.get(f"{EUTILS}/{endpoint}", {**params, "retmode": "json", **self._identity}).data

    def esearch(self, db: str, term: str, retmax: int = 20, sort: str | None = None) -> list[str]:
        params: dict[str, Any] = {"db": db, "term": term, "retmax": retmax}
        if sort:
            params["sort"] = sort
        return list((self._get("esearch.fcgi", params).get("esearchresult") or {}).get("idlist") or [])

    def esummary(self, db: str, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        result = self._get("esummary.fcgi", {"db": db, "id": ",".join(ids)}).get("result") or {}
        return {uid: result[uid] for uid in result.get("uids") or [] if uid in result}

    @staticmethod
    def _gene(doc: dict[str, Any]) -> GeneRecord:
        aliases = [a.strip() for a in (doc.get("otheraliases") or "").split(",") if a.strip()]
        return GeneRecord(entrez_id=str(doc.get("uid")), symbol=doc.get("name") or "",
                          description=doc.get("description") or "", chromosome=doc.get("chromosome") or "",
                          map_location=doc.get("maplocation") or "", aliases=aliases,
                          summary=doc.get("summary") or "", taxid=(doc.get("organism") or {}).get("taxid"))

    def search_gene(self, symbol: str, taxid: int = 9606) -> list[GeneRecord]:
        """Genes matching a symbol (or alias) in one species, in NCBI's order."""
        ids = self.esearch("gene", f"{symbol}[sym] AND {taxid}[taxid]", retmax=10)
        docs = self.esummary("gene", ids)
        return [self._gene(docs[i]) for i in ids if i in docs]

    def gene(self, entrez_id: str) -> GeneRecord | None:
        docs = self.esummary("gene", [entrez_id])
        return self._gene(docs[entrez_id]) if entrez_id in docs else None

    def search_pubmed(self, term: str, retmax: int = 10) -> list[PubmedRecord]:
        ids = self.esearch("pubmed", term, retmax=retmax, sort="relevance")
        docs = self.esummary("pubmed", ids)
        out = []
        for i in ids:
            d = docs.get(i)
            if not d:
                continue
            m = re.match(r"(\d{4})", d.get("pubdate") or "")
            out.append(PubmedRecord(pmid=i, title=d.get("title") or "", year=int(m.group(1)) if m else None,
                                    journal=d.get("source") or ""))
        return out
