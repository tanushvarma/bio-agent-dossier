"""Europe PMC REST client: peer-reviewed (PubMed-indexed) abstracts."""
from __future__ import annotations

import html
import re

from ..schemas import Article
from ._http import JsonHttp

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_BLOCK_TAG = re.compile(r"</?(?:h\d|p|br|div|li|ul|ol|section|abstract)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")


def clean_text(s: str | None) -> str:
    text = _TAG.sub("", _BLOCK_TAG.sub(" ", s or ""))      # block tags -> space, inline (<i>, <sup>) -> nothing
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class EuropePmcClient:
    def __init__(self, http: JsonHttp, url: str = EPMC_SEARCH):
        self.http = http
        self.url = url

    def search(self, query: str, max_results: int = 10, peer_reviewed: bool = True) -> list[Article]:
        """Relevance-ranked abstracts. peer_reviewed=True keeps PubMed-indexed records with an
        abstract and excludes preprints and retraction notices."""
        q = f"({query})"
        if peer_reviewed:
            q += ' AND SRC:MED AND HAS_ABSTRACT:Y NOT PUB_TYPE:"Retraction of Publication"'
        data = self.http.get(self.url, {"query": q, "format": "json", "resultType": "core",
                                        "pageSize": max(1, min(max_results, 100))}).data
        out: list[Article] = []
        for r in (data.get("resultList") or {}).get("result") or []:
            pmid = str(r.get("pmid") or "").strip()
            if not pmid.isdigit():
                continue
            year = str(r.get("pubYear") or "")
            out.append(Article(pmid=pmid, title=clean_text(r.get("title")),
                               pub_year=int(year) if year.isdigit() else None,
                               abstract_text=clean_text(r.get("abstractText")), doi=r.get("doi")))
        return out[:max_results]
