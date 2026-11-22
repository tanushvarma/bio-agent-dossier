"""HTTP layer and API clients, all against mocked httpx transports (no network)."""
from __future__ import annotations

import httpx
import pytest

from bioagent.clients import (
    ApiError,
    DiskCache,
    EuropePmcClient,
    NcbiClient,
    OpenTargetsClient,
    RetryableHTTPError,
    Throttle,
)
from bioagent.clients.ncbi import ncbi_throttle
from bioagent.config import ConfigError
from conftest import EGFR, make_http


def test_retries_429_then_succeeds(world):
    world.fail["europepmc"] = [429]
    sleeps: list[float] = []
    http = make_http("europepmc", world)
    http._sleep = sleeps.append                         # observe backoff without waiting
    arts = EuropePmcClient(http).search("EGFR")
    assert len(world.calls) == 2 and http.n_requests == 2
    assert sleeps == [0.0]                              # honoured Retry-After: 0
    assert [a.pmid for a in arts] == ["30000001", "30000002"]


def test_retries_5xx_then_gives_up(world):
    world.fail["opentargets"] = [503, 502, 500, 504, 503]
    http = make_http("opentargets", world, max_attempts=3)
    with pytest.raises(RetryableHTTPError) as exc:
        OpenTargetsClient(http).search("EGFR")
    assert exc.value.status == 500 and len(world.calls) == 3


def test_client_error_is_not_retried(world):
    world.fail["europepmc"] = [404]
    http = make_http("europepmc", world)
    with pytest.raises(ApiError) as exc:
        EuropePmcClient(http).search("EGFR")
    assert exc.value.status == 404 and len(world.calls) == 1


def test_transport_errors_are_retried():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"resultList": {"result": []}})

    from bioagent.clients import JsonHttp
    http = JsonHttp("europepmc", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert EuropePmcClient(http).search("x") == []
    assert len(attempts) == 2


def test_disk_cache_makes_repeat_queries_free_and_ignores_identity(world, tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite")
    a = NcbiClient(make_http("ncbi", world, cache=cache), "a@asu.edu", api_key="KEY1")
    first = a.search_gene("EGFR")
    n = len(world.calls)
    b = NcbiClient(make_http("ncbi", world, cache=cache), "b@asu.edu", api_key="KEY2")
    assert b.search_gene("EGFR") == first
    assert len(world.calls) == n and b.http.n_cache_hits == 2     # esearch + esummary from disk
    stored = cache._db.execute("SELECT value FROM cache").fetchall()
    assert not any("KEY1" in v[0] or "a@asu.edu" in v[0] for v in stored)


def test_cache_disabled_always_hits_network(world, tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite", enabled=False)
    http = make_http("europepmc", world, cache=cache)
    EuropePmcClient(http).search("EGFR")
    EuropePmcClient(http).search("EGFR")
    assert len(world.calls) == 2


def test_throttle_spaces_requests():
    t = [0.0]
    slept: list[float] = []

    def sleep(s):
        slept.append(round(s, 3))
        t[0] += s

    th = Throttle(0.34, clock=lambda: t[0], sleep=sleep)
    th.wait()
    th.wait()
    t[0] += 1.0
    th.wait()
    assert slept == [0.34]


def test_ncbi_requires_real_email(world):
    for bad in (None, "", "noat", "bio-agent-dossier@example.com"):
        with pytest.raises(ConfigError):
            NcbiClient(make_http("ncbi", world), bad)


def test_ncbi_gene_and_pubmed_parsing(world):
    c = NcbiClient(make_http("ncbi", world), "lab@asu.edu")
    [g] = c.search_gene("EGFR")
    assert (g.entrez_id, g.symbol, g.aliases, g.taxid) == ("1956", "EGFR", ["ERBB", "ERBB1", "HER1"], 9606)
    assert c.search_gene("NOTAGENE") == []
    pubs = c.search_pubmed("EGFR inhibitor")
    assert [(p.pmid, p.year) for p in pubs] == [("111", 2020), ("222", None)]
    sent = world.calls[0].url.params
    assert sent["email"] == "lab@asu.edu" and sent["tool"] == "bioagent" and sent["retmode"] == "json"


def test_opentargets_search_and_target(world):
    ot = OpenTargetsClient(make_http("opentargets", world))
    hits = ot.search("EGFR")
    assert hits[0].id == EGFR and hits[0].approved_symbol == "EGFR"
    p = ot.target(EGFR)
    assert p.approved_symbol == "EGFR" and p.biotype == "protein_coding" and p.disease_count == 2
    assert [d.chembl_id for d in p.drugs] == ["CHEMBL939", "CHEMBL3353410"]
    # mechanism rows about other targets are dropped
    assert p.drugs[0].mechanisms == ["Epidermal growth factor receptor erbB1 inhibitor"]
    assert p.drugs[0].diseases == ["non-small cell lung carcinoma"]
    assert ot.target("ENSG00000000000") is None


def test_opentargets_graphql_errors_raise():
    def handler(request):
        return httpx.Response(200, json={"errors": [{"message": "Cannot query field 'knownDrugs'"}]})

    from bioagent.clients import JsonHttp
    ot = OpenTargetsClient(JsonHttp("opentargets", transport=httpx.MockTransport(handler)))
    with pytest.raises(ApiError, match="knownDrugs"):
        ot.target(EGFR)


def test_europepmc_cleans_html_and_skips_records_without_pmid(world):
    arts = EuropePmcClient(make_http("europepmc", world)).search("EGFR")
    assert [a.pmid for a in arts] == ["30000001", "30000002"]
    assert arts[0].title == "Osimertinib in EGFR-mutant NSCLC"
    assert arts[0].abstract_text == "Background Third-generation EGFR inhibitor & T790M."
    q = world.calls[0].url.params["query"]
    assert "SRC:MED" in q and "HAS_ABSTRACT:Y" in q and world.calls[0].url.params["resultType"] == "core"


def test_ncbi_throttle_is_shared_process_wide():
    assert ncbi_throttle(None) is ncbi_throttle(None)                # all eval workers share one limiter
    assert ncbi_throttle("key") is not ncbi_throttle(None)
    assert ncbi_throttle(None).min_interval_s == 0.34 and ncbi_throttle("key").min_interval_s == 0.11
