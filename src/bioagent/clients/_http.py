"""Shared JSON-over-HTTP client: httpx timeouts, throttling, tenacity retry on 429/5xx, disk cache."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

from .cache import DiskCache

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "bioagent/0.2 (+https://github.com/tanushvarma/bio-agent-dossier)"


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RetryableHTTPError(ApiError):
    def __init__(self, message: str, status: int, retry_after: float | None = None):
        super().__init__(message, status)
        self.retry_after = retry_after


class Throttle:
    """Minimum spacing between requests (NCBI allows 3/s without a key, 10/s with one)."""

    def __init__(self, min_interval_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval_s = min_interval_s
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                gap = self.min_interval_s - (now - self._last)
                if gap > 0:
                    self._sleep(gap)
                    now += gap
            self._last = now


@dataclass
class HttpResult:
    data: Any
    cached: bool
    latency_ms: float


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None


class JsonHttp:
    def __init__(self, namespace: str, *, timeout_s: float = 20.0, cache: DiskCache | None = None,
                 throttle: Throttle | None = None, transport: httpx.BaseTransport | None = None,
                 max_attempts: int = 4, backoff_base_s: float = 0.5, backoff_max_s: float = 20.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.namespace = namespace
        self.cache = cache
        self.throttle = throttle
        self.max_attempts = max_attempts
        self._backoff = (backoff_base_s, backoff_max_s)
        self._sleep = sleep
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport,
                                    headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        self.n_requests = 0      # network attempts, including retries
        self.n_cache_hits = 0

    def get(self, url: str, params: dict[str, Any] | None = None) -> HttpResult:
        return self._request("GET", url, params=params)

    def post(self, url: str, json_body: Any) -> HttpResult:
        return self._request("POST", url, json_body=json_body)

    def _wait(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        base, cap = self._backoff
        backoff = min(cap, base * 2 ** (state.attempt_number - 1))
        if isinstance(exc, RetryableHTTPError) and exc.retry_after is not None:
            return min(cap, max(exc.retry_after, 0.0))
        return backoff

    def _request(self, method: str, url: str, params: dict[str, Any] | None = None,
                 json_body: Any = None) -> HttpResult:
        t0 = time.perf_counter()
        key = DiskCache.key(self.namespace, method, url, params, json_body) if self.cache else None
        if key:
            hit = self.cache.get(key)  # type: ignore[union-attr]
            if hit is not None:
                self.n_cache_hits += 1
                return HttpResult(hit, True, (time.perf_counter() - t0) * 1000)

        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=self._wait,
            retry=retry_if_exception_type((RetryableHTTPError, httpx.TransportError)),
            sleep=self._sleep,
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                if self.throttle:
                    self.throttle.wait()
                self.n_requests += 1
                r = self._client.request(method, url, params=params, json=json_body)
                if r.status_code in RETRY_STATUS:
                    raise RetryableHTTPError(f"{self.namespace}: HTTP {r.status_code}", r.status_code,
                                             _retry_after(r))
                if r.status_code >= 400:
                    raise ApiError(f"{self.namespace}: HTTP {r.status_code}: {r.text[:200]}", r.status_code)
                try:
                    data = r.json()
                except ValueError as exc:
                    raise ApiError(f"{self.namespace}: response was not JSON") from exc
        if key:
            self.cache.set(key, data)  # type: ignore[union-attr]
        return HttpResult(data, False, (time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        self._client.close()
