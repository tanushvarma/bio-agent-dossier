"""SQLite disk cache keyed by a hash of the request, so repeat queries and evals are fast and repeatable."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

# Request fields that identify the caller, not the query. Never part of a key, never stored.
_IDENTITY_KEYS = {"api_key", "email", "tool"}


class DiskCache:
    def __init__(self, path: Path | str | None, enabled: bool = True):
        self.enabled = enabled and path is not None
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if self.enabled:
            p = Path(path)  # type: ignore[arg-type]
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(p), check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)"
            )
            self._db.commit()

    @staticmethod
    def key(namespace: str, method: str, url: str, params: dict | None, body: Any) -> str:
        clean = {k: v for k, v in (params or {}).items() if k not in _IDENTITY_KEYS}
        blob = json.dumps([namespace, method.upper(), url, clean, body], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        if not self._db:
            return None
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        if not self._db:
            return
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            self._db.commit()

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
