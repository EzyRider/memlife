"""Thread-safe SQLite connection proxy.

Moves the reentrant-lock wrapper out of store.py so it can be imported
by schema and other mixins without creating circular imports.
"""

from __future__ import annotations

import sqlite3
import threading


class _LockedConn:
    """Proxy that serialises all DB access through a reentrant lock.

    Wraps a sqlite3.Connection so every method call acquires the store's
    lock first. This makes individual statements thread-safe when the
    connection is shared across threads (e.g. the MCP server thread pool).
    For multi-statement atomicity, use MemoryStore.transaction().
    """

    __slots__ = ("_raw", "_lock")

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._raw = conn
        self._lock = lock

    def execute(self, sql, params=()):
        with self._lock:
            return self._raw.execute(sql, params)

    def executemany(self, sql, params_seq):
        with self._lock:
            return self._raw.executemany(sql, params_seq)

    def executescript(self, script):
        with self._lock:
            return self._raw.executescript(script)

    def commit(self):
        with self._lock:
            return self._raw.commit()

    def rollback(self):
        with self._lock:
            return self._raw.rollback()

    def close(self):
        with self._lock:
            return self._raw.close()

    @property
    def row_factory(self):
        return self._raw.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._raw.row_factory = value
