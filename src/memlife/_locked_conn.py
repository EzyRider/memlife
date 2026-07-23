"""Thread-safe SQLite connection proxy.

Moves the reentrant-lock wrapper out of store.py so it can be imported
by schema and other mixins without creating circular imports.
"""

from __future__ import annotations

import sqlite3
import threading
from types import TracebackType

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


class _LockedCursor:
    """Context-managed cursor that holds the store lock for its lifetime.

    HF-001: cursors must not outlive the lock that serialises access to the
    underlying connection. This wrapper acquires the lock on entry, closes
    the real cursor on exit, and releases the lock even when the iteration
    body raises.
    """

    __slots__ = ("_cur", "_lock")

    def __init__(self, lock: threading.RLock, cur: sqlite3.Cursor):
        self._lock = lock
        self._cur = cur

    def __enter__(self) -> Self:
        self._lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            self._cur.close()
        finally:
            self._lock.release()

    def __iter__(self):
        return iter(self._cur)

    def __getattr__(self, name: str):
        # Forward everything else (fetchone, fetchall, fetchmany,
        # rowcount, description, etc.) to the real cursor.
        return getattr(self._cur, name)


class _LockedConn:
    """Proxy that serialises all DB access through a reentrant lock.

    Wraps a sqlite3.Connection so every method call acquires the store's
    lock first. This makes individual statements thread-safe when the
    connection is shared across threads (e.g. the MCP server thread pool).
    For multi-statement atomicity, use MemoryStore.transaction().
    """

    __slots__ = ("_lock", "_raw")

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

    def cursor(self) -> _LockedCursor:
        """Return a context-managed cursor that holds the lock while open."""
        with self._lock:
            return _LockedCursor(self._lock, self._raw.cursor())

    @property
    def row_factory(self):
        # HF-002: row_factory is a Python-level attribute on the underlying
        # connection. Read and write it under the same lock used for execute.
        with self._lock:
            return self._raw.row_factory

    @row_factory.setter
    def row_factory(self, value):
        with self._lock:
            self._raw.row_factory = value

    @property
    def isolation_level(self):
        # Like row_factory, isolation_level mutates connection state and is
        # accessed by consumers. Guard it under the lock.
        with self._lock:
            return self._raw.isolation_level

    @isolation_level.setter
    def isolation_level(self, value):
        with self._lock:
            self._raw.isolation_level = value

    @property
    def text_factory(self):
        # text_factory affects how SQLite values are converted to Python
        # objects; concurrent mutation would race.
        with self._lock:
            return self._raw.text_factory

    @text_factory.setter
    def text_factory(self, value):
        with self._lock:
            self._raw.text_factory = value

    def __getattr__(self, name: str):
        # Forward any other attribute access to the raw connection. Read-only
        # attributes (total_changes, iterdump, backup, etc.) do not need the
        # lock because sqlite3.Connection itself handles them safely; methods
        # that return a new object (backup, iterdump) create their own
        # resources. Callers that mutate state should use the explicit
        # properties above or go through MemoryStore.transaction().
        return getattr(self._raw, name)
