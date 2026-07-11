"""Optional sqlite-vec vector backend adapter.

Provides runtime detection and a thin wrapper around ``sqlite_vec`` if it is
installed.  The store only uses this backend when a sqlite-vec backend is
configured and the extension can be loaded; otherwise it falls back to JSON
embeddings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

SQLITE_VEC_AVAILABLE = False
_sqlite_vec = None
_pysqlite3_conn = None
_PYSQLITE3_AVAILABLE = None


def _import_sqlite_vec() -> object | None:
    global _sqlite_vec, SQLITE_VEC_AVAILABLE
    if _sqlite_vec is not None:
        return _sqlite_vec
    try:
        import sqlite_vec as sv  # type: ignore[import-untyped]

        _sqlite_vec = sv
        SQLITE_VEC_AVAILABLE = True
        return sv
    except Exception:
        SQLITE_VEC_AVAILABLE = False
        return None


def _import_pysqlite3() -> bool:
    global _PYSQLITE3_AVAILABLE
    if _PYSQLITE3_AVAILABLE is not None:
        return _PYSQLITE3_AVAILABLE
    try:
        import pysqlite3.dbapi2  # type: ignore[import-untyped]

        _ = pysqlite3.dbapi2  # appease linters: we only need the import to succeed
        _PYSQLITE3_AVAILABLE = True
        return True
    except Exception:
        _PYSQLITE3_AVAILABLE = False
        return False


def available() -> bool:
    """True if the sqlite-vec python package is installed."""
    return _import_sqlite_vec() is not None


def _has_load_extension(conn: sqlite3.Connection) -> bool:
    """True if this connection/interpreter supports loading extensions."""
    return hasattr(conn, "enable_load_extension") and hasattr(conn, "load_extension")


def _pysqlite3_fallback(conn: sqlite3.Connection) -> sqlite3.Connection | None:
    """Return a pysqlite3 connection to the same DB if stdlib sqlite3 lacks
    extension loading. Returns ``None`` if pysqlite3 is not installed.
    """
    global _pysqlite3_conn
    if _pysqlite3_conn is not None:
        try:
            _pysqlite3_conn.execute("SELECT 1")
            return _pysqlite3_conn
        except Exception:
            _pysqlite3_conn = None
    if not _import_pysqlite3():
        return None
    try:
        import pysqlite3.dbapi2 as pysqlite3_sqlite3

        # Resolve the DB file path from the connection.
        path_row = conn.execute("PRAGMA database_list").fetchone()
        db_path = path_row[2] if path_row else ""
        if not db_path:
            return None
        fallback = pysqlite3_sqlite3.connect(db_path)
        fallback.execute("PRAGMA journal_mode=WAL")
        fallback.execute("PRAGMA busy_timeout=5000")
        _pysqlite3_conn = fallback
        return fallback
    except Exception as exc:
        logger.debug("pysqlite3 fallback connection failed: %s", exc)
        return None


def _use_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Return ``conn`` if it can load extensions, else a pysqlite3 fallback."""
    if _has_load_extension(conn):
        return conn
    fallback = _pysqlite3_fallback(conn)
    return fallback if fallback is not None else conn


def can_load(conn: sqlite3.Connection) -> bool:
    """True if sqlite-vec can be loaded into this connection."""
    sv = _import_sqlite_vec()
    if sv is None:
        return False
    conn = _use_conn(conn)
    if not _has_load_extension(conn):
        logger.debug("sqlite-vec: interpreter lacks SQLite extension loading support")
        return False
    try:
        conn.enable_load_extension(True)
        sv.load(conn)
        return True
    except Exception as exc:
        logger.debug("sqlite-vec load failed: %s", exc)
        return False


def table_name(dim: int) -> str:
    """Virtual table name for a fixed embedding dimension."""
    return f"memlife_vec_{dim}"


def ensure_schema(conn: sqlite3.Connection, dim: int) -> bool:
    """Create the dimension-specific virtual table and metadata table."""
    sv = _import_sqlite_vec()
    if sv is None:
        return False
    conn = _use_conn(conn)
    if not can_load(conn):
        return False
    vec_table = table_name(dim)
    meta_table = f"{vec_table}_meta"
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_table} USING vec0("
            f"rowid INTEGER PRIMARY KEY, embedding float[{dim}] distance=cosine)"
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {meta_table} (
                rowid INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                item_id TEXT NOT NULL
            )
            """
        )
        return True
    except Exception as exc:
        logger.debug("sqlite-vec schema create failed: %s", exc)
        return False


def rowid_for(kind: str, item_id: str, dim: int) -> int:
    """Stable, deterministic rowid derived from kind + item_id + dim."""
    import hashlib

    key = f"{kind}:{item_id}:{dim}".encode()
    h = int(hashlib.sha256(key).hexdigest()[:16], 16)
    # Keep rowids within SQLite's signed 64-bit integer range and avoid
    # negative values, which behave unexpectedly in virtual-table scans.
    return h & 0x7FFFFFFFFFFFFFFF


def store(
    conn: sqlite3.Connection,
    kind: str,
    item_id: str,
    vec: list[float],
) -> bool:
    """Store a vector in sqlite-vec if possible."""
    if not vec or not ensure_schema(conn, len(vec)):
        return False
    sv = _import_sqlite_vec()
    if sv is None:
        return False
    conn = _use_conn(conn)
    dim = len(vec)
    vec_table = table_name(dim)
    meta_table = f"{vec_table}_meta"
    rid = rowid_for(kind, item_id, dim)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
            (rid, sv.serialize_float32(vec)),
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {meta_table}(rowid, kind, item_id) "
            f"VALUES (?, ?, ?)",
            (rid, kind, item_id),
        )
        return True
    except Exception as exc:
        logger.debug("sqlite-vec store failed: %s", exc)
        return False


def delete(
    conn: sqlite3.Connection,
    kind: str,
    item_id: str,
    dim: int,
) -> bool:
    """Remove a vector from sqlite-vec if the virtual table exists."""
    sv = _import_sqlite_vec()
    if sv is None or not can_load(conn):
        return False
    conn = _use_conn(conn)
    vec_table = table_name(dim)
    meta_table = f"{vec_table}_meta"
    rid = rowid_for(kind, item_id, dim)
    try:
        conn.execute(f"DELETE FROM {vec_table} WHERE rowid = ?", (rid,))
        conn.execute(
            f"DELETE FROM {meta_table} WHERE rowid = ? AND kind = ?",
            (rid, kind),
        )
        return True
    except Exception as exc:
        logger.debug("sqlite-vec delete failed: %s", exc)
        return False


def search(
    conn: sqlite3.Connection,
    kind: str,
    query_vec: list[float],
    *,
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Return ``(item_id, similarity)`` tuples via sqlite-vec KNN."""
    if not query_vec or not ensure_schema(conn, len(query_vec)):
        return []
    sv = _import_sqlite_vec()
    if sv is None:
        return []
    conn = _use_conn(conn)
    dim = len(query_vec)
    vec_table = table_name(dim)
    meta_table = f"{vec_table}_meta"
    try:
        rows = conn.execute(
            f"""
            SELECT v.rowid, v.distance
            FROM {vec_table} AS v
            WHERE v.embedding MATCH ? AND v.k = ?
              AND v.rowid IN (
                  SELECT rowid FROM {meta_table} WHERE kind = ?
              )
            ORDER BY v.distance
            """,
            (sv.serialize_float32(query_vec), limit, kind),
        ).fetchall()
    except Exception as exc:
        logger.debug("sqlite-vec search failed: %s", exc)
        return []
    results: list[tuple[str, float]] = []
    for rid, distance in rows:
        meta = conn.execute(
            f"SELECT item_id FROM {meta_table} WHERE rowid = ? AND kind = ?",
            (rid, kind),
        ).fetchone()
        if not meta:
            continue
        # cosine distance: 0.0 = identical, 2.0 = opposite
        sim = max(0.0, 1.0 - 0.5 * float(distance))
        results.append((meta[0], sim))
    return results
