"""Optional sqlite-vec vector backend adapter.

Provides runtime detection and a thin wrapper around ``sqlite_vec`` if it is
installed.  The store only uses this backend when ``config.use_sqlite_vec`` is
True and the extension can be loaded; otherwise it falls back to JSON
embeddings.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

SQLITE_VEC_AVAILABLE = False
_sqlite_vec = None

try:
    import sqlite_vec as _sqlite_vec  # type: ignore[import-untyped]

    SQLITE_VEC_AVAILABLE = True
except Exception:
    pass


def available() -> bool:
    """True if the sqlite-vec python package is installed."""
    return SQLITE_VEC_AVAILABLE


def can_load(conn: sqlite3.Connection) -> bool:
    """True if sqlite-vec can be loaded into this connection."""
    if not _sqlite_vec:
        return False
    try:
        conn.enable_load_extension(True)
        _sqlite_vec.load(conn)
        return True
    except Exception as exc:
        logger.debug("sqlite-vec load failed: %s", exc)
        return False


def table_name(dim: int) -> str:
    """Virtual table name for a fixed embedding dimension."""
    return f"memlife_vec_{dim}"


def ensure_schema(conn: sqlite3.Connection, dim: int) -> bool:
    """Create the dimension-specific virtual table and metadata table."""
    if not _sqlite_vec or not can_load(conn):
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
    return h & 0x7FFFFFFF


def store(
    conn: sqlite3.Connection,
    kind: str,
    item_id: str,
    vec: list[float],
) -> bool:
    """Store a vector in sqlite-vec if possible."""
    if not vec or not ensure_schema(conn, len(vec)):
        return False
    dim = len(vec)
    vec_table = table_name(dim)
    meta_table = f"{vec_table}_meta"
    rid = rowid_for(kind, item_id, dim)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
            (rid, json.dumps(vec)),
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
    dim = len(query_vec)
    vec_table = table_name(dim)
    meta_table = f"{vec_table}_meta"
    rid_low = rowid_for(kind, "", dim)
    rid_high = rowid_for(kind + "\xff", "", dim)
    try:
        rows = conn.execute(
            f"""
            SELECT v.rowid, v.distance
            FROM {vec_table} AS v
            WHERE v.embedding MATCH ? AND v.k = ?
              AND v.rowid BETWEEN ? AND ?
            ORDER BY v.distance
            LIMIT ?
            """,
            (json.dumps(query_vec), limit, rid_low, rid_high, limit),
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
