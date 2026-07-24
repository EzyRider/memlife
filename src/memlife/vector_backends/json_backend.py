"""Default vector backend: embeddings stored in ``embedding_json`` columns.

This backend is always available. It supports optional binary compression via
``MemoryConfig.use_binary_vectors``. It does not create any extra tables.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

from memlife import binary_vectors
from memlife.vector_backends.base import VectorBackend, VectorSearchResult
from memlife.vectors import cosine

if TYPE_CHECKING:
    from memlife.store import MemoryStore

logger = logging.getLogger(__name__)


_TABLES = {"facts", "episodes", "journal"}


class JsonVectorBackend(VectorBackend):
    """Store embeddings as JSON (or compressed binary) strings inline.

    Search is performed by loading all vectors for a kind into memory and
    computing cosine similarity. This is simple, portable, and sufficient for
    small-to-medium databases.
    """

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(store)

    @property
    def name(self) -> str:
        return "json"

    def available(self) -> bool:
        return True

    def serialize(self, vec: list[float]) -> str:
        """Serialize a float vector.

        When ``config.use_binary_vectors`` is True the vector is binarized and
        base64-encoded as ``binary:<dim>:<bytes>``; otherwise it is stored as
        JSON floats.
        """
        if not vec:
            return ""
        if self._memory_store.config.use_binary_vectors:
            packed = binary_vectors.binarize(vec)
            return f"binary:{len(vec)}:{base64.b64encode(packed).decode()}"
        return json.dumps(vec)

    def deserialize(self, raw: str) -> list[float] | None:
        """Reconstruct a float vector from its stored form."""
        if not raw:
            return None
        if raw.startswith("binary:"):
            try:
                _, dim_str, b64 = raw.split(":", 2)
                dim = int(dim_str)
                packed = base64.b64decode(b64)
                return binary_vectors.debinarize(packed, dim)
            except Exception:
                return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def store(
        self,
        kind: str,
        item_id: str,
        vec: list[float],
    ) -> bool:
        """JSON backend stores vectors inline; no extra action needed.

        The caller already updates ``embedding_json`` via ``serialize()``.
        """
        return bool(vec)

    def delete(
        self,
        kind: str,
        item_id: str,
        dim: int,
    ) -> bool:
        """Clear the ``embedding_json`` column for the item.

        The caller is responsible for committing.
        """
        if kind not in _TABLES:
            raise ValueError(f"invalid vector table kind: {kind}")
        table = self._table_for_kind(kind)
        if not table:
            return False
        self._memory_store.conn.execute(
            f"UPDATE {table} SET embedding_json = '', embedding_model = '' WHERE id = ?",
            (item_id,),
        )
        return True

    def search(
        self,
        kind: str,
        query_vec: list[float],
        *,
        limit: int = 20,
    ) -> list[VectorSearchResult]:
        """Brute-force cosine search over all stored vectors for ``kind``."""
        if kind not in _TABLES:
            raise ValueError(f"invalid vector table kind: {kind}")
        table, extra_filter = self._table_and_filter_for_kind(kind)
        if not table:
            return []
        rows = self._memory_store.conn.execute(
            f"SELECT id, embedding_json FROM {table} "
            f"WHERE embedding_json != ''{extra_filter}"
        ).fetchall()
        results: list[VectorSearchResult] = []
        for item_id, raw in rows:
            vec = self.deserialize(raw)
            if not vec:
                continue
            sim = cosine(query_vec, vec)
            results.append(VectorSearchResult(item_id, sim, kind))
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def _table_for_kind(self, kind: str) -> str | None:
        """Map a vector kind to its SQLite table name."""
        mapping = {
            "facts": "facts",
            "episodes": "episodes",
            "journal": "journal",
        }
        return mapping.get(kind)

    def _table_and_filter_for_kind(self, kind: str) -> tuple[str | None, str]:
        """Map a vector kind to table name and any extra SQL filter."""
        table = self._table_for_kind(kind)
        if table == "journal":
            # Contradictions are not retrievable via vector recall.
            return table, "AND type != 'contradiction'"
        return table, ""
