"""sqlite-vec vector backend.

Wraps the existing ``memlife.vec_backend`` adapter but scopes it to a single
MemoryStore instance and namespace. The store must request this backend
explicitly via ``MemoryConfig.vector_backend = "sqlite_vec"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memlife import vec_backend
from memlife.vector_backends.base import VectorBackend, VectorSearchResult

if TYPE_CHECKING:
    from memlife.store import MemoryStore

logger = logging.getLogger(__name__)


class SqliteVecBackend(VectorBackend):
    """Use sqlite-vec virtual tables for fast approximate nearest neighbours.

    Falls back to reporting ``available() == False`` when the extension or
    extension-loading support is missing. The store is responsible for
    switching to ``JsonVectorBackend`` in that case.
    """

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(store)

    @property
    def name(self) -> str:
        return "sqlite_vec"

    def available(self) -> bool:
        """True only if sqlite-vec can be loaded into this store's connection."""
        raw = self._raw_conn()
        return vec_backend.can_load(raw)

    def serialize(self, vec: list[float]) -> str:
        """sqlite-vec still stores a JSON copy in ``embedding_json`` for portability."""
        if not vec:
            return ""
        import json

        return json.dumps(vec)

    def deserialize(self, raw: str) -> list[float] | None:
        """Reconstruct a float vector from its stored JSON form."""
        if not raw:
            return None
        import json

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
        """Persist the vector in a sqlite-vec virtual table."""
        if not vec:
            return False
        raw = self._raw_conn()
        # Touch the schema first on the same raw handle the store will use.
        vec_backend.ensure_schema(raw, len(vec))
        return vec_backend.store(raw, kind, item_id, vec)

    def delete(
        self,
        kind: str,
        item_id: str,
        dim: int,
    ) -> bool:
        """Remove the vector from the sqlite-vec virtual table."""
        raw = self._raw_conn()
        return vec_backend.delete(raw, kind, item_id, dim)

    def search(
        self,
        kind: str,
        query_vec: list[float],
        *,
        limit: int = 20,
    ) -> list[VectorSearchResult]:
        """Return sqlite-vec KNN results for ``kind``."""
        if not query_vec:
            return []
        raw = self._raw_conn()
        matches = vec_backend.search(raw, kind, query_vec, limit=limit)
        return [VectorSearchResult(item_id, sim, kind) for item_id, sim in matches]

    def _raw_conn(self):
        """Return the underlying sqlite3.Connection from the store's locked wrapper."""
        conn = self._memory_store.conn
        return conn._raw if hasattr(conn, "_raw") else conn
