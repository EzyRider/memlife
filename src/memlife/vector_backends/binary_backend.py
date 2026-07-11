"""Binary vector backend.

Stores embeddings as bit-packed binary blobs (MV2-I002).  This gives a ~32x
storage reduction over JSON floats and uses Hamming distance for similarity.
The binary backend is still a JSON-backend variant under the hood — vectors
are base64-encoded into ``embedding_json`` — but it overrides search to use
Hamming distance on the packed representation instead of expanding to floats.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

from memlife import binary_vectors
from memlife.vector_backends.base import VectorBackend, VectorSearchResult

if TYPE_CHECKING:
    from memlife.store import MemoryStore

logger = logging.getLogger(__name__)


class BinaryVectorBackend(VectorBackend):
    """Store embeddings as compact binary vectors and search with Hamming distance.

    This backend is selected when ``MemoryConfig.vector_backend = "binary"``.
    It is always available and shares the same inline-storage approach as
    ``JsonVectorBackend``, but uses ``binary_vectors`` for serialization and
    Hamming similarity for ranking.
    """

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(store)

    @property
    def name(self) -> str:
        return "binary"

    def available(self) -> bool:
        return True

    def serialize(self, vec: list[float]) -> str:
        """Serialize a float vector as ``binary:<dim>:<bytes>``."""
        if not vec:
            return ""
        packed = binary_vectors.binarize(vec)
        return f"binary:{len(vec)}:{base64.b64encode(packed).decode()}"

    def deserialize(self, raw: str) -> list[float] | None:
        """Reconstruct a float vector from its stored form.

        Accepts the native ``binary:<dim>:<bytes>`` prefix and falls back to
        JSON deserialization for rows stored by ``JsonVectorBackend`` before a
        backend switch. This prevents data loss when a user flips from the
        default JSON backend to binary on an existing store.
        """
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

    def _unpack(self, raw: str) -> tuple[bytes, int] | None:
        """Return the packed bytes and dimension for a stored vector.

        Falls back to full deserialization for JSON-encoded vectors so that
        Hamming distance search can still rank embeddings created before the
        binary backend was selected.  The debinarizer converts floats back to
        packed form, which is acceptable for mixed-format recall.
        """
        if not raw:
            return None
        if raw.startswith("binary:"):
            try:
                _, dim_str, b64 = raw.split(":", 2)
                dim = int(dim_str)
                return base64.b64decode(b64), dim
            except Exception:
                return None
        # JSON-stored vector: deserialize and binarize on the fly.
        try:
            vec = json.loads(raw)
            if not vec:
                return None
            return binary_vectors.binarize(vec), len(vec)
        except Exception:
            return None

    def store(
        self,
        kind: str,
        item_id: str,
        vec: list[float],
    ) -> bool:
        """Binary backend stores vectors inline; no extra action needed."""
        return bool(vec)

    def delete(
        self,
        kind: str,
        item_id: str,
        dim: int,
    ) -> bool:
        """Clear the ``embedding_json`` column for the item."""
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
        """Hamming-distance search over packed binary vectors for ``kind``."""
        table, extra_filter = self._table_and_filter_for_kind(kind)
        if not table:
            return []
        query_packed = binary_vectors.binarize(query_vec)
        query_dim = len(query_vec)
        rows = self._memory_store.conn.execute(
            f"SELECT id, embedding_json FROM {table} "
            f"WHERE embedding_json != ''{extra_filter}"
        ).fetchall()
        results: list[VectorSearchResult] = []
        for item_id, raw in rows:
            unpacked = self._unpack(raw)
            if unpacked is None:
                continue
            packed, dim = unpacked
            # Guard against dimension mismatch.
            if dim != query_dim:
                continue
            sim = binary_vectors.hamming_similarity(query_packed, packed, dim)
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
            return table, " AND type != 'contradiction'"
        return table, ""
