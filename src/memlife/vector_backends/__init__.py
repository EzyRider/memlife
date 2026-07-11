"""Pluggable vector backends for memlife.

Backends are scoped to a single MemoryStore / namespace / SQLite connection.
The default ``JsonVectorBackend`` stores embeddings in the existing
``embedding_json`` columns. ``SqliteVecBackend`` uses the sqlite-vec
extension when it is available.
"""

from __future__ import annotations

from memlife.vector_backends.base import VectorBackend, VectorSearchResult
from memlife.vector_backends.binary_backend import BinaryVectorBackend
from memlife.vector_backends.json_backend import JsonVectorBackend
from memlife.vector_backends.sqlite_vec_backend import SqliteVecBackend

__all__ = [
    "VectorBackend",
    "VectorSearchResult",
    "JsonVectorBackend",
    "BinaryVectorBackend",
    "SqliteVecBackend",
    "create_vector_backend",
]


def create_vector_backend(name: str, store) -> VectorBackend:
    """Factory: create a backend instance bound to ``store``.

    Args:
        name: Backend identifier. Supported values:
            - "json" (default)
            - "binary"
            - "sqlite_vec"
        store: The MemoryStore instance that owns this backend.

    Returns:
        A backend instance scoped to the store's connection/namespace.

    Raises:
        ValueError: If ``name`` is not a recognised backend.
    """
    name = (name or "json").lower().strip()
    if name == "json":
        return JsonVectorBackend(store)
    if name == "binary":
        return BinaryVectorBackend(store)
    if name in ("sqlite_vec", "sqlite-vec"):
        return SqliteVecBackend(store)
    raise ValueError(f"Unknown vector backend: {name!r}")
