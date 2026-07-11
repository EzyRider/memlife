"""Abstract base class for memlife vector backends.

A vector backend is scoped to exactly one MemoryStore instance. It owns
embedding storage, retrieval, and deletion for that store's namespace and
SQLite connection. This prevents cross-namespace leakage that would happen if
a backend were shared between stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlife.store import MemoryStore


@dataclass(frozen=True)
class VectorSearchResult:
    """One nearest-neighbour result from a vector backend search."""

    item_id: str
    similarity: float
    kind: str


class VectorBackend(ABC):
    """Abstract vector backend.

    Implementations must be bound to a single ``MemoryStore`` instance and
    must only operate on that store's connection/namespace. They may store
    vectors in SQLite columns, virtual tables, external files, etc.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._memory_store = store

    @property
    def memory_store(self) -> MemoryStore:
        """The MemoryStore instance this backend is scoped to."""
        return self._memory_store

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend identifier (e.g. ``json``, ``sqlite_vec``)."""

    @abstractmethod
    def available(self) -> bool:
        """True if this backend can operate in the current environment.

        This is a runtime check (e.g. an optional extension is installed).
        The store uses it to decide whether the requested backend can be used
        or should fall back to JSON.
        """

    @abstractmethod
    def serialize(self, vec: list[float]) -> str:
        """Convert a float vector into the form stored in ``embedding_json``."""

    @abstractmethod
    def deserialize(self, raw: str) -> list[float] | None:
        """Reconstruct a float vector from its stored form."""

    @abstractmethod
    def store(
        self,
        kind: str,
        item_id: str,
        vec: list[float],
    ) -> bool:
        """Persist a vector for ``kind``/``item_id``.

        Returns True if the vector was stored successfully. Implementations
        may choose to store only in their native format, or may also update
        the ``embedding_json`` column for redundancy.
        """

    @abstractmethod
    def delete(
        self,
        kind: str,
        item_id: str,
        dim: int,
    ) -> bool:
        """Remove a vector from the backend.

        ``dim`` is provided because some backends (e.g. sqlite-vec) use
        dimension-specific tables.
        """

    @abstractmethod
    def search(
        self,
        kind: str,
        query_vec: list[float],
        *,
        limit: int = 20,
    ) -> list[VectorSearchResult]:
        """Return nearest neighbours of ``query_vec`` restricted to ``kind``."""

    def health(self) -> dict:
        """Optional diagnostics. Override to add backend-specific metrics."""
        return {"backend": self.name, "available": self.available()}
