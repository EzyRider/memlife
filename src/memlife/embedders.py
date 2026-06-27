"""DummyEmbedder — hash-based embeddings for testing and quickstart.

Zero external dependencies, no API calls, deterministic.
Produces 128-dimensional vectors from SHA-256 hashes.
Good enough for keyword-like similarity, not for real semantic recall.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


class DummyEmbedder:
    """Hash-based embeddings. No external dependencies.

    Implements the Embedder protocol: ``await embedder.embed(texts)``.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        return [self._hash_vector(t) for t in texts]

    @staticmethod
    def _hash_vector(text: str, dim: int = 128) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [((h[i % len(h)] / 255.0) - 0.5) * 2.0 for i in range(dim)]