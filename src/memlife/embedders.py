"""DummyEmbedder — bag-of-words embeddings for testing and quickstart.

Zero external dependencies, no API calls, deterministic.
Produces 128-dimensional vectors from bag-of-words token hashing.
Similar sentences get positive cosine similarity (unlike raw hash vectors).
Good enough for keyword-like similarity, not for real semantic recall.
"""

from __future__ import annotations

import re
import hashlib
from typing import Sequence


class DummyEmbedder:
    """Bag-of-words embeddings. No external dependencies.

    Implements the Embedder protocol: ``await embedder.embed(texts)``.
    MF-014: replaced hash-based vectors with bag-of-words so similar
    sentences receive positive cosine similarity instead of misleading
    near-zero or negative values.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        return [self._bow_vector(t) for t in texts]

    @staticmethod
    def _bow_vector(text: str, dim: int = 128) -> list[float]:
        """Bag-of-words vector: tokens hash into bins, counts accumulate.

        Two sentences sharing words will have overlapping non-zero bins,
        giving positive cosine similarity. Unrelated sentences share few
        or no bins, giving near-zero similarity.
        """
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        vec = [0.0] * dim
        for tok in tokens:
            idx = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16) % dim
            vec[idx] += 1.0
        return vec