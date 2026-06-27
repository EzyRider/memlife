"""Sentence Transformers adapter — local embedder for memlife.

Requires the ``sentence-transformers`` package:
``pip install memlife[sentence-transformers]``
or ``pip install sentence-transformers``

Runs entirely locally — no API key, no network calls. Good for
privacy-sensitive use cases and offline development.

Usage:
    from memlife.adapters.sentence_transformers import STEmbedder

    embedder = STEmbedder(model="all-MiniLM-L6-v2")
    store = MemoryStore(config=config, embedder=embedder)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class STEmbedder:
    """Embedder backed by a local Sentence Transformers model.

    Implements the memlife Embedder protocol: ``await embedder.embed(texts)``.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ):
        self.model_name = model
        self._device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if self._device:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings locally via Sentence Transformers."""
        if not texts:
            return []
        try:
            model = self._get_model()
            # Sentence Transformers encode is sync — run in executor
            import asyncio
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: model.encode(texts, convert_to_numpy=True)
            )
            return [list(e) for e in embeddings]
        except Exception as exc:
            logger.warning("Sentence Transformers embed failed: %s", exc)
            return None