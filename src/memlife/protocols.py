"""Protocols and type aliases for injectable embedder and LLM."""

from __future__ import annotations

from typing import Protocol, Sequence


class Embedder(Protocol):
    """Turns text into float vectors.

    Implementations: DummyEmbedder (zero deps), OllamaEmbedder,
    OpenAIEmbedder, SentenceTransformersEmbedder.

    The embedder is optional — when absent, semantic and episodic
    recall degrade to keyword search. The system still works.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        ...


class ChatCallable(Protocol):
    """An LLM chat function for the reflection loop.

    Implementations: DummyChat (canned output), OllamaChat, OpenAIChat.

    Only needed for reflection. Store, retrieval, decay, and GC
    work without any LLM.
    """

    async def chat(self, messages: list[dict], model: str) -> str:
        ...