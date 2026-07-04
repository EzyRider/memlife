"""Ollama adapter — embedder and chat for memlife.

Requires the ``aiohttp`` package: ``pip install memlife[ollama]``
or ``pip install aiohttp``.

Usage:
    from memlife.adapters.ollama import OllamaEmbedder, OllamaChat

    embedder = OllamaEmbedder(
        base_url="http://localhost:11434",
        model="mxbai-embed-large:latest",
    )
    chat = OllamaChat(
        base_url="http://localhost:11434",
        model="your-model-name",
    )

    store = MemoryStore(config=config, embedder=embedder)
    reflector = Reflector(memory=store, model_chat=chat)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when the Ollama API returns an error."""


class OllamaEmbedder:
    """Embedder backed by Ollama's /api/embed endpoint.

    Implements the memlife Embedder protocol: ``await embedder.embed(texts)``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mxbai-embed-large:latest",
        *,
        timeout: float = 30.0,
        _session: aiohttp.ClientSession | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session = _session

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings via Ollama."""
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        try:
            async with self.session.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self._timeout,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("Ollama embed error %d: %s", resp.status, text[:300])
                    return None
                data = await resp.json()
                embeddings = data.get("embeddings", [])
                return list(embeddings) if embeddings else None
        except Exception as exc:
            logger.warning("Ollama embed failed: %s", exc)
            return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


class OllamaChat:
    """Chat callable backed by Ollama's /api/chat endpoint.

    Implements the memlife ChatCallable protocol: ``await chat.chat(messages, model)``.
    Returns the raw text content from the model response.

    Supports model fallback: if the primary model fails, fallback models
    are tried in order. Each model is retried up to ``max_retries`` times.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "",  # MF-011: caller must provide — no deployment-specific default
        fallback_models: list[str] | None = None,
        *,
        temperature: float = 0.7,
        num_ctx: int = 32768,
        max_retries: int = 3,
        timeout: float = 120.0,
        _session: aiohttp.ClientSession | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_models = fallback_models or []
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session = _session

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def chat(self, messages: list[dict], model: str) -> str:
        """Send a chat completion request and return the text content.

        Tries ``model`` first (overrides the default), then fallback models.
        """
        models = [model] + self.fallback_models if model not in self.fallback_models else [model] + [m for m in self.fallback_models if m != model]
        last_error: Exception | None = None

        for m in models:
            try:
                return await self._chat_one(messages, m)
            except OllamaError as exc:
                last_error = exc
                logger.warning("Ollama chat model %r failed: %s", m, exc)
                continue

        raise OllamaError(
            f"All models failed (tried {models}): {last_error}"
        )

    async def _chat_one(
        self, messages: list[dict], model: str,
    ) -> str:
        """One model, with retries."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self._timeout,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("message", {}).get("content", "")
                    text = await resp.text()
                    if resp.status == 404:
                        raise OllamaError(f"model not found: {model}")
                    logger.warning(
                        "Ollama error %d (attempt %d/%d): %s",
                        resp.status, attempt + 1, self.max_retries, text[:300],
                    )
                    last_error = OllamaError(
                        f"Ollama API error {resp.status}: {text[:500]}"
                    )
            except OllamaError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Chat attempt %d/%d for %s failed: %s",
                        attempt + 1, self.max_retries, model, exc,
                    )
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))

        raise OllamaError(
            f"Ollama chat failed for {model} after {self.max_retries} attempts: {last_error}"
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None