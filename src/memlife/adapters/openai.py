"""OpenAI adapter — embedder and chat for memlife.

Requires the ``openai`` package: ``pip install memlife[openai]``
or ``pip install openai``.

Usage:
    from memlife.adapters.openai import OpenAIEmbedder, OpenAIChat

    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    chat = OpenAIChat(model="gpt-4o-mini")

    store = MemoryStore(config=config, embedder=embedder)
    reflector = Reflector(memory=store, model_chat=chat)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """Embedder backed by OpenAI's embeddings API.

    Implements the memlife Embedder protocol: ``await embedder.embed(texts)``.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings via OpenAI."""
        if not texts:
            return []
        try:
            client = self._get_client()
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.warning("OpenAI embed failed: %s", exc)
            return None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


class OpenAIChat:
    """Chat callable backed by OpenAI's chat completions API.

    Implements the memlife ChatCallable protocol: ``await chat.chat(messages, model)``.
    Returns the raw text content from the model response.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs = {"max_retries": self.max_retries}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def chat(self, messages: list[dict], model: str) -> str:
        """Send a chat completion request and return the text content."""
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
        )
        # MF-016: handle empty choices list (API error / content filter).
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None