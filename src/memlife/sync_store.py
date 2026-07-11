"""SyncMemoryStore — synchronous wrapper for non-async codebases.

Wraps MemoryStore's async methods with asyncio.run() so users without
an event loop can use the full API synchronously.

Usage:
    from memlife import SyncMemoryStore, MemoryConfig, DummyEmbedder

    store = SyncMemoryStore(
        config=MemoryConfig(db_path="./mem.db"),
        embedder=DummyEmbedder(),
    )
    store.remember(task="hello", outcome="success")
    fact_id = store.store_fact("Test fact", confidence=0.7)
    context = store.retrieve("test")
    store.run_gc()

The sync wrapper delegates all sync methods directly to the underlying
MemoryStore (which is already sync for those). Only the async methods
are wrapped.
"""

from __future__ import annotations

import asyncio
from typing import Any

from memlife.config import MemoryConfig
from memlife.store import MemoryStore
from memlife.protocols import Embedder


class SyncMemoryStore:
    """Synchronous wrapper around MemoryStore.

    All sync methods (remember, recall, recent, run_gc, search_episodes_by_tool,
    etc.) pass through directly. Async methods (store_fact, recall_facts,
    retrieve, embed_episode, backfill_embeddings, etc.) are wrapped with
    asyncio.run() so they can be called without an event loop.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
    ):
        self._store = MemoryStore(config=config, embedder=embedder)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _run(self, coro: Any) -> Any:
        """Run a coroutine, creating a new event loop if needed."""
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context — can't use asyncio.run().
            raise RuntimeError(
                "SyncMemoryStore cannot be used from within a running "
                "event loop. Use MemoryStore directly instead."
            )
        except RuntimeError as exc:
            # "no running event loop" (expected) — fall through to asyncio.run().
            # Any other RuntimeError (e.g. our own message above) is re-raised.
            if "cannot be used from within a running" in str(exc):
                raise
        return asyncio.run(coro)

    # --- Passthrough properties ---
    @property
    def db_path(self) -> str:
        return self._store.db_path

    @property
    def config(self) -> MemoryConfig:
        return self._store.config

    @property
    def embedder(self) -> Embedder | None:
        return self._store.embedder

    @property
    def embedding_model_name(self) -> str:
        return self._store.embedding_model_name

    @embedding_model_name.setter
    def embedding_model_name(self, value: str) -> None:
        self._store.embedding_model_name = value

    @property
    def fact_merge_threshold(self) -> float:
        return self._store.fact_merge_threshold

    @fact_merge_threshold.setter
    def fact_merge_threshold(self, value: float) -> None:
        self._store.fact_merge_threshold = value

    @property
    def fact_conflict_threshold(self) -> float:
        return self._store.fact_conflict_threshold

    @fact_conflict_threshold.setter
    def fact_conflict_threshold(self, value: float) -> None:
        self._store.fact_conflict_threshold = value

    # --- Sync passthrough methods ---
    def remember(self, task: str, outcome: str = "success",
                 summary: str = "", tool_calls: list[dict] | None = None) -> str:
        return self._store.remember(task, outcome, summary, tool_calls)

    def recall(self, query: str, limit: int = 10):
        return self._store.recall(query, limit=limit)

    def recent(self, limit: int = 10):
        return self._store.recent(limit=limit)

    def search_episodes_by_tool(self, tool_name: str, outcome: str | None = None,
                                limit: int = 20):
        return self._store.search_episodes_by_tool(tool_name, outcome, limit)

    def search_episodes_by_keyword(self, query: str, limit: int = 10):
        return self._store.search_episodes_by_keyword(query, limit=limit)

    def search_journal(self, query: str, limit: int = 10):
        return self._store.search_journal(query, limit=limit)

    def fact_by_id(self, fact_id: str):
        return self._store.fact_by_id(fact_id)

    def journal_recent(self, limit: int = 5):
        return self._store.journal_recent(limit=limit)

    def embedding_health(self) -> dict:
        return self._store.embedding_health()

    def run_gc(self, **kwargs) -> dict:
        return self._store.run_gc(**kwargs)

    def list_sessions(self):
        return self._store.list_sessions()

    def close(self) -> None:
        self._store.close()

    # --- Async-wrapped methods ---
    def store_fact(self, content: str, source: str = "agent",
                   confidence: float = 0.5, embed: bool = True) -> str:
        return self._run(self._store.store_fact(content, source, confidence, embed))

    def recall_facts(self, query: str, limit: int = 5,
                     query_vector: list[float] | None = None):
        return self._run(self._store.recall_facts(query, limit, query_vector))

    def retrieve(self, query: str, config: MemoryConfig | None = None) -> str:
        return self._run(self._store.retrieve(query, config))

    def embed_episode(self, ep_id: str) -> None:
        return self._run(self._store.embed_episode(ep_id))

    def embed_journal_entry(self, jid: str) -> None:
        return self._run(self._store.embed_journal_entry(jid))

    def revise_fact(self, fact_id: str, new_content: str,
                    confidence: float = 0.7) -> str:
        return self._run(self._store.revise_fact(fact_id, new_content, confidence))

    def check_conflicts(self, content: str) -> list[dict]:
        return self._run(self._store.check_conflicts(content))

    def backfill_embeddings(self, batch_size: int = 20) -> dict:
        return self._run(self._store.backfill_embeddings(batch_size))