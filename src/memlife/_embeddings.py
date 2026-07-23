"""Embed texts, serialise vectors, backfill stale embeddings, and cache vectors.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlife.vector_backends.base import VectorBackend

logger = logging.getLogger(__name__)


class EmbedMixin:
    """Embed texts, serialise vectors, backfill stale embeddings, and cache vectors."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object
    embedder: object
    embedding_model_name: str
    _embed_failures: int
    vector_backend: VectorBackend

    def _serialize_vec(self, vec: list[float]) -> str:
        """Serialize a float vector for storage."""
        return self.vector_backend.serialize(vec)

    def _deserialize_vec(self, raw: str) -> list[float] | None:
        """Reconstruct a float vector from its stored form."""
        return self.vector_backend.deserialize(raw)

    def _maybe_store_vec(self, kind: str, item_id: str, vec: list[float]) -> None:
        """Persist the vector through the configured backend."""
        if not vec:
            return
        self.vector_backend.store(kind, item_id, vec)

    def _cache_key(self, model_name: str, text: str) -> str:
        """Content-addressable key: model name + sha256 of text."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{model_name}:{text_hash}"

    def _text_hash(self, text: str) -> str:
        """SHA-256 hex digest of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_enabled(self) -> bool:
        """True if the embedding cache is enabled and we have a model name."""
        return bool(
            getattr(self.config, "embedding_cache_enabled", True)
            and self.embedding_model_name
        )

    def _cache_lookup(self, texts: list[str]) -> tuple[list[list[float] | None], int]:
        """Look up cached vectors for ``texts`` under the current model.

        Returns a parallel list of vectors (None for cache misses) and the
        count of hits.  Updates ``last_used_at`` for each hit.

        Queries are batched so a request with N texts costs 1-2 round-trips
        instead of N individual SELECTs plus N UPDATEs.
        """
        if not self._cache_enabled() or not texts:
            return [None] * len(texts), 0

        now = time.time()
        model = self.embedding_model_name
        keys = [self._cache_key(model, text) for text in texts]
        hits: list[list[float] | None] = [None] * len(texts)
        hit_count = 0
        key_to_vec: dict[str, list[float]] = {}

        # SQLite has a default 999-parameter limit; stay well under it.
        chunk = 900
        for i in range(0, len(keys), chunk):
            batch_keys = keys[i : i + chunk]
            placeholders = ",".join("?" * len(batch_keys))
            rows = self.conn.execute(
                f"SELECT cache_key, vector_json FROM embedding_cache "
                f"WHERE cache_key IN ({placeholders})",
                tuple(batch_keys),
            ).fetchall()
            for key, raw in rows:
                try:
                    vec = json.loads(raw)
                    if vec and isinstance(vec, list) and all(
                        isinstance(x, (int, float)) for x in vec
                    ):
                        key_to_vec[key] = vec
                except (json.JSONDecodeError, TypeError):
                    pass

        hit_keys: list[str] = []
        for i, text in enumerate(texts):
            vec = key_to_vec.get(keys[i])
            if vec is not None:
                hits[i] = vec
                hit_count += 1
                hit_keys.append(keys[i])

        if hit_keys:
            # A single text may appear multiple times in ``texts``; only
            # update last_used_at once per distinct cache key.
            unique_hit_keys = list(dict.fromkeys(hit_keys))
            for i in range(0, len(unique_hit_keys), chunk):
                batch_keys = unique_hit_keys[i : i + chunk]
                placeholders = ",".join("?" * len(batch_keys))
                self.conn.execute(
                    f"UPDATE embedding_cache SET last_used_at = ? "
                    f"WHERE cache_key IN ({placeholders})",
                    (now, *batch_keys),
                )
            self.conn.commit()
        return hits, hit_count

    def _cache_store(self, texts: list[str], vectors: list[list[float] | None]) -> int:
        """Store non-None vectors in the embedding cache.  Returns stored count.

        Inserts are batched into multi-row ``INSERT OR REPLACE`` statements
        so a batch of N vectors costs a small number of round-trips instead
        of N individual INSERTs.
        """
        if not self._cache_enabled():
            return 0
        now = time.time()
        model = self.embedding_model_name

        rows: list[tuple[str, str, str, str, float, float]] = []
        for text, vec in zip(texts, vectors):
            if not vec:
                continue
            # Guard against buggy embedders that return non-numeric vectors.
            if not all(isinstance(x, (int, float)) for x in vec):
                logger.warning(
                    "Skipping non-numeric embedding cache entry for text %r", text
                )
                continue
            rows.append(
                (
                    self._cache_key(model, text),
                    model,
                    self._text_hash(text),
                    json.dumps(vec),
                    now,
                    now,
                )
            )

        # 6 parameters per row; keep total params under SQLite's 999 limit.
        chunk = 150
        stored = 0
        for i in range(0, len(rows), chunk):
            batch = rows[i : i + chunk]
            placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(batch))
            params = [value for row in batch for value in row]
            try:
                cur = self.conn.execute(
                    f"INSERT OR REPLACE INTO embedding_cache "
                    f"(cache_key, model_name, text_hash, vector_json, created_at, last_used_at) "
                    f"VALUES {placeholders}",
                    params,
                )
                if cur.rowcount > 0:
                    stored += cur.rowcount
                else:
                    stored += len(batch)
            except Exception:
                logger.warning("Failed to store embedding cache entries", exc_info=True)
        if stored:
            self.conn.commit()
        return stored

    async def embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Best-effort embedding; returns None if no embedder or it fails.

        Uses the embedding cache when enabled, so repeated text with the
        same model is cheap.  Cache entries are canonical float vectors,
        independent of the configured vector_backend.

        Failures are logged at WARNING (not DEBUG) so silent degradation to
        keyword recall is visible to operators — an absent or misconfigured
        embedder otherwise looks like the system "just works" on keywords.
        """
        if self.embedder is None or not texts:
            return None

        # 1. Try cache.
        cached, hits = self._cache_lookup(texts)
        if hits == len(texts):
            return cached  # type: ignore[return-value]

        # 2. Collect texts that missed the cache.
        missing_indices = [i for i, v in enumerate(cached) if v is None]
        missing_texts = [texts[i] for i in missing_indices]

        try:
            result = await self.embedder.embed(missing_texts)
            if result is not None:
                dims = {len(v) for v in result if v}
                if len(dims) > 1:
                    logger.warning(
                        "embedder returned inconsistent vector dimensions: %s", dims
                    )
                    self._embed_failures += 1
                    return None
                self._embed_failures = 0  # reset on success
        except Exception as exc:
            self._embed_failures += 1
            logger.warning(
                "embed_texts failed (attempt #%d, falling back to keyword recall): %s",
                self._embed_failures, exc,
            )
            if self._embed_failures >= 5:
                logger.error(
                    "embed_texts has failed %d consecutive times — semantic "
                    "recall is degraded to keyword-only. Check the embedder "
                    "and Ollama availability.",
                    self._embed_failures,
                )
            return None

        if result is None:
            return None

        # 3. Merge cached + fresh, write fresh to cache.
        final: list[list[float] | None] = list(cached)
        for idx, vec in zip(missing_indices, result):
            final[idx] = vec

        self._cache_store(missing_texts, result)
        return final  # type: ignore[return-value]

    def embedding_health(self) -> dict:
        """Return a snapshot of embedding coverage across all memory layers.

        Reports how many facts, journal entries, and episodes have embeddings
        vs how many are missing them, plus the consecutive failure counter.
        Also reports how many embeddings are stale (created with a different
        model than the currently configured one). Use this to detect silent
        degradation of the semantic layer.
        """
        def _count(table: str) -> dict:
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            with_vec = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE embedding_json != ''"
            ).fetchone()[0]
            # Stale = has a vector but the model name doesn't match current.
            # Only counts if we actually have a model name configured.
            stale = 0
            if self.embedding_model_name:
                stale = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE embedding_json != '' "
                    f"AND embedding_model != ?",
                    (self.embedding_model_name,),
                ).fetchone()[0]
            return {"total": total, "with_embeddings": with_vec,
                    "missing": total - with_vec, "stale": stale}

        cache_stats = self._embedding_cache_stats()
        health = {
            "facts": _count("facts"),
            "journal": _count("journal"),
            "episodes": _count("episodes"),
            "embedding_model": self.embedding_model_name,
            "consecutive_failures": self._embed_failures,
            "embedder_present": self.embedder is not None,
            "embedding_cache": cache_stats,
        }
        health.update(self.vector_backend.health())
        return health

    def _embedding_cache_stats(self) -> dict:
        """Return cache row count and estimated size in bytes."""
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(vector_json)), 0) "
            "FROM embedding_cache"
        ).fetchone()
        return {
            "enabled": self._cache_enabled(),
            "entries": row[0] if row else 0,
            "vector_json_bytes": row[1] if row else 0,
        }

    async def backfill_embeddings(self, batch_size: int = 20) -> dict:
        """Re-embed facts, journal entries, and episodes that are missing vectors
        or whose vectors were created with a different embedding model.

        Processes in batches to avoid hammering the embedder. Skips items where
        the content is empty. Returns counts of how many were embedded and how
        many failed. Safe to run repeatedly — only processes items with empty
        embedding_json or a mismatched embedding_model.  The embedding cache is
        primed for every text that is sent to the embedder.
        """
        results = {"facts_embedded": 0, "journal_embedded": 0,
                    "episodes_embedded": 0, "failed": 0}

        # Build the stale-condition: missing OR model mismatch.
        # When no model name is configured, only missing embeddings are
        # re-embedded (backward compatible).
        model_clause = ""
        model_params: list = []
        if self.embedding_model_name:
            model_clause = " OR embedding_model != ?"
            model_params = [self.embedding_model_name]

        # Facts without embeddings or with stale model.
        fact_rows = self.conn.execute(
            f"SELECT id, content FROM facts "
            f"WHERE (embedding_json = ''{model_clause}) AND content != ''",
            model_params,
        ).fetchall()
        for i in range(0, len(fact_rows), batch_size):
            batch = fact_rows[i:i + batch_size]
            texts = [r[1] for r in batch]
            vecs = await self.embed_texts(texts)
            if vecs is None:
                results["failed"] += len(batch)
                continue
            for row, vec in zip(batch, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE facts SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (self._serialize_vec(vec), self.embedding_model_name, row[0]),
                    )
                    self._maybe_store_vec("facts", row[0], vec)
                    results["facts_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        # Journal entries without embeddings (skip contradictions — they're
        # not retrieved into context).
        journal_rows = self.conn.execute(
            f"SELECT id, content FROM journal "
            f"WHERE type != 'contradiction' "
            f"AND (embedding_json = ''{model_clause}) AND content != ''",
            model_params,
        ).fetchall()
        for i in range(0, len(journal_rows), batch_size):
            batch = journal_rows[i:i + batch_size]
            texts = [r[1] for r in batch]
            vecs = await self.embed_texts(texts)
            if vecs is None:
                results["failed"] += len(batch)
                continue
            for row, vec in zip(batch, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE journal SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (self._serialize_vec(vec), self.embedding_model_name, row[0]),
                    )
                    self._maybe_store_vec("journal", row[0], vec)
                    results["journal_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        # Episodes without embeddings or with stale model.
        ep_rows = self.conn.execute(
            f"SELECT id, task, summary FROM episodes "
            f"WHERE is_gap_marker = 0 "
            f"AND (embedding_json = ''{model_clause}) "
            f"AND (task != '' OR summary != '')",
            model_params,
        ).fetchall()
        for i in range(0, len(ep_rows), batch_size):
            batch = ep_rows[i:i + batch_size]
            texts = [f"{r[1]}\n{r[2] or ''}".strip() for r in batch]
            vecs = await self.embed_texts(texts)
            if vecs is None:
                results["failed"] += len(batch)
                continue
            for row, vec in zip(batch, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE episodes SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (self._serialize_vec(vec), self.embedding_model_name, row[0]),
                    )
                    self._maybe_store_vec("episodes", row[0], vec)
                    results["episodes_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        return results
