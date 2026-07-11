"""Embed texts, serialise vectors, and backfill stale embeddings.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING
from memlife import binary_vectors, vec_backend


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbedMixin:
    """Embed texts, serialise vectors, and backfill stale embeddings."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object
    embedder: object
    embedding_model_name: str
    _embed_failures: int

    def _serialize_vec(self, vec: list[float]) -> str:
        """Serialize a float vector for storage.

        When ``config.use_binary_vectors`` is True the vector is binarized
        and base64-encoded as ``binary:<dim>:<bytes>``; otherwise it is
        stored as JSON floats.
        """
        if not vec:
            return ""
        if self.config.use_binary_vectors:
            packed = binary_vectors.binarize(vec)
            return f"binary:{len(vec)}:{base64.b64encode(packed).decode()}"
        return json.dumps(vec)

    def _deserialize_vec(self, raw: str) -> list[float] | None:
        """Reconstruct a float vector from its stored form."""
        if not raw:
            return None
        if raw.startswith("binary:"):
            try:
                _, dim_str, b64 = raw.split(":", 2)
                dim = int(dim_str)
                packed = base64.b64decode(b64)
                return binary_vectors.debinarize(packed, dim)
            except Exception:
                return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _maybe_store_vec(self, kind: str, item_id: str, vec: list[float]) -> None:
        """Try to persist the vector in sqlite-vec; ignore failures silently."""
        if not self.config.use_sqlite_vec or not vec:
            return
        raw = self.conn._raw if hasattr(self.conn, "_raw") else self.conn
        # ensure_schema is called inside store(), but it needs a raw connection
        # with extension loading enabled.  Touch the connection first so the
        # virtual table is created on the same raw handle the store will use.
        vec_backend.ensure_schema(raw, len(vec))
        vec_backend.store(raw, kind, item_id, vec)

    async def embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Best-effort embedding; returns None if no embedder or it fails.

        Failures are logged at WARNING (not DEBUG) so silent degradation to
        keyword recall is visible to operators — an absent or misconfigured
        embedder otherwise looks like the system "just works" on keywords.
        """
        if self.embedder is None or not texts:
            return None
        try:
            result = await self.embedder.embed(texts)
            if result is not None:
                dims = {len(v) for v in result if v}
                if len(dims) > 1:
                    logger.warning(
                        "embedder returned inconsistent vector dimensions: %s", dims
                    )
                    self._embed_failures += 1
                    return None
                self._embed_failures = 0  # reset on success
            return result
        except Exception as exc:  # noqa: BLE001
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

        return {
            "facts": _count("facts"),
            "journal": _count("journal"),
            "episodes": _count("episodes"),
            "embedding_model": self.embedding_model_name,
            "consecutive_failures": self._embed_failures,
            "embedder_present": self.embedder is not None,
        }

    async def backfill_embeddings(self, batch_size: int = 20) -> dict:
        """Re-embed facts, journal entries, and episodes that are missing vectors
        or whose vectors were created with a different embedding model.

        Processes in batches to avoid hammering the embedder. Skips items where
        the content is empty. Returns counts of how many were embedded and how
        many failed. Safe to run repeatedly — only processes items with empty
        embedding_json or a mismatched embedding_model.
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
                        (json.dumps(vec), self.embedding_model_name, row[0]),
                    )
                    results["facts_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        # Journal entries without embeddings (skip contradictions — they're
        # not retrieved, so embedding them is wasted work) or with stale model.
        # MF-005: contradictions with existing but stale embeddings ARE
        # backfilled so they don't become orphaned vectors after a model swap.
        # Only contradictions with NO embedding are skipped.
        if self.embedding_model_name:
            j_rows = self.conn.execute(
                f"SELECT id, content FROM journal "
                f"WHERE (embedding_json = ''{model_clause}) "
                f"AND content != '' AND type != 'contradiction'",
                model_params,
            ).fetchall()
            # Contradictions: only backfill stale-model ones, not missing.
            j_contradiction_rows = self.conn.execute(
                "SELECT id, content FROM journal "
                "WHERE embedding_json != '' AND embedding_model != ? "
                "AND content != '' AND type = 'contradiction'",
                [self.embedding_model_name],
            ).fetchall()
            j_rows = j_rows + j_contradiction_rows
        else:
            j_rows = self.conn.execute(
                "SELECT id, content FROM journal "
                "WHERE embedding_json = '' "
                "AND content != '' AND type != 'contradiction'",
            ).fetchall()
        for i in range(0, len(j_rows), batch_size):
            batch = j_rows[i:i + batch_size]
            texts = [r[1] for r in batch]
            vecs = await self.embed_texts(texts)
            if vecs is None:
                results["failed"] += len(batch)
                continue
            for row, vec in zip(batch, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE journal SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (json.dumps(vec), self.embedding_model_name, row[0]),
                    )
                    results["journal_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        # Episodes without embeddings or with stale model.
        ep_rows = self.conn.execute(
            f"SELECT id, task, summary FROM episodes "
            f"WHERE (embedding_json = ''{model_clause}) AND task != ''",
            model_params,
        ).fetchall()
        ep_texts = [f"{r[1]}\n{r[2] or ''}".strip() for r in ep_rows]
        for i in range(0, len(ep_rows), batch_size):
            batch_texts = ep_texts[i:i + batch_size]
            batch_rows = ep_rows[i:i + batch_size]
            vecs = await self.embed_texts(batch_texts)
            if vecs is None:
                results["failed"] += len(batch_rows)
                continue
            for row, vec in zip(batch_rows, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE episodes SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (json.dumps(vec), self.embedding_model_name, row[0]),
                    )
                    results["episodes_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        return results

