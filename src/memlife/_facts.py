"""Storage and retrieval of durable facts.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

from memlife._utils import _parse_annotations
import json
import logging
import re
import time
import uuid
from memlife._schema import MAX_FACT_CONFIDENCE
from memlife.models import Fact
from memlife.vectors import cosine, recency_weight


logger = logging.getLogger(__name__)


class FactStore:
    """Storage and retrieval of durable facts."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object
    fact_merge_threshold: float
    fact_conflict_threshold: float
    embedding_model_name: str

    async def store_fact(
        self, content: str, source: str = "agent",
        confidence: float = 0.5, embed: bool = True,
    ) -> str:
        """Store a fact, optionally embedding it. Returns the fact ID.

        Confidence is clamped to ``MAX_FACT_CONFIDENCE`` so no fact is ever
        stored as immutable (1.0).

        Dedup is layered (later layers only run when the earlier miss):
          1. Exact-normalised content (case/whitespace/punctuation-tail) → same
             fact; return the existing ID.
          2. Containment — one content is a substring of the other → keep the
             longer (more specific) wording, bump its updated_at, don't store
             the shorter redundant one.
          3. Semantic — cosine ≥ ``fact_merge_threshold`` over stored embeddings
             → the same fact reworded, or a correction. Merge by superseding the
             lower-confidence one: the higher-confidence content wins, and on a
             tie the newer content wins (so an equal-confidence correction takes
             effect instead of leaving the stale fact active alongside it).

        With no embedder, only layers 1–2 run (rephrased near-duplicates are
        stored, as before — degrades, doesn't break).
        """
        confidence = min(float(confidence), MAX_FACT_CONFIDENCE)

        norm = self._normalize(content)
        new_lower = content.strip().lower()

        # Layer 1: exact-normalised dedup (cheap; skips the embedding call).
        for fid, raw in self.conn.execute(
            "SELECT id, content FROM facts WHERE superseded_by = ''"
        ).fetchall():
            if self._normalize(raw) == norm:
                return fid

        # Embed once; reused for both the cosine merge and the insert.
        new_vec = None
        if embed:
            vecs = await self.embed_texts([content])
            if vecs:
                new_vec = vecs[0]
        embedding_json = self._serialize_vec(new_vec) if new_vec else ""

        active = self._active_facts()

        # Layer 2: weighted containment — keep the longer (more specific)
        # content, but only when the shorter content doesn't add meaningful
        # tokens the longer one lacks. MF-007: blind substring containment
        # conflated string length with semantic value, erasing nuanced facts.
        for f in active:
            ex = f.content.strip().lower()
            if not ex or ex == new_lower:
                continue
            if ex in new_lower or new_lower in ex:
                # Compute non-stop-word tokens in the symmetric difference.
                shorter = ex if len(ex) < len(new_lower) else new_lower
                longer = new_lower if len(ex) < len(new_lower) else ex
                shorter_tokens = set(re.findall(r"[a-z0-9_]+", shorter))
                longer_tokens = set(re.findall(r"[a-z0-9_]+", longer))
                extra = shorter_tokens - longer_tokens
                _STOP = {"the","a","an","is","are","was","were","be","been",
                         "of","in","on","at","to","for","and","or","but",
                         "not","it","this","that","with","from","by","as"}
                meaningful_extra = extra - _STOP
                # If the shorter fact has meaningful tokens the longer one
                # lacks, skip containment and let the semantic-merge layer
                # (cosine similarity) decide.
                if meaningful_extra:
                    continue
                # Confidence tie-break: if the shorter fact has much higher
                # confidence, prefer it as the retained core truth.
                if confidence > f.confidence + 0.15:
                    return await self._supersede_fact(
                        f.id, content, source, confidence, embedding_json)
                if len(f.content) >= len(content):
                    self._refresh_fact(f.id, confidence)
                    return f.id
                return await self._supersede_fact(
                    f.id, content, source, confidence, embedding_json)

        # Layer 3: semantic merge over stored embeddings.
        if new_vec is not None:
            best: Fact | None = None
            best_sim = 0.0
            for f in active:
                if f.embedding is None:
                    continue
                sim = cosine(new_vec, f.embedding)
                if sim > best_sim:
                    best, best_sim = f, sim
            if best is not None and best_sim >= self.fact_merge_threshold:
                # Higher confidence wins; tie → newer (new) wins.
                if confidence >= best.confidence:
                    return await self._supersede_fact(
                        best.id, content, source, confidence, embedding_json)
                # Existing wins: refresh it, drop the weaker duplicate.
                self._refresh_fact(best.id, confidence)
                return best.id

        # No merge: insert as a new fact.
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self.conn.execute(
            "INSERT INTO facts (id, content, source, confidence, "
            "embedding_json, embedding_model, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fact_id, content, source, confidence, embedding_json,
             self.embedding_model_name if new_vec else "", now, now),
        )
        if new_vec:
            self._maybe_store_vec("facts", fact_id, new_vec)
        self.conn.commit()
        return fact_id

    def _refresh_fact(self, fact_id: str, extra_confidence: float) -> None:
        """Bump a kept fact's updated_at and raise its confidence to the max —
        used when a near-duplicate is folded into an existing fact.
        Confidence is capped at ``MAX_FACT_CONFIDENCE``."""
        now = time.time()
        self.conn.execute(
            "UPDATE facts SET updated_at = ?, "
            "confidence = MIN(MAX(confidence, ?), ?) WHERE id = ?",
            (now, extra_confidence, MAX_FACT_CONFIDENCE, fact_id),
        )
        self.conn.commit()

    def adjust_confidence(self, fact_id: str, delta: float) -> None:
        """Adjust a fact's confidence by delta, clamped to [0.0, MAX_FACT_CONFIDENCE]."""
        self.conn.execute(
            "UPDATE facts SET confidence = MAX(0.0, MIN(?, confidence + ?)), "
            "updated_at = ? WHERE id = ?",
            (MAX_FACT_CONFIDENCE, delta, time.time(), fact_id),
        )
        self.conn.commit()

    async def _supersede_fact(
        self, old_id: str, new_content: str, source: str,
        confidence: float, embedding_json: str,
    ) -> str:
        """Atomically store a new fact and mark ``old_id`` superseded by it
        (the revise_fact SAVEPOINT pattern). Returns the new fact id.
        Incoming confidence is capped at ``MAX_FACT_CONFIDENCE``."""
        new_id = f"fact_{uuid.uuid4().hex[:12]}"
        now = time.time()
        capped = min(float(confidence), MAX_FACT_CONFIDENCE)
        try:
            with self.transaction():
                self.conn.execute("SAVEPOINT supersede_fact")
                self.conn.execute(
                    "INSERT INTO facts (id, content, source, confidence, "
                    "embedding_json, embedding_model, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (new_id, new_content, source, capped, embedding_json,
                     self.embedding_model_name if embedding_json else "", now, now),
                )
                if embedding_json:
                    vec = self._deserialize_vec(embedding_json)
                    if vec:
                        self._maybe_store_vec("facts", new_id, vec)
                self.conn.execute(
                    "UPDATE facts SET superseded_by = ?, updated_at = ? WHERE id = ?",
                    (new_id, now, old_id),
                )
                self.conn.execute("RELEASE SAVEPOINT supersede_fact")
        except Exception:
            try:
                self.conn.execute("ROLLBACK TO SAVEPOINT supersede_fact")
                self.conn.execute("RELEASE SAVEPOINT supersede_fact")
            except Exception:
                logger.warning("savepoint rollback failed for _supersede_fact", exc_info=True)
            raise

        # MV2-003: expire any open triples tied to the old fact.
        self.expire_triples_for_fact(old_id, now)

        return new_id

    def _active_facts(self) -> list[Fact]:
        """All non-superseded facts (with or without embeddings)."""
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by, annotations_json FROM facts "
            "WHERE superseded_by = ''"
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
            annotations_json=r[8] or "[]",
        ) for r in rows]

    def fact_by_id(self, fact_id: str) -> Fact | None:
        """Fetch a single fact by id, regardless of supersession status."""
        row = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by, annotations_json "
            "FROM facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        if not row:
            return None
        return Fact(
            id=row[0], content=row[1], source=row[2], confidence=row[3],
            embedding_json=row[4] or "", created_at=row[5], updated_at=row[6],
            superseded_by=row[7] or "",
            annotations_json=row[8] or "[]",
        )

    def annotate_fact(
        self, fact_id: str, label: str, *, dedupe: bool = True,
    ) -> bool:
        """Attach a veracity annotation to a fact.

        Common labels: ``confirmed``, ``unsure``, ``contradicted``,
        ``retired``.  If ``dedupe`` is True the label is not added twice.
        Returns True if a row was updated.
        """
        if not label or not label.strip():
            return False
        label = label.strip()
        row = self.conn.execute(
            "SELECT annotations_json FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            return False
        annotations = _parse_annotations(row[0])
        if dedupe and label in annotations:
            return False
        annotations.append(label)
        self.conn.execute(
            "UPDATE facts SET annotations_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(annotations), time.time(), fact_id),
        )
        self.conn.commit()
        return True

    def _annotations_for(
        self, table: str, row_id: str,
    ) -> list[str]:
        """Load annotations list for a facts/journal row."""
        row = self.conn.execute(
            f"SELECT annotations_json FROM {table} WHERE id = ?", (row_id,)
        ).fetchone()
        if not row or not row[0]:
            return []
        return _parse_annotations(row[0])

    def resolve_fact(self, fact_id: str) -> Fact | None:
        """Resolve a fact id to the currently-active version.

        If the fact has been superseded by another fact, follow the chain
        until an active (non-superseded) fact is found. Returns None if the
        fact is missing, retired, or the chain ends in a missing/retired fact.
        """
        seen: set[str] = set()
        current_id = fact_id
        while current_id and current_id not in seen:
            seen.add(current_id)
            fact = self.fact_by_id(current_id)
            if fact is None:
                return None
            if not fact.superseded_by:
                return fact
            if fact.retired:
                return None
            current_id = fact.superseded_by
        return None

    @staticmethod
    def _normalize(content: str) -> str:
        """Normalise a fact for dedup: collapse whitespace, lowercase, strip trailing punctuation."""
        # MF-016: was rstrip(".") only — now strips all trailing punctuation
        # so facts ending with !, ?, ;, etc. match their counterparts.
        return re.sub(r"\s+", " ", content.strip().lower()).rstrip(".,;:!?")

    async def check_conflicts(self, content: str) -> list[dict]:
        """Check for existing active facts that semantically overlap with new content.

        Returns a list of dicts with id, content, similarity, confidence for
        each candidate conflict (cosine >= fact_conflict_threshold). Uses
        embeddings when available, falls back to keyword overlap otherwise.
        Embedding failures are logged by embed_texts() — not silently swallowed.
        """
        conflicts: list[dict] = []
        new_vec = None
        # embed_texts handles failure logging internally — no silent swallow.
        vecs = await self.embed_texts([content])
        if vecs:
            new_vec = vecs[0]

        active = self._active_facts()

        if new_vec is not None:
            for f in active:
                if f.embedding is None:
                    continue
                sim = cosine(new_vec, f.embedding)
                if sim >= self.fact_conflict_threshold:
                    conflicts.append({
                        "id": f.id,
                        "content": f.content,
                        "similarity": round(sim, 3),
                        "confidence": f.confidence,
                    })
            conflicts.sort(key=lambda c: c["similarity"], reverse=True)
            return conflicts[:5]

        # Keyword fallback — token overlap, same as recall_facts uses.
        tokens = self._tokenize(content)
        if not tokens:
            return []
        for f in active:
            hay = f.content.lower()
            match_count = sum(1 for t in tokens if t in hay)
            if match_count == 0:
                continue
            score = match_count / max(1, len(tokens))
            if score >= 0.5:
                conflicts.append({
                    "id": f.id,
                    "content": f.content,
                    "similarity": round(score, 3),
                    "confidence": f.confidence,
                })
        conflicts.sort(key=lambda c: c["similarity"], reverse=True)
        return conflicts[:5]

    def expire_fact(self, fact_id: str, reason: str = "expired") -> bool:
        """Mark a fact as expired/superseded without a replacement fact.

        Uses the ``__retired__:<reason>`` sentinel, same pattern as
        retire_journal. Returns True if a row was updated, False if the
        fact was missing or already superseded.
        """
        cur = self.conn.execute(
            "UPDATE facts SET superseded_by = ?, updated_at = ? "
            "WHERE id = ? AND superseded_by = ''",
            (f"__retired__:{reason}", time.time(), fact_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    async def recall_facts(
        self, query: str, limit: int = 5, query_vector: list[float] | None = None,
    ) -> list[Fact]:
        """Vector recall over facts when embeddings are available, scored by
        cosine × confidence × recency; keyword fallback with same weighting."""
        if query_vector is None and self.embedder is not None and query.strip():
            query_vector = (await self.embed_texts([query]) or [None])[0]  # type: ignore[arg-type]
        if query_vector is not None:
            facts_with_vec = self._facts_with_embeddings()
            if self.config.use_sqlite_vec:
                return await self._recall_facts_sqlite_vec(
                    query_vector, facts_with_vec, limit
                )
            scored = []
            for f in facts_with_vec:
                if f.embedding is None:
                    continue
                sim = cosine(query_vector, f.embedding)
                f._relevance = sim  # type: ignore[attr-defined]
                f._vector_sim = sim  # type: ignore[attr-defined]
                f._score = sim * f.confidence * recency_weight(f.updated_at)  # type: ignore[attr-defined]
                scored.append((f._score, f))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = [f for s, f in scored[:limit] if s > 0.0]
            if top:
                return top
        # Keyword fallback — tokenised LIKE, scored by confidence × recency.
        tokens = self._tokenize(query)
        if not tokens:
            return []
        clauses = " OR ".join("content LIKE ?" for _ in tokens)
        params = [f"%{t}%" for t in tokens]
        rows = self.conn.execute(
            f"SELECT id, content, source, confidence, embedding_json, "
            f"created_at, updated_at, superseded_by, annotations_json FROM facts "
            f"WHERE ({clauses}) AND superseded_by = '' "
            f"ORDER BY updated_at DESC LIMIT ?",
            (*params, limit * 3),
        ).fetchall()
        facts = [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
            annotations_json=r[8] or "[]",
        ) for r in rows]
        n_tokens = max(1, len(tokens))
        for f in facts:
            match = sum(1 for t in tokens if t in f.content.lower())
            f._relevance = match / n_tokens  # type: ignore[attr-defined]
            f._text_score = match / n_tokens  # type: ignore[attr-defined]
            f._score = f._relevance * f.confidence * recency_weight(f.updated_at)  # type: ignore[attr-defined]
        facts.sort(key=lambda f: getattr(f, "_score", 0), reverse=True)
        return facts[:limit]

    async def _recall_facts_sqlite_vec(
        self,
        query_vector: list[float],
        facts_with_vec: list[Fact],
        limit: int,
    ) -> list[Fact]:
        """Use sqlite-vec KNN for fact vector recall, then score in Python."""
        from memlife import vec_backend

        raw = self.conn._raw if hasattr(self.conn, "_raw") else self.conn
        matches = vec_backend.search(
            raw, "facts", query_vector, limit=max(limit * 4, 20)
        )
        if not matches:
            return []
        by_id = {f.id: f for f in facts_with_vec if f.embedding is not None}
        scored = []
        for item_id, sim in matches:
            f = by_id.get(item_id)
            if f is None:
                continue
            f._relevance = sim  # type: ignore[attr-defined]
            f._vector_sim = sim  # type: ignore[attr-defined]
            f._score = sim * f.confidence * recency_weight(f.updated_at)  # type: ignore[attr-defined]
            scored.append((f._score, f))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for s, f in scored[:limit] if s > 0.0]

    def _facts_with_embeddings(self) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by, annotations_json FROM facts "
            "WHERE embedding_json != '' AND superseded_by = ''"
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
            annotations_json=r[8] or "[]",
        ) for r in rows]

    def _facts_with_embeddings_since(self, since: float) -> list[Fact]:
        """Active facts with embeddings that were created or updated since
        ``since``. Used by incremental contradiction detection so we only
        compare new/changed facts against the full set, not all pairs."""
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by, annotations_json FROM facts "
            "WHERE embedding_json != '' AND superseded_by = '' "
            "AND (created_at >= ? OR updated_at >= ?)",
            (since, since),
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
            annotations_json=r[8] or "[]",
        ) for r in rows]

    def _active_facts_since(self, since: float) -> list[Fact]:
        """Active facts (with or without embeddings) created or updated since
        ``since``. Lexical fallback path for incremental contradiction detection."""
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by, annotations_json FROM facts "
            "WHERE superseded_by = '' AND (created_at >= ? OR updated_at >= ?)",
            (since, since),
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
            annotations_json=r[8] or "[]",
        ) for r in rows]

    async def revise_fact(self, fact_id: str, new_content: str, confidence: float = 0.7) -> str:
        """Mark an old fact superseded and store a revised one with a fresh embedding.

        The revised fact gets its own ``created_at`` (now) so decay/recency treat
        it as fresh, and a fresh embedding for the *new* content. Returns the new
        fact id, or ``""`` if ``fact_id`` does not exist (no silent no-op).
        """
        row = self.conn.execute(
            "SELECT source FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            logger.warning("revise_fact: fact %r not found; no revision stored.", fact_id)
            return ""
        embedding_json = ""
        vecs = await self.embed_texts([new_content])
        if vecs:
            embedding_json = self._serialize_vec(vecs[0])
        return await self._supersede_fact(
            fact_id, new_content, row[0], confidence, embedding_json)

