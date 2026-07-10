"""Journal entries, contradictions, reflection queue, and retrieval.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

from memlife._utils import _parse_annotations
import json
import logging
import time
import uuid
from memlife.models import JournalEntry
from memlife.vectors import cosine, recency_weight
from memlife.config import MemoryConfig


logger = logging.getLogger(__name__)




class JournalStore:
    """Journal entries, contradictions, reflection queue, and retrieval."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object
    embedder: object
    embedding_model_name: str
    fact_merge_threshold: float
    fact_conflict_threshold: float
    _embed_failures: int
    _recall_counters: dict[str, int]

    def annotate_journal(
        self, entry_id: str, label: str, *, dedupe: bool = True,
    ) -> bool:
        """Attach a veracity annotation to a journal entry."""
        if not label or not label.strip():
            return False
        label = label.strip()
        row = self.conn.execute(
            "SELECT annotations_json FROM journal WHERE id = ?", (entry_id,)
        ).fetchone()
        if not row:
            return False
        annotations = _parse_annotations(row[0])
        if dedupe and label in annotations:
            return False
        annotations.append(label)
        self.conn.execute(
            "UPDATE journal SET annotations_json = ? WHERE id = ?",
            (json.dumps(annotations), entry_id),
        )
        self.conn.commit()
        return True

    def _load_links(self, entry_id: str) -> list[dict]:
        """Load journal link list for an entry."""
        row = self.conn.execute(
            "SELECT links_json FROM journal WHERE id = ?", (entry_id,)
        ).fetchone()
        if not row or not row[0]:
            return []
        try:
            value = json.loads(row[0])
            if isinstance(value, list):
                return value
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def link_journal_entries(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        strength: float = 1.0,
        *,
        bidirectional: bool = False,
        belief_type: str = "world",
        evidence: str = "",
    ) -> bool:
        """Create a belief-network link between two journal entries.

        ``relation`` is one of ``supports``, ``undermines``, or ``related``.
        ``strength`` is clamped to [0.0, 1.0].
        ``belief_type`` is ``user`` or ``world`` (MV2-009).
        ``evidence`` is an optional free-form provenance note.
        Returns True if ``from_id`` was updated; raises ValueError for
        unsupported relations or belief types.
        """
        if relation not in {"supports", "undermines", "related"}:
            raise ValueError(f"unsupported relation: {relation}")
        if belief_type not in {"user", "world"}:
            raise ValueError(f"unsupported belief_type: {belief_type}")
        if from_id == to_id:
            return False
        strength = max(0.0, min(1.0, float(strength)))
        row = self.conn.execute(
            "SELECT 1 FROM journal WHERE id = ?", (from_id,)
        ).fetchone()
        if not row:
            return False
        links = self._load_links(from_id)
        # Replace any existing link to the same target.
        links = [ln for ln in links if ln.get("target") != to_id]
        links.append({
            "target": to_id,
            "relation": relation,
            "strength": strength,
            "belief_type": belief_type,
            "evidence": evidence,
        })
        self.conn.execute(
            "UPDATE journal SET links_json = ? WHERE id = ?",
            (json.dumps(links), from_id),
        )
        self.conn.commit()
        if bidirectional:
            self.link_journal_entries(
                to_id, from_id, "related", strength,
                bidirectional=False, belief_type=belief_type, evidence=evidence,
            )
        return True

    def list_contradictions(self, limit: int = 20) -> list[dict]:
        """Return stored contradiction journal entries resolved to current facts.

        Each item contains the contradiction journal id, when it was detected,
        and the two conflicting facts resolved to their currently-active
        versions (following supersession chains). If a fact was retired or the
        chain cannot be followed, the original fact id and content are
        returned marked as missing/retired so the contradiction stays visible.
        """
        rows = self.conn.execute(
            "SELECT id, content, confidence, source_episodes_json, created_at "
            "FROM journal WHERE type = 'contradiction' AND superseded_by = '' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: list[dict] = []
        for row in rows:
            jid, content, conf, source_json, created_at = row
            fact_ids: list[str] = []
            try:
                fact_ids = json.loads(source_json or "[]")
            except json.JSONDecodeError:
                pass
            facts: list[dict] = []
            for fid in fact_ids[:2]:
                resolved = self.resolve_fact(fid)
                if resolved:
                    facts.append({
                        "id": resolved.id,
                        "content": resolved.content,
                        "confidence": resolved.confidence,
                        "active": True,
                    })
                else:
                    original = self.fact_by_id(fid)
                    facts.append({
                        "id": fid,
                        "content": original.content if original else "(unknown)",
                        "confidence": original.confidence if original else 0.0,
                        "active": False,
                        "missing": original is None,
                        "retired": bool(original and original.retired),
                    })
            # A contradiction is considered unresolved if both sides still
            # resolve to active, distinct facts.
            unresolved = (
                len(facts) == 2
                and facts[0]["active"]
                and facts[1]["active"]
                and facts[0]["id"] != facts[1]["id"]
            )
            results.append({
                "id": jid,
                "content": content,
                "confidence": conf,
                "created_at": created_at,
                "facts": facts,
                "unresolved": unresolved,
            })
        return results

    def add_journal_entry(
        self, type: str, content: str, confidence: float = 0.5,
        source_episodes: list[str] | None = None, private: bool = True,
        last_detected: int = 0,
    ) -> str:
        """Store a journal entry. Returns its id. (Sync; embedding is a
        separate async step via :meth:`embed_journal_entry`.)"""
        jid = f"jrn_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO journal (id, type, content, confidence, "
            "source_episodes_json, private, created_at, last_detected) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (jid, type, content, confidence,
             json.dumps(source_episodes or []),
             1 if private else 0, time.time(), last_detected),
        )
        self.conn.commit()
        return jid

    async def embed_journal_entry(self, jid: str) -> None:
        """Compute and store an embedding for a journal entry's content."""
        row = self.conn.execute(
            "SELECT content FROM journal WHERE id = ?", (jid,)
        ).fetchone()
        if not row or not row[0]:
            return
        vecs = await self.embed_texts([row[0]])
        if vecs:
            self.conn.execute(
                "UPDATE journal SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                (self._serialize_vec(vecs[0]), self.embedding_model_name, jid),
            )
            self.conn.commit()
            self._maybe_store_vec("journal", jid, vecs[0])

    async def recall_journal_vector(
        self, query_vector: list[float], limit: int = 5,
    ) -> list[JournalEntry]:
        """Vector recall over journal entries that have embeddings.

        Contradictions are excluded — they're unresolved tensions recorded
        for reflection, not beliefs to inject into a turn (see
        ``_active_journal_sql`` for the same filter on the keyword paths).
        """
        rows = self.conn.execute(
            "SELECT id, type, content, confidence, source_episodes_json, "
            "private, created_at, superseded_by, embedding_json, "
            "last_detected, annotations_json, links_json "
            "FROM journal WHERE superseded_by = '' AND embedding_json != '' "
            "AND type != 'contradiction' "
            "ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        entries = [self._journal_from_row(r) for r in rows]
        scored = []
        for j in entries:
            v = j.embedding
            if v is None:
                continue
            sim = cosine(query_vector, v)
            # MF-016: use the unified score formula (sim x confidence x recency)
            # matching other recall methods, instead of raw cosine similarity.
            rec = recency_weight(j.created_at, halflife_days=30.0)
            score = sim * j.confidence * rec
            j._relevance = sim  # type: ignore[attr-defined]
            j._vector_sim = sim  # type: ignore[attr-defined]
            j._score = score  # type: ignore[attr-defined]
            scored.append((score, j))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [j for s, j in scored[:limit] if s > 0.0]

    def search_journal(self, query: str, limit: int = 10) -> list[JournalEntry]:
        """Keyword search over active journal entries (observations and
        hypotheses only, no contradictions). Used by the memory_search_journal
        tool so the agent can pull relevant past reflections."""
        tokens = self._tokenize(query)
        if not tokens:
            return self.journal_recent(limit=limit)
        clauses = " OR ".join("content LIKE ?" for _ in tokens)
        params = [f"%{t}%" for t in tokens]
        rows = self.conn.execute(
            f"SELECT id, type, content, confidence, source_episodes_json, "
            f"private, created_at, superseded_by, embedding_json, "
            f"last_detected, annotations_json, links_json FROM journal "
            f"WHERE superseded_by = '' AND type IN ('observation', 'hypothesis') "
            f"AND ({clauses}) "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit * 4),
        ).fetchall()
        entries = [self._journal_from_row(r) for r in rows]
        n_tokens = max(1, len(tokens))
        scored = []
        for j in entries:
            hay = j.content.lower()
            hits = sum(1 for t in tokens if t in hay)
            if hits > 0:
                j._match_score = hits  # type: ignore[attr-defined]
                j._relevance = hits / n_tokens  # type: ignore[attr-defined]
                j._text_score = hits / n_tokens  # type: ignore[attr-defined]
                scored.append((j._match_score, j))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [j for _, j in scored[:limit]]

    async def retrieve(self, query: str, config: MemoryConfig | None = None) -> str:
        """Retrieve and rank memories across all layers.

        This is the canonical retrieval method. Delegates to
        :func:`memlife.retrieval.retrieve`. If ``config`` is None,
        falls back to ``self.config``.
        """
        from memlife.retrieval import retrieve as _retrieve
        return await _retrieve(self, query, config or self.config)

    def supersede_journal(self, target_id: str, by_id: str) -> bool:
        """Mark a prior journal entry superseded by a new one (closes the loop).

        Returns True if a row was updated; False (with a warning) if the target
        is missing or already superseded/retired — a revision that closes no
        loop is almost certainly a bogus ``revises`` id from the model.
        """
        cur = self.conn.execute(
            "UPDATE journal SET superseded_by = ? WHERE id = ? AND superseded_by = ''",
            (by_id, target_id),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            logger.warning(
                "supersede_journal: target %r not updated (missing or already "
                "superseded/retired); revision %r closes no loop.",
                target_id, by_id,
            )
            return False
        return True

    def retire_journal(self, entry_id: str, reason: str = "low-confidence") -> None:
        """Retire a journal entry from retrieval without deleting it.

        Uses the ``__retired__:<reason>`` sentinel in ``superseded_by``; this is
        distinct from real supersession (see :meth:`supersede_journal`) and is
        reported separately via :attr:`JournalEntry.retired`.
        """
        self.conn.execute(
            "UPDATE journal SET superseded_by = ? WHERE id = ? AND superseded_by = ''",
            (f"__retired__:{reason}", entry_id),
        )
        self.conn.commit()

    def has_active_contradiction(self, fact_a: str, fact_b: str) -> bool:
        """Return True if an active contradiction already covers this fact pair.

        The pair is considered the same regardless of ordering.
        """
        row = self.conn.execute(
            "SELECT 1 FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "AND ("
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?) "
            " OR "
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?)"
            ") "
            "LIMIT 1",
            (fact_a, fact_b, fact_b, fact_a),
        ).fetchone()
        return row is not None

    def touch_active_contradiction(
        self, fact_a: str, fact_b: str, cycle: int
    ) -> int:
        """Update last_detected for an active contradiction covering this pair.

        Returns the number of rows updated (0 or 1). The pair is considered
        the same regardless of ordering.
        """
        cur = self.conn.execute(
            "UPDATE journal SET last_detected = ? "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "AND ("
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?) "
            " OR "
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?)"
            ")",
            (cycle, fact_a, fact_b, fact_b, fact_a),
        )
        self.conn.commit()
        return cur.rowcount

    def reinforce_unresolved_contradictions(self, current_cycle: int) -> int:
        """Update last_detected for contradictions that are still unresolved.

        A contradiction is unresolved if both source facts still resolve to
        active, distinct facts. These tensions stay alive across reflection
        passes and should not be retired just because they were first created
        a while ago. Returns the number of contradictions reinforced.
        """
        rows = self.conn.execute(
            "SELECT id, source_episodes_json FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction'"
        ).fetchall()
        reinforced = 0
        for jid, source_json in rows:
            fact_ids: list[str] = []
            try:
                fact_ids = json.loads(source_json or "[]")
            except json.JSONDecodeError:
                continue
            if len(fact_ids) != 2:
                continue
            a = self.resolve_fact(fact_ids[0])
            b = self.resolve_fact(fact_ids[1])
            if a is None or b is None or a.id == b.id:
                continue
            self.conn.execute(
                "UPDATE journal SET last_detected = ? WHERE id = ?",
                (current_cycle, jid),
            )
            reinforced += 1
        self.conn.commit()
        return reinforced

    def retire_stale_contradictions(
        self, current_cycle: int, retirement_cycles: int
    ) -> list[str]:
        """Retire active contradictions not seen in ``retirement_cycles`` passes.

        Returns the IDs of contradictions retired in this pass. Retired entries
        are marked with ``__retired__:stale-contradiction`` so they remain in
        the table but are excluded from active retrieval.
        """
        if retirement_cycles <= 0:
            return []
        cutoff = current_cycle - retirement_cycles
        rows = self.conn.execute(
            "SELECT id FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "AND last_detected <= ?",
            (cutoff,),
        ).fetchall()
        retired_ids: list[str] = []
        for (jid,) in rows:
            self.retire_journal(jid, reason="stale-contradiction")
            retired_ids.append(jid)
        return retired_ids

    def _journal_from_row(self, r) -> JournalEntry:
        return JournalEntry(
            id=r[0], type=r[1], content=r[2], confidence=r[3],
            source_episodes_json=r[4] or "[]", private=bool(r[5]),
            created_at=r[6], superseded_by=r[7] or "",
            embedding_json=(r[8] or "") if len(r) > 8 else "",
            last_detected=r[9] if len(r) > 9 else 0,
            annotations_json=(r[10] or "[]") if len(r) > 10 else "[]",
            links_json=(r[11] or "[]") if len(r) > 11 else "[]",
        )

    def _active_journal_sql(self, select: str, extra_where: str = "") -> str:
        return (
            f"SELECT {select} FROM journal WHERE superseded_by = '' "
            f"AND type != 'contradiction' "
            f"{extra_where} ORDER BY created_at DESC"
        )

    def journal_contradictions(self, limit: int = 20) -> list[JournalEntry]:
        """Active contradiction journal entries, newest first."""
        rows = self.conn.execute(
            "SELECT id, type, content, confidence, source_episodes_json, "
            "private, created_at, superseded_by, embedding_json, last_detected "
            "FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def journal_recent(self, limit: int = 5) -> list[JournalEntry]:
        rows = self.conn.execute(
            self._active_journal_sql(
                "id, type, content, confidence, source_episodes_json, "
                "private, created_at, superseded_by, embedding_json, "
                "last_detected, annotations_json, links_json"
            ) + " LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def journal_by_type(self, type: str, limit: int = 10) -> list[JournalEntry]:
        rows = self.conn.execute(
            self._active_journal_sql(
                "id, type, content, confidence, source_episodes_json, "
                "private, created_at, superseded_by, embedding_json, "
                "last_detected, annotations_json, links_json",
                "AND type = ?"
            ) + " LIMIT ?",
            (type, limit),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def journal_relevant(self, query: str, limit: int = 3) -> list[JournalEntry]:
        """Keyword recall over active journal entries, ranked by recency×confidence."""
        tokens = self._tokenize(query)
        if not tokens:
            rows = self.conn.execute(
                self._active_journal_sql(
                    "id, type, content, confidence, source_episodes_json, "
                    "private, created_at, superseded_by, embedding_json, "
                    "last_detected, annotations_json, links_json"
                ) + " LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._journal_from_row(r) for r in rows]
        clauses = " OR ".join("content LIKE ?" for _ in tokens)
        params = [f"%{t}%" for t in tokens]
        rows = self.conn.execute(
            f"SELECT id, type, content, confidence, source_episodes_json, "
            f"private, created_at, superseded_by, embedding_json, "
            f"last_detected, annotations_json, links_json FROM journal "
            f"WHERE superseded_by = '' AND type != 'contradiction' "
            f"AND ({clauses}) "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit * 4),
        ).fetchall()
        entries = [self._journal_from_row(r) for r in rows]
        n_tokens = max(1, len(tokens))
        for j in entries:
            hay = j.content.lower()
            j._match_score = sum(1 for t in tokens if t in hay)  # type: ignore[attr-defined]
            j._relevance = j._match_score / n_tokens  # type: ignore[attr-defined]
            j._text_score = j._match_score / n_tokens  # type: ignore[attr-defined]
        entries.sort(
            key=lambda e: (getattr(e, "_match_score", 0), e.confidence, e.created_at),
            reverse=True,
        )
        return entries[:limit]

    def consolidate_journal(
        self, halflife_days: float = 30.0, floor: float = 0.15,
    ) -> dict:
        """Retire entries whose decayed confidence has fallen below ``floor``,
        and merge near-duplicate active observations. Returns counts.

        Retirement is batched into a single transaction (one commit, not one
        per entry). Merging uses token-Jaccard similarity (>= 0.8) rather than a
        raw 40-char prefix, so two observations that share a long prefix but
        diverge substantially are no longer collapsed — and two paraphrases
        that share no prefix are.
        """
        retired = 0
        merged = 0
        active = self.journal_recent(limit=500)
        # Retire low effective-confidence observations/hypotheses — batched.
        to_retire = [
            j.id for j in active
            if j.type in ("observation", "hypothesis")
            and j.effective_confidence(halflife_days, floor) <= floor
        ]
        if to_retire:
            placeholders = ",".join("?" * len(to_retire))
            self.conn.execute(
                f"UPDATE journal SET superseded_by = '__retired__:low-confidence' "
                f"WHERE id IN ({placeholders}) AND superseded_by = ''",
                tuple(to_retire),
            )
            self.conn.commit()
            retired = len(to_retire)
        # Merge near-duplicate active observations by token-Jaccard similarity.
        surviving = self.journal_by_type("observation", limit=500)
        token_sets = [set(self._tokenize(j.content)) for j in surviving]
        superseded_ids: set[str] = set()
        merge_pairs: list[tuple[str, str]] = []  # (old_id, new_id)
        for i, j in enumerate(surviving):
            ti = token_sets[i]
            if not ti or j.id in superseded_ids:
                continue
            for k in range(i + 1, len(surviving)):
                if surviving[k].id in superseded_ids:
                    continue
                tk = token_sets[k]
                if not tk:
                    continue
                union = ti | tk
                sim = len(ti & tk) / len(union)
                # surviving is newest-first, so j (i) is newer than k (i<k).
                # Supersede the older (k) with the newer (j).
                if sim >= 0.8:
                    merge_pairs.append((surviving[k].id, j.id))
                    superseded_ids.add(surviving[k].id)
        # MF-016: batch all supersessions into one commit instead of
        # calling supersede_journal() (which commits) per merge.
        for old_id, new_id in merge_pairs:
            self.conn.execute(
                "UPDATE journal SET superseded_by = ? WHERE id = ? AND superseded_by = ''",
                (new_id, old_id),
            )
        if merge_pairs:
            self.conn.commit()
        merged = len(merge_pairs)
        return {"retired": retired, "merged": merged}

    def queue_reflection(self, episode_id: str) -> None:
        """Flag an episode for the nightly reflection to consider.

        Idempotent at the DB level: the unique index on ``episode_id`` means a
        re-queue of the same episode is a no-op (``INSERT OR IGNORE``), with no
        TOCTOU window between a SELECT and an INSERT.
        """
        qid = f"rq_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT OR IGNORE INTO reflection_queue (id, episode_id, queued_at) "
            "VALUES (?, ?, ?)",
            (qid, episode_id, time.time()),
        )
        self.conn.commit()

    def pending_reflections(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT episode_id FROM reflection_queue WHERE reflected = 0 "
            "ORDER BY queued_at ASC"
        ).fetchall()
        return [r[0] for r in rows]

    def mark_reflected(self, episode_id: str) -> None:
        self.conn.execute(
            "UPDATE reflection_queue SET reflected = 1 WHERE episode_id = ?",
            (episode_id,),
        )
        self.conn.commit()

