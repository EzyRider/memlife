"""Storage and retrieval of raw episodes.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from memlife.models import Episode, JournalEntry
from memlife.vectors import cosine, recency_weight


logger = logging.getLogger(__name__)


class EpisodeStore:
    """Storage and retrieval of raw episodes."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object

    def remember(
        self, task: str, outcome: str, summary: str = "",
        tool_calls: list[dict] | None = None,
    ) -> str:
        """Store an episode. Returns the episode ID. (Sync; embedding is a
        separate async step via :meth:`embed_episode`.)

        Tool call names are indexed in ``episode_tools`` for fast
        ``search_episodes_by_tool`` queries.

        If ``gap_marker_threshold_hours`` is set and the new episode is
        more than that many hours after the previous episode, a synthetic
        "time passed" episode is inserted to preserve narrative continuity
        (MV2-008).
        """
        ep_id = f"ep_{uuid.uuid4().hex[:12]}"
        now = time.time()

        # MV2-008: insert a synthetic gap marker if the silence is long.
        # The gap marker and the real episode are committed together so an
        # orphaned marker is impossible if the process crashes mid-write.
        threshold_hours = getattr(self.config, "gap_marker_threshold_hours", 0.0)
        if threshold_hours > 0:
            last_row = self.conn.execute(
                "SELECT id, created_at FROM episodes ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if last_row and last_row[0]:
                last_time = last_row[1]
                gap_hours = (now - last_time) / 3600.0
                if gap_hours > threshold_hours:
                    gap_id = f"gap_{uuid.uuid4().hex[:12]}"
                    gap_label = self._format_gap(gap_hours)
                    # Place marker midway through the gap so it sorts between
                    # the two real episodes.
                    marker_at = last_time + (now - last_time) / 2.0
                    self.conn.execute(
                        "INSERT INTO episodes (id, task, outcome, summary, "
                        "tool_calls_json, created_at, is_gap_marker) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (gap_id, gap_label, "", "",
                         json.dumps([]), marker_at, 1),
                    )

        try:
            self.conn.execute(
                "INSERT INTO episodes (id, task, outcome, summary, "
                "tool_calls_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (ep_id, task, outcome, summary,
                 json.dumps(tool_calls or []), now),
            )
            # Index tool names for episodic search by behaviour.
            if tool_calls:
                seen_tools: set[str] = set()
                for tc in tool_calls:
                    name = tc.get("tool") if isinstance(tc, dict) else None
                    if name and name not in seen_tools:
                        seen_tools.add(name)
                        self.conn.execute(
                            "INSERT OR IGNORE INTO episode_tools "
                            "(episode_id, tool_name, created_at) VALUES (?, ?, ?)",
                            (ep_id, name, now),
                        )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return ep_id

    def _format_gap(self, gap_hours: float) -> str:
        """Human-readable gap marker label for a time gap."""
        if gap_hours < 48:
            return f"[gap: {int(round(gap_hours))} hours passed]"
        days = gap_hours / 24.0
        if days < 60:
            return f"[gap: {int(round(days))} days passed]"
        months = days / 30.44
        return f"[gap: {int(round(months))} months passed]"

    async def embed_episode(self, ep_id: str) -> None:
        """Compute and store an embedding for an episode's task+summary."""
        row = self.conn.execute(
            "SELECT task, summary FROM episodes WHERE id = ?", (ep_id,)
        ).fetchone()
        if not row:
            return
        text = f"{row[0]}\n{row[1] or ''}".strip()
        if not text:
            return
        vecs = await self.embed_texts([text])
        if vecs:
            self.conn.execute(
                "UPDATE episodes SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                (self._serialize_vec(vecs[0]), self.embedding_model_name, ep_id),
            )
            self.conn.commit()
            self._maybe_store_vec("episodes", ep_id, vecs[0])

    def _tokenize(self, query: str) -> list[str]:
        # MF-013: was len >= 3, which dropped important short terms like
        # "AI", "ML", "Go", "C", "OS", "Py". Lowered to 2.
        return [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) >= 2]

    def recall(self, query: str, limit: int = 5) -> list[Episode]:
        """Tokenised keyword recall over task and summary, ranked by match count
        × recency. Falls back to recent() if the query has no usable tokens.
        """
        # MF-016: validate limit — negative values return all rows in SQLite.
        limit = max(0, limit)
        tokens = self._tokenize(query)
        if not tokens:
            return self.recent(limit=limit)
        clauses = []
        params: list = []
        for t in tokens:
            clauses.append("(task LIKE ? OR summary LIKE ?)")
            params += [f"%{t}%", f"%{t}%"]
        where = " OR ".join(clauses)
        rows = self.conn.execute(
            f"SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            f"embedding_json, is_gap_marker FROM episodes WHERE ({where}) "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit * 4),
        ).fetchall()
        eps = [Episode.from_row(tuple(r)) for r in rows]
        # Rank by normalised relevance × recency. _relevance ∈ [0,1] so the
        # agent's cross-layer ranker (relevance × confidence × recency) is on a
        # common scale with facts/journal.
        n_tokens = max(1, len(tokens))
        for ep in eps:
            hay = (ep.task + " " + ep.summary).lower()
            match_count = sum(1 for t in tokens if t in hay)
            ep._relevance = match_count / n_tokens  # type: ignore[attr-defined]
            ep._text_score = match_count / n_tokens  # type: ignore[attr-defined]
            ep._score = ep._relevance * recency_weight(ep.created_at)  # type: ignore[attr-defined]
        eps.sort(key=lambda e: getattr(e, "_score", 0), reverse=True)
        return eps[:limit]

    async def recall_episodes_vector(
        self, query_vector: list[float], limit: int = 5,
    ) -> list[Episode]:
        """Vector recall over episodes that have embeddings, scored by
        cosine × recency_weight."""
        t0 = time.time()
        if self.vector_backend.name == "sqlite_vec":
            return await self._recall_episodes_sqlite_vec(query_vector, limit)
        rows = self.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            "embedding_json, is_gap_marker FROM episodes WHERE embedding_json != '' "
            "ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        eps = [Episode.from_row(tuple(r)) for r in rows]
        scored = []
        for ep in eps:
            v = ep.embedding
            if v is None:
                continue
            sim = cosine(query_vector, v)
            rw = recency_weight(ep.created_at)
            # Attach the normalised relevance so the agent's cross-layer ranker
            # (relevance × confidence × recency) actually includes cosine —
            # previously this was computed only to sort and then discarded.
            ep._relevance = sim  # type: ignore[attr-defined]
            ep._vector_sim = sim  # type: ignore[attr-defined]
            ep._score = sim * rw  # type: ignore[attr-defined]
            scored.append((sim * rw, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        dt = (time.time() - t0) * 1000
        if dt > 50:
            import logging
            logging.getLogger(__name__).info(
                "episode vector recall: %.0fms for %d episodes (%d with embeddings)",
                dt, len(rows), len(scored),
            )
        return [ep for s, ep in scored[:limit] if s > 0.0]

    def recent(self, limit: int = 10) -> list[Episode]:
        """Return most recent episodes."""
        # MF-016: validate limit — negative values return all rows in SQLite.
        limit = max(0, limit)
        rows = self.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            "embedding_json, is_gap_marker FROM episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def episodes_since(self, since: float, limit: int = 200) -> list[Episode]:
        """Episodes created at/after a timestamp (for nightly reflection)."""
        rows = self.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            "embedding_json, is_gap_marker FROM episodes WHERE created_at >= ? "
            "ORDER BY created_at DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    async def _recall_episodes_sqlite_vec(
        self, query_vector: list[float], limit: int,
    ) -> list[Episode]:
        """Use sqlite-vec KNN for episode vector recall, then score in Python."""
        matches = self.vector_backend.search(
            "episodes", query_vector, limit=max(limit * 4, 20)
        )
        if not matches:
            return []
        ids = [result.item_id for result in matches]
        eps = self.episodes_by_ids(ids)
        by_id = {ep.id: ep for ep in eps}
        scored = []
        for result in matches:
            ep = by_id.get(result.item_id)
            if ep is None or ep.embedding is None:
                continue
            rw = recency_weight(ep.created_at)
            ep._relevance = result.similarity  # type: ignore[attr-defined]
            ep._vector_sim = result.similarity  # type: ignore[attr-defined]
            ep._score = result.similarity * rw  # type: ignore[attr-defined]
            scored.append((ep._score, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for s, ep in scored[:limit] if s > 0.0]

    def episodes_sample_historical(
        self, before: float, limit: int = 15, bins: int = 5,
    ) -> list[Episode]:
        """Sample representative episodes from before ``before`` (epoch),
        spread across time so the reflector sees long-term patterns, not
        just a contiguous block.

        Divides the time range from the oldest episode to ``before`` into
        ``bins`` equal segments and takes ``limit // bins`` episodes from
        each segment (most recent in each). This gives the model a view
        that spans days or weeks rather than just the last N episodes.
        """
        oldest = self.conn.execute(
            "SELECT MIN(created_at) FROM episodes"
        ).fetchone()[0]
        if not oldest or oldest >= before:
            return []
        per_bin = max(1, limit // bins)
        results: list[Episode] = []
        seg_size = (before - oldest) / bins
        for i in range(bins):
            seg_start = oldest + (seg_size * i)
            seg_end = oldest + (seg_size * (i + 1))
            rows = self.conn.execute(
                "SELECT id, task, outcome, summary, tool_calls_json, "
                "created_at, embedding_json, is_gap_marker FROM episodes "
                "WHERE created_at >= ? AND created_at < ? "
                "ORDER BY created_at DESC LIMIT ?",
                (seg_start, seg_end, per_bin),
            ).fetchall()
            results.extend(Episode.from_row(tuple(r)) for r in rows)
        return results[:limit]

    def journal_thematic_summary(self, limit: int = 10) -> list[JournalEntry]:
        """Return a spread of older journal entries (observations and
        hypotheses only, no contradictions) for the reflection prompt's
        historical context. Samples across time rather than just the most
        recent entries."""
        # _active_journal_sql already appends ORDER BY created_at DESC,
        # so we just add the LIMIT via parameter.
        rows = self.conn.execute(
            self._active_journal_sql(
                "id, type, content, confidence, source_episodes_json, "
                "private, created_at, superseded_by, embedding_json, "
                "last_detected, annotations_json, links_json",
                "AND type IN ('observation', 'hypothesis')"
            ) + " LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def episodes_by_ids(self, ids: list[str]) -> list[Episode]:
        """Fetch specific episodes by id (preserving no particular order)."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            f"embedding_json, is_gap_marker FROM episodes WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        by_id = {r[0]: Episode.from_row(tuple(r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def search_episodes_by_tool(
        self, tool_name: str, outcome: str | None = None, limit: int = 20,
    ) -> list[Episode]:
        """Find episodes where a specific tool was used. Optionally filter
        by outcome ('success', 'failed', etc.). Uses the episode_tools
        index for fast lookups without parsing JSON blobs."""
        sql = (
            "SELECT e.id, e.task, e.outcome, e.summary, "
            "e.tool_calls_json, e.created_at, e.embedding_json, e.is_gap_marker "
            "FROM episodes e "
            "JOIN episode_tools et ON et.episode_id = e.id "
            "WHERE et.tool_name = ?"
        )
        params: list = [tool_name]
        if outcome:
            sql += " AND e.outcome = ?"
            params.append(outcome)
        sql += " ORDER BY e.created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def search_episodes_by_keyword(
        self, query: str, limit: int = 10,
    ) -> list[Episode]:
        """Keyword search over episodes by task and summary text."""
        tokens = self._tokenize(query)
        if not tokens:
            return self.recent(limit=limit)
        clauses = " OR ".join("(task LIKE ? OR summary LIKE ?)" for _ in tokens)
        params: list[str] = []
        for t in tokens:
            like = f"%{t}%"
            params.extend([like, like])
        rows = self.conn.execute(
            f"SELECT id, task, outcome, summary, tool_calls_json, "
            f"created_at, embedding_json, is_gap_marker FROM episodes "
            f"WHERE {clauses} "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit * 4),
        ).fetchall()
        episodes = [Episode.from_row(tuple(r)) for r in rows]
        scored = []
        for ep in episodes:
            text = (ep.task + " " + (ep.summary or "")).lower()
            hits = sum(1 for t in tokens if t in text)
            if hits > 0:
                ep._match_score = hits  # type: ignore[attr-defined]
                scored.append((hits, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

