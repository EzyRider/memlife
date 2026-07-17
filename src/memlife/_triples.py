"""Subject-predicate-object facts with valid time ranges.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid

from memlife._schema import MAX_FACT_CONFIDENCE


logger = logging.getLogger(__name__)


class TripleMixin:
    """Subject-predicate-object facts with valid time ranges."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object

    def store_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
        valid_from: float | None = None,
        valid_until: float | None = None,
        provenance: list[dict] | None = None,
    ) -> str:
        """Store a standalone subject-predicate-object triple.

        Auto-creates entity records for ``subject`` and ``object`` (resolving
        aliases first) and attaches optional provenance links.  Returns the
        triple id.
        """
        now = time.time()
        triple_id = f"triple_{uuid.uuid4().hex[:12]}"
        subj = self.resolve_entity(subject.strip()) or subject.strip()
        obj = self.resolve_entity(object.strip()) or object.strip()
        pred = predicate.strip()

        self._ensure_entity(subj)
        self._ensure_entity(obj)

        self.conn.execute(
            "INSERT INTO temporal_triples "
            "(id, subject, predicate, object, valid_from, valid_until, "
            "fact_id, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (triple_id, subj, pred, obj,
             valid_from if valid_from is not None else now, valid_until,
             None, min(float(confidence), MAX_FACT_CONFIDENCE), now),
        )
        if provenance:
            self._add_triple_provenance(triple_id, provenance)
        self.conn.commit()
        return triple_id

    def store_fact_triple(
        self,
        fact_id: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
        valid_from: float | None = None,
        valid_until: float | None = None,
    ) -> str:
        """Record that ``fact_id`` asserts ``subject predicate object``.

        If the fact is currently active, ``valid_from`` defaults to now and
        ``valid_until`` is left open (current truth).  Returns the triple id.
        """
        provenance = [{"kind": "fact", "id": fact_id}]
        triple_id = self.store_triple(
            subject, predicate, object,
            confidence=confidence,
            valid_from=valid_from,
            valid_until=valid_until,
            provenance=provenance,
        )
        # Preserve the original fact_id link on the temporal_triples row.
        self.conn.execute(
            "UPDATE temporal_triples SET fact_id = ? WHERE id = ?",
            (fact_id, triple_id),
        )
        self.conn.commit()
        return triple_id

    def store_mention_triple(
        self,
        source_kind: str,
        source_id: str,
        entity: str,
        confidence: float = 0.6,
    ) -> str:
        """Record that ``source_id`` (a fact/episode/journal) mentions ``entity``.

        This creates a lightweight ``mentions`` triple whose subject is the
        source row id and whose object is the canonical entity name. Only the
        object is treated as an entity; the source id is a foreign key, not a
        graph node. The triple is tagged with provenance so GC can remove it
        when the source row is pruned. Returns the triple id.
        """
        now = time.time()
        triple_id = f"triple_{uuid.uuid4().hex[:12]}"
        canonical = self.resolve_entity(entity.strip()) or entity.strip()
        self._ensure_entity(canonical)
        self.conn.execute(
            "INSERT INTO temporal_triples "
            "(id, subject, predicate, object, valid_from, valid_until, "
            "fact_id, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (triple_id, source_id.strip(), "mentions", canonical,
             now, None, None, min(float(confidence), MAX_FACT_CONFIDENCE), now),
        )
        self._add_triple_provenance(triple_id, [{"kind": source_kind, "id": source_id}])
        self.conn.commit()
        return triple_id

    def extract_and_link_entities(
        self,
        source_kind: str,
        source_id: str,
        text: str,
    ) -> None:
        """Extract entities from ``text`` and link them to ``source_id``.

        This is a no-op if ``auto_entity_extraction`` is disabled or if the
        extractor finds nothing. Aliases and mention triples are only created
        when ``auto_entity_mentions`` is True (default).
        """
        from memlife.entity_extractor import extract_entities

        if not getattr(self.config, "auto_entity_extraction", False):
            return
        mentions_enabled = getattr(self.config, "auto_entity_mentions", True)
        confidence = getattr(self.config, "auto_entity_confidence", 0.6)
        allowlist = getattr(self.config, "entity_extraction_allowlist", None)
        blocklist = getattr(self.config, "entity_extraction_blocklist", None)

        for canonical, alias in extract_entities(
            text, allowlist=allowlist, blocklist=blocklist
        ):
            self._ensure_entity(canonical)
            # Store an alias if the original casing differs from the canonical.
            if alias and alias != canonical:
                self.add_entity_alias(canonical, alias)
            if mentions_enabled:
                self.store_mention_triple(source_kind, source_id, canonical, confidence)

    def expire_triples_for_fact(self, fact_id: str, valid_until: float | None = None) -> int:
        """Close currently-open triples linked to ``fact_id``.

        Used when a fact is superseded or revised. Returns the number of
        triples expired.
        """
        until = valid_until if valid_until is not None else time.time()
        cur = self.conn.execute(
            "UPDATE temporal_triples SET valid_until = ? "
            "WHERE fact_id = ? AND valid_until IS NULL",
            (until, fact_id),
        )
        self.conn.commit()
        return cur.rowcount

    def effective_triple_confidence(
        self,
        triple: dict,
        halflife_days: float | None = None,
        floor: float | None = None,
    ) -> float:
        """Return the confidence of a triple with age decay applied.

        Uses the same exponential decay formula as ``Fact.effective_confidence``.
        Open triples (``valid_until`` is None) decay from ``created_at``;
        closed triples decay from their ``valid_until`` so stale, expired
        assertions fade rather than contributing to veracity forever.

        Defaults are taken from ``self.config`` so callers can simply pass the
        raw triple dict.
        """
        if halflife_days is None:
            halflife_days = getattr(self.config, "fact_decay_halflife_days", 365.0)
        if floor is None:
            floor = getattr(self.config, "fact_decay_floor", 0.1)
        raw_conf = min(float(triple.get("confidence", 0.5)), MAX_FACT_CONFIDENCE)
        anchor = triple.get("valid_until") or triple.get("created_at") or time.time()
        age_days = max(0.0, (time.time() - anchor) / 86400.0)
        decay = math.pow(0.5, age_days / max(1e-6, halflife_days))
        return max(raw_conf * decay, floor)

    def current_truth(
        self, subject: str, predicate: str,
    ) -> tuple[str | None, float, str | None]:
        """Return the current object for ``subject predicate``.

        Returns ``(object, confidence, triple_id)`` or ``(None, 0.0, None)``
        if no open triple exists.
        """
        now = time.time()
        subj = self.resolve_entity(subject.strip()) or subject.strip()
        row = self.conn.execute(
            "SELECT id, object, confidence FROM temporal_triples "
            "WHERE subject = ? AND predicate = ? "
            "AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?) "
            "ORDER BY valid_from DESC, confidence DESC LIMIT 1",
            (subj, predicate.strip(), now, now),
        ).fetchone()
        if not row:
            return None, 0.0, None
        return row[1], row[2], row[0]

    def truth_as_of(
        self, subject: str, predicate: str, timestamp: float,
    ) -> tuple[str | None, float, str | None]:
        """Return the object that was true at ``timestamp``.

        Returns ``(object, confidence, triple_id)`` or ``(None, 0.0, None)``.
        """
        subj = self.resolve_entity(subject.strip()) or subject.strip()
        row = self.conn.execute(
            "SELECT id, object, confidence FROM temporal_triples "
            "WHERE subject = ? AND predicate = ? "
            "AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?) "
            "ORDER BY valid_from DESC, confidence DESC LIMIT 1",
            (subj, predicate.strip(), timestamp, timestamp),
        ).fetchone()
        if not row:
            return None, 0.0, None
        return row[1], row[2], row[0]

    def triples_for_fact(self, fact_id: str) -> list[dict]:
        """Return all triples associated with a fact, enriched with provenance."""
        rows = self.conn.execute(
            "SELECT id, subject, predicate, object, valid_from, valid_until, "
            "confidence, created_at FROM temporal_triples WHERE fact_id = ? "
            "ORDER BY valid_from DESC",
            (fact_id,),
        ).fetchall()
        triple_ids = [r[0] for r in rows]
        prov = self._triples_with_provenance(triple_ids)
        return [
            {
                "id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                "valid_from": r[4], "valid_until": r[5], "confidence": r[6],
                "created_at": r[7],
                "provenance": prov.get(r[0], []),
            }
            for r in rows
        ]

    def triples_about(
        self, entity: str, predicate: str | None = None, limit: int = 20,
    ) -> list[dict]:
        """Return triples where ``entity`` appears as subject or object."""
        canonical = self.resolve_entity(entity.strip()) or entity.strip()
        sql = (
            "SELECT id, subject, predicate, object, valid_from, valid_until, "
            "confidence, created_at FROM temporal_triples "
            "WHERE (subject = ? OR object = ?)"
        )
        params: list = [canonical, canonical]
        if predicate:
            sql += " AND predicate = ?"
            params.append(predicate.strip())
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        triple_ids = [r[0] for r in rows]
        prov = self._triples_with_provenance(triple_ids)
        return [
            {
                "id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                "valid_from": r[4], "valid_until": r[5], "confidence": r[6],
                "created_at": r[7],
                "provenance": prov.get(r[0], []),
            }
            for r in rows
        ]

    def triples_from(
        self, entity: str, predicate: str | None = None, limit: int = 20,
    ) -> list[dict]:
        """Return outgoing triples from ``entity``."""
        canonical = self.resolve_entity(entity.strip()) or entity.strip()
        sql = (
            "SELECT id, subject, predicate, object, valid_from, valid_until, "
            "confidence, created_at FROM temporal_triples WHERE subject = ?"
        )
        params: list = [canonical]
        if predicate:
            sql += " AND predicate = ?"
            params.append(predicate.strip())
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        triple_ids = [r[0] for r in rows]
        prov = self._triples_with_provenance(triple_ids)
        return [
            {
                "id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                "valid_from": r[4], "valid_until": r[5], "confidence": r[6],
                "created_at": r[7],
                "provenance": prov.get(r[0], []),
            }
            for r in rows
        ]

    def source_scores_linked_to_entity(
        self,
        entity: str,
        predicate: str | None = None,
        source_kinds: set[str] | None = None,
        limit: int = 100,
    ) -> dict[str, list[tuple[str, dict]]]:
        """Return source rows linked to ``entity`` with linking triple records.

        Returns ``{source_kind: [(source_id, triple_dict), ...]}``.  Only
        currently-valid triples (``valid_until IS NULL``) are considered, and a
        source may appear multiple times if it is linked by several triples.
        """
        canonical = self.resolve_entity_ci(entity.strip()) or entity.strip()
        source_kinds = source_kinds or {"fact", "episode", "journal"}
        triples = self.triples_about(canonical, predicate=predicate, limit=limit)
        result: dict[str, list[tuple[str, dict]]] = {}
        for t in triples:
            if t.get("valid_until") is not None:
                continue
            for prov in t.get("provenance", []):
                kind = prov.get("kind", "").lower()
                sid = prov.get("id", "").strip()
                if kind in source_kinds and sid and t.get("id"):
                    result.setdefault(kind, []).append((sid, t))
        return result

    def source_ids_linked_to_entity(
        self,
        entity: str,
        predicate: str | None = None,
        source_kinds: set[str] | None = None,
        limit: int = 100,
    ) -> dict[str, list[str]]:
        """Return source row ids that mention ``entity`` via triple provenance."""
        result: dict[str, list[str]] = {}
        for kind, entries in self.source_scores_linked_to_entity(
            entity, predicate=predicate, source_kinds=source_kinds, limit=limit
        ).items():
            result[kind] = [sid for sid, _triple in entries]
        return result

    def source_scores_linked_via_relationship(
        self,
        entity: str,
        predicate: str | None = None,
        source_kinds: set[str] | None = None,
        limit: int = 100,
    ) -> dict[str, list[tuple[str, dict]]]:
        """Return source rows linked to ``entity`` through relationship triples.

        For each currently-valid outgoing relationship triple from ``entity``,
        the object is treated as a related entity and its mention-triple
        provenance is collected along with the full mention triple record.
        """
        canonical = self.resolve_entity_ci(entity.strip()) or entity.strip()
        source_kinds = source_kinds or {"fact", "episode", "journal"}
        result: dict[str, list[tuple[str, dict]]] = {}
        # Outgoing relationship triples from the entity (subject == entity).
        sql = (
            "SELECT id, object FROM temporal_triples WHERE subject = ? "
            "AND predicate != 'mentions' AND valid_until IS NULL "
        )
        params: list = [canonical]
        if predicate:
            sql += "AND predicate = ? "
            params.append(predicate.strip())
        sql += "ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        for _triple_id, obj in rows:
            for kind, entries in self.source_scores_linked_to_entity(
                obj, predicate="mentions", source_kinds=source_kinds, limit=limit
            ).items():
                result.setdefault(kind, []).extend(entries)
        return result

    def source_ids_linked_via_relationship(
        self,
        entity: str,
        predicate: str | None = None,
        source_kinds: set[str] | None = None,
        limit: int = 100,
    ) -> dict[str, list[str]]:
        """Return source row ids linked to ``entity`` through relationship triples."""
        result: dict[str, list[str]] = {}
        for kind, entries in self.source_scores_linked_via_relationship(
            entity, predicate=predicate, source_kinds=source_kinds, limit=limit
        ).items():
            result[kind] = [sid for sid, _triple in entries]
        return result

    def triples_to(
        self, entity: str, predicate: str | None = None, limit: int = 20,
    ) -> list[dict]:
        """Return incoming triples to ``entity``."""
        canonical = self.resolve_entity(entity.strip()) or entity.strip()
        sql = (
            "SELECT id, subject, predicate, object, valid_from, valid_until, "
            "confidence, created_at FROM temporal_triples WHERE object = ?"
        )
        params: list = [canonical]
        if predicate:
            sql += " AND predicate = ?"
            params.append(predicate.strip())
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        triple_ids = [r[0] for r in rows]
        prov = self._triples_with_provenance(triple_ids)
        return [
            {
                "id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                "valid_from": r[4], "valid_until": r[5], "confidence": r[6],
                "created_at": r[7],
                "provenance": prov.get(r[0], []),
            }
            for r in rows
        ]

    def entity_neighbors(
        self,
        entity: str,
        predicate: str | None = None,
        depth: int = 1,
        limit: int = 100,
    ) -> list[dict]:
        """BFS traversal of the triple graph starting at ``entity``.

        Returns a list of ``{"entity": canonical_name, "depth": int,
        "via": [triple_dict, ...]}`` entries.  The start entity itself is
        not included.  ``depth`` controls how many edge-hops to follow.
        """
        start = self.resolve_entity(entity.strip()) or entity.strip()
        seen: set[str] = {start}
        frontier: set[str] = {start}
        results: list[dict] = []
        by_entity: dict[str, dict] = {}

        for d in range(1, depth + 1):
            if len(results) >= limit:
                break
            next_frontier: set[str] = set()
            for current in frontier:
                for t in self.triples_about(current, predicate=predicate, limit=limit):
                    other = t["object"] if t["subject"] == current else t["subject"]
                    if other in seen:
                        continue
                    seen.add(other)
                    next_frontier.add(other)
                    entry = by_entity.get(other)
                    if entry is None:
                        entry = {
                            "entity": other,
                            "depth": d,
                            "via": [],
                        }
                        by_entity[other] = entry
                        results.append(entry)
                    entry["via"].append(t)
                    if len(results) >= limit:
                        break
            frontier = next_frontier

        return results[:limit]

    def add_entity_alias(self, canonical_name: str, alias: str) -> bool:
        """Map ``alias`` -> ``canonical_name`` for entity resolution.

        Returns True if a new alias was recorded.
        """
        canonical = canonical_name.strip()
        alias = alias.strip()
        if not canonical or not alias or alias == canonical:
            return False
        self._ensure_entity(canonical)
        # Update JSON aliases list on the entity row.
        row = self.conn.execute(
            "SELECT aliases_json FROM entities WHERE canonical_name = ?",
            (canonical,),
        ).fetchone()
        aliases = _parse_json_list(row[0]) if row else []
        if alias not in aliases:
            aliases.append(alias)
            self.conn.execute(
                "UPDATE entities SET aliases_json = ? WHERE canonical_name = ?",
                (json.dumps(aliases), canonical),
            )
        self.conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (alias, canonical_name) VALUES (?, ?)",
            (alias, canonical),
        )
        self.conn.commit()
        return True

    def resolve_entity(self, name: str) -> str | None:
        """Return the canonical name for ``name`` if known, else None."""
        name = name.strip()
        row = self.conn.execute(
            "SELECT canonical_name FROM entities WHERE canonical_name = ?",
            (name,),
        ).fetchone()
        if row:
            return row[0]
        row = self.conn.execute(
            "SELECT canonical_name FROM entity_aliases WHERE alias = ?",
            (name,),
        ).fetchone()
        return row[0] if row else None

    def resolve_entity_ci(self, name: str) -> str | None:
        """Case-insensitive entity resolution.

        First tries exact match, then falls back to a case-insensitive lookup
        on canonical names and aliases. This is used by graph retrieval so a
        query mentioning "james" can follow triples stored for "James".
        """
        exact = self.resolve_entity(name)
        if exact:
            return exact
        lower = name.strip().lower()
        row = self.conn.execute(
            "SELECT canonical_name FROM entities WHERE lower(canonical_name) = ?",
            (lower,),
        ).fetchone()
        if row:
            return row[0]
        row = self.conn.execute(
            "SELECT canonical_name FROM entity_aliases WHERE lower(alias) = ?",
            (lower,),
        ).fetchone()
        return row[0] if row else None

    def _ensure_entity(self, name: str) -> None:
        """Ensure ``name`` exists as a canonical entity."""
        now = time.time()
        self.conn.execute(
            "INSERT OR IGNORE INTO entities (canonical_name, aliases_json, created_at) "
            "VALUES (?, '[]', ?)",
            (name, now),
        )

    def _add_triple_provenance(
        self, triple_id: str, provenance: list[dict],
    ) -> None:
        """Store provenance links for a triple."""
        now = time.time()
        for p in provenance:
            kind = p.get("kind", "").strip().lower()
            source_id = p.get("id", "").strip()
            if not kind or not source_id:
                continue
            self.conn.execute(
                "INSERT OR IGNORE INTO triple_provenance "
                "(triple_id, source_kind, source_id, created_at) VALUES (?, ?, ?, ?)",
                (triple_id, kind, source_id, now),
            )

    def _triples_with_provenance(
        self, triple_ids: list[str],
    ) -> dict[str, list[dict]]:
        """Return {triple_id: [provenance, ...]} for the given triple ids."""
        if not triple_ids:
            return {}
        placeholders = ",".join("?" * len(triple_ids))
        rows = self.conn.execute(
            f"SELECT triple_id, source_kind, source_id FROM triple_provenance "
            f"WHERE triple_id IN ({placeholders})",
            tuple(triple_ids),
        ).fetchall()
        result: dict[str, list[dict]] = {}
        for triple_id, kind, source_id in rows:
            result.setdefault(triple_id, []).append({"kind": kind, "id": source_id})
        return result


def _parse_json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return value
    except (json.JSONDecodeError, TypeError):
        pass
    return []
