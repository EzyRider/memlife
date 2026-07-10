"""Subject-predicate-object facts with valid time ranges.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import logging
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
        now = time.time()
        triple_id = f"triple_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO temporal_triples "
            "(id, subject, predicate, object, valid_from, valid_until, "
            "fact_id, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (triple_id, subject.strip(), predicate.strip(), object.strip(),
             valid_from if valid_from is not None else now, valid_until,
             fact_id, min(float(confidence), MAX_FACT_CONFIDENCE), now),
        )
        self.conn.commit()
        return triple_id

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

    def current_truth(
        self, subject: str, predicate: str,
    ) -> tuple[str | None, float, str | None]:
        """Return the current object for ``subject predicate``.

        Returns ``(object, confidence, triple_id)`` or ``(None, 0.0, None)``
        if no open triple exists.
        """
        now = time.time()
        row = self.conn.execute(
            "SELECT id, object, confidence FROM temporal_triples "
            "WHERE subject = ? AND predicate = ? "
            "AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?) "
            "ORDER BY valid_from DESC, confidence DESC LIMIT 1",
            (subject.strip(), predicate.strip(), now, now),
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
        row = self.conn.execute(
            "SELECT id, object, confidence FROM temporal_triples "
            "WHERE subject = ? AND predicate = ? "
            "AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?) "
            "ORDER BY valid_from DESC, confidence DESC LIMIT 1",
            (subject.strip(), predicate.strip(), timestamp, timestamp),
        ).fetchone()
        if not row:
            return None, 0.0, None
        return row[1], row[2], row[0]

    def triples_for_fact(self, fact_id: str) -> list[dict]:
        """Return all triples associated with a fact."""
        rows = self.conn.execute(
            "SELECT id, subject, predicate, object, valid_from, valid_until, "
            "confidence FROM temporal_triples WHERE fact_id = ? "
            "ORDER BY valid_from DESC",
            (fact_id,),
        ).fetchall()
        return [
            {
                "id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                "valid_from": r[4], "valid_until": r[5], "confidence": r[6],
            }
            for r in rows
        ]

