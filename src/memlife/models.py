"""Data classes for memory items: Episode, Fact, JournalEntry."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field


@dataclass
class Episode:
    """A stored episode — one agent run."""

    id: str
    task: str
    outcome: str  # 'success', 'failed', 'cancelled'
    summary: str = ""
    tool_calls_json: str = "[]"
    created_at: float = field(default_factory=time.time)
    embedding_json: str = ""

    @classmethod
    def from_row(cls, row: tuple) -> "Episode":
        # Tolerate rows from pre-migration schemas that lack embedding_json.
        if len(row) >= 7:
            return cls(
                id=row[0], task=row[1], outcome=row[2],
                summary=row[3] or "", tool_calls_json=row[4] or "[]",
                created_at=row[5], embedding_json=row[6] or "",
            )
        return cls(
            id=row[0], task=row[1], outcome=row[2],
            summary=row[3] or "", tool_calls_json=row[4] or "[]",
            created_at=row[5],
        )

    @property
    def tool_calls(self) -> list[dict]:
        try:
            return json.loads(self.tool_calls_json)
        except json.JSONDecodeError:
            return []

    @property
    def embedding(self) -> list[float] | None:
        if not self.embedding_json:
            return None
        try:
            return json.loads(self.embedding_json)
        except json.JSONDecodeError:
            return None

    def index_text(self) -> str:
        """Text used for embedding/recall: task + summary."""
        return f"{self.task}\n{self.summary}".strip()


@dataclass
class Fact:
    """A stored fact / preference / entity relationship."""

    id: str
    content: str
    source: str = "agent"  # 'user', 'agent', 'journal'
    confidence: float = 0.5
    embedding_json: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    superseded_by: str = ""

    @property
    def embedding(self) -> list[float] | None:
        if not self.embedding_json:
            return None
        try:
            return json.loads(self.embedding_json)
        except json.JSONDecodeError:
            return None


@dataclass
class JournalEntry:
    """A journal entry — observation, hypothesis, revision, or contradiction."""

    id: str
    type: str
    content: str
    confidence: float = 0.5
    source_episodes_json: str = "[]"
    private: bool = True
    created_at: float = field(default_factory=time.time)
    superseded_by: str = ""
    embedding_json: str = ""
    last_detected: int = 0  # reflection cycle when last re-detected (contradictions)

    @property
    def source_episodes(self) -> list[str]:
        try:
            return json.loads(self.source_episodes_json)
        except json.JSONDecodeError:
            return []

    @property
    def superseded(self) -> bool:
        # A real supersession links to another journal entry id; the
        # ``__retired__:`` sentinel is retirement, not supersession.
        return bool(self.superseded_by) and not self.superseded_by.startswith("__retired__")

    @property
    def retired(self) -> bool:
        return self.superseded_by.startswith("__retired__")

    @property
    def embedding(self) -> list[float] | None:
        if not self.embedding_json:
            return None
        try:
            return json.loads(self.embedding_json)
        except json.JSONDecodeError:
            return None

    def effective_confidence(self, halflife_days: float = 30.0, floor: float = 0.15) -> float:
        """Confidence decayed by age, floored at ``floor`` (never falls to 0).

        Used for retrieval ranking and retirement: a stale entry's confidence
        bottoms out at ``floor`` rather than decaying to nothing, so very old
        but once-confident beliefs still nudge retrieval slightly.
        """
        age_days = max(0.0, (time.time() - self.created_at) / 86400.0)
        decay = math.pow(0.5, age_days / max(1e-6, halflife_days))
        return max(self.confidence * decay, floor)


