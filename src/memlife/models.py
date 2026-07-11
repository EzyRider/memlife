"""Data classes for memory items: Episode, Fact, JournalEntry."""

from __future__ import annotations

import json
import math
import time
import base64
from dataclasses import dataclass, field


def _decode_embedding(raw: str) -> list[float] | None:
    """Decode an embedding from JSON or binary ``binary:dim:base64`` form."""
    if not raw:
        return None
    if raw.startswith("binary:"):
        try:
            _, dim_str, b64 = raw.split(":", 2)
            dim = int(dim_str)
            packed = base64.b64decode(b64)
            vec = []
            for i in range(dim):
                byte_idx = i // 8
                bit_idx = 7 - (i % 8)
                if byte_idx >= len(packed):
                    return None
                bit = (packed[byte_idx] >> bit_idx) & 1
                vec.append(1.0 if bit else -1.0)
            return vec
        except Exception:
            return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


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
    is_gap_marker: bool = False

    @classmethod
    def from_row(cls, row: tuple) -> "Episode":
        # Tolerate rows from pre-migration schemas that lack embedding_json
        # or is_gap_marker.
        if len(row) >= 8:
            return cls(
                id=row[0], task=row[1], outcome=row[2],
                summary=row[3] or "", tool_calls_json=row[4] or "[]",
                created_at=row[5], embedding_json=row[6] or "",
                is_gap_marker=bool(row[7]) if row[7] is not None else False,
            )
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
    def has_tool_calls(self) -> bool:
        """True if the episode recorded at least one tool invocation."""
        try:
            return len(json.loads(self.tool_calls_json)) > 0
        except Exception:
            return False

    @property
    def is_success(self) -> bool:
        """A successful episode outcome, case-insensitive."""
        return self.outcome.lower() in {"success", "succeeded", "ok"}

    @property
    def is_failure(self) -> bool:
        """An explicitly failed outcome."""
        return self.outcome.lower() in {"failed", "failure", "error"}

    @property
    def embedding(self) -> list[float] | None:
        if not self.embedding_json:
            return None
        return _decode_embedding(self.embedding_json)

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
    annotations_json: str = "[]"

    @property
    def embedding(self) -> list[float] | None:
        if not self.embedding_json:
            return None
        return _decode_embedding(self.embedding_json)

    @property
    def annotations(self) -> list[str]:
        try:
            return json.loads(self.annotations_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def retired(self) -> bool:
        """True if this fact was explicitly expired rather than superseded."""
        return self.superseded_by.startswith("__retired__")

    def effective_confidence(
        self, halflife_days: float = 365.0, floor: float = 0.1
    ) -> float:
        """Confidence decayed by age, floored at ``floor``."""
        age_days = max(0.0, (time.time() - self.updated_at) / 86400.0)
        decay = math.pow(0.5, age_days / max(1e-6, halflife_days))
        return max(self.confidence * decay, floor)


@dataclass
class Metrics:
    """Snapshot of memory system health, counts, and diagnostics."""

    # Database / config
    db_path: str = ""
    db_size_bytes: int = 0
    db_size_mb: float = 0.0
    journal_mode: str = ""
    busy_timeout_ms: int = 0
    vector_backend: str = ""
    namespace: str = ""
    embedding_model: str = ""

    # Counts
    episodes: int = 0
    facts: int = 0
    active_facts: int = 0
    journal_entries: int = 0
    active_journal: int = 0
    contradictions: int = 0
    unresolved_contradictions: int = 0
    user_corrections: int = 0
    sessions: int = 0
    agent_runs: int = 0
    triples: int = 0
    entities: int = 0

    # Embeddings
    embedded_episodes: int = 0
    embedded_facts: int = 0
    embedded_journal: int = 0
    pending_embeddings: int = 0
    embedding_health: dict = field(default_factory=dict)

    # Reflection aggregates
    total_reflections: int = 0
    last_reflection_at: float | None = None
    avg_keep_rate: float | None = None
    avg_confidence: float | None = None
    total_observations_kept: int = 0
    total_hypotheses_kept: int = 0
    total_revisions_kept: int = 0
    total_contradictions_found: int = 0
    total_retired: int = 0
    total_merged: int = 0

    # Recall diagnostics
    recall: dict = field(default_factory=dict)

    # Schema migration health
    migration: dict = field(default_factory=dict)


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
    annotations_json: str = "[]"
    links_json: str = "[]"  # [{"target": id, "relation": "supports"|"undermines"|"related", "strength": float}]

    @property
    def source_episodes(self) -> list[str]:
        try:
            return json.loads(self.source_episodes_json)
        except json.JSONDecodeError:
            return []

    @property
    def annotations(self) -> list[str]:
        try:
            return json.loads(self.annotations_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def links(self) -> list[dict]:
        try:
            links = json.loads(self.links_json)
            if isinstance(links, list):
                return links
        except (json.JSONDecodeError, TypeError):
            pass
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
        return _decode_embedding(self.embedding_json)

    def effective_confidence(self, halflife_days: float = 30.0, floor: float = 0.15) -> float:
        """Confidence decayed by age, floored at ``floor`` (never falls to 0).

        Used for retrieval ranking and retirement: a stale entry's confidence
        bottoms out at ``floor`` rather than decaying to nothing, so very old
        but once-confident beliefs still nudge retrieval slightly.
        """
        age_days = max(0.0, (time.time() - self.created_at) / 86400.0)
        decay = math.pow(0.5, age_days / max(1e-6, halflife_days))
        return max(self.confidence * decay, floor)


