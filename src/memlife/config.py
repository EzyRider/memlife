"""Memory configuration — memory fields only, no agent config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for the memory system.

    All fields have sensible defaults. The store works with zero
    configuration — just ``MemoryConfig()``.

    Env vars (optional, override defaults):
      MEMLIFE_DB_PATH, MEMLIFE_EMBEDDING_MODEL, etc.
    """

    # Database
    db_path: str = "./memlife.db"

    # Embedding model name (stored with each vector for versioning)
    embedding_model: str = ""

    # Retrieval — how much context to inject before responding
    recall_episodes: int = 5
    recall_facts: int = 5
    recall_journal: int = 3
    working_window: int = 20
    max_context_chars: int = 4000
    recency_halflife_days: float = 14.0

    # Strict recall cut-off + density dedup
    recall_min_score: float = 0.0
    recall_score_cutoff_ratio: float = 0.0
    recall_dedup_threshold: float = 0.75
    recall_dedup_method: str = "jaccard"

    # Fact memory — cosine bands for merge and conflict detection
    fact_merge_threshold: float = 0.90
    fact_conflict_threshold: float = 0.75

    # Reflection quality
    reflect_critic: bool = True
    critic_model: str = ""  # empty = use primary model; set to a cheap model for the critic pass
    significance_model: str = ""
    journal_decay_halflife_days: float = 30.0
    journal_decay_floor: float = 0.15

    # Reflection timeouts
    reflection_timeout: float = 120.0
    reflection_total_timeout: float = 300.0

    # GC retention (days)
    gc_superseded_facts_days: int = 90
    gc_superseded_journal_days: int = 90
    gc_completed_runs_days: int = 60
    gc_metrics_days: int = 30
    gc_reflected_queue_days: int = 30

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Load from environment variables with MEMLIFE_ prefix."""
        import os

        def _bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            db_path=os.getenv("MEMLIFE_DB_PATH", "./memlife.db"),
            embedding_model=os.getenv("MEMLIFE_EMBEDDING_MODEL", ""),
            recall_episodes=int(os.getenv("MEMLIFE_RECALL_EPISODES", "5")),
            recall_facts=int(os.getenv("MEMLIFE_RECALL_FACTS", "5")),
            recall_journal=int(os.getenv("MEMLIFE_RECALL_JOURNAL", "3")),
            working_window=int(os.getenv("MEMLIFE_WORKING_WINDOW", "20")),
            max_context_chars=int(os.getenv("MEMLIFE_MAX_CONTEXT_CHARS", "4000")),
            recency_halflife_days=float(os.getenv("MEMLIFE_RECENCY_HALFLIFE_DAYS", "14")),
            recall_min_score=float(os.getenv("MEMLIFE_RECALL_MIN_SCORE", "0")),
            recall_score_cutoff_ratio=float(os.getenv("MEMLIFE_RECALL_SCORE_CUTOFF_RATIO", "0")),
            recall_dedup_threshold=float(os.getenv("MEMLIFE_RECALL_DEDUP_THRESHOLD", "0.75")),
            recall_dedup_method=os.getenv("MEMLIFE_RECALL_DEDUP_METHOD", "jaccard"),
            fact_merge_threshold=float(os.getenv("MEMLIFE_FACT_MERGE_THRESHOLD", "0.90")),
            fact_conflict_threshold=float(os.getenv("MEMLIFE_FACT_CONFLICT_THRESHOLD", "0.75")),
            reflect_critic=_bool("MEMLIFE_REFLECT_CRITIC", True),
            critic_model=os.getenv("MEMLIFE_CRITIC_MODEL", ""),
            journal_decay_halflife_days=float(os.getenv("MEMLIFE_JOURNAL_HALFLIFE_DAYS", "30")),
            journal_decay_floor=float(os.getenv("MEMLIFE_JOURNAL_DECAY_FLOOR", "0.15")),
            reflection_timeout=float(os.getenv("MEMLIFE_REFLECTION_TIMEOUT", "120")),
            reflection_total_timeout=float(os.getenv("MEMLIFE_REFLECTION_TOTAL_TIMEOUT", "300")),
            gc_superseded_facts_days=int(os.getenv("MEMLIFE_GC_SUPERSEDED_FACTS_DAYS", "90")),
            gc_superseded_journal_days=int(os.getenv("MEMLIFE_GC_SUPERSEDED_JOURNAL_DAYS", "90")),
            gc_completed_runs_days=int(os.getenv("MEMLIFE_GC_COMPLETED_RUNS_DAYS", "60")),
            gc_metrics_days=int(os.getenv("MEMLIFE_GC_METRICS_DAYS", "30")),
            gc_reflected_queue_days=int(os.getenv("MEMLIFE_GC_REFLECTED_QUEUE_DAYS", "30")),
        )