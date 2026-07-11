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
    db_path: str = ""

    # Namespace layout.  When db_path is not set explicitly, the store resolves
    # to data_dir / namespace / "memlife.db".  This gives each user a fully
    # isolated SQLite file while keeping the public API unchanged.
    data_dir: str = "./memlife_data"
    namespace: str = "default"

    # SQLite pragmas — WAL + busy_timeout protect against corruption under
    # concurrent writers (MF-002). Both on by default, overridable via env.
    sqlite_journal_mode: str = "WAL"
    sqlite_busy_timeout_ms: int = 5000

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

    # Hybrid retrieval scoring weights.  Vector, text and source signals are
    # normalised per query before blending.  Weights default to the Mnemosyne-
    # inspired 0.5 / 0.3 / 0.2 split (MV2-002).
    recall_vector_weight: float = 0.5
    recall_text_weight: float = 0.3
    recall_source_weight: float = 0.2

    # Veracity weighting (MV2-005).  A small bonus for confident facts and
    # journal entries. 0.0 disables the signal.
    recall_veracity_weight: float = 0.05

    # Layer-aware decay halflifes (MV2-007).  Facts decay slowly;
    # episodes decay fast; journal sits in the middle.
    fact_decay_halflife_days: float = 365.0
    episode_decay_halflife_days: float = 7.0
    journal_decay_halflife_days: float = 30.0
    journal_decay_floor: float = 0.15
    fact_decay_floor: float = 0.1  # MV2-007: floor for fact confidence decay

    # Tiered episodic degradation (MV2-001).  Successful tool episodes linger;
    # plain observations fade fast.
    episode_tool_success_halflife_days: float = 21.0
    episode_failure_halflife_days: float = 3.0
    episode_observation_halflife_days: float = 1.0

    # Temporal gap markers (MV2-008).  When consecutive episode timestamps
    # exceed this threshold, a synthetic "time passed" episode is inserted
    # to preserve narrative continuity.  0 disables the feature.
    gap_marker_threshold_hours: float = 24.0

    # Fact memory — cosine bands for merge and conflict detection
    fact_merge_threshold: float = 0.90
    fact_conflict_threshold: float = 0.75

    # Reflection quality
    reflect_critic: bool = True
    critic_model: str = ""  # empty = use primary model; set to a cheap model for the critic pass
    significance_model: str = ""

    # Reflection timeouts
    reflection_timeout: float = 120.0
    reflection_total_timeout: float = 300.0

    # Contradiction retirement — retire active contradictions not
    # re-detected in N reflection passes (MF-004). 0 disables retirement.
    contradiction_retirement_cycles: int = 14

    # GC retention (days)
    gc_superseded_facts_days: int = 90
    gc_superseded_journal_days: int = 90
    gc_completed_runs_days: int = 60
    gc_metrics_days: int = 30
    gc_reflected_queue_days: int = 30
    gc_episodes_days: int = 180  # MF-009: episodes were never pruned
    gc_closed_triples_days: int = 90  # MV2-003: closed temporal triples

    # Optional infrastructure backends (MV2-I001..I004).  All default off.
    use_sqlite_vec: bool = False
    use_binary_vectors: bool = False
    use_polyphonic_recall: bool = False
    memorias_extraction: bool = False  # structured MEMORIA extraction (I003)

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
            db_path=os.getenv("MEMLIFE_DB_PATH", ""),
            data_dir=os.getenv("MEMLIFE_DATA_DIR", "./memlife_data"),
            namespace=os.getenv("MEMLIFE_NAMESPACE", "default"),
            sqlite_journal_mode=os.getenv("MEMLIFE_SQLITE_JOURNAL_MODE", "WAL"),
            sqlite_busy_timeout_ms=int(os.getenv("MEMLIFE_SQLITE_BUSY_TIMEOUT_MS", "5000")),
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
            recall_vector_weight=float(os.getenv("MEMLIFE_RECALL_VECTOR_WEIGHT", "0.5")),
            recall_text_weight=float(os.getenv("MEMLIFE_RECALL_TEXT_WEIGHT", "0.3")),
            recall_source_weight=float(os.getenv("MEMLIFE_RECALL_SOURCE_WEIGHT", "0.2")),
            recall_veracity_weight=float(os.getenv("MEMLIFE_RECALL_VERACITY_WEIGHT", "0.05")),
            fact_decay_halflife_days=float(os.getenv("MEMLIFE_FACT_DECAY_HALFLIFE_DAYS", "365")),
            episode_decay_halflife_days=float(os.getenv("MEMLIFE_EPISODE_DECAY_HALFLIFE_DAYS", "7")),
            episode_tool_success_halflife_days=float(os.getenv("MEMLIFE_EPISODE_TOOL_SUCCESS_HALFLIFE_DAYS", "21")),
            episode_failure_halflife_days=float(os.getenv("MEMLIFE_EPISODE_FAILURE_HALFLIFE_DAYS", "3")),
            episode_observation_halflife_days=float(os.getenv("MEMLIFE_EPISODE_OBSERVATION_HALFLIFE_DAYS", "1")),
            journal_decay_halflife_days=float(os.getenv("MEMLIFE_JOURNAL_DECAY_HALFLIFE_DAYS", "30")),
            journal_decay_floor=float(os.getenv("MEMLIFE_JOURNAL_DECAY_FLOOR", "0.15")),
            fact_decay_floor=float(os.getenv("MEMLIFE_FACT_DECAY_FLOOR", "0.1")),
            gap_marker_threshold_hours=float(os.getenv("MEMLIFE_GAP_MARKER_THRESHOLD_HOURS", "24")),
            fact_merge_threshold=float(os.getenv("MEMLIFE_FACT_MERGE_THRESHOLD", "0.90")),
            fact_conflict_threshold=float(os.getenv("MEMLIFE_FACT_CONFLICT_THRESHOLD", "0.75")),
            reflect_critic=_bool("MEMLIFE_REFLECT_CRITIC", True),
            critic_model=os.getenv("MEMLIFE_CRITIC_MODEL", ""),
            significance_model=os.getenv("MEMLIFE_SIGNIFICANCE_MODEL", ""),
            reflection_timeout=float(os.getenv("MEMLIFE_REFLECTION_TIMEOUT", "120")),
            reflection_total_timeout=float(os.getenv("MEMLIFE_REFLECTION_TOTAL_TIMEOUT", "300")),
            contradiction_retirement_cycles=int(
                os.getenv("MEMLIFE_CONTRADICTION_RETIREMENT_CYCLES", "14")
            ),
            gc_superseded_facts_days=int(os.getenv("MEMLIFE_GC_SUPERSEDED_FACTS_DAYS", "90")),
            gc_superseded_journal_days=int(os.getenv("MEMLIFE_GC_SUPERSEDED_JOURNAL_DAYS", "90")),
            gc_completed_runs_days=int(os.getenv("MEMLIFE_GC_COMPLETED_RUNS_DAYS", "60")),
            gc_metrics_days=int(os.getenv("MEMLIFE_GC_METRICS_DAYS", "30")),
            gc_reflected_queue_days=int(os.getenv("MEMLIFE_GC_REFLECTED_QUEUE_DAYS", "30")),
            gc_episodes_days=int(os.getenv("MEMLIFE_GC_EPISODES_DAYS", "180")),
            gc_closed_triples_days=int(os.getenv("MEMLIFE_GC_CLOSED_TRIPLES_DAYS", "90")),
            use_sqlite_vec=_bool("MEMLIFE_USE_SQLITE_VEC", False),
            use_binary_vectors=_bool("MEMLIFE_USE_BINARY_VECTORS", False),
            use_polyphonic_recall=_bool("MEMLIFE_USE_POLYPHONIC_RECALL", False),
            memorias_extraction=_bool("MEMLIFE_MEMORIAS_EXTRACTION", False),
        )
