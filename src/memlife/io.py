"""Import/export for the memory store.

JSONL dump/restore for backup, migration, or moving between stores.
"""

from __future__ import annotations

import json

from memlife.store import MemoryStore

# MF-017: every schema table must be round-trippable through JSONL export/import.
# These column lists are the single source of truth for both directions.
_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "episodes": (
        "id", "task", "outcome", "summary", "tool_calls_json",
        "created_at", "embedding_json", "embedding_model", "is_gap_marker",
    ),
    "agent_runs": (
        "id", "task", "status", "created_at", "completed_at",
        "model_used", "total_tokens", "error_message", "trace_json",
    ),
    "checkpoints": (
        "id", "run_id", "step_index", "step_description", "state_json",
        "tool_calls_json", "observation", "outcome", "tokens_used", "created_at",
    ),
    "facts": (
        "id", "content", "source", "confidence", "embedding_json",
        "embedding_model", "created_at", "updated_at", "superseded_by",
        "annotations_json",
    ),
    "journal": (
        "id", "type", "content", "confidence", "source_episodes_json",
        "private", "created_at", "superseded_by", "embedding_json",
        "embedding_model", "last_detected", "annotations_json", "links_json",
    ),
    "reflection_queue": (
        "id", "episode_id", "queued_at", "reflected",
    ),
    "sessions": (
        "id", "name", "created_at", "updated_at", "model_used",
        "conversation_json", "rolling_summary",
    ),
    "reflection_metrics": (
        "id", "created_at", "episodes_considered", "observations_proposed",
        "observations_kept", "hypotheses_proposed", "hypotheses_kept",
        "revisions_proposed", "revisions_kept", "contradictions_found",
        "avg_confidence", "keep_rate", "consolidated_retired",
        "consolidated_merged", "total_journal_entries", "total_facts",
        "total_episodes",
    ),
    "reflection_passes": (
        "id", "created_at", "episode_ids_json", "proposed_json", "kept_json",
        "dropped_json", "model_used", "critic_model_used", "total_timeout",
        "elapsed_seconds",
    ),
    "episode_tools": (
        "episode_id", "tool_name", "created_at",
    ),
    "temporal_triples": (
        "id", "subject", "predicate", "object", "valid_from", "valid_until",
        "fact_id", "confidence", "created_at",
    ),
    "entities": (
        "canonical_name", "aliases_json", "created_at",
    ),
    "entity_aliases": (
        "alias", "canonical_name",
    ),
    "triple_provenance": (
        "triple_id", "source_kind", "source_id", "created_at",
    ),
    "embedding_cache": (
        "cache_key", "model_name", "text_hash", "vector_json", "created_at",
        "last_used_at",
    ),
}


# Derived whitelist used for import validation.
_ALLOWED_COLUMNS: dict[str, set[str]] = {
    table: set(cols) for table, cols in _TABLE_COLUMNS.items()
}


def _dump_table(f, store: MemoryStore, table: str, columns: tuple[str, ...]) -> int:
    """Write every row of ``table`` as a JSONL line and return the row count."""
    col_list = ", ".join(columns)
    count = 0
    for row in store.conn.execute(f"SELECT {col_list} FROM {table}").fetchall():
        f.write(json.dumps({
            "table": table,
            "data": dict(row),
        }) + "\n")
        count += 1
    return count


def export_jsonl(store: MemoryStore, path: str) -> dict:
    """Export all memory data to a JSONL file.

    Each line is a JSON object with a 'table' field and the row data.
    Useful for backup, migration, or moving between stores.
    """
    counts: dict[str, int] = {table: 0 for table in _TABLE_COLUMNS}

    with open(path, "w") as f:
        for table, columns in _TABLE_COLUMNS.items():
            counts[table] = _dump_table(f, store, table, columns)

    counts["total"] = sum(counts.values())
    counts["path"] = path
    return counts


def import_jsonl(store: MemoryStore, path: str) -> dict:
    """Import memory data from a JSONL file.

    Each line is a JSON object with 'table' and 'data' fields.
    Rows are inserted with INSERT OR IGNORE to avoid duplicates.
    """
    # MF-012: whitelist allowed columns per table to prevent SQL injection
    # via crafted column names in JSONL keys.
    counts: dict[str, int] = {table: 0 for table in _TABLE_COLUMNS}

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                table = obj["table"]
                data = obj["data"]

                if table not in _ALLOWED_COLUMNS:
                    continue

                allowed = _ALLOWED_COLUMNS[table]
                # Filter to whitelisted columns only; reject unknown keys.
                safe_data = {k: v for k, v in data.items() if k in allowed}
                if not safe_data:
                    continue

                cols = ", ".join(safe_data.keys())
                placeholders = ", ".join("?" * len(safe_data))
                store.conn.execute(
                    f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})",
                    list(safe_data.values()),
                )
                counts[table] += 1
    except Exception:
        # MF-016: rollback on failure so a partial import doesn't leave
        # the database in an inconsistent state.
        store.conn.rollback()
        raise

    store.conn.commit()
    counts["total"] = sum(counts.values())
    counts["path"] = path
    return counts