"""Import/export for the memory store.

JSONL dump/restore for backup, migration, or moving between stores.
"""

from __future__ import annotations

import json

from memlife.store import MemoryStore



def export_jsonl(store: MemoryStore, path: str) -> dict:
    """Export all memory data to a JSONL file.

    Each line is a JSON object with a 'table' field and the row data.
    Useful for backup, migration, or moving between stores.
    """
    counts = {"episodes": 0, "facts": 0, "journal": 0, "sessions": 0}

    with open(path, "w") as f:
        # Episodes
        for row in store.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, "
            "created_at, embedding_json, embedding_model FROM episodes"
        ).fetchall():
            f.write(json.dumps({
                "table": "episodes",
                "data": dict(row),
            }) + "\n")
            counts["episodes"] += 1

        # Facts
        for row in store.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "embedding_model, created_at, updated_at, superseded_by FROM facts"
        ).fetchall():
            f.write(json.dumps({
                "table": "facts",
                "data": dict(row),
            }) + "\n")
            counts["facts"] += 1

        # Journal
        for row in store.conn.execute(
            "SELECT id, type, content, confidence, source_episodes_json, "
            "private, created_at, superseded_by, embedding_json, "
            "embedding_model FROM journal"
        ).fetchall():
            f.write(json.dumps({
                "table": "journal",
                "data": dict(row),
            }) + "\n")
            counts["journal"] += 1

        # Sessions
        for row in store.conn.execute(
            "SELECT id, name, created_at, updated_at, model_used, "
            "conversation_json, rolling_summary FROM sessions"
        ).fetchall():
            f.write(json.dumps({
                "table": "sessions",
                "data": dict(row),
            }) + "\n")
            counts["sessions"] += 1

    counts["total"] = sum(counts.values())
    counts["path"] = path
    return counts


def import_jsonl(store: MemoryStore, path: str) -> dict:
    """Import memory data from a JSONL file.

    Each line is a JSON object with 'table' and 'data' fields.
    Rows are inserted with INSERT OR IGNORE to avoid duplicates.
    """
    counts = {"episodes": 0, "facts": 0, "journal": 0, "sessions": 0}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            table = obj["table"]
            data = obj["data"]

            if table == "episodes":
                cols = ", ".join(data.keys())
                placeholders = ", ".join("?" * len(data))
                store.conn.execute(
                    f"INSERT OR IGNORE INTO episodes ({cols}) VALUES ({placeholders})",
                    list(data.values()),
                )
                counts["episodes"] += 1
            elif table == "facts":
                cols = ", ".join(data.keys())
                placeholders = ", ".join("?" * len(data))
                store.conn.execute(
                    f"INSERT OR IGNORE INTO facts ({cols}) VALUES ({placeholders})",
                    list(data.values()),
                )
                counts["facts"] += 1
            elif table == "journal":
                cols = ", ".join(data.keys())
                placeholders = ", ".join("?" * len(data))
                store.conn.execute(
                    f"INSERT OR IGNORE INTO journal ({cols}) VALUES ({placeholders})",
                    list(data.values()),
                )
                counts["journal"] += 1
            elif table == "sessions":
                cols = ", ".join(data.keys())
                placeholders = ", ".join("?" * len(data))
                store.conn.execute(
                    f"INSERT OR IGNORE INTO sessions ({cols}) VALUES ({placeholders})",
                    list(data.values()),
                )
                counts["sessions"] += 1

    store.conn.commit()
    counts["total"] = sum(counts.values())
    counts["path"] = path
    return counts