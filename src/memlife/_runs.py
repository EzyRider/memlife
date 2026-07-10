"""Agent run, checkpoint, and session storage.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import json
import logging
import time
import uuid


logger = logging.getLogger(__name__)

# Cap on trace events stored per run (oldest are dropped beyond this).
TRACE_EVENT_LIMIT = 200


class RunMixin:
    """Agent run, checkpoint, and session storage."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object

    def start_run(self, task: str, model: str = "") -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO agent_runs (id, task, status, created_at, model_used) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, task, time.time(), model),
        )
        self.conn.commit()
        return run_id

    def save_checkpoint(
        self, run_id: str, step_index: int, step_description: str,
        state: dict, tool_calls: list[dict] | None = None,
        observation: str = "", outcome: str = "success",
        tokens_used: int = 0,
    ) -> str:
        cp_id = f"cp_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(id, run_id, step_index, step_description, state_json, "
            "tool_calls_json, observation, outcome, tokens_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cp_id, run_id, step_index, step_description,
             json.dumps(state), json.dumps(tool_calls or []),
             observation, outcome, tokens_used, time.time()),
        )
        self.conn.commit()
        return cp_id

    def get_last_checkpoint(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT state_json FROM checkpoints WHERE run_id = ? "
            "ORDER BY step_index DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                logger.warning(
                    "get_last_checkpoint: corrupt state_json for run %s; returning None",
                    run_id,
                )
                return None
        return None

    def complete_run(
        self, run_id: str, total_tokens: int = 0, error: str = ""
    ) -> None:
        status = "failed" if error else "completed"
        self.conn.execute(
            "UPDATE agent_runs SET status = ?, completed_at = ?, "
            "total_tokens = ?, error_message = ? WHERE id = ?",
            (status, time.time(), total_tokens, error or None, run_id),
        )
        self.conn.commit()

    def trace_event(self, run_id: str, event: str, detail: dict | None = None) -> None:
        """Append a structured trace event to a run.

        Capped at ``TRACE_EVENT_LIMIT`` events (oldest dropped) so the trace
        blob can't grow without bound across a long run. Atomic via
        transaction() so concurrent trace_event calls don't lose events.
        """
        with self.transaction():
            row = self.conn.execute(
                "SELECT trace_json FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not row:
                return
            try:
                trace = json.loads(row[0])
            except json.JSONDecodeError:
                trace = []
            trace.append({"ts": time.time(), "event": event, "detail": detail or {}})
            # Drop oldest beyond the cap.
            if len(trace) > TRACE_EVENT_LIMIT:
                trace = trace[-TRACE_EVENT_LIMIT:]
            self.conn.execute(
                "UPDATE agent_runs SET trace_json = ? WHERE id = ?",
                (json.dumps(trace), run_id),
            )
            self.conn.commit()

    def get_incomplete_run(self) -> dict | None:
        row = self.conn.execute(
            "SELECT id, task, model_used FROM agent_runs "
            "WHERE status = 'running' ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        if row:
            return {"id": row[0], "task": row[1], "model_used": row[2]}
        return None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at, updated_at, model_used "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "created_at": r[2],
             "updated_at": r[3], "model_used": r[4]}
            for r in rows
        ]

    def create_session(self, name: str, model: str = "") -> str:
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self.conn.execute(
            "INSERT INTO sessions (id, name, created_at, updated_at, model_used) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, name, now, now, model),
        )
        self.conn.commit()
        return sid

    def load_session(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, name, model_used, conversation_json, rolling_summary "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        try:
            conversation = json.loads(row[3])
        except json.JSONDecodeError:
            conversation = []
        return {
            "id": row[0], "name": row[1], "model_used": row[2],
            "conversation": conversation, "rolling_summary": row[4] or "",
        }

    def save_session(self, session_id: str, conversation: list[dict],
                     model: str = "", rolling_summary: str = "") -> None:
        self.conn.execute(
            "UPDATE sessions SET conversation_json = ?, updated_at = ?, "
            "model_used = CASE WHEN ? != '' THEN ? ELSE model_used END, "
            "rolling_summary = CASE WHEN ? != '' THEN ? ELSE rolling_summary END "
            "WHERE id = ?",
            (json.dumps(conversation), time.time(), model, model,
             rolling_summary, rolling_summary, session_id),
        )
        self.conn.commit()

    def delete_session(self, session_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,),
        )
        self.conn.commit()
        return cur.rowcount > 0

