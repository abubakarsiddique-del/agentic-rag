"""Local store for post-response evaluation metrics; never stores answer text."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DB = Path(__file__).parents[1] / "data" / "production_eval_results.db"


class SamplingResultStore:
    def __init__(self, path: str | Path = DEFAULT_RESULTS_DB) -> None:
        self.path = Path(path)

    def save(self, result: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    judge_type TEXT NOT NULL DEFAULT 'live',
                    metrics_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO evaluation_results (message_id, conversation_id, trace_id, evaluated_at, judge_type, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,
                    trace_id=excluded.trace_id,
                    evaluated_at=excluded.evaluated_at,
                    judge_type=excluded.judge_type,
                    metrics_json=excluded.metrics_json
                """,
                (
                    result["message_id"],
                    result["conversation_id"],
                    result["trace_id"],
                    result["evaluated_at"],
                    str(result.get("judge_type") or "live"),
                    json.dumps(result["metrics"], separators=(",", ":"), allow_nan=False),
                ),
            )

    def get(self, message_id: str) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT message_id, conversation_id, trace_id, evaluated_at, judge_type, metrics_json "
                "FROM evaluation_results WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "message_id": row["message_id"],
            "conversation_id": row["conversation_id"],
            "trace_id": row["trace_id"],
            "evaluated_at": row["evaluated_at"],
            "judge_type": row["judge_type"],
            "metrics": json.loads(row["metrics_json"]),
        }