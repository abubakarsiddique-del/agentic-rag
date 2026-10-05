"""Summarize downvoted answers by likely failing stage using persisted traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / ".rag_history.db"
FAILURE_STAGES = ("route", "retrieve", "rerank", "grade", "rewrite", "unclassified")


def _load_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def load_feedback_records(database: Path = DEFAULT_DATABASE) -> list[dict[str, Any]]:
    if not database.is_file():
        raise FileNotFoundError("feedback database not found")
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, conversation_id, role, content,
                COALESCE(trace_json, trace) AS trace_value, rating, status, rowid
            FROM messages ORDER BY conversation_id, rowid
            """
        ).fetchall()
    finally:
        connection.close()

    last_user_by_conversation: dict[str, str] = {}
    records = []
    for row in rows:
        conversation_id = str(row["conversation_id"])
        if row["role"] == "user":
            last_user_by_conversation[conversation_id] = str(row["content"] or "")
            continue
        if row["role"] != "assistant" or row["rating"] not in (-1, 0, 1):
            continue
        if row["status"] != "complete" or not last_user_by_conversation.get(conversation_id):
            continue
        trace = _load_json(row["trace_value"])
        records.append({
            "message_id": str(row["id"]),
            "conversation_id": conversation_id,
            "rating": int(row["rating"]),
            "trace": trace if isinstance(trace, list) else [],
        })
    return records


def classify_failure_stage(trace: list[dict[str, Any]]) -> str:
    by_step: dict[str, list[dict[str, Any]]] = {}
    for item in trace:
        if isinstance(item, dict) and item.get("step"):
            by_step.setdefault(str(item["step"]), []).append(item)

    routes = by_step.get("route", [])
    if any(any(word in str(item.get("detail", "")).lower() for word in ("failed", "fallback")) for item in routes):
        return "route"

    retrievals = by_step.get("retrieve", [])
    if not retrievals or any(int(item.get("passage_count", 0) or 0) == 0 for item in retrievals):
        return "retrieve"

    reranks = by_step.get("rerank", [])
    if any(item.get("fallback") is True or int(item.get("kept_count", 0) or 0) == 0 for item in reranks):
        return "rerank"

    grades = by_step.get("grade", [])
    if not grades or grades[-1].get("sufficient") is False:
        return "grade"

    if by_step.get("rewrite"):
        return "rewrite"
    return "unclassified"


def build_feedback_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {-1: 0, 0: 0, 1: 0}
    grouped: dict[str, list[str]] = {stage: [] for stage in FAILURE_STAGES}
    for record in records:
        rating = int(record["rating"])
        if rating not in counts:
            continue
        counts[rating] += 1
        if rating == -1:
            stage = classify_failure_stage(record.get("trace") or [])
            grouped[stage].append(str(record["message_id"]))
    return {
        "rated_completed_answers": sum(counts.values()),
        "thumbs_down": counts[-1],
        "neutral": counts[0],
        "thumbs_up": counts[1],
        "low_rated_by_stage": {
            stage: {"count": len(message_ids), "message_ids": message_ids[:10]}
            for stage, message_ids in grouped.items()
            if message_ids
        },
        "classification": "heuristic trace-stage attribution; not causal diagnosis",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_feedback_report(load_feedback_records(args.database))
    except Exception as exc:
        parser.error(f"feedback report failed ({type(exc).__name__}); message content was not logged")
    print(f"Rated completed answers: {report['rated_completed_answers']}")
    print(f"Thumbs up/down/neutral: {report['thumbs_up']}/{report['thumbs_down']}/{report['neutral']}")
    if not report["rated_completed_answers"]:
        print("No feedback records are available yet.")
    for stage, values in report["low_rated_by_stage"].items():
        print(f"{stage:<14} {values['count']} low-rated answers")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())