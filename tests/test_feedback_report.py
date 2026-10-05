import json
import sqlite3

from eval.feedback_report import (
    build_feedback_report,
    classify_failure_stage,
    load_feedback_records,
)


def test_feedback_stage_heuristics_cover_pipeline_failures():
    traces = {
        "route": [{"step": "route", "detail": "route failed, defaulting to retrieve"}],
        "retrieve": [{"step": "route"}, {"step": "retrieve", "passage_count": 0}],
        "rerank": [{"step": "retrieve", "passage_count": 2}, {"step": "rerank", "fallback": True}],
        "grade": [{"step": "retrieve", "passage_count": 2}, {"step": "grade", "sufficient": False}],
        "rewrite": [
            {"step": "retrieve", "passage_count": 2},
            {"step": "grade", "sufficient": False},
            {"step": "rewrite"},
            {"step": "grade", "sufficient": True},
        ],
    }
    assert {name: classify_failure_stage(trace) for name, trace in traces.items()} == {
        name: name for name in traces
    }


def test_feedback_loader_joins_ratings_to_assistant_trace_read_only(tmp_path):
    database = tmp_path / "history.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE messages (id TEXT, conversation_id TEXT, role TEXT, content TEXT, "
            "trace TEXT, trace_json TEXT, rating INTEGER, status TEXT, rowid INTEGER PRIMARY KEY)"
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("u1", "conv", "user", "Question?", None, None, None, "complete", 1),
                ("a1", "conv", "assistant", "Answer", None,
                 json.dumps([{"step": "retrieve", "passage_count": 1}, {"step": "grade", "sufficient": False}]),
                 -1, "complete", 2),
                ("a2", "conv", "assistant", "Unrated", None, None, None, "complete", 3),
            ],
        )
        connection.commit()

    records = load_feedback_records(database)
    report = build_feedback_report(records)
    assert report["rated_completed_answers"] == 1
    assert report["low_rated_by_stage"]["grade"]["message_ids"] == ["a1"]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3


def test_empty_feedback_has_no_low_rated_stage_groups():
    report = build_feedback_report([])
    assert report["rated_completed_answers"] == 0
    assert report["low_rated_by_stage"] == {}