import json
import sqlite3

import pytest

import eval.production.judge_calibration as calibration_module
from eval.production.feedback_correlation import analyze_feedback, load_rated_turns
from eval.answer_eval import confirm_live_call_count, create_stub_judge
from eval.production.judge_calibration import rating_template, run_calibration


def test_calibration_compares_judge_with_manual_ratings():
    cases = [
        {"id": f"case-{index}", "question": str(index), "answer": "Answer", "sources": []}
        for index in range(1, 21)
    ]
    ratings = {
        case["id"]: {"correctness": (int(case["question"]) - 1) % 5 + 1,
                     "completeness": (int(case["question"]) - 1) % 5 + 1,
                     "conciseness": (int(case["question"]) - 1) % 5 + 1}
        for case in cases
    }
    calls = []

    def fake_judge(payload):
        calls.append(payload)
        score = (int(payload["question"]) - 1) % 5 + 1
        return {
            "claims": [],
            "scores": {
                name: {"score": score, "justification": "stub"}
                for name in ("correctness", "completeness", "conciseness")
            },
        }

    report = run_calibration(cases, ratings, judge=fake_judge)
    assert len(calls) == 20
    assert report["agreement"]["correctness"]["spearman_rho"] == pytest.approx(1.0)
    assert report["agreement"]["correctness"]["weak"] is False
    assert len(rating_template(cases)["ratings"]) == 20


def test_calibration_flags_weak_agreement():
    cases = [
        {"id": f"case-{index}", "question": str(index), "answer": "Answer", "sources": []}
        for index in range(1, 21)
    ]
    ratings = {
        case["id"]: {"correctness": 6 - int(case["question"]),
                     "completeness": 6 - int(case["question"]),
                     "conciseness": 6 - int(case["question"])}
        for case in cases
    }
    judge = lambda _: {
        "claims": [],
        "scores": {
            name: {"score": 3, "justification": "stub"}
            for name in ("correctness", "completeness", "conciseness")
        },
    }
    report = run_calibration(cases, ratings, judge=judge)
    assert report["agreement"]["correctness"]["weak"] is True


def test_calibration_repeats_each_case_and_reports_agreement_variance():
    cases = [
        {"id": f"case-{index}", "question": str(index), "answer": "Answer", "sources": []}
        for index in range(1, 21)
    ]
    ratings = {
        case["id"]: {dimension: (int(case["question"]) - 1) % 5 + 1
                     for dimension in ("correctness", "completeness", "conciseness")}
        for case in cases
    }
    call_count = 0

    def varying_judge(payload):
        nonlocal call_count
        call_count += 1
        score = (int(payload["question"]) - 1) % 5 + 1
        if call_count > len(cases):
            score = 6 - score
        return {
            "claims": [],
            "scores": {
                name: {"score": score, "justification": "stub"}
                for name in ("correctness", "completeness", "conciseness")
            },
        }

    report = run_calibration(cases, ratings, judge=varying_judge, runs=2, mode="LIVE")
    assert call_count == 40
    assert report["mode"] == "LIVE"
    assert report["agreement"]["correctness"]["per_run_spearman_rho"] == pytest.approx([1.0, -1.0])
    assert report["agreement"]["correctness"]["mean"] == 0.0
    assert report["agreement"]["correctness"]["std_dev"] == pytest.approx(2 ** 0.5)


def test_live_call_guard_prints_estimate_and_requires_confirmation(capsys):
    assert confirm_live_call_count(75, input_fn=lambda _prompt: "no") is False
    assert "75" in capsys.readouterr().out
    assert confirm_live_call_count(75, yes=True, input_fn=lambda _prompt: "unexpected") is True


def test_default_stub_judge_is_neutral_and_deterministic():
    judge = create_stub_judge()
    result = judge({"question": "ignored"})
    assert set(result["scores"]) == {"correctness", "completeness", "conciseness"}
    assert all(item["score"] == 3 for item in result["scores"].values())


def test_calibration_cli_defaults_to_stub_without_creating_groq_client(tmp_path, monkeypatch, capsys):
    cases = [
        {"id": f"case-{index}", "question": str(index), "answer": "Answer", "sources": []}
        for index in range(1, 21)
    ]
    case_path = tmp_path / "cases.json"
    rating_path = tmp_path / "ratings.json"
    case_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    ratings = [
        {"case_id": case["id"], **{dimension: 3 for dimension in ("correctness", "completeness", "conciseness")}}
        for case in cases
    ]
    rating_path.write_text(json.dumps({"ratings": ratings}), encoding="utf-8")

    def fail_if_created():
        raise AssertionError("default CLI path must not construct Groq")

    monkeypatch.setattr(calibration_module, "create_groq_judge", fail_if_created)
    assert calibration_module.main(["--cases", str(case_path), "--ratings", str(rating_path)]) == 0
    output = capsys.readouterr().out
    assert "[STUB]" in output
    assert '"groq_call_attempts": 0' in output


def test_live_calibration_decline_stops_before_client_creation(tmp_path, monkeypatch, capsys):
    cases = [
        {"id": f"case-{index}", "question": str(index), "answer": "Answer", "sources": []}
        for index in range(1, 21)
    ]
    case_path = tmp_path / "cases.json"
    rating_path = tmp_path / "ratings.json"
    case_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    ratings = [
        {"case_id": case["id"], **{dimension: 3 for dimension in ("correctness", "completeness", "conciseness")}}
        for case in cases
    ]
    rating_path.write_text(json.dumps({"ratings": ratings}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    monkeypatch.setattr(
        calibration_module,
        "create_groq_judge",
        lambda: (_ for _ in ()).throw(AssertionError("declined live run must not create Groq client")),
    )
    with pytest.raises(SystemExit):
        calibration_module.main([
            "--cases", str(case_path), "--ratings", str(rating_path), "--live", "--runs", "3"
        ])
    assert "60" in capsys.readouterr().out


def test_feedback_records_join_latest_user_and_read_without_writes(tmp_path):
    database = tmp_path / "history.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE messages (conversation_id TEXT, role TEXT, content TEXT, "
            "trace TEXT, trace_json TEXT, sources TEXT, sources_json TEXT, rating INTEGER, "
            "status TEXT, rowid INTEGER PRIMARY KEY)"
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv", "user", "What is covered?", None, None, None, None, None, "complete", 1),
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv", "assistant", "The plan covers X.", None,
             json.dumps([{"step": "retrieve"}]), None,
             json.dumps([{"filename": "plan.pdf", "page": 1, "snippet": "Plan text"}]),
             1, "complete", 2),
        )
        connection.commit()

    records = load_rated_turns(database)
    assert len(records) == 1
    assert records[0]["question"] == "What is covered?"
    assert records[0]["rating"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_feedback_correlation_flags_weak_metrics_and_trace_stages():
    turns = [
        {"question": "Q", "answer": str(rating), "sources": [], "trace": [{"step": "grade"}], "rating": rating}
        for rating in (-1, 1, -1, 1)
    ]

    def scorer(_question, answer, _sources):
        positive = float(answer == "1")
        return {
            "faithfulness": positive,
            "citation_precision": positive,
            "citation_recall": positive,
            "judge_scores": {"correctness": positive, "completeness": positive, "conciseness": positive},
        }

    report = analyze_feedback(turns, scorer=scorer, min_samples=4)
    assert report["correlations"]["faithfulness"]["point_biserial_r"] == pytest.approx(1.0)
    assert report["correlations"]["faithfulness"]["predictive"] is True
    assert report["stage_feedback"]["grade"]["thumbs_up_rate"] == 0.5


def test_feedback_empty_data_does_not_call_scorer():
    def fail_if_called(*_args):
        raise AssertionError("scorer should not run without ratings")

    report = analyze_feedback([], scorer=fail_if_called)
    assert report["sample_count"] == 0
    assert "no judge calls" in report["warning"]