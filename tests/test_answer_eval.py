import json

import pytest

import eval.answer_eval as answer_eval_module
from eval.answer_eval import evaluate_golden_set, load_golden_cases
from eval.answer_eval import _invoke_with_rate_limit_retry, score_answer
from eval.metrics import pearson_correlation, spearman_correlation


def _judge_result():
    return {
        "claims": [
            {"claim": "Claim one", "supported": True, "has_citation": True, "citation_supported": False},
            {"claim": "Claim two", "supported": False, "has_citation": False, "citation_supported": False},
        ],
        "scores": {
            "correctness": {"score": 3, "justification": "Some support."},
            "completeness": {"score": 4, "justification": "Mostly complete."},
            "conciseness": {"score": 5, "justification": "Brief."},
        },
    }


def test_answer_scorer_calculates_faithfulness_and_citation_metrics():
    captured = {}

    def fake_judge(payload):
        captured.update(payload)
        return _judge_result()

    result = score_answer(
        "Question?",
        "Answer [source]",
        [{"filename": "guide.pdf", "page": 2, "snippet": "Supporting passage."}],
        judge=fake_judge,
    )
    assert result["faithfulness"] == 0.5
    assert result["citation_precision"] == 0.0
    assert result["citation_recall"] == 0.5
    assert result["judge_scores"]["correctness"] == 3
    assert captured["sources"][0]["id"] == "guide.pdf page 2"


def test_answer_scorer_rejects_malformed_judge_scores():
    result = _judge_result()
    result["scores"]["correctness"]["score"] = 6
    with pytest.raises(ValueError, match="between 1 and 5"):
        score_answer("Q", "A", [], judge=lambda _: result)


def test_golden_eval_skips_explicit_chitchat_cases_and_predictions():
    cases = [
        {
            "id": "document-case",
            "question": "What is the policy?",
            "sources": [],
            "expected_answer": "Policy answer",
            "expected_citations": [],
        },
        {
            "id": "greeting-case",
            "question": "Hello!",
            "sources": [],
            "expected_answer": "Hi!",
            "expected_citations": [],
            "category": "chitchat",
        },
    ]
    predictions = [
        {"id": "document-case", "answer": "Policy answer", "sources": []},
        {"id": "greeting-case", "answer": "Hi!", "sources": [], "trace": [{"route": "chitchat"}]},
    ]
    report = evaluate_golden_set(cases, predictions, judge=lambda _payload: _judge_result())

    assert report["case_count"] == 1
    assert report["skipped_case_count"] == 1
    assert report["skipped_case_ids"] == ["greeting-case"]


def test_rank_correlations_handle_ties_and_constant_values():
    assert spearman_correlation([1, 2, 2, 4], [1, 3, 2, 4]) == pytest.approx(0.948683298)
    assert pearson_correlation([1, 1], [0, 1]) is None


def test_groq_rate_limit_retries_then_returns_result():
    class RateLimited:
        calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            if self.calls < 3:
                error = RuntimeError("429 rate limit")
                error.status_code = 429
                raise error
            return "ok"

    delays = []
    attempts = []
    client = RateLimited()
    assert _invoke_with_rate_limit_retry(
        client, "prompt", sleep_fn=delays.append, on_attempt=lambda: attempts.append(1)
    ) == "ok"
    assert client.calls == 3
    assert len(attempts) == 3
    assert delays == [1.0, 2.0]


def test_groq_rate_limit_exhaustion_is_not_silently_swallowed():
    class AlwaysLimited:
        def invoke(self, _prompt):
            error = RuntimeError("429 rate limit")
            error.status_code = 429
            raise error

    with pytest.raises(RuntimeError, match="persisted after 4 attempts"):
        _invoke_with_rate_limit_retry(AlwaysLimited(), "prompt", sleep_fn=lambda _delay: None)


def test_answer_eval_cli_defaults_to_stub_and_never_creates_groq(tmp_path, monkeypatch, capsys):
    cases = load_golden_cases()
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"answers": [
        {"id": case["id"], "answer": case["expected_answer"], "sources": case["sources"]}
        for case in cases
    ]}), encoding="utf-8")
    monkeypatch.setattr(
        answer_eval_module,
        "create_groq_judge",
        lambda: (_ for _ in ()).throw(AssertionError("default answer eval must not create Groq")),
    )
    assert answer_eval_module.main(["--answers", str(answers)]) == 1
    output = capsys.readouterr().out
    assert "[STUB]" in output
    assert '"groq_call_attempts": 0' in output


def test_answer_eval_live_decline_prints_call_estimate_and_stops(tmp_path, monkeypatch, capsys):
    cases = load_golden_cases()
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"answers": [
        {"id": case["id"], "answer": case["expected_answer"], "sources": case["sources"]}
        for case in cases
    ]}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    monkeypatch.setattr(
        answer_eval_module,
        "create_groq_judge",
        lambda: (_ for _ in ()).throw(AssertionError("declined live eval must not create Groq")),
    )
    with pytest.raises(SystemExit):
        answer_eval_module.main(["--answers", str(answers), "--live"])
    assert "12" in capsys.readouterr().out