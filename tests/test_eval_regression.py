from eval.answer_eval import (
    DEFAULT_GOLDEN_THRESHOLDS,
    check_golden_thresholds,
    evaluate_golden_set,
    load_golden_cases,
)


def _predictions(cases):
    return [
        {"id": case["id"], "answer": case["expected_answer"], "sources": case["sources"]}
        for case in cases
    ]


def _judge(payload):
    citation_ids = payload.get("expected_citations", [])
    return {
        "claims": [{
            "claim": payload.get("expected_answer", "answer"),
            "supported": True,
            "has_citation": True,
            "citation_supported": True,
            "citation_ids": citation_ids,
        }],
        "scores": {
            name: {"score": 5, "justification": "Matches the reviewed golden answer and cited evidence."}
            for name in ("correctness", "completeness", "conciseness")
        },
    }


def test_golden_regression_set_passes_with_stubbed_judge():
    cases = load_golden_cases()
    assert len(cases) == 12
    report = evaluate_golden_set(cases, _predictions(cases), judge=_judge)
    assert report["metrics"]["faithfulness"] == 1.0
    assert report["metrics"]["citation_precision"] == 1.0
    assert report["metrics"]["citation_recall"] == 1.0
    assert report["metrics"]["gold_citation_recall"] == 1.0
    assert check_golden_thresholds(report, DEFAULT_GOLDEN_THRESHOLDS) == []


def test_golden_regression_gate_fails_on_unsupported_uncited_claims():
    cases = load_golden_cases()
    bad_judge = lambda _payload: {
        "claims": [{"claim": "unsupported", "supported": False, "has_citation": False,
                    "citation_supported": False, "citation_ids": []}],
        "scores": {
            name: {"score": 1, "justification": "Does not match the reference."}
            for name in ("correctness", "completeness", "conciseness")
        },
    }
    report = evaluate_golden_set(cases, _predictions(cases), judge=bad_judge)
    failures = check_golden_thresholds(report, DEFAULT_GOLDEN_THRESHOLDS)
    assert "faithfulness" in " ".join(failures)
    assert "citation_recall" in " ".join(failures)


def test_answer_eval_passes_gold_fields_to_mockable_judge():
    cases = load_golden_cases()
    captured = {}
    evaluate_golden_set(cases[:1], _predictions(cases[:1]), judge=lambda payload: captured.update(payload) or _judge(payload))
    assert captured["expected_answer"] == cases[0]["expected_answer"]
    assert captured["expected_citations"] == cases[0]["expected_citations"]


def test_golden_report_includes_optional_stage_latency_and_llm_calls():
    cases = load_golden_cases()[:2]
    predictions = _predictions(cases)
    predictions[0].update({"stage_latency_ms": {"route": 10.0}, "llm_calls": {"route": 1, "grade": 1}})
    predictions[1].update({"stage_latency_ms": {"route": 20.0}, "llm_calls": {"route": 1, "grade": 2}})
    report = evaluate_golden_set(cases, predictions, judge=_judge)
    operations = report["operational_metrics"]
    assert operations["stage_latency_ms"]["route"]["p50_ms"] == 15.0
    assert operations["stage_latency_ms"]["route"]["p95_ms"] == 19.5
    assert operations["llm_calls_per_answer"]["total_mean"] == 2.5


def test_golden_report_does_not_count_missing_call_telemetry_as_zero():
    cases = load_golden_cases()[:2]
    predictions = _predictions(cases)
    predictions[0]["llm_calls"] = {"route": 2}
    report = evaluate_golden_set(cases, predictions, judge=_judge)
    assert report["operational_metrics"]["llm_calls_per_answer"]["total_mean"] == 2.0