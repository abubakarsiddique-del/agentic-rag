import pytest

from eval.production.drift_check import run_drift_check
from eval.production.shadow_compare import compare_reports, run_shadow_compare


def _report(quality, latency, calls):
    return {
        "metrics": {"retrieval_mrr": {"mean": quality}},
        "latency_ms": {
            "end_to_end": {"p50_ms": latency, "p95_ms": latency * 2},
            "stages": {"retrieve": {"p50_ms": latency / 2, "p95_ms": latency}},
        },
        "llm_calls": {"total_per_run": {"mean": calls}},
    }


def test_drift_check_appends_timestamped_metric_movement(tmp_path):
    history = tmp_path / "drift.jsonl"
    reports = iter((_report(0.5, 100, 8), _report(0.7, 120, 9)))
    evaluator = lambda *_args, **_kwargs: next(reports)
    cases = [{"id": "opaque-1"}]

    first = run_drift_check(cases, history, evaluator=evaluator, timestamp="t1")
    second = run_drift_check(cases, history, evaluator=evaluator, timestamp="t2")
    assert first["movement_vs_previous"] == {}
    assert second["movement_vs_previous"]["retrieval_mrr"] == pytest.approx(0.2)
    assert len(history.read_text(encoding="utf-8").splitlines()) == 2


def test_shadow_comparison_reports_quality_latency_and_call_deltas():
    report = compare_reports("base", _report(0.5, 100, 8), "candidate", _report(0.7, 120, 9))
    assert report["quality"]["retrieval_mrr"]["delta_right_minus_left"] == pytest.approx(0.2)
    assert report["latency_ms"]["end_to_end"]["delta_right_minus_left"]["p95_ms"] == 40
    assert report["cost_proxy_llm_calls_per_run"]["delta_right_minus_left"] == 1


def test_shadow_uses_same_cases_for_each_named_config():
    calls = []

    def evaluator(cases, **settings):
        calls.append((cases, settings))
        return _report(settings["quality"], 100, settings["calls"])

    cases = [{"id": "opaque-1"}]
    result = run_shadow_compare(
        cases,
        "base",
        {"quality": 0.5, "calls": 8},
        "candidate",
        {"quality": 0.7, "calls": 9},
        evaluator=evaluator,
    )
    assert calls[0][0] is cases and calls[1][0] is cases
    assert result["quality"]["retrieval_mrr"]["delta_right_minus_left"] == pytest.approx(0.2)