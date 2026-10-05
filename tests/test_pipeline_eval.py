from eval.pipeline_eval import (
    check_thresholds,
    configured_thresholds,
    evaluate,
    load_cases,
)


def test_pipeline_fixture_meets_regression_thresholds():
    report = evaluate(load_cases())
    assert check_thresholds(report, configured_thresholds()) == []


def test_pipeline_metrics_capture_rerank_and_rewrite_deltas():
    report = evaluate(load_cases())
    assert report["retrieval_hit_at_k"] == 0.5
    assert report["rerank"]["mrr_delta"] > 0
    assert report["rewrite"]["mrr_delta"] == 1.0
    assert report["rewrite"]["evaluated_rewrites"] == 1


def test_threshold_failure_is_reported():
    report = evaluate(load_cases())
    failures = check_thresholds(report, {"route_accuracy": ("min", 1.1)})
    assert failures == ["route_accuracy: 1.000 violates minimum 1.100"]