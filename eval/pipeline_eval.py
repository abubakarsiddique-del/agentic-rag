"""Evaluate labeled, offline RAG trace snapshots without calling the live pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path(__file__).parent / "data" / "pipeline_fixtures.json"
DEFAULT_THRESHOLDS = {
    "route_accuracy": ("min", 0.8),
    "retrieval_hit_at_k": ("min", 0.5),
    "grading_precision": ("min", 0.75),
    "grading_recall": ("min", 0.75),
    "rerank_mrr_delta": ("min", -0.2),
    "rewrite_mrr_delta": ("min", -0.2),
    "false_abstain_rate": ("max", 0.2),
    "false_answer_rate": ("max", 0.2),
}


def load_cases(path: Path = DEFAULT_DATA) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation data must contain a non-empty cases list")
    for case in cases:
        for field in ("id", "question", "gold_passages", "expected_route", "expected_sufficiency", "trace"):
            if field not in case:
                raise ValueError(f"case is missing required field: {field}")
        if not isinstance(case["trace"], list):
            raise ValueError(f"case {case['id']} trace must be a list")
    return cases


def _steps(trace: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [item for item in trace if item.get("step") == name]


def _normalize_route(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return "broad" if normalized in {"broad", "global"} else normalized


def _ranking_metrics(ranking: list[str], gold: set[str], k: int) -> tuple[float, float]:
    hit = float(bool(set(ranking[:k]) & gold))
    reciprocal_rank = next(
        (1.0 / rank for rank, passage_id in enumerate(ranking, start=1) if passage_id in gold),
        0.0,
    )
    return hit, reciprocal_rank


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _classification_metrics(cases: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    true_positive = false_positive = false_negative = 0
    for case in cases:
        observed = _steps(case["trace"], "grade")
        if not observed:
            continue
        predicted = observed[-1].get("sufficient") is True
        expected = case["expected_sufficiency"] is True
        true_positive += int(predicted and expected)
        false_positive += int(predicted and not expected)
        false_negative += int(not predicted and expected)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    return precision, recall


def evaluate(cases: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    if k < 1:
        raise ValueError("k must be at least 1")

    route_matches: list[float] = []
    retrieval_hits: list[float] = []
    retrieval_mrr: list[float] = []
    rerank_hits_before: list[float] = []
    rerank_hits_after: list[float] = []
    rerank_mrr_before: list[float] = []
    rerank_mrr_after: list[float] = []
    rewrite_hit_deltas: list[float] = []
    rewrite_mrr_deltas: list[float] = []
    false_abstains = answerable = false_answers = unanswerable = 0

    for case in cases:
        trace = case["trace"]
        gold = {str(value) for value in case["gold_passages"]}
        route_steps = _steps(trace, "route")
        if route_steps:
            observed_route = _normalize_route(route_steps[-1].get("scope") or route_steps[-1].get("route"))
            expected_route = _normalize_route(case["expected_route"])
            route_matches.append(float(observed_route == expected_route))

        latest_retrieval: tuple[float, float] | None = None
        retrieval_before_rewrite: tuple[float, float] | None = None
        initial_retrieval_recorded = False
        for item in trace:
            if item.get("step") == "retrieve":
                passage_ids = [str(value) for value in item.get("passage_ids", [])]
                if passage_ids:
                    latest_retrieval = _ranking_metrics(passage_ids, gold, k)
                    if not initial_retrieval_recorded:
                        retrieval_hits.append(latest_retrieval[0])
                        retrieval_mrr.append(latest_retrieval[1])
                        initial_retrieval_recorded = True
                    if retrieval_before_rewrite is not None:
                        rewrite_hit_deltas.append(latest_retrieval[0] - retrieval_before_rewrite[0])
                        rewrite_mrr_deltas.append(latest_retrieval[1] - retrieval_before_rewrite[1])
                        retrieval_before_rewrite = None
            elif item.get("step") == "rerank":
                ranked_ids = [str(value) for value in item.get("passage_ids", [])]
                if latest_retrieval is not None and ranked_ids:
                    rerank_hits_before.append(latest_retrieval[0])
                    rerank_mrr_before.append(latest_retrieval[1])
                    hit_after, mrr_after = _ranking_metrics(ranked_ids, gold, k)
                    rerank_hits_after.append(hit_after)
                    rerank_mrr_after.append(mrr_after)
            elif item.get("step") == "rewrite":
                retrieval_before_rewrite = latest_retrieval

        abstained = bool(_steps(trace, "abstain"))
        answered = any(_steps(trace, step) for step in ("generate", "direct_answer", "map_reduce"))
        sufficient = case["expected_sufficiency"] is True
        if sufficient:
            answerable += 1
            false_abstains += int(abstained and not answered)
        else:
            unanswerable += 1
            false_answers += int(answered and not abstained)

    grading_precision, grading_recall = _classification_metrics(cases)
    hit_before = _mean(rerank_hits_before)
    hit_after = _mean(rerank_hits_after)
    mrr_before = _mean(rerank_mrr_before)
    mrr_after = _mean(rerank_mrr_after)
    return {
        "case_count": len(cases),
        "k": k,
        "route_accuracy": _mean(route_matches),
        "retrieval_hit_at_k": _mean(retrieval_hits),
        "retrieval_mrr": _mean(retrieval_mrr),
        "rerank": {
            "hit_at_k_before": hit_before,
            "hit_at_k_after": hit_after,
            "hit_at_k_delta": hit_after - hit_before if hit_before is not None and hit_after is not None else None,
            "mrr_before": mrr_before,
            "mrr_after": mrr_after,
            "mrr_delta": mrr_after - mrr_before if mrr_before is not None and mrr_after is not None else None,
        },
        "grading": {"precision": grading_precision, "recall": grading_recall},
        "rewrite": {
            "hit_at_k_delta": _mean(rewrite_hit_deltas),
            "mrr_delta": _mean(rewrite_mrr_deltas),
            "evaluated_rewrites": len(rewrite_mrr_deltas),
        },
        "abstention": {
            "false_abstain_rate": false_abstains / answerable if answerable else None,
            "false_answer_rate": false_answers / unanswerable if unanswerable else None,
            "answerable_cases": answerable,
            "unanswerable_cases": unanswerable,
        },
    }


def configured_thresholds() -> dict[str, tuple[str, float]]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    for metric, (direction, default) in thresholds.items():
        env_name = f"PIPELINE_EVAL_{metric.upper()}"
        if env_name in os.environ:
            thresholds[metric] = (direction, float(os.environ[env_name]))
    return thresholds


def check_thresholds(report: dict[str, Any], thresholds: dict[str, tuple[str, float]]) -> list[str]:
    values = {
        "route_accuracy": report["route_accuracy"],
        "retrieval_hit_at_k": report["retrieval_hit_at_k"],
        "grading_precision": report["grading"]["precision"],
        "grading_recall": report["grading"]["recall"],
        "rerank_mrr_delta": report["rerank"]["mrr_delta"],
        "rewrite_mrr_delta": report["rewrite"]["mrr_delta"],
        "false_abstain_rate": report["abstention"]["false_abstain_rate"],
        "false_answer_rate": report["abstention"]["false_answer_rate"],
    }
    failures = []
    for metric, (direction, limit) in thresholds.items():
        value = values[metric]
        if value is None:
            failures.append(f"{metric}: no trace evidence available")
        elif (direction == "min" and value < limit) or (direction == "max" and value > limit):
            failures.append(f"{metric}: {value:.3f} violates {direction}imum {limit:.3f}")
    return failures


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_report(report: dict[str, Any], failures: list[str]) -> str:
    rows = [
        ("Route accuracy", report["route_accuracy"]),
        (f"Retrieval hit@{report['k']}", report["retrieval_hit_at_k"]),
        ("Retrieval MRR", report["retrieval_mrr"]),
        ("Rerank hit@k delta", report["rerank"]["hit_at_k_delta"]),
        ("Rerank MRR delta", report["rerank"]["mrr_delta"]),
        ("Grading precision", report["grading"]["precision"]),
        ("Grading recall", report["grading"]["recall"]),
        ("Rewrite hit@k delta", report["rewrite"]["hit_at_k_delta"]),
        ("Rewrite MRR delta", report["rewrite"]["mrr_delta"]),
        ("False-abstain rate", report["abstention"]["false_abstain_rate"]),
        ("False-answer rate", report["abstention"]["false_answer_rate"]),
    ]
    label_width = max(len(label) for label, _ in rows)
    lines = [f"Pipeline evaluation ({report['case_count']} cases, k={report['k']})", "-" * 54]
    lines.extend(f"{label:<{label_width}}  {_format(value)}" for label, value in rows)
    lines.append(f"Thresholds: {'PASS' if not failures else 'FAIL'}")
    lines.extend(f"  - {failure}" for failure in failures)
    return "\n".join(lines)


def render_variance_report(report: dict[str, Any]) -> str:
    """Render repeated-run metrics with mean, sample deviation, and observed range."""
    metrics = report["metrics"]
    label_width = max((len(name) for name in metrics), default=6)
    lines = [
        f"Production variance evaluation ({report['case_count']} cases, "
        f"{report['runs_per_case']} runs/case, k={report['k']})",
        f"{'Metric':<{label_width}}  {'Mean':>8}  {'Std dev':>8}  {'Min':>8}  {'Max':>8}",
        "-" * (label_width + 44),
    ]
    for name, values in metrics.items():
        lines.append(
            f"{name:<{label_width}}  {_format(values['mean']):>8}  "
            f"{_format(values['std_dev']):>8}  {_format(values['min']):>8}  "
            f"{_format(values['max']):>8}"
        )
    high_variance = report.get("high_variance_cases", [])
    lines.append(f"High-variance cases: {len(high_variance)}")
    lines.extend(
        f"  - {item['case_id']}: {', '.join(item['metrics']) or 'low answer agreement'}"
        for item in high_variance
    )
    latency = report.get("latency_ms", {})
    if latency:
        lines.append("Stage latency (ms, p50/p95)")
        end_to_end = latency["end_to_end"]
        lines.append(
            f"  {'end_to_end':<12} {_format(end_to_end['p50_ms'])}/"
            f"{_format(end_to_end['p95_ms'])}"
        )
        for stage, values in latency.get("stages", {}).items():
            lines.append(f"  {stage:<12} {_format(values['p50_ms'])}/{_format(values['p95_ms'])}")
    calls = report.get("llm_calls", {})
    if calls:
        lines.append(f"LLM calls/run: mean {_format(calls['total_per_run']['mean'])}")
        for stage, values in calls.get("by_stage_per_run", {}).items():
            lines.append(f"  {stage:<12} mean {_format(values['mean'])}")
    settings = report.get("settings")
    if settings:
        lines.append("Settings: " + json.dumps(settings, sort_keys=True))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSON labeled trace snapshots")
    parser.add_argument("--k", type=int, default=5, help="cutoff for hit@k")
    parser.add_argument("--json-out", type=Path, help="also write the complete report to this path")
    for metric, (direction, default) in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{metric.replace('_', '-')}", type=float, default=None,
                            help=f"override {direction}imum threshold (default: {default})")
    args = parser.parse_args(argv)

    cases = load_cases(args.data)
    report = evaluate(cases, args.k)
    thresholds = configured_thresholds()
    for metric in thresholds:
        override = getattr(args, metric)
        if override is not None:
            direction, _ = thresholds[metric]
            thresholds[metric] = (direction, override)
    failures = check_thresholds(report, thresholds)
    report["thresholds"] = {name: {"direction": direction, "value": limit}
                            for name, (direction, limit) in thresholds.items()}
    report["regressions"] = failures
    report["passed"] = not failures
    print(render_report(report, failures))
    print("\nJSON report:")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())