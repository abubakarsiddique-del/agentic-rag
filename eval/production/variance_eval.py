"""Repeat reviewed cases against the real RAG model and report score variance.

Run on demand with a local, reviewed dataset. Each repetition makes fresh Groq
calls; no question, answer, passage text, or source content is written to the report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from time import perf_counter
from pathlib import Path
from typing import Any

from eval.add_case import DEFAULT_DATA, validate_case
from eval.pipeline_eval import _normalize_route, _ranking_metrics, render_variance_report


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"production case file not found: {path}; add reviewed cases with python -m eval.add_case")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("production dataset must contain at least one reviewed case")
    for case in cases:
        validate_case(case)
    return cases


def summarize(values: list[float | int | None]) -> dict[str, float | int | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"n": 0, "mean": None, "std_dev": None, "min": None, "max": None}
    return {
        "n": len(present),
        "mean": statistics.fmean(present),
        "std_dev": statistics.stdev(present) if len(present) > 1 else 0.0,
        "min": min(present),
        "max": max(present),
    }


def run_repeated_cases(cases: list[Any], runs: int, evaluator) -> list[list[Any]]:
    """Apply a caller-supplied scorer to the same cases for each repeat."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    return [[evaluator(case) for case in cases] for _ in range(runs)]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _passage_id(document: Any) -> str:
    metadata = dict(getattr(document, "metadata", {}) or {})
    value = metadata.get("passage_id") or metadata.get("id")
    if value:
        return str(value)
    parts = [metadata.get("document_id"), metadata.get("page"), metadata.get("start_index")]
    return ":".join(str(part) for part in parts if part is not None)


def create_instrumented_service(conversation_id: str, top_k: int) -> tuple[Any, dict[str, Any]]:
    """Capture IDs and telemetry on an eval-only service without altering trace/SSE data."""
    from agentic_rag import RAGService
    from langchain_core.callbacks import BaseCallbackHandler

    captured: dict[str, Any] = {
        "retrievals": [],
        "reranks": [],
        "stage_latency_ms": {},
        "llm_calls": {},
        "current_stage": None,
    }
    stage_methods = (
        "_contextualize", "_route_question", "_map_reduce", "_retrieve", "_rerank",
        "_grade", "_rewrite_query", "_generate", "_abstain", "_direct_answer",
    )
    original_methods = {name: getattr(RAGService, name) for name in stage_methods}

    def instrument(stage_name, original):
        stage = stage_name.removeprefix("_")

        def wrapped(service, *args, **kwargs):
            state = args[0] if args else {}
            previous_stage = captured["current_stage"]
            captured["current_stage"] = stage
            started = perf_counter()
            try:
                result = original(service, *args, **kwargs)
                if stage == "retrieve":
                    captured["retrievals"].append({
                        "query": state.get("search_query", ""),
                        "passage_ids": [
                            _passage_id(document) for document in result.get("retrieved_docs", [])
                        ],
                    })
                elif stage == "rerank":
                    captured["reranks"].append({
                        "candidate_ids": [
                            _passage_id(document) for document in state.get("retrieved_docs", [])
                        ],
                        "passage_ids": [
                            _passage_id(document) for document in result.get("retrieved_docs", [])
                        ],
                    })
                return result
            finally:
                elapsed = (perf_counter() - started) * 1000
                captured["stage_latency_ms"].setdefault(stage, []).append(elapsed)
                captured["current_stage"] = previous_stage

        return wrapped

    class CallCounter(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            stage = captured["current_stage"] or "unattributed"
            captured["llm_calls"][stage] = captured["llm_calls"].get(stage, 0) + 1

    for name, original in original_methods.items():
        setattr(RAGService, name, instrument(name, original))
    try:
        service = RAGService(conversation_id=conversation_id, top_k=top_k)
    finally:
        for name, original in original_methods.items():
            setattr(RAGService, name, original)
    service.llm.callbacks = [*(getattr(service.llm, "callbacks", None) or []), CallCounter()]
    return service, captured


def score_run(
    case: dict[str, Any],
    trace: list[dict[str, Any]],
    captured: dict[str, list[dict[str, Any]]],
    k: int,
) -> dict[str, float | None]:
    expected_route = _normalize_route(case["expected_route"])
    route_entries = [item for item in trace if item.get("step") == "route"]
    route_value = (route_entries[-1].get("scope") or route_entries[-1].get("route")) if route_entries else None
    observed_route = _normalize_route(route_value)
    expected_sufficiency = case["expected_sufficiency"] is True
    grades = [item for item in trace if item.get("step") == "grade"]
    predicted_sufficiency = grades[-1].get("sufficient") is True if grades else None
    answerable = expected_sufficiency and bool(case["gold_passages"])
    gold = {str(value) for value in case["gold_passages"]}

    retrievals = captured.get("retrievals", [])
    reranks = captured.get("reranks", [])
    initial_retrieval = retrievals[0].get("passage_ids", []) if retrievals else None
    retrieval_hit = retrieval_mrr = None
    if answerable and initial_retrieval is not None:
        retrieval_hit, retrieval_mrr = _ranking_metrics(initial_retrieval, gold, k)

    rerank_hit_deltas: list[float] = []
    rerank_mrr_deltas: list[float] = []
    if answerable:
        for item in reranks:
            before = _ranking_metrics(item.get("candidate_ids", []), gold, k)
            after = _ranking_metrics(item.get("passage_ids", []), gold, k)
            rerank_hit_deltas.append(after[0] - before[0])
            rerank_mrr_deltas.append(after[1] - before[1])

    rewrite_count = sum(item.get("step") == "rewrite" for item in trace)
    rewrite_hit_deltas: list[float] = []
    rewrite_mrr_deltas: list[float] = []
    if answerable:
        for index in range(min(rewrite_count, max(0, len(retrievals) - 1))):
            before = _ranking_metrics(retrievals[index].get("passage_ids", []), gold, k)
            after = _ranking_metrics(retrievals[index + 1].get("passage_ids", []), gold, k)
            rewrite_hit_deltas.append(after[0] - before[0])
            rewrite_mrr_deltas.append(after[1] - before[1])

    abstained = any(item.get("step") == "abstain" for item in trace)
    answered = any(item.get("step") in {"generate", "direct_answer", "map_reduce"} for item in trace)
    return {
        "route_accuracy": float(observed_route == expected_route) if route_entries else None,
        "retrieval_hit_at_k": retrieval_hit,
        "retrieval_mrr": retrieval_mrr,
        "rerank_hit_at_k_delta": statistics.fmean(rerank_hit_deltas) if rerank_hit_deltas else None,
        "rerank_mrr_delta": statistics.fmean(rerank_mrr_deltas) if rerank_mrr_deltas else None,
        "grading_accuracy": float(predicted_sufficiency == expected_sufficiency) if predicted_sufficiency is not None else None,
        "false_abstain": float(abstained and not answered) if expected_sufficiency else None,
        "false_answer": float(answered and not abstained) if not expected_sufficiency else None,
        "rewrite_hit_at_k_delta": statistics.fmean(rewrite_hit_deltas) if rewrite_hit_deltas else None,
        "rewrite_mrr_delta": statistics.fmean(rewrite_mrr_deltas) if rewrite_mrr_deltas else None,
    }


def _answer_agreement(answers: list[str]) -> float | None:
    normalized = [" ".join(answer.casefold().split()) for answer in answers]
    if len(normalized) < 2:
        return None
    counts: dict[str, int] = {}
    for answer in normalized:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.values()) / len(normalized)


def build_report(
    cases: list[dict[str, Any]],
    run_scores: dict[str, list[dict[str, float | None]]],
    answer_agreements: dict[str, float | None],
    runs_per_case: int,
    k: int,
    high_variance_std: float,
    min_answer_agreement: float,
) -> dict[str, Any]:
    metric_names = list(next(iter(run_scores.values()))[0]) if run_scores else []
    metrics: dict[str, Any] = {
        name: summarize([score[name] for scores in run_scores.values() for score in scores])
        for name in metric_names
    }

    case_details = []
    high_variance_cases = []
    for case in cases:
        case_id = case["id"]
        scores = run_scores[case_id]
        case_metrics = {name: summarize([score[name] for score in scores]) for name in metric_names}
        agreement = answer_agreements.get(case_id)
        variable = [
            name for name, summary in case_metrics.items()
            if summary["std_dev"] is not None and summary["std_dev"] >= high_variance_std
        ]
        case_details.append({
            "case_id": case_id,
            "strata": case["strata"],
            "metrics": case_metrics,
            "answer_exact_agreement": agreement,
        })
        if variable or (agreement is not None and agreement < min_answer_agreement):
            high_variance_cases.append({
                "case_id": case_id,
                "metrics": variable,
                "answer_exact_agreement": agreement,
            })

    return {
        "case_count": len(cases),
        "runs_per_case": runs_per_case,
        "k": k,
        "metrics": metrics,
        "case_results": case_details,
        "high_variance_cases": high_variance_cases,
        "variance_threshold_std_dev": high_variance_std,
        "minimum_answer_exact_agreement": min_answer_agreement,
    }


def run_evaluation(
    cases: list[dict[str, Any]],
    *,
    runs: int = 5,
    k: int = 5,
    top_k: int = 4,
    rerank_enabled: bool = True,
    rerank_candidates: int = 20,
    rerank_top_n: int = 5,
    max_retries: int = 2,
    map_reduce_mode: str = "auto",
    high_variance_std: float = 0.35,
    min_answer_agreement: float = 0.6,
    service_factory=create_instrumented_service,
) -> dict[str, Any]:
    if runs < 2:
        raise ValueError("runs must be at least 2 to estimate variance")
    services: dict[str, tuple[Any, dict[str, list[dict[str, Any]]]]] = {}
    run_scores: dict[str, list[dict[str, float | None]]] = {}
    answer_agreements: dict[str, float | None] = {}
    stage_latencies: dict[str, list[float]] = {}
    llm_calls_by_run: list[dict[str, int]] = []
    total_latencies: list[float] = []

    for case in cases:
        conversation_id = str(case["conversation_id"])
        if conversation_id not in services:
            services[conversation_id] = service_factory(conversation_id, top_k)
        service, captured = services[conversation_id]
        scores: list[dict[str, float | None]] = []
        answers: list[str] = []
        for _ in range(runs):
            captured.setdefault("retrievals", []).clear()
            captured.setdefault("reranks", []).clear()
            captured.setdefault("stage_latency_ms", {}).clear()
            captured.setdefault("llm_calls", {}).clear()
            service.max_retries = max_retries
            started = perf_counter()
            answer, _ = service.ask(
                case["question"],
                document_ids=case["document_ids"],
                rerank_enabled=rerank_enabled,
                rerank_candidates=rerank_candidates,
                rerank_top_n=rerank_top_n,
                map_reduce_mode=map_reduce_mode,
            )
            total_latencies.append((perf_counter() - started) * 1000)
            answers.append(str(answer or ""))
            scores.append(score_run(case, list(service.last_trace or []), captured, k))
            for stage, values in captured["stage_latency_ms"].items():
                stage_latencies.setdefault(stage, []).extend(values)
            llm_calls_by_run.append(dict(captured["llm_calls"]))
        run_scores[case["id"]] = scores
        answer_agreements[case["id"]] = _answer_agreement(answers)

    report = build_report(
        cases,
        run_scores,
        answer_agreements,
        runs,
        k,
        high_variance_std,
        min_answer_agreement,
    )
    report["metrics"]["answer_exact_agreement"] = summarize(list(answer_agreements.values()))
    grade_precision: list[float | None] = []
    grade_recall: list[float | None] = []
    for repeat in range(runs):
        true_positive = false_positive = false_negative = 0
        for case in cases:
            accuracy = run_scores[case["id"]][repeat]["grading_accuracy"]
            if accuracy is None:
                continue
            expected = case["expected_sufficiency"] is True
            predicted = (accuracy == 1.0) if expected else (accuracy == 0.0)
            true_positive += int(predicted and expected)
            false_positive += int(predicted and not expected)
            false_negative += int(not predicted and expected)
        grade_precision.append(true_positive / (true_positive + false_positive) if true_positive + false_positive else None)
        grade_recall.append(true_positive / (true_positive + false_negative) if true_positive + false_negative else None)
    report["metrics"]["grading_precision"] = summarize(grade_precision)
    report["metrics"]["grading_recall"] = summarize(grade_recall)
    call_stages = sorted({stage for run in llm_calls_by_run for stage in run})
    report["latency_ms"] = {
        "end_to_end": _latency_summary(total_latencies),
        "stages": {
            stage: _latency_summary(values)
            for stage, values in sorted(stage_latencies.items())
        },
    }
    report["llm_calls"] = {
        "total_per_run": summarize([sum(run.values()) for run in llm_calls_by_run]),
        "by_stage_per_run": {
            stage: summarize([run.get(stage, 0) for run in llm_calls_by_run])
            for stage in call_stages
        },
    }
    report["settings"] = {
        "runs_per_case": runs,
        "k": k,
        "top_k": top_k,
        "rerank_enabled": rerank_enabled,
        "rerank_candidates": rerank_candidates,
        "rerank_top_n": rerank_top_n,
        "max_retries": max_retries,
        "map_reduce_mode": map_reduce_mode,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--map-reduce-mode", choices=("auto", "off", "force"), default="auto")
    parser.add_argument("--rerank-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high-variance-std", type=float, default=0.35)
    parser.add_argument("--min-answer-agreement", type=float, default=0.6)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f"reviewed case file not found: {args.data}; add reviewed cases with python -m eval.add_case")
    try:
        cases = load_cases(args.data)
        report = run_evaluation(
            cases,
            runs=args.runs,
            k=args.k,
            top_k=args.top_k,
            rerank_enabled=args.rerank_enabled,
            rerank_candidates=args.rerank_candidates,
            rerank_top_n=args.rerank_top_n,
            max_retries=args.max_retries,
            map_reduce_mode=args.map_reduce_mode,
            high_variance_std=args.high_variance_std,
            min_answer_agreement=args.min_answer_agreement,
        )
    except Exception as exc:  # Keep private questions and provider errors out of logs.
        parser.error(f"evaluation could not run ({type(exc).__name__}); no case content was logged")
    print(render_variance_report(report))
    print("\nJSON report:")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())