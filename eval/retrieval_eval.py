"""Evaluate ranked retrieval artifacts against stable, text-anchored evidence labels."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).parent
DEFAULT_DATA = EVAL_DIR / "data" / "retrieval_cases_v1.json"
DEFAULT_RESULTS = EVAL_DIR / "data" / "retrieval_results_v1.example.json"
STAGES = ("vector_candidates", "post_rerank")
METRICS = ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _normalize_source(value: Any) -> str:
    source = Path(str(value or "")).name
    source = re.sub(r"\.[^.]+$", "", source)
    return normalize_text(source)


def passage_matches_evidence(
    passage: dict[str, Any],
    evidence: dict[str, Any],
) -> bool:
    """Match by stable source/page/answer-span, never by generated chunk ID."""
    passage_source = passage.get("filename") or passage.get("source") or passage.get("document")
    evidence_source = evidence.get("filename") or evidence.get("source") or evidence.get("document")
    if not passage_source or _normalize_source(passage_source) != _normalize_source(evidence_source):
        return False
    if str(passage.get("page", "")) != str(evidence.get("page", "")):
        return False
    anchor = normalize_text(str(evidence.get("anchor") or ""))
    content = normalize_text(str(passage.get("text") or passage.get("page_content") or ""))
    return bool(anchor) and anchor in content


def _matched_evidence(
    passage: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> set[int]:
    return {
        index
        for index, item in enumerate(evidence)
        if passage_matches_evidence(passage, item)
    }


def score_ranking(
    ranking: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    k: int,
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not evidence:
        raise ValueError("retrieval ranking metrics require at least one evidence label")

    ranked = ranking[:k]
    covered: set[int] = set()
    relevant_count = 0
    reciprocal_rank = 0.0
    dcg = 0.0
    for rank, passage in enumerate(ranked, start=1):
        matched = _matched_evidence(passage, evidence)
        is_relevant = bool(matched)
        relevant_count += int(is_relevant)
        covered.update(matched)
        if is_relevant and reciprocal_rank == 0.0:
            reciprocal_rank = 1.0 / rank
        if is_relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant = min(len(evidence), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    precision_denominator = min(k, len(ranking))
    return {
        "precision_at_k": relevant_count / precision_denominator if precision_denominator else 0.0,
        "recall_at_k": len(covered) / len(evidence),
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_dataset(path: Path = DEFAULT_DATA) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("retrieval dataset must be an object with schema_version 1")
    corpus = payload.get("corpus")
    cases = payload.get("cases")
    if not isinstance(corpus, list) or not corpus or not isinstance(cases, list) or not cases:
        raise ValueError("retrieval dataset requires non-empty corpus and cases lists")

    known_sources: set[str] = set()
    for document in corpus:
        filename = str(document.get("filename") or "")
        pages = document.get("pages")
        if not filename or not isinstance(pages, list) or not pages:
            raise ValueError("each corpus document needs a filename and non-empty pages list")
        source_key = _normalize_source(filename)
        if source_key in known_sources:
            raise ValueError(f"duplicate corpus filename: {filename}")
        known_sources.add(source_key)
        for page in pages:
            if "page" not in page or not str(page.get("text") or "").strip():
                raise ValueError(f"each page in {filename} needs a page number and text")

    seen: set[str] = set()
    allowed_types = {
        "single_fact", "multi_part", "comparison", "global_summary",
        "follow_up", "unanswerable",
    }
    for case in cases:
        required = ("id", "type", "question", "expected_route", "answerable", "evidence")
        if not all(key in case for key in required):
            raise ValueError("each retrieval case needs id, type, question, route, answerability, and evidence")
        case_id = str(case["id"])
        if case_id in seen:
            raise ValueError(f"duplicate retrieval case ID: {case_id}")
        seen.add(case_id)
        if case["type"] not in allowed_types:
            raise ValueError(f"unsupported case type: {case['type']}")
        if case["expected_route"] not in {"local", "broad"}:
            raise ValueError(f"{case_id}: expected_route must be local or broad")
        if not isinstance(case["answerable"], bool) or not isinstance(case["evidence"], list):
            raise ValueError(f"{case_id}: answerable must be boolean and evidence must be a list")
        if case["answerable"] != bool(case["evidence"]):
            raise ValueError(f"{case_id}: answerable cases need evidence and unanswerable cases must not")
        for evidence in case["evidence"]:
            if not all(key in evidence for key in ("filename", "page", "anchor")):
                raise ValueError(f"{case_id}: each evidence label needs filename, page, and anchor")
            if _normalize_source(evidence["filename"]) not in known_sources:
                raise ValueError(f"{case_id}: evidence references an unknown corpus document")
            if not str(evidence["anchor"]).strip():
                raise ValueError(f"{case_id}: evidence anchor must not be empty")
    return payload


def load_results(path: Path = DEFAULT_RESULTS) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("retrieval results file must contain a results list")
    seen: set[str] = set()
    for result in results:
        case_id = str(result.get("case_id") or "")
        if not case_id or case_id in seen:
            raise ValueError("each retrieval result needs a unique case_id")
        seen.add(case_id)
        for stage in STAGES:
            ranking = result.get(stage)
            if not isinstance(ranking, list):
                raise ValueError(f"{case_id}: {stage} must be a list")
            for passage in ranking:
                if not isinstance(passage, dict) or not (
                    passage.get("filename") or passage.get("source") or passage.get("document")
                ):
                    raise ValueError(f"{case_id}: every candidate needs a source filename")
                if "page" not in passage or not (
                    passage.get("text") or passage.get("page_content")
                ):
                    raise ValueError(f"{case_id}: every candidate needs page and text")
    return results


def evaluate_retrieval(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    if k < 1:
        raise ValueError("k must be at least 1")
    case_by_id = {str(case["id"]): case for case in cases}
    results_by_id = {str(result["case_id"]): result for result in results}
    if set(case_by_id) != set(results_by_id):
        raise ValueError("retrieval results must cover exactly the dataset case IDs")

    per_case: list[dict[str, Any]] = []
    collected: dict[str, dict[str, list[float]]] = {
        stage: {metric: [] for metric in METRICS} for stage in STAGES
    }
    type_metrics: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {stage: {metric: [] for metric in METRICS} for stage in STAGES}
    )
    route_metrics: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {stage: {metric: [] for metric in METRICS} for stage in STAGES}
    )
    lift_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    answerable_count = 0
    unanswerable_count = 0

    for case in cases:
        case_id = str(case["id"])
        result = results_by_id[case_id]
        evidence = case["evidence"]
        item: dict[str, Any] = {
            "case_id": case_id,
            "type": case["type"],
            "expected_route": case["expected_route"],
            "answerable": case["answerable"],
        }
        if not case["answerable"]:
            unanswerable_count += 1
            item["retrieval_metrics"] = None
            per_case.append(item)
            continue

        answerable_count += 1
        item["retrieval_metrics"] = {}
        for stage in STAGES:
            scores = score_ranking(result[stage], evidence, k=k)
            item["retrieval_metrics"][stage] = scores
            for metric, value in scores.items():
                collected[stage][metric].append(value)
                type_metrics[case["type"]][stage][metric].append(value)
                route_metrics[case["expected_route"]][stage][metric].append(value)

        item["rerank_lift"] = {}
        for metric in METRICS:
            delta = (
                item["retrieval_metrics"]["post_rerank"][metric]
                - item["retrieval_metrics"]["vector_candidates"][metric]
            )
            item["rerank_lift"][metric] = delta
            lift_values[metric].append(delta)
        per_case.append(item)

    def summarize_groups(groups: dict[str, dict[str, dict[str, list[float]]]]) -> dict[str, Any]:
        return {
            group: {
                stage: {metric: _mean(values) for metric, values in metrics.items()}
                for stage, metrics in stages.items()
            }
            for group, stages in sorted(groups.items())
        }

    return {
        "case_count": len(cases),
        "answerable_case_count": answerable_count,
        "unanswerable_case_count": unanswerable_count,
        "k": k,
        "metric_definitions": {
            "precision_at_k": "Relevant returned passages / min(k, number returned).",
            "recall_at_k": "Distinct gold evidence anchors matched by the top k / all gold anchors.",
            "mrr": "Reciprocal rank of the first passage matching any gold evidence anchor, within k.",
            "ndcg_at_k": "Binary-relevance nDCG using matched evidence passages, normalized by min(gold anchors, k).",
            "rerank_lift": "Post-rerank metric minus vector-candidate metric for the same case.",
        },
        "metrics": {
            stage: {metric: _mean(values) for metric, values in metrics.items()}
            for stage, metrics in collected.items()
        },
        "rerank_lift": {metric: _mean(values) for metric, values in lift_values.items()},
        "by_type": summarize_groups(type_metrics),
        "by_route": summarize_groups(route_metrics),
        "per_case": per_case,
        "judge": "none (deterministic retrieval metrics)",
        "source": "offline ranked-passage artifact; no model/provider calls",
    }


def render_report(report: dict[str, Any]) -> str:
    def format_metric(value: float | None, *, width: int) -> str:
        return f"{value:>{width}.3f}" if value is not None else f"{'n/a':>{width}}"

    lines = [
        f"Offline retrieval evaluation ({report['case_count']} cases, "
        f"{report['answerable_case_count']} answerable, k={report['k']})",
        "Judge: none (deterministic; no model/provider calls)",
        "",
        f"{'Metric':<18} {'Vector candidates':>18} {'Post-rerank':>14} {'Rerank lift':>13}",
        "-" * 67,
    ]
    for metric in METRICS:
        vector = report["metrics"]["vector_candidates"][metric]
        reranked = report["metrics"]["post_rerank"][metric]
        lift = report["rerank_lift"][metric]
        lines.append(
            f"{metric:<18} {format_metric(vector, width=18)} "
            f"{format_metric(reranked, width=14)} "
            f"{format_metric(lift, width=13)}"
        )
    lines.append(f"Unanswerable cases excluded from retrieval quality: {report['unanswerable_case_count']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        results = load_results(args.results)
        report = evaluate_retrieval(dataset["cases"], results, k=args.k)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"retrieval evaluation failed ({type(exc).__name__}): {exc}")
    print(render_report(report))
    print("\nJSON report:")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
