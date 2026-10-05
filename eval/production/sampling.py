"""Opt-in post-response answer evaluation; default sampling rate is zero."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import statistics
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from eval.answer_eval import score_answer, should_skip_factual_evaluation
from eval.production.sampling_store import DEFAULT_RESULTS_DB, SamplingResultStore


logger = logging.getLogger(__name__)


def should_sample_request(random_value: float | None = None) -> bool:
    try:
        rate = float(os.getenv("RAG_EVAL_SAMPLE_RATE", "0"))
    except ValueError:
        return False
    if not 0 < rate <= 1:
        return False
    draw = random.random() if random_value is None else random_value
    return draw < rate


def _trace_id(trace: Any) -> str:
    encoded = json.dumps(trace or [], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_sampled_message_evaluation(
    message: dict[str, Any],
    *,
    scorer=None,
    results_path=DEFAULT_RESULTS_DB,
) -> dict[str, Any] | None:
    """Score a completed persisted turn after SSE delivery; persist metrics only."""
    if message.get("status") != "complete":
        return None
    if should_skip_factual_evaluation(trace=message.get("trace")):
        return None
    message_id = str(message.get("message_id") or "")
    conversation_id = str(message.get("conversation_id") or "")
    if not message_id or not conversation_id:
        return None

    started = perf_counter()
    try:
        scoring_function = scorer or score_answer
        scores = scoring_function(
            str(message.get("question") or ""),
            str(message.get("answer") or ""),
            message.get("sources") or [],
        )
        judge_type = str(scores.get("judge_type") or "live")
        judge_scores = scores.get("judge_scores", {})
        numeric_scores = [
            float(judge_scores[name])
            for name in ("correctness", "completeness", "conciseness")
            if isinstance(judge_scores.get(name), (int, float))
        ]
        metrics = {
            "faithfulness": scores.get("faithfulness"),
            "citation_precision": scores.get("citation_precision"),
            "citation_recall": scores.get("citation_recall"),
            "judge_scores": judge_scores,
            "judge_score": statistics.fmean(numeric_scores) if numeric_scores else None,
            "judge_latency_ms": round((perf_counter() - started) * 1000, 2),
            "judge_call_count": 1,
        }
        stored = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "trace_id": _trace_id(message.get("trace")),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "metrics": metrics,
            "judge_type": judge_type,
        }
        SamplingResultStore(results_path).save(stored)
        return stored
    except Exception as exc:  # Never let eval failures affect a delivered answer.
        logger.warning("Post-response evaluation failed for message %s (%s)", message_id, type(exc).__name__)
        return None