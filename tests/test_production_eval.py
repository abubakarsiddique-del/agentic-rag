import pytest

from eval.add_case import append_case, validate_case
from eval.production.variance_eval import run_evaluation, run_repeated_cases, score_run, summarize


def test_case_authoring_validates_answerability_and_strata(tmp_path):
    case = {
        "id": "case-1",
        "question": "What is the policy?",
        "conversation_id": "conversation-1",
        "document_ids": ["document-1"],
        "gold_passages": ["document-1:1"],
        "expected_route": "local",
        "expected_sufficiency": True,
        "strata": {
            "scope": "narrow",
            "document_count": "single",
            "answerability": "answerable",
            "language": "en",
            "document_profile": "structured",
        },
    }
    path = tmp_path / "cases.json"
    append_case(path, case)
    assert validate_case(case) is None
    with pytest.raises(ValueError, match="already exists"):
        append_case(path, case)


def test_unanswerable_case_cannot_claim_a_gold_passage():
    case = {
        "id": "case-2",
        "question": "Unknown?",
        "conversation_id": "conversation-1",
        "document_ids": ["document-1"],
        "gold_passages": ["document-1:1"],
        "expected_route": "local",
        "expected_sufficiency": False,
        "strata": {},
    }
    with pytest.raises(ValueError, match="must not list a gold passage"):
        validate_case(case)


def test_repeated_run_summary_reports_sample_variance():
    result = summarize([0, 1, 1])
    assert result["mean"] == pytest.approx(2 / 3)
    assert result["std_dev"] == pytest.approx(0.577350269)
    assert result["min"] == 0
    assert result["max"] == 1


def test_shared_repeat_helper_scores_same_cases_for_each_run():
    calls = []
    results = run_repeated_cases(["a", "b"], 3, lambda case: calls.append(case) or case)
    assert results == [["a", "b"], ["a", "b"], ["a", "b"]]
    assert calls == ["a", "b"] * 3


def test_run_scoring_uses_captured_candidates_without_trace_changes():
    case = {
        "expected_route": "local",
        "expected_sufficiency": True,
        "gold_passages": ["doc:p2"],
    }
    trace = [
        {"step": "route", "scope": "local"},
        {"step": "grade", "sufficient": True},
        {"step": "generate"},
    ]
    captured = {
        "retrievals": [{"passage_ids": ["doc:p1", "doc:p2"]}],
        "reranks": [{"candidate_ids": ["doc:p1", "doc:p2"], "passage_ids": ["doc:p2", "doc:p1"]}],
    }
    result = score_run(case, trace, captured, k=5)
    assert result["route_accuracy"] == 1
    assert result["retrieval_mrr"] == 0.5
    assert result["rerank_mrr_delta"] == 0.5
    assert result["grading_accuracy"] == 1


def test_repeated_eval_aggregates_variance_without_live_model_calls():
    case = {
        "id": "variable-case",
        "question": "Question text is not reported",
        "conversation_id": "conversation-1",
        "document_ids": ["document-1"],
        "gold_passages": ["document-1:p1"],
        "expected_route": "local",
        "expected_sufficiency": True,
        "strata": {"scope": "narrow"},
    }

    class FakeService:
        def __init__(self, captured):
            self.calls = 0
            self.last_trace = []
            self.captured = captured

        def ask(self, question, **kwargs):
            self.calls += 1
            self.captured["stage_latency_ms"].setdefault("route", []).append(float(self.calls))
            self.captured["llm_calls"]["route"] = 1
            observed_scope = "local" if self.calls == 1 else "global"
            self.last_trace = [
                {"step": "route", "scope": observed_scope},
                {"step": "grade", "sufficient": True},
                {"step": "generate"},
            ]
            return ("stable answer" if self.calls == 1 else "different answer"), []

    def fake_factory(conversation_id, top_k):
        captured = {"retrievals": [], "reranks": [], "stage_latency_ms": {}, "llm_calls": {}}
        return FakeService(captured), captured

    report = run_evaluation(
        [case],
        runs=2,
        service_factory=fake_factory,
    )
    assert report["metrics"]["route_accuracy"]["mean"] == 0.5
    assert report["metrics"]["route_accuracy"]["std_dev"] == pytest.approx(0.70710678)
    assert report["metrics"]["answer_exact_agreement"]["mean"] == 0.5
    assert report["high_variance_cases"][0]["case_id"] == "variable-case"
    assert report["latency_ms"]["stages"]["route"]["p50_ms"] == 1.5
    assert report["latency_ms"]["stages"]["route"]["p95_ms"] == 1.95
    assert report["llm_calls"]["by_stage_per_run"]["route"]["mean"] == 1.0
    assert report["settings"]["map_reduce_mode"] == "auto"