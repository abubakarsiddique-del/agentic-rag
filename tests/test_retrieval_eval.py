import json
import math

import pytest

from eval.retrieval_eval import (
    DEFAULT_DATA,
    DEFAULT_RESULTS,
    evaluate_retrieval,
    load_dataset,
    load_results,
    main,
    passage_matches_evidence,
    score_ranking,
)


def test_metric_math_matches_hand_computed_rankings():
    evidence = [
        {"filename": "policy.txt", "page": 1, "anchor": "first answer-bearing fact"},
        {"filename": "policy.txt", "page": 2, "anchor": "second answer-bearing fact"},
    ]
    irrelevant = {"filename": "policy.txt", "page": 3, "text": "Unrelated paragraph."}
    first = {"filename": "policy.txt", "page": 1, "text": "FIRST answer-bearing fact."}
    second = {"filename": "policy.txt", "page": 2, "text": "Second answer-bearing fact."}

    result = score_ranking([irrelevant, first, second], evidence, k=3)
    rank_two_gain = 1 / math.log2(3)
    ideal_dcg = 1 + rank_two_gain
    actual_dcg = rank_two_gain + 1 / 2
    assert result == pytest.approx({
        "precision_at_k": 2 / 3,
        "recall_at_k": 1.0,
        "mrr": 0.5,
        "ndcg_at_k": actual_dcg / ideal_dcg,
    })

    improved = score_ranking([first, irrelevant, second], evidence, k=3)
    assert improved["precision_at_k"] == pytest.approx(2 / 3)
    assert improved["recall_at_k"] == 1.0
    assert improved["mrr"] == 1.0
    assert improved["ndcg_at_k"] == pytest.approx((1 + 0.5) / ideal_dcg)


def test_evidence_matching_survives_rechunking_and_ignores_chunk_ids():
    evidence = {
        "filename": "employee_handbook.test.txt",
        "page": 2,
        "anchor": "expire on March 31 of the following year",
    }
    rechunked_passage = {
        "passage_id": "completely-different-chunk-id",
        "filename": "/uploads/EMPLOYEE_HANDBOOK.test.txt",
        "page": "2",
        "text": "The rollover window ends when unused days expire on march 31 of the following year.",
    }
    assert passage_matches_evidence(rechunked_passage, evidence)


def test_evidence_requires_matching_source_page_and_answer_span():
    evidence = {"filename": "policy.pdf", "page": 4, "anchor": "renewal requires written approval"}
    assert not passage_matches_evidence(
        {"filename": "other.pdf", "page": 4, "text": "Renewal requires written approval."},
        evidence,
    )
    assert not passage_matches_evidence(
        {"filename": "policy.pdf", "page": 5, "text": "Renewal requires written approval."},
        evidence,
    )
    assert not passage_matches_evidence(
        {"filename": "policy.pdf", "page": 4, "text": "Renewal requires manager review."},
        evidence,
    )


def test_starter_dataset_is_complete_and_excludes_unanswerable_from_retrieval_means():
    dataset = load_dataset(DEFAULT_DATA)
    results = load_results(DEFAULT_RESULTS)
    report = evaluate_retrieval(dataset["cases"], results, k=5)

    assert report["case_count"] == 12
    assert report["answerable_case_count"] == 10
    assert report["unanswerable_case_count"] == 2
    assert report["judge"] == "none (deterministic retrieval metrics)"
    assert report["metrics"]["post_rerank"]["recall_at_k"] == 1.0
    assert report["metrics"]["post_rerank"]["precision_at_k"] > report["metrics"]["vector_candidates"]["precision_at_k"]
    assert report["by_type"]["global_summary"]["post_rerank"]["recall_at_k"] == 1.0
    assert report["per_case"][-1]["retrieval_metrics"] is None


def test_offline_cli_prints_metrics_without_provider_calls(capsys):
    assert main(["--dataset", str(DEFAULT_DATA), "--results", str(DEFAULT_RESULTS)]) == 0
    output = capsys.readouterr().out
    assert "Offline retrieval evaluation (12 cases, 10 answerable, k=5)" in output
    assert "Judge: none (deterministic; no model/provider calls)" in output
    assert "recall_at_k" in output


def test_results_must_cover_dataset_cases_exactly():
    cases = load_dataset(DEFAULT_DATA)["cases"]
    results = load_results(DEFAULT_RESULTS)[:-1]
    with pytest.raises(ValueError, match="cover exactly"):
        evaluate_retrieval(cases, results)


def test_results_loader_rejects_candidates_without_evidence_text(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": [{
        "case_id": "case",
        "vector_candidates": [{"filename": "policy.pdf", "page": 1}],
        "post_rerank": [],
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="page and text"):
        load_results(path)
