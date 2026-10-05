import copy

from eval.memory_eval import evaluate_memory_threads, load_threads


def test_memory_fixture_measures_resolution_on_and_off_without_leakage():
    report = evaluate_memory_threads(load_threads())
    assert report["resolution_accuracy"]["memory_on"] == 1.0
    assert report["resolution_accuracy"]["memory_off"] == 0.0
    assert report["memory_resolution_delta"] == 1.0
    assert report["leakage_pass"] is True


def test_memory_hint_as_claim_evidence_is_a_hard_failure():
    threads = copy.deepcopy(load_threads())
    threads[0]["memory_on"]["claims"][0]["evidence_ids"] = ["hint:pto-turn"]
    threads[0]["memory_on"]["claims"][0]["citation_ids"] = ["hint:pto-turn"]
    report = evaluate_memory_threads(threads)
    assert report["passed"] is False
    assert any("uses_memory_hint_as_evidence" in item["reason"] for item in report["leakage_failures"])
    assert any("cites_memory_hint" in item["reason"] for item in report["leakage_failures"])


def test_factual_claim_without_retrieved_provenance_fails_hard():
    threads = copy.deepcopy(load_threads())
    threads[1]["memory_on"]["retrieved_passage_ids"] = []
    report = evaluate_memory_threads(threads)
    assert report["passed"] is False
    assert any("evidence_not_retrieved" in item["reason"] for item in report["leakage_failures"])