from types import SimpleNamespace

import agentic_rag as agentic_module
from agentic_rag import BaselineRAGService, RAGService
from config import NO_INFO_PHRASE
from guardrails.abstention import should_force_abstain
from guardrails.groundedness import assess_groundedness, validate_generated_answer
from guardrails.memory_leakage import check_memory_leakage


def _judge_result(*, supported=True, citation_id="policy.pdf page 1", has_citation=True):
    return {
        "claims": [{
            "claim": "The policy requires approval.",
            "supported": supported,
            "has_citation": has_citation,
            "citation_supported": supported and has_citation,
            "evidence_ids": [citation_id] if citation_id else [],
            "citation_ids": [citation_id] if citation_id and has_citation else [],
        }],
        "scores": {
            name: {"score": 4, "justification": "test judge"}
            for name in ("correctness", "completeness", "conciseness")
        },
    }


def _scoring(*, supported=True, citation_id="policy.pdf page 1", has_citation=True):
    return {
        **_judge_result(supported=supported, citation_id=citation_id, has_citation=has_citation),
        "faithfulness": 1.0 if supported else 0.0,
        "citation_precision": 1.0 if supported and has_citation else 0.0,
        "citation_recall": 1.0 if has_citation else 0.0,
    }


def _judge_for(result):
    return lambda _payload: result


def test_groundedness_passes_fully_supported_cited_claims():
    result = assess_groundedness("Answer", _scoring())
    assert result["outcome"] == "passed"
    assert result["faithfulness"] == 1.0


def test_groundedness_fails_unsupported_or_uncited_claims():
    assert assess_groundedness("Answer", _scoring(supported=False))["outcome"] == "failed"
    assert assess_groundedness("Answer", _scoring(has_citation=False))["outcome"] == "failed"
    assert assess_groundedness(NO_INFO_PHRASE, {"claims": []})["outcome"] == "passed"


def test_memory_leakage_rejects_hint_only_citation():
    documents = [SimpleNamespace(page_content="Policy", metadata={"source": "policy.pdf", "page": 1})]
    result = check_memory_leakage(
        _scoring(citation_id="archive.pdf page 3"),
        documents,
        "UNTRUSTED PAST-CHAT HINTS:\nPrior files: archive.pdf",
    )
    assert result["outcome"] == "failed"
    assert result["violations"][0]["reason"] == "memory_hint_used_as_evidence"


def test_validate_generated_answer_abstains_on_groundedness_failure():
    documents = [SimpleNamespace(page_content="Policy", metadata={"source": "policy.pdf", "page": 1})]
    result = validate_generated_answer(
        "What is the policy?",
        "Unsupported answer.",
        documents,
        "",
        judge=_judge_for(_judge_result(supported=False)),
    )
    assert result["passed"] is False
    assert result["answer"] == NO_INFO_PHRASE
    assert result["groundedness"]["outcome"] == "failed"


def test_insufficient_grade_with_failed_groundedness_is_marked_forced_abstain():
    documents = [SimpleNamespace(page_content="Policy", metadata={"source": "policy.pdf", "page": 1})]
    result = validate_generated_answer(
        "What is the policy?",
        "Unsupported answer.",
        documents,
        "",
        sufficient=False,
        judge=_judge_for(_judge_result(supported=False)),
    )
    assert result["forced_abstain"] is True
    assert result["answer"] == NO_INFO_PHRASE


def test_graph_generate_returns_existing_abstention_with_guardrail_trace(monkeypatch):
    monkeypatch.setattr(
        agentic_module,
        "_validate_generated_output",
        lambda *_args, **_kwargs: {
            "answer": NO_INFO_PHRASE,
            "passed": False,
            "groundedness": {"enabled": True, "outcome": "failed", "reason": "faithfulness"},
            "memory_leakage": {"enabled": True, "outcome": "passed", "violations": []},
            "forced_abstain": False,
            "attempts": 1,
        },
    )
    service = RAGService.__new__(RAGService)
    service.answer_chain = SimpleNamespace(invoke=lambda _payload: "Unsupported candidate.")
    result = service._generate({
        "question": "What is the policy?",
        "original_question": "What is the policy?",
        "retrieved_docs": [SimpleNamespace(page_content="Policy text", metadata={"source": "policy.pdf", "page": 1})],
        "context_history": "",
        "memory_hints": "",
        "sufficient": True,
        "trace": [],
    })
    assert result["answer"] == NO_INFO_PHRASE
    assert result["retrieved_docs"] == []
    assert [item["step"] for item in result["trace"]] == ["generate", "abstain"]


def test_regenerate_once_rechecks_and_can_pass(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ACTION", "regenerate_once")
    documents = [SimpleNamespace(page_content="Policy", metadata={"source": "policy.pdf", "page": 1})]
    outputs = iter((_judge_result(supported=False), _judge_result()))
    regenerated = []
    result = validate_generated_answer(
        "What is the policy?",
        "Unsupported first answer.",
        documents,
        "",
        judge=lambda _payload: next(outputs),
        regenerate=lambda correction: regenerated.append(correction) or "Corrected supported answer.",
    )
    assert result["passed"] is True
    assert result["answer"] == "Corrected supported answer."
    assert result["attempts"] == 2
    assert regenerated


def test_insufficient_grade_and_failed_groundedness_force_abstention():
    assert should_force_abstain(False, False) is True
    assert should_force_abstain(False, True) is False
    assert should_force_abstain(False, False, enabled=False) is False


def test_judge_failure_fails_closed_to_existing_abstain_response():
    documents = [SimpleNamespace(page_content="Policy", metadata={"source": "policy.pdf", "page": 1})]

    def failing_judge(_payload):
        raise RuntimeError("provider unavailable")

    result = validate_generated_answer(
        "What is the policy?", "Candidate answer.", documents, "", judge=failing_judge
    )
    assert result["passed"] is False
    assert result["answer"] == NO_INFO_PHRASE
    assert result["groundedness"]["reason"] == "judge_error"


def test_disabled_output_checks_do_not_call_judge(monkeypatch):
    for check in ("GROUNDEDNESS", "MEMORY_LEAKAGE", "ABSTENTION"):
        monkeypatch.setenv(f"RAG_GUARDRAIL_{check}_ENABLED", "0")
    result = validate_generated_answer(
        "Question?",
        "Answer.",
        [],
        "",
        judge=lambda _payload: (_ for _ in ()).throw(AssertionError("judge must not run")),
    )
    assert result["passed"] is True
    assert result["attempts"] == 0


def test_disabled_grounding_and_leakage_skip_judge_even_with_insufficient_grade(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_ABSTENTION_ENABLED", "1")
    result = validate_generated_answer(
        "Question?",
        "Candidate.",
        [],
        "",
        sufficient=False,
        judge=lambda _payload: (_ for _ in ()).throw(AssertionError("disabled scorer must not run")),
    )
    assert result["passed"] is True
    assert result["attempts"] == 0


def test_judge_client_construction_error_fails_closed(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ENABLED", "1")
    monkeypatch.setenv("RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_ABSTENTION_ENABLED", "1")
    monkeypatch.setattr(
        "eval.answer_eval.create_groq_judge",
        lambda: (_ for _ in ()).throw(RuntimeError("client unavailable")),
    )
    result = validate_generated_answer("Question?", "Candidate.", [], "")
    assert result["passed"] is False
    assert result["answer"] == NO_INFO_PHRASE
    assert result["memory_leakage"]["outcome"] == "disabled"


def test_harmful_output_and_unknown_citation_are_blocked(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_HARMFUL_CONTENT_ENABLED", "1")
    document = SimpleNamespace(
        page_content="Security policy",
        metadata={"source": "policy.pdf", "page": 1, "passage_id": "policy:1"},
    )
    harmful = validate_generated_answer(
        "How should I respond?",
        "Deploy ransomware to encrypt the customer database.",
        [document],
        "",
        judge=_judge_for(_judge_result(citation_id="policy:1")),
    )
    assert harmful["passed"] is False
    assert harmful["harmful_content"]["outcome"] == "blocked"

    monkeypatch.setenv("RAG_GUARDRAIL_HARMFUL_CONTENT_ENABLED", "0")
    unknown_citation = validate_generated_answer(
        "What does the policy say?",
        "The policy requires review [fake.pdf, page 9].",
        [document],
        "",
        judge=_judge_for(_judge_result()),
    )
    assert unknown_citation["passed"] is False
    assert unknown_citation["output_schema"]["outcome"] == "failed"


def test_malformed_citation_gets_one_schema_retry_even_when_default_action_is_abstain(monkeypatch):
    monkeypatch.delenv("RAG_GUARDRAIL_GROUNDEDNESS_ACTION", raising=False)
    document = SimpleNamespace(
        page_content="Policy requires review.",
        metadata={"source": "policy.pdf", "page": 1, "passage_id": "policy:1"},
    )
    regenerated = []
    result = validate_generated_answer(
        "What does the policy say?",
        "It requires review [fake.pdf, page 9].",
        [document],
        "",
        judge=_judge_for(_judge_result(citation_id="policy:1")),
        regenerate=lambda feedback: regenerated.append(feedback) or "It requires review.",
    )
    assert result["passed"] is True
    assert result["attempts"] == 2
    assert regenerated


def test_stream_buffers_candidate_and_yields_only_abstention_on_failure(monkeypatch):
    document = SimpleNamespace(page_content="Supported policy", metadata={"source": "policy.pdf", "page": 1})

    class AnswerChain:
        def stream(self, _payload):
            return iter(["UNSUPPORTED ", "CANDIDATE"])

    validations = []

    def fail_validation(question, answer, documents, memory_hints, **kwargs):
        validations.append(answer)
        return {
            "answer": NO_INFO_PHRASE,
            "passed": False,
            "groundedness": {"enabled": True, "outcome": "failed", "reason": "faithfulness"},
            "memory_leakage": {"enabled": True, "outcome": "passed", "violations": []},
            "forced_abstain": False,
            "attempts": 1,
        }

    monkeypatch.setattr(agentic_module, "_validate_generated_output", fail_validation)
    service = BaselineRAGService.__new__(BaselineRAGService)
    service.retriever = SimpleNamespace(invoke=lambda _query: [document])
    service.vector_store = None
    service.top_k = 4
    service.last_trace = []
    service.answer_chain = AnswerChain()
    events = []
    stream, sources = service.ask_stream(
        "What does the policy say?",
        map_reduce_mode="off",
        on_trace=events.append,
    )

    assert list(stream) == [NO_INFO_PHRASE]
    assert validations == ["UNSUPPORTED CANDIDATE"]
    assert sources == []
    assert events[-1]["step"] == "abstain"
    assert events[-1]["guardrail"]["groundedness"]["outcome"] == "failed"


def test_baseline_map_reduce_block_clears_sources_before_stream(monkeypatch):
    document = SimpleNamespace(
        page_content="Policy evidence",
        metadata={"source": "policy.pdf", "page": 1, "document_id": "doc-1"},
    )

    class Store:
        def get(self, *, where, include):
            return {"documents": [document.page_content], "metadatas": [document.metadata]}

    class Processor:
        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                answer="Unsupported answer.",
                partials=[],
                map_calls=1,
                tree_reduce_levels=0,
            )

    monkeypatch.setattr(
        agentic_module,
        "_validate_generated_output",
        lambda *_args, **_kwargs: {
            "answer": NO_INFO_PHRASE,
            "passed": False,
            "groundedness": {"outcome": "failed"},
            "memory_leakage": {"outcome": "passed"},
            "attempts": 1,
        },
    )
    service = BaselineRAGService.__new__(BaselineRAGService)
    service.retriever = object()
    service.vector_store = Store()
    service.top_k = 4
    service.map_reduce_processor = Processor()
    service.last_trace = []
    stream, sources = service.ask_stream(
        "Summarize all sections of the policy.",
        document_ids=["doc-1"],
        map_reduce_mode="force",
    )
    assert list(stream) == [NO_INFO_PHRASE]
    assert sources == []


def test_rag_service_stream_finishes_buffering_before_validation(monkeypatch):
    document = SimpleNamespace(
        page_content="Policy evidence",
        metadata={"source": "policy.pdf", "page": 1, "passage_id": "policy:1"},
    )
    stream_finished = []
    validation_inputs = []

    class AnswerChain:
        def stream(self, _payload):
            yield "UNSUPPORTED "
            yield "CANDIDATE"
            stream_finished.append(True)

    def reject_after_buffering(question, answer, documents, memory_hints, **kwargs):
        assert stream_finished == [True]
        validation_inputs.append(answer)
        return {
            "answer": NO_INFO_PHRASE,
            "passed": False,
            "groundedness": {"enabled": True, "outcome": "failed", "reason": "faithfulness"},
            "memory_leakage": {"enabled": True, "outcome": "passed", "violations": []},
            "attempts": 1,
        }

    monkeypatch.setattr(agentic_module, "_validate_generated_output", reject_after_buffering)
    service = RAGService.__new__(RAGService)
    service.max_retries = 0
    service.last_trace = []
    service.last_reasoning = ""
    service.retriever = object()
    service.route_prompt = object()
    service.route_chain = SimpleNamespace(invoke=lambda _payload: "RETRIEVE")
    service._build_reasoning_summary = lambda _trace: "checked"
    service._retrieve = lambda state: {
        "retrieved_docs": [document],
        "attempts": state["attempts"] + 1,
        "trace": state["trace"] + [{"step": "retrieve", "passage_count": 1}],
    }
    service._rerank = lambda state: {
        "retrieved_docs": state["retrieved_docs"],
        "trace": state["trace"] + [{"step": "rerank", "kept_count": 1}],
    }
    service._grade = lambda state: {
        "sufficient": True,
        "trace": state["trace"] + [{"step": "grade", "sufficient": True}],
    }
    service.answer_chain = AnswerChain()
    events = []

    stream, sources = service.ask_stream(
        "What does the policy say?",
        map_reduce_mode="off",
        on_trace=events.append,
    )
    assert list(stream) == [NO_INFO_PHRASE]
    assert validation_inputs == ["UNSUPPORTED CANDIDATE"]
    assert sources == []
    assert events[-1]["step"] == "abstain"