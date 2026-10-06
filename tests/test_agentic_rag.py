import os
import sqlite3
import warnings

from types import SimpleNamespace

import agentic_rag as agentic_module
from agentic_rag import BaselineRAGService, RAGService
from app_helpers import (
    SourcePreview,
    format_conversation_export,
    group_sources_by_document,
    normalize_document_name,
    serialize_sources,
    summarize_answer_effort,
)
from persistence.models import DocumentRecord, IndexedDocument, MessageRecord
from persistence.store import SQLiteConversationStore
from map_reduce import MapCallCapExceeded, MapPartial, MapReduceResult, SourceRef
from backend.app import derive_auto_title, prune_default_titles


def test_derive_auto_title_uses_answer_and_falls_back_to_question():
    question = "  Compare the Australian Constitution with India's Constitution in detail, please.  "
    answer = "The Australian Constitution is a codified document with federal and state powers, while India's Constitution is a longer, more detailed framework."
    assert derive_auto_title(question, answer, maximum=32) == "The Australian Constitution is a"
    assert derive_auto_title(question, "", maximum=50).startswith("Compare the Australian Constitution")


def test_rag_service_initialization_defers_graph_and_checkpoint_import(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(agentic_module, "get_embedding_model", lambda: object())
    monkeypatch.setattr(agentic_module, "get_logger", lambda *_args: object())
    monkeypatch.setattr(agentic_module, "get_tracer", lambda *_args: object())
    monkeypatch.setattr(
        agentic_module,
        "get_langfuse_handler",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(RAGService, "_attach_existing_index", lambda _self: None)

    class FakeRunnable:
        def __or__(self, _other):
            return self

    class FakePrompt:
        def __or__(self, _other):
            return FakeRunnable()

    class FakeLLM:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks")

        def with_structured_output(self, _schema):
            return FakeRunnable()

    monkeypatch.setattr(agentic_module, "ChatGroq", FakeLLM)
    monkeypatch.setattr(
        agentic_module.ChatPromptTemplate,
        "from_messages",
        staticmethod(lambda _messages: FakePrompt()),
    )

    def unexpected_graph_build(_self):
        raise AssertionError("graph construction must be deferred")

    monkeypatch.setattr(RAGService, "_build_graph", unexpected_graph_build)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        service = RAGService()

    assert service.graph is None
    assert not any(
        "allowed_objects" in str(warning.message)
        for warning in caught
    )


def test_route_question_routes_direct_for_greetings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    service.route_chain = type(
        "DummyRouteChain",
        (),
        {"invoke": lambda self, payload: "DIRECT"},
    )()

    result = service._route_question({"question": "hello", "trace": []})

    assert result["route"] == "direct"
    assert result["trace"][-1]["step"] == "route"
    assert "direct" in result["trace"][-1]["detail"]


def test_grade_decision_uses_retry_and_generate_logic(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    service.max_retries = 2

    assert service._grade_decision({"sufficient": True, "attempts": 1}) == "generate"
    # After retry exhaustion we now abstain instead of forcing a weak generation
    assert service._grade_decision({"sufficient": False, "attempts": 3}) == "abstain"
    assert service._grade_decision({"sufficient": False, "attempts": 1}) == "rewrite"


def test_grade_decision_abstains_when_injection_guard_removed_every_passage():
    service = RAGService.__new__(RAGService)
    service.max_retries = 2
    assert service._grade_decision({
        "sufficient": True,
        "attempts": 1,
        "passage_injection_blocked": True,
    }) == "abstain"


def test_retrieve_respects_document_scope(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    matching = SimpleNamespace(metadata={"source": "report.pdf"})
    other = SimpleNamespace(metadata={"source": "notes.txt"})
    service.retriever = type(
        "DummyRetriever",
        (),
        {"invoke": lambda self, query: [matching, other]},
    )()

    scoped = service._retrieve(
        {
            "search_query": "question",
            "document_scope": ["report.pdf"],
            "attempts": 0,
            "trace": [],
        }
    )
    unscoped = service._retrieve(
        {
            "search_query": "question",
            "document_scope": [],
            "attempts": 0,
            "trace": [],
        }
    )

    assert scoped["retrieved_docs"] == [matching]
    assert unscoped["retrieved_docs"] == [matching, other]


def test_retrieve_balances_passages_across_selected_documents(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    first = SimpleNamespace(metadata={"document_id": "doc-1", "source": "first.pdf"})
    second = SimpleNamespace(metadata={"document_id": "doc-2", "source": "second.pdf"})

    class DominatedSearchStore:
        def similarity_search(self, query, *, k, filter):
            if filter == {"document_id": {"$in": ["doc-1", "doc-2"]}}:
                return [first] * k
            document = filter["document_id"]
            return [first if document == "doc-1" else second] * k

    service.top_k = 4
    service.vector_store = DominatedSearchStore()
    service.retriever = object()

    result = service._retrieve({
        "search_query": "compare the uploaded files",
        "document_ids": ["doc-1", "doc-2"],
        "document_sources": ["first.pdf", "second.pdf"],
        "document_scope": [],
        "attempts": 0,
        "trace": [],
    })

    assert {doc.metadata["document_id"] for doc in result["retrieved_docs"]} == {"doc-1", "doc-2"}

    focused = service._retrieve({
        "search_query": "What are the key takeaways from rag db1?",
        "question": "What are the key takeaways from rag db1?",
        "original_question": "What are the key takeaways from rag db1?",
        "document_ids": ["doc-1", "doc-2"],
        "document_sources": ["rag db1.pdf", "rag db3.pdf"],
        "document_scope": [],
        "attempts": 0,
        "trace": [],
    })

    assert {doc.metadata["document_id"] for doc in focused["retrieved_docs"]} == {"doc-1"}


def test_route_path_selects_map_reduce_for_auto_and_force_modes():
    assert RAGService._route_path({
        "route": "retrieve",
        "scope": "global",
        "map_reduce_mode": "auto",
    }) == "map_reduce"
    assert RAGService._route_path({
        "route": "retrieve",
        "scope": "global",
        "map_reduce_mode": "off",
    }) == "retrieve"
    assert RAGService._route_path({
        "route": "retrieve",
        "scope": "local",
        "map_reduce_mode": "force",
    }) == "map_reduce"
    assert RAGService._route_path({
        "route": "direct",
        "scope": "global",
        "map_reduce_mode": "force",
    }) == "direct"


def test_map_reduce_reads_selected_documents_and_preserves_citation_sources(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED", "0")
    document = SimpleNamespace(
        page_content="Supported facts",
        metadata={"source": "selected.pdf", "document_name": "selected.pdf", "page": 2, "document_id": "doc-1"},
    )

    class Store:
        def get(self, *, where, include):
            assert where == {"document_id": "doc-1"}
            assert include == ["documents", "metadatas"]
            return {"documents": [document.page_content], "metadatas": [document.metadata]}

    class Processor:
        def run(self, question, documents, *, cancel_event, on_progress):
            assert len(documents) == 1
            return MapReduceResult(
                answer="Supported answer [Source 1: Page 2]",
                partials=[MapPartial("Supported facts", (SourceRef(1, "selected.pdf", "2"),))],
                map_calls=1,
                tree_reduce_levels=0,
            )

    service = RAGService.__new__(RAGService)
    service.vector_store = Store()
    service.map_reduce_processor = Processor()
    result = service._map_reduce({
        "question": "summarize the selected file",
        "original_question": "summarize the selected file",
        "document_ids": ["doc-1", "doc-2"],
        "document_sources": ["selected.pdf", "other.pdf"],
        "document_scope": [],
        "scope": "global",
        "trace": [],
    })

    assert result["answer"].endswith("[Source 1: Page 2]")
    assert len(result["retrieved_docs"]) == 1
    assert result["trace"][-1]["fallback"] is False


def test_map_reduce_call_cap_records_fallback_for_standard_retrieval():
    class Store:
        def get(self, *, where, include):
            return {"documents": ["passage"], "metadatas": [{"source": "doc.pdf", "page": 1}]}

    class Processor:
        def run(self, *args, **kwargs):
            raise MapCallCapExceeded("Map-reduce needs 31 calls; limit is 30")

    service = RAGService.__new__(RAGService)
    service.vector_store = Store()
    service.map_reduce_processor = Processor()
    result = service._map_reduce({
        "question": "summarize all sections",
        "original_question": "summarize all sections",
        "document_ids": [],
        "document_sources": [],
        "document_scope": [],
        "scope": "global",
        "trace": [],
    })

    assert result["map_reduce_fallback"] is True
    assert result["trace"][-1]["reason"] == "map_call_cap"


def test_agentic_stream_uses_auto_map_reduce_for_global_question():
    service = RAGService.__new__(RAGService)
    service.max_retries = 2
    service.route_prompt = object()
    service.route_chain = type("Route", (), {"invoke": lambda self, payload: "RETRIEVE"})()
    service.last_trace = []
    service._build_reasoning_summary = lambda trace: "summary"
    progress = []

    def run_map_reduce(state):
        state["on_map_progress"]({"stage": "map", "mapped": 1, "total": 1, "document": "guide.pdf"})
        return {
            "answer": "Grounded summary [Source 1: Page 1]",
            "retrieved_docs": [SimpleNamespace(metadata={"source": "guide.pdf", "page": 1})],
            "map_reduce_fallback": False,
            "trace": state["trace"] + [{"step": "map_reduce", "fallback": False}],
        }

    service._map_reduce = run_map_reduce
    stream, documents = service.ask_stream(
        "Summarize all sections of the guide",
        on_map_progress=progress.append,
    )

    assert list(stream) == ["Grounded summary [Source 1: Page 1]"]
    assert documents[0].metadata["source"] == "guide.pdf"
    assert progress == [{"stage": "map", "mapped": 1, "total": 1, "document": "guide.pdf"}]
    assert [event["step"] for event in service.last_trace][-2:] == ["map_reduce", "generate"]


def test_agentic_graph_takes_map_reduce_path_for_broad_questions():
    service = RAGService.__new__(RAGService)
    service.top_k = 4
    service.retriever = object()
    service.route_prompt = object()
    service.route_chain = type("Route", (), {"invoke": lambda self, payload: "RETRIEVE"})()
    service._map_reduce = lambda state: {
        "answer": "Document summary",
        "retrieved_docs": [],
        "map_reduce_fallback": False,
        "trace": state["trace"] + [{"step": "map_reduce", "fallback": False}],
    }
    assert getattr(service, "graph", None) is None

    answer, documents = service.ask("Summarize the whole report", map_reduce_mode="auto")

    assert answer == "Document summary"
    assert documents == []
    assert service.graph is not None
    assert service.graph.checkpointer is None
    assert any(event["step"] == "map_reduce" for event in service.last_trace)


def test_baseline_stream_can_run_map_reduce_for_broad_question(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED", "0")
    document = SimpleNamespace(
        page_content="Evidence",
        metadata={"source": "guide.pdf", "page": 1, "document_id": "doc-1"},
    )

    class Store:
        def get(self, *, where, include):
            return {"documents": [document.page_content], "metadatas": [document.metadata]}

    class Processor:
        def run(self, question, documents, *, cancel_event, on_progress):
            return MapReduceResult(
                answer="Mapped answer [Source 1: Page 1]",
                partials=[MapPartial("Evidence", (SourceRef(1, "guide.pdf", "1"),))],
                map_calls=1,
                tree_reduce_levels=0,
            )

    service = BaselineRAGService.__new__(BaselineRAGService)
    service.retriever = object()
    service.vector_store = Store()
    service.top_k = 4
    service.contextualize_chain = None
    service.answer_chain = None
    service.map_reduce_processor = Processor()
    service.last_trace = []

    stream, documents = service.ask_stream(
        "Summarize the guide",
        document_ids=["doc-1"],
        document_sources=["guide.pdf"],
        map_reduce_mode="auto",
    )

    assert list(stream) == ["Mapped answer [Source 1: Page 1]"]
    assert len(documents) == 1
    assert any(event["step"] == "map_reduce" for event in service.last_trace)


def test_baseline_contextualization_receives_memory_hints(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED", "0")
    calls = []

    class Contextualizer:
        def invoke(self, payload):
            calls.append(payload)
            return SimpleNamespace(standalone_question="What is the review period?")

    class Store:
        def similarity_search(self, query, *, k, filter=None):
            return []

    class AnswerChain:
        def stream(self, payload):
            return iter(["answer"])

    service = BaselineRAGService.__new__(BaselineRAGService)
    service.retriever = type("Retriever", (), {"invoke": lambda self, query: []})()
    service.vector_store = Store()
    service.top_k = 4
    service.contextualize_chain = Contextualizer()
    service.answer_chain = AnswerChain()
    service.last_trace = []
    stream, _ = service.ask_stream(
        "How long is it?",
        history=[],
        memory_hints="UNTRUSTED PAST-CHAT HINTS: review period",
        map_reduce_mode="off",
    )

    assert list(stream) == ["answer"]
    assert calls[0]["memory_hints"].startswith("UNTRUSTED PAST-CHAT HINTS")


def test_contextualize_skips_chain_without_history():
    service = RAGService.__new__(RAGService)
    state = {
        "question": "What about the second one?",
        "trace": [],
    }

    result = service._contextualize(state)

    assert result["standalone_question"] == "What about the second one?"
    assert result["trace"][-1]["step"] == "contextualize"
    assert result["trace"][-1]["skipped"] is True


def test_contextualize_rewrites_follow_up_with_structured_output():
    service = RAGService.__new__(RAGService)
    calls = []

    class FakeChain:
        def invoke(self, payload):
            calls.append(payload)
            return SimpleNamespace(standalone_question="What is the second document's amendment process?")

    service.contextualize_chain = FakeChain()
    result = service._contextualize({
        "question": "What about the second one?",
        "original_question": "What about the second one?",
        "history": [
            {"role": "user", "content": "Compare amendment processes in these files."},
            {"role": "assistant", "content": "The second file is the India constitution."},
        ],
        "trace": [],
    })

    assert calls[0]["question"] == "What about the second one?"
    assert "India constitution" in calls[0]["history"]
    assert result["question"] == "What is the second document's amendment process?"
    assert result["search_query"] == result["question"]
    assert result["trace"][-1]["skipped"] is False


def test_contextualize_can_use_memory_hints_without_local_history():
    calls = []
    service = RAGService.__new__(RAGService)

    class FakeChain:
        def invoke(self, payload):
            calls.append(payload)
            return SimpleNamespace(standalone_question="What is the review period?")

    service.contextualize_chain = FakeChain()
    result = service._contextualize({
        "question": "How long is it?",
        "original_question": "How long is it?",
        "history": [],
        "memory_hints": "UNTRUSTED PAST-CHAT HINTS: Prior question: review period.",
        "trace": [],
    })

    assert calls[0]["memory_hints"].startswith("UNTRUSTED PAST-CHAT HINTS")
    assert result["question"] == "What is the review period?"


def test_final_generation_does_not_receive_cross_session_memory_hints(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED", "0")
    calls = []
    service = RAGService.__new__(RAGService)

    class FakeChain:
        def invoke(self, payload):
            calls.append(payload)
            return "Answer from the uploaded excerpt."

    service.answer_chain = FakeChain()
    service._generate({
        "question": "What does the document say?",
        "original_question": "What does the document say?",
        "retrieved_docs": [SimpleNamespace(page_content="Document facts", metadata={"source": "doc.pdf", "page": 1})],
        "context_history": "",
        "memory_hints": "UNTRUSTED PAST-CHAT HINTS: unrelated prior summary",
        "sufficient": True,
        "trace": [],
    })

    assert calls[0] == {
        "context": calls[0]["context"],
        "question": "What does the document say?",
        "history": "",
    }
    assert "memory_hints" not in calls[0]


def test_ask_stream_emits_chitchat_route_before_contextualization(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    service.max_retries = 1
    service.last_trace = []
    service.route_prompt = object()
    service.route_chain = type("Route", (), {"invoke": lambda self, payload: "DIRECT"})()
    service.direct_chain = type(
        "Direct",
        (),
        {
            "stream": lambda self, payload: iter(["Hello."]),
            "invoke": lambda self, payload: "Hello.",
        },
    )()
    service._build_reasoning_summary = lambda trace: "summary"
    events = []

    stream, docs = service.ask_stream("hello", on_trace=events.append)

    assert list(stream)[0] in __import__("chitchat.responses", fromlist=["RESPONSES"]).RESPONSES["greeting"]
    assert docs == []
    assert [event["step"] for event in events] == ["route"]
    assert events[0]["route"] == "chitchat"
    assert events[0]["retrieval_skipped"] is True


def test_add_documents_appends_with_document_id_metadata(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.added = []
            self.deleted = []

        def add_documents(self, documents, ids):
            self.added.extend(zip(ids, documents))

        def as_retriever(self, **kwargs):
            return object()

        def delete(self, **kwargs):
            self.deleted.append(kwargs)

    def split_documents(documents, **_kwargs):
        for document in documents:
            document.metadata["chunk_index"] = 1
        return list(documents), [f"{document.metadata['document_id']}:1" for document in documents]

    monkeypatch.setattr("agentic_rag.split_documents_for_cloud", split_documents)
    service = RAGService.__new__(RAGService)
    service.chunk_size = 800
    service.chunk_overlap = 100
    service.top_k = 4
    service.conversation_id = "test-conversation"
    service.documents = []
    service.chunks = []
    service.vector_store = FakeStore()
    service.embedding_model = object()
    service.retriever = object()
    first = SimpleNamespace(page_content="first", metadata={"page": 1})
    second = SimpleNamespace(page_content="second", metadata={"page": 2})

    first_stats = service.add_documents([first], document_id="doc-1", filename="first.pdf")
    second_stats = service.add_documents([second], document_id="doc-2", filename="second.pdf")
    service.delete_document("doc-2")

    assert first_stats == {"pages": 1, "chunks": 1}
    assert second_stats == {"pages": 1, "chunks": 1}
    assert [item[0] for item in service.vector_store.added] == ["doc-1:1", "doc-2:1"]
    assert first.metadata["document_id"] == "doc-1"
    assert second.metadata["source"] == "second.pdf"
    assert service.vector_store.deleted == [{"where": {"document_id": "doc-2"}}]


def test_parse_route_and_grade_safe_fallbacks(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)

    assert service._parse_route_decision("direct") == "direct"
    assert service._parse_route_decision("not DIRECT") == "retrieve"
    assert service._parse_route_decision("maybe") == "retrieve"
    assert service._parse_grade_decision("YES") is True
    assert service._parse_grade_decision("definitely not YES") is False
    assert service._parse_grade_decision("hmm") is False


def test_ask_supports_direct_questions_without_index(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    service.retriever = None
    service.route_chain = type("DummyRouteChain", (), {"invoke": lambda self, payload: "DIRECT"})()
    service.direct_chain = type("DummyDirectChain", (), {"invoke": lambda self, payload: "Hello there!"})()
    service.graph = type("DummyGraph", (), {"invoke": lambda self, state: {"answer": "Hello there!", "retrieved_docs": [], "trace": [{"step": "direct_answer", "detail": "direct"}]}})()

    answer, docs = service.ask("hello")

    assert answer == "Hello there!"
    assert docs == []


def test_summarize_answer_effort_counts_retry_round_trips():
    trace = [
        {"step": "route", "detail": "routed to retrieve"},
        {"step": "retrieve", "detail": 'searched "first", found 4 passages'},
        {"step": "grade", "detail": "insufficient"},
        {"step": "rewrite", "detail": 'new search: "refined"'},
        {"step": "retrieve", "detail": 'searched "refined", found 2 passages'},
        {"step": "grade", "detail": "sufficient"},
        {"step": "generate", "detail": "answered from retrieved passages"},
    ]

    assert summarize_answer_effort(trace) == "Answered after 1 retry"

    direct_trace = [
        {"step": "route", "detail": "routed to direct"},
        {"step": "direct_answer", "detail": "answered without retrieval"},
    ]
    assert summarize_answer_effort(direct_trace) == "Answered directly"


def test_ask_stream_returns_answer_and_sources_for_compatible_frontend(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    service.retriever = object()
    service.ask = lambda question: ("Answer text", ["doc-1"])
    service.last_reasoning = "I checked retrieval first and then answered."

    stream, docs = service.ask_stream("What does the document say?")
    chunks = list(stream())

    assert chunks == ["Answer text"]
    assert docs == ["doc-1"]


def test_baseline_stream_overfetches_and_reranks_before_answer(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_GROUNDEDNESS_ENABLED", "0")
    monkeypatch.setenv("RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED", "0")
    service = BaselineRAGService.__new__(BaselineRAGService)
    first = SimpleNamespace(page_content="first passage", metadata={"source": "first.pdf"})
    second = SimpleNamespace(page_content="second passage", metadata={"source": "second.pdf"})

    class Store:
        def __init__(self):
            self.query = None

        def similarity_search(self, query, *, k, filter=None):
            self.query = (query, k, filter)
            return [first, second]

    class AnswerChain:
        def stream(self, payload):
            return iter(["answer"])

    def rerank(query, documents, *, enabled, top_n, legacy_top_n):
        assert enabled is True
        assert top_n == 1
        assert legacy_top_n == 4
        return [second], {
            "step": "rerank",
            "candidate_count": len(documents),
            "kept_count": 1,
            "top_score": 0.9,
            "latency_ms": 1.0,
            "enabled": True,
            "fallback": False,
            "detail": "reranked",
        }

    store = Store()
    service.vector_store = store
    service.retriever = object()
    service.top_k = 4
    service.contextualize_chain = None
    service.answer_chain = AnswerChain()
    service.last_trace = []
    monkeypatch.setattr("agentic_rag._rerank_passages", rerank)

    stream, docs = service.ask_stream(
        "compare these files",
        document_ids=["doc-1", "doc-2"],
        document_sources=["first.pdf", "second.pdf"],
        rerank_enabled=True,
        rerank_candidates=20,
        rerank_top_n=1,
    )

    assert store.query[1] == 10
    assert docs == [second]
    assert list(stream) == ["answer"]
    assert [event["step"] for event in service.last_trace][-2:] == ["rerank", "generate"]


def test_ask_with_reasoning_returns_answer_docs_and_reasoning(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    service.retriever = object()
    service.ask = lambda question: ("Answer text", ["doc-1"])
    service.last_reasoning = "I checked retrieval first and then answered."

    answer, docs, reasoning = service.ask_with_reasoning("hello")

    assert answer == "Answer text"
    assert docs == ["doc-1"]
    assert "retrieval" in reasoning.lower()


def test_format_conversation_export_handles_legacy_messages_without_reasoning():
    messages = [
        SimpleNamespace(role="user", content="What is this?"),
        SimpleNamespace(role="assistant", content="It is a document.", is_error=False, sources=None),
    ]

    export = format_conversation_export(messages)

    assert "You: What is this?" in export
    assert "Assistant: It is a document." in export


def test_load_uploaded_documents_sets_document_metadata_basename():
    from agentic_rag import load_uploaded_dcouments, load_uploaded_documents

    uploaded = SimpleNamespace(name="/tmp/nested/reports/final_report.txt", getvalue=lambda: b"hello world")

    docs = load_uploaded_dcouments([uploaded])
    docs_2 = load_uploaded_documents([uploaded])

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "final_report.txt"
    assert docs[0].metadata["document_name"] == "final_report.txt"
    assert docs_2[0].metadata["source"] == "final_report.txt"


def test_ask_with_reasoning_preserves_document_name_and_page_metadata(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = RAGService.__new__(RAGService)
    service.retriever = object()
    service.ask = lambda question: (
        "Answer text",
        [
            SimpleNamespace(
                metadata={"source": "/tmp/nested/report.pdf", "document_name": "report.pdf", "page": 3},
                page_content="report text",
            )
        ],
    )
    service.last_reasoning = "I checked retrieval first and then answered."

    answer, docs, reasoning = service.ask_with_reasoning("hello")

    assert answer == "Answer text"
    assert docs[0].metadata["document_name"] == "report.pdf"
    assert docs[0].metadata["page"] == 3
    assert "retrieval" in reasoning.lower()


def test_build_index_replaces_stale_collection_for_same_conversation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeVectorStore:
        def __init__(self):
            self.added = []

        def add_documents(self, documents, *, ids=None):
            self.added.extend(documents)

        def as_retriever(self, **kwargs):
            return object()

    # tests now use the renamed, canonical loader; patch both names for safety
    monkeypatch.setattr("agentic_rag.load_uploaded_documents", lambda files: [SimpleNamespace(page_content="new text", metadata={"source": "new.pdf", "page": 1})])
    monkeypatch.setattr("agentic_rag.load_uploaded_dcouments", lambda files: [SimpleNamespace(page_content="new text", metadata={"source": "new.pdf", "page": 1})])
    monkeypatch.setattr(
        "agentic_rag.split_documents_for_cloud",
        lambda docs, **kwargs: (list(docs), ["chunk-1"]),
    )
    fake_store = FakeVectorStore()
    monkeypatch.setattr("agentic_rag._create_conversation_vector_store", lambda conversation_id: fake_store)

    service = RAGService.__new__(RAGService)
    service.conversation_id = "same_conversation"
    service.chunk_size = 200
    service.chunk_overlap = 20
    service.top_k = 3
    service.max_retries = 2
    service.llm = object()
    service.documents = []
    service.chunks = []
    service.vector_store = None
    service.retriever = None

    service.build_index([SimpleNamespace(name="new.pdf", getvalue=lambda: b"new")])

    assert len(fake_store.added) == 1
    assert fake_store.added[0].page_content == "new text"


def test_phase_3_helper_layer_preserves_metadata_and_grouping():
    source_a = SourcePreview(
        filename="/tmp/alpha.pdf",
        page=1,
        snippet="first",
        document_name="alpha.pdf",
        document_path="/tmp/alpha.pdf",
        score=0.91,
    )
    source_b = SourcePreview(
        filename="/tmp/alpha.pdf",
        page=2,
        snippet="second",
        document_name="alpha.pdf",
        document_path="/tmp/alpha.pdf",
        score=0.84,
    )
    source_c = SourcePreview(
        filename="beta.txt",
        page=5,
        snippet="third",
        document_name="beta.txt",
        document_path="/tmp/beta.txt",
        score=0.72,
    )

    assert normalize_document_name("/tmp/alpha.pdf") == "alpha.pdf"
    assert normalize_document_name("  ") == "Unknown file"
    grouped = group_sources_by_document([source_a, source_b, source_c])
    assert list(grouped.keys()) == ["alpha.pdf", "beta.txt"]
    assert len(grouped["alpha.pdf"]) == 2

    serialized = serialize_sources([source_a, source_c])
    assert serialized[0]["document_name"] == "alpha.pdf"
    assert serialized[0]["score"] == 0.91
    assert serialized[1]["page"] == 5


def test_prune_default_titles_updates_placeholder_titles(tmp_path):
    store = SQLiteConversationStore(tmp_path / "prune.db")
    conv = store.create_conversation()
    store.append_message(
        conv.id,
        MessageRecord(
            id="user-1",
            conversation_id=conv.id,
            role="user",
            content="  Compare the Australian Constitution with India's Constitution in detail, please.  ",
            created_at="2024-01-01T00:00:00Z",
        ),
    )
    store.rename_conversation(conv.id, "New conversation")

    assert prune_default_titles(store) == 1
    assert store.get_conversation(conv.id).title.startswith("Compare the Australian Constitution")


def test_store_crud_round_trip_and_delete(tmp_path):
    db_path = tmp_path / "rag_history.db"
    store = SQLiteConversationStore(db_path)

    conv = store.create_conversation()
    assert conv.title == "New conversation"
    assert store.list_conversations() == []

    first_message = MessageRecord(
        id="demo-user-msg",
        conversation_id=conv.id,
        role="user",
        content="  Compare the Australian Constitution with India's Constitution in detail, please.  ",
        created_at="2024-01-01T00:01:00Z",
    )
    store.append_message(conv.id, first_message)
    listed = store.list_conversations()
    assert [item.id for item in listed] == [conv.id]
    assert listed[0].title == "Compare the Australian Constitution with India's"
    assert listed[0].message_count == 1

    fetched = store.get_conversation(conv.id)
    assert fetched is not None
    assert fetched.title == listed[0].title
    assert fetched.message_count == 1

    renamed = store.rename_conversation(conv.id, "Updated title")
    assert renamed is not None and renamed.title == "Updated title"

    deleted = store.delete_conversation(conv.id)
    assert deleted is True
    assert store.get_conversation(conv.id) is None


def test_store_message_round_trip_preserves_json_fields(tmp_path):
    db_path = tmp_path / "rag_history.db"
    store = SQLiteConversationStore(db_path)
    conv = store.create_conversation("JSON"
    )

    message = MessageRecord(
        id="msg_1",
        conversation_id=conv.id,
        role="assistant",
        content="Answer",
        created_at="2024-01-01T00:00:00Z",
        reasoning="Step 1: route",
        trace=[{"step": "route", "detail": "direct"}],
        sources=[{"filename": "doc.pdf", "page": 2, "snippet": "text", "score": 0.9}],
    )

    stored = store.append_message(conv.id, message)
    reloaded = store.get_messages(conv.id)

    assert stored.reasoning == "Step 1: route"
    assert reloaded[0].trace == [{"step": "route", "detail": "direct"}]
    assert reloaded[0].sources[0]["filename"] == "doc.pdf"
    assert reloaded[0].sources[0]["score"] == 0.9


def test_store_register_documents_preserves_source_metadata(tmp_path):
    db_path = tmp_path / "rag_history.db"
    store = SQLiteConversationStore(db_path)
    conv = store.create_conversation("Docs")

    docs = [
        IndexedDocument(
            id="doc_1",
            conversation_id=conv.id,
            document_id="doc_1",
            filename="report.pdf",
            source_path="/tmp/report.pdf",
            chunk_count=4,
            metadata={"source": "report.pdf", "document_name": "report.pdf", "page": 2},
            created_at="2024-01-01T00:00:00Z",
        )
    ]

    registered = store.register_documents(conv.id, docs)
    reloaded = store.get_documents(conv.id)

    assert len(registered) == 1
    assert reloaded[0].metadata["document_name"] == "report.pdf"
    assert reloaded[0].metadata["page"] == 2


def test_store_migrates_legacy_rows_idempotently(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, message_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, created_at TEXT NOT NULL, reasoning TEXT,
            trace TEXT, sources TEXT
        );
        CREATE TABLE indexed_documents (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, document_id TEXT NOT NULL,
            filename TEXT NOT NULL, source_path TEXT, chunk_count INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
        );
        INSERT INTO conversations VALUES
            ('legacy-conv', 'Legacy title', '2024-01-01 00:00:00', '2024-01-02 00:00:00', 1);
        INSERT INTO messages VALUES
            ('legacy-msg', 'legacy-conv', 'assistant', 'Existing answer',
             '2024-01-02 00:00:00', NULL,
             '[{"step":"generate"}]',
             '[{"filename":"old.pdf","page":1,"snippet":"kept"}]');
        INSERT INTO indexed_documents VALUES
            ('legacy-doc', 'legacy-conv', 'legacy-doc', 'old.pdf', NULL, 7,
             '{"source":"old.pdf","page":1}', '2024-01-01 00:00:00');
        """
    )
    connection.commit()
    connection.close()

    store = SQLiteConversationStore(db_path)
    conversation = store.get_conversation("legacy-conv")
    message = store.get_messages("legacy-conv")[0]
    documents = store.list_document_records("legacy-conv")
    SQLiteConversationStore(db_path)

    assert conversation is not None
    assert conversation.title == "Legacy title"
    assert conversation.created_at == "2024-01-01T00:00:00Z"
    assert conversation.updated_at == "2024-01-02T00:00:00Z"
    assert (conversation.message_count, conversation.document_count) == (1, 1)
    assert message.content == "Existing answer"
    assert message.created_at == "2024-01-02T00:00:00Z"
    assert message.trace == [{"step": "generate"}]
    assert message.sources[0]["filename"] == "old.pdf"
    assert message.status == "complete"
    assert len(documents) == 1
    assert (documents[0].filename, documents[0].pages, documents[0].chunks) == ("old.pdf", 1, 7)
    assert documents[0].status == "ready"
    assert documents[0].created_at == "2024-01-01T00:00:00Z"
    assert len(store.list_document_records("legacy-conv")) == 1
    assert store.delete_conversation("legacy-conv") is True
    assert store.get_messages("legacy-conv") == []
    assert store.get_documents("legacy-conv") == []
    assert store.list_document_records("legacy-conv") == []


def test_document_records_status_update_and_delete(tmp_path):
    store = SQLiteConversationStore(tmp_path / "documents.db")
    conversation = store.create_conversation()
    document = DocumentRecord(
        id="document-id",
        conversation_id=conversation.id,
        filename="report.pdf",
        sha256="abc123",
        size_bytes=1024,
        pages=0,
        chunks=0,
        status="queued",
        error_code=None,
        error_message=None,
        created_at="2024-01-01T00:00:00Z",
    )

    store.create_document(document)
    assert store.find_document_by_sha(conversation.id, "abc123").id == document.id
    updated = store.update_document(
        document.id,
        status="ready",
        pages=12,
        chunks=32,
    )
    assert updated.status == "ready"
    assert (updated.pages, updated.chunks) == (12, 32)
    assert store.delete_document(conversation.id, document.id) is True
    SQLiteConversationStore(tmp_path / "documents.db")
    assert store.list_document_records(conversation.id) == []


def test_conversation_history_orders_nonempty_rows_by_updated_at(tmp_path):
    store = SQLiteConversationStore(tmp_path / "history-order.db")
    older = store.create_conversation("Older")
    newer = store.create_conversation("Newer")
    store.append_message(older.id, MessageRecord(
        id="older-message",
        conversation_id=older.id,
        role="user",
        content="Old question",
        created_at="2024-01-01T00:00:00Z",
    ))
    store.append_message(newer.id, MessageRecord(
        id="newer-message",
        conversation_id=newer.id,
        role="user",
        content="Recent question",
        created_at="2024-01-02T00:00:00Z",
    ))

    with store._connect() as connection:
        connection.execute("UPDATE conversations SET updated_at = '2024-01-01T00:00:00Z' WHERE id = ?", (older.id,))
        connection.execute("UPDATE conversations SET updated_at = '2024-01-02T00:00:00Z' WHERE id = ?", (newer.id,))

    assert [conversation.id for conversation in store.list_conversations()] == [newer.id, older.id]


def test_format_conversation_export_handles_null_json_fields():
    messages = [
        SimpleNamespace(role="user", content="What is this?", reasoning=None, trace=None, sources=None),
        SimpleNamespace(
            role="assistant",
            content="It is a document.",
            is_error=False,
            reasoning=None,
            trace=None,
            sources=None,
        ),
    ]

    export = format_conversation_export(messages)

    assert "You: What is this?" in export
    assert "Assistant: It is a document." in export
    assert "Reasoning:" not in export


def test_format_conversation_export_handles_structured_sources_and_stopped_status():
    message = SimpleNamespace(
        role="assistant",
        content="Partial answer",
        reasoning=None,
        trace={"step": "retrieve", "query": "follow up"},
        sources={"filename": "guide.pdf", "page": 4, "snippet": "Relevant passage"},
        status="stopped",
    )

    exported = format_conversation_export([message])

    assert "Status: stopped" in exported
    assert "Retrieve:" in exported
    assert "guide.pdf (page 4): Relevant passage" in exported


def test_legacy_message_round_trip_with_null_fields_renders_without_crashing(tmp_path):
    db_path = tmp_path / "legacy.db"
    store = SQLiteConversationStore(db_path)
    conv = store.create_conversation("Legacy")

    legacy = MessageRecord(
        id="legacy_1",
        conversation_id=conv.id,
        role="assistant",
        content="No answer found.",
        created_at="2024-01-01T00:00:00Z",
        reasoning=None,
        trace=None,
        sources=None,
    )

    store.append_message(conv.id, legacy)
    reloaded = store.get_messages(conv.id)
    exported = format_conversation_export(
        [
            SimpleNamespace(role="user", content="Hello", reasoning=None, trace=None, sources=None),
            SimpleNamespace(
                role="assistant",
                content=reloaded[0].content,
                reasoning=reloaded[0].reasoning,
                trace=reloaded[0].trace,
                sources=reloaded[0].sources,
            ),
        ]
    )

    assert reloaded[0].reasoning is None
    assert reloaded[0].trace is None
    assert reloaded[0].sources is None
    assert "No answer found." in exported
