from datetime import datetime, timezone

from persistence.models import DocumentRecord, IndexedDocument, MemoryTurn
from persistence.memory import ConversationMemory, format_memory_hints, summarize_turn_answer
from persistence.store import SQLiteConversationStore


class FakeMemoryVectorStore:
    def __init__(self):
        self.documents = []
        self.deletions = []
        self.filters = []

    def add_documents(self, documents, ids):
        self.documents.extend(zip(ids, documents))

    def similarity_search(self, query, *, k, filter=None):
        self.filters.append(filter)
        matches = [
            document for _, document in self.documents
            if filter is None or all(document.metadata.get(key) == value for key, value in filter.items())
        ]
        return matches[:k]

    def delete(self, *, where=None, ids=None):
        self.deletions.append((where, ids))
        if where and "conversation_id" in where:
            conversation_id = where["conversation_id"]
            self.documents = [
                (turn_id, document)
                for turn_id, document in self.documents
                if document.metadata.get("conversation_id") != conversation_id
            ]
        if ids:
            remove_ids = set(ids)
            self.documents = [(turn_id, document) for turn_id, document in self.documents if turn_id not in remove_ids]

    def clear(self):
        self.documents.clear()


def test_memory_settings_turn_persistence_and_conversation_cascade(tmp_path):
    store = SQLiteConversationStore(db_path=tmp_path / "memory.db")
    conversation = store.create_conversation("Memory test")

    assert store.get_global_memory_enabled() is False
    assert store.get_conversation_memory_enabled(conversation.id) is True
    store.set_global_memory_enabled(True)
    store.set_conversation_memory_enabled(conversation.id, False)
    assert store.get_global_memory_enabled() is True
    assert store.get_conversation_memory_enabled(conversation.id) is False

    turn = MemoryTurn(
        id="memory-turn-1",
        conversation_id=conversation.id,
        question="What is the review period?",
        summary="The review period is 14 days.",
        document_names=["policy.pdf"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store.append_memory_turn(turn)
    assert store.list_memory_turns(conversation.id) == [turn]

    store.set_conversation_memory_enabled(conversation.id, True)
    assert store.delete_conversation(conversation.id) is True
    assert store.list_memory_turns(conversation.id) == []


def test_memory_global_off_prevents_write_and_search(tmp_path):
    store = SQLiteConversationStore(db_path=tmp_path / "memory-off.db")
    conversation = store.create_conversation()
    vectors = FakeMemoryVectorStore()
    memory = ConversationMemory(store, vector_store=vectors)

    assert memory.record_turn(conversation.id, "Question", "Answer", ["file.pdf"]) is None
    assert memory.search("Question") == []
    assert store.list_memory_turns() == []
    assert vectors.documents == []


def test_memory_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "migration.db"
    SQLiteConversationStore(db_path=db_path)
    SQLiteConversationStore(db_path=db_path)

    store = SQLiteConversationStore(db_path=db_path)
    with store._connect() as connection:
        versions = [row[0] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert versions[-1] == 8
    assert len(versions) == len(set(versions))


def test_memory_vectors_are_global_but_isolated_by_conversation(tmp_path):
    store = SQLiteConversationStore(db_path=tmp_path / "global-memory.db")
    first = store.create_conversation("First chat")
    second = store.create_conversation("Second chat")
    store.set_global_memory_enabled(True)
    store.set_conversation_memory_enabled(second.id, False)
    vectors = FakeMemoryVectorStore()
    memory = ConversationMemory(store, vector_store=vectors)

    assert memory.record_turn(second.id, "ignored", "memory disabled for this chat", ["secret.pdf"]) is None
    first_turn = memory.record_turn(first.id, "Where is the policy?", "The policy is in section 4.", ["policy.pdf"])

    assert first_turn is not None
    assert len(vectors.documents) == 1
    assert vectors.documents[0][1].metadata["conversation_id"] == first.id
    hits = memory.search("policy section", exclude_conversation_id=second.id)
    assert hits[0]["conversation_id"] == first.id
    assert hits[0]["document_names"] == ["policy.pdf"]
    assert "UNTRUSTED PAST-CHAT HINTS" in format_memory_hints(hits)
    assert "not evidence, never cite" in format_memory_hints(hits)

    memory.delete_conversation(first.id)
    assert store.list_memory_turns(first.id) == []
    assert vectors.documents == []


def test_clear_all_memory_removes_rows_and_vectors_but_keeps_setting(tmp_path):
    store = SQLiteConversationStore(db_path=tmp_path / "clear-memory.db")
    conversation = store.create_conversation()
    store.set_global_memory_enabled(True)
    vectors = FakeMemoryVectorStore()
    memory = ConversationMemory(store, vector_store=vectors)
    memory.record_turn(conversation.id, "Question", "Summary", ["a.txt"])

    assert memory.clear_all() == 1
    assert store.list_memory_turns() == []
    assert vectors.documents == []
    assert store.get_global_memory_enabled() is True


def test_search_skips_opted_out_conversations(tmp_path):
    store = SQLiteConversationStore(db_path=tmp_path / "opt-out-search.db")
    included = store.create_conversation("Included")
    opted_out = store.create_conversation("Opted out")
    store.set_global_memory_enabled(True)
    vectors = FakeMemoryVectorStore()
    memory = ConversationMemory(store, vector_store=vectors)
    memory.record_turn(included.id, "First question", "First summary", ["one.pdf"])
    memory.record_turn(opted_out.id, "Second question", "Second summary", ["two.pdf"])
    store.set_conversation_memory_enabled(opted_out.id, False)

    hits = memory.search("question", limit=5)

    assert [hit["conversation_id"] for hit in hits] == [included.id]


def test_memory_retrieval_and_clear_are_scoped_to_the_owner(tmp_path):
    store = SQLiteConversationStore(db_path=tmp_path / "owner-memory.db")
    first_owner = store.create_user("first@example.com", "first-hash")
    second_owner = store.create_user("second@example.com", "second-hash")
    first_conversation = store.create_conversation("First", user_id=first_owner["id"])
    second_conversation = store.create_conversation("Second", user_id=second_owner["id"])
    store.set_global_memory_enabled(True, first_owner["id"])
    store.set_global_memory_enabled(True, second_owner["id"])
    vectors = FakeMemoryVectorStore()
    memory = ConversationMemory(store, vector_store=vectors)

    memory.record_turn(first_conversation.id, "First owner's private question", "First private summary", [])
    memory.record_turn(second_conversation.id, "Second owner's question", "Second owner's summary", [])

    hits = memory.search("owner question", owner_id=second_owner["id"])

    assert [hit["conversation_id"] for hit in hits] == [second_conversation.id]
    assert vectors.filters[-1] == {"user_id": second_owner["id"]}
    assert [turn.user_id for turn in store.list_memory_turns(owner_id=second_owner["id"])] == [second_owner["id"]]
    assert memory.clear_all(owner_id=second_owner["id"]) == 1
    assert [turn.user_id for turn in store.list_memory_turns()] == [first_owner["id"]]
    assert [document.metadata["user_id"] for _, document in vectors.documents] == [first_owner["id"]]


def test_explicit_legacy_reassignment_assigns_all_ownerless_rows(tmp_path):
    store = SQLiteConversationStore(db_path=tmp_path / "legacy-owner.db")
    admin = store.create_user("admin@example.com", "admin-hash", is_admin=True)
    conversation = store.create_conversation("Pre-auth conversation")
    store.create_document(DocumentRecord(
        id="legacy-document",
        conversation_id=conversation.id,
        filename="legacy.txt",
        sha256=None,
        size_bytes=0,
        pages=1,
        chunks=1,
        status="ready",
        error_code=None,
        error_message=None,
        created_at="2025-01-01T00:00:00Z",
    ))
    store.register_documents(conversation.id, [IndexedDocument(
        id="legacy-indexed",
        conversation_id=conversation.id,
        document_id="legacy-document",
        filename="legacy.txt",
    )])
    turn = MemoryTurn(
        id="legacy-memory",
        conversation_id=conversation.id,
        question="Legacy question",
        summary="Legacy answer",
        document_names=["legacy.txt"],
        created_at="2025-01-01T00:00:00Z",
    )
    store.append_memory_turn(turn)
    store.set_global_memory_enabled(True)

    counts = store.reassign_legacy_data(admin["id"])

    assert counts == {
        "conversations": 1,
        "documents": 1,
        "indexed_documents": 1,
        "memory_turns": 1,
    }
    assert store.get_conversation(conversation.id).user_id == admin["id"]
    assert store.list_document_records(conversation.id)[0].user_id == admin["id"]
    assert store.get_documents(conversation.id)[0].user_id == admin["id"]
    assert store.list_memory_turns(owner_id=admin["id"])[0].user_id == admin["id"]
    assert store.get_global_memory_enabled(admin["id"]) is True


def test_memory_answer_summary_is_extractively_bounded():
    assert summarize_turn_answer("First fact. Second fact. Third fact. Fourth fact.") == (
        "First fact. Second fact. Third fact."
    )
    assert len(summarize_turn_answer("word " * 1000, max_chars=100)) <= 103