from __future__ import annotations

import pytest
from chromadb import Knn, Rrf, SparseVectorIndexConfig, VectorIndexConfig
from chromadb.execution.expression.operator import GroupBy
from langchain_core.documents import Document

import chroma_cloud
from chroma_cloud import ChromaCloudVectorStore, create_cloud_client, split_documents_for_cloud


def test_cloud_client_requires_credentials_without_disclosing_values(monkeypatch):
    monkeypatch.delenv("CHROMA_API_KEY", raising=False)
    monkeypatch.delenv("CHROMA_TENANT", raising=False)
    monkeypatch.delenv("CHROMA_DATABASE", raising=False)

    with pytest.raises(RuntimeError, match="CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE"):
        create_cloud_client()


def test_cloud_client_uses_configured_cloud_parameters(monkeypatch):
    configured = {}
    monkeypatch.setenv("CHROMA_API_KEY", "test-secret")
    monkeypatch.setenv("CHROMA_TENANT", "tenant-id")
    monkeypatch.setenv("CHROMA_DATABASE", "database-name")
    monkeypatch.setenv("CHROMA_HOST", "https://api.example.test/")
    monkeypatch.setattr(
        chroma_cloud.chromadb,
        "CloudClient",
        lambda **kwargs: configured.update(kwargs) or object(),
    )

    create_cloud_client()

    assert configured == {
        "tenant": "tenant-id",
        "database": "database-name",
        "api_key": "test-secret",
        "cloud_host": "api.example.test",
    }


def test_collections_use_qwen_dense_and_splade_sparse_indexes(monkeypatch):
    monkeypatch.setenv("CHROMA_API_KEY", "test-secret")
    chroma_cloud.get_dense_embedding_function.cache_clear()
    chroma_cloud.get_sparse_embedding_function.cache_clear()

    try:
        schema, dense_embedding = chroma_cloud._collection_schema()
        dense_index = schema.defaults.float_list.vector_index.config
        sparse_index = schema.keys[chroma_cloud.SPARSE_VECTOR_KEY].sparse_vector.sparse_vector_index.config

        assert isinstance(dense_index, VectorIndexConfig)
        assert dense_index.space == "cosine"
        assert dense_embedding.model.value == "Qwen/Qwen3-Embedding-0.6B"
        assert isinstance(sparse_index, SparseVectorIndexConfig)
        assert sparse_index.embedding_function.model.value == "prithivida/Splade_PP_en_v1"
    finally:
        chroma_cloud.get_dense_embedding_function.cache_clear()
        chroma_cloud.get_sparse_embedding_function.cache_clear()


def test_split_documents_limits_utf8_size_and_adds_grouping_metadata():
    source = Document(
        page_content="🍵" * 6000,
        metadata={"source": "guide.txt", "document_id": "doc-1"},
    )

    chunks, ids = split_documents_for_cloud(
        [source],
        chunk_size=20_000,
        chunk_overlap=0,
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content.encode("utf-8")) <= chroma_cloud.MAX_CHROMA_DOCUMENT_BYTES for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.metadata["source_document_id"] == "doc-1" for chunk in chunks)
    assert ids == [f"doc-1:{index}" for index in range(1, len(chunks) + 1)]


def test_cloud_vector_store_uses_rrf_hybrid_search_and_groupby():
    class FakeResult:
        def rows(self):
            return [[{
                "document": "The answer is in section 4.",
                "metadata": {"document_id": "doc-1", "chunk_index": 2},
                "score": 0.03,
            }]]

    class FakeCollection:
        name = "conversation_test"

        def search(self, request):
            self.request = request
            return FakeResult()

    collection = FakeCollection()
    store = ChromaCloudVectorStore(object(), collection)

    matches = store.similarity_search("Where is the answer?", k=5, filter={"document_id": "doc-1"})

    assert matches == [
        Document(
            page_content="The answer is in section 4.",
            metadata={"document_id": "doc-1", "chunk_index": 2},
        )
    ]
    request = collection.request
    rank = request._rank
    assert isinstance(rank, Rrf)
    assert len(rank.ranks) == 2
    assert all(isinstance(candidate, Knn) and candidate.return_rank for candidate in rank.ranks)
    assert rank.ranks[1].key == chroma_cloud.SPARSE_VECTOR_KEY
    assert isinstance(request._group_by, GroupBy)
    assert request._where.key == "document_id"
    assert request._where.value == "doc-1"


def test_cloud_vector_store_rejects_records_over_cloud_limit():
    class FakeCollection:
        def add(self, **kwargs):
            raise AssertionError("oversized content must not be sent to Chroma")

    store = ChromaCloudVectorStore(object(), FakeCollection())
    oversized = Document(page_content="x" * (chroma_cloud.MAX_CHROMA_DOCUMENT_BYTES + 1))

    with pytest.raises(ValueError, match="Chroma Cloud allows at most"):
        store.add_documents([oversized], ids=["too-large"])
