from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import chromadb
from chromadb import K, Knn, Rrf, Search, Schema, SparseVectorIndexConfig, VectorIndexConfig
from chromadb.execution.expression.operator import GroupBy, MinK
from chromadb.utils.embedding_functions import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
    ChromaCloudQwenEmbeddingModel,
    ChromaCloudQwenEmbeddingTarget,
)
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

MAX_CHROMA_DOCUMENT_BYTES = 16 * 1024
SAFE_CHUNK_BYTES = MAX_CHROMA_DOCUMENT_BYTES - 1024
SPARSE_VECTOR_KEY = "sparse_embedding"
DENSE_EMBEDDING_MODEL = "Chroma Cloud Qwen/Qwen3-Embedding-0.6B"
DENSE_EMBEDDING_DIMENSION = 1024


@lru_cache(maxsize=1)
def get_dense_embedding_function() -> ChromaCloudQwenEmbeddingFunction:
    return ChromaCloudQwenEmbeddingFunction(
        model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
        task="rag_retrieval",
        instructions={
            "rag_retrieval": {
                ChromaCloudQwenEmbeddingTarget.DOCUMENTS: (
                    "Represent this passage from an uploaded document for retrieval."
                ),
                ChromaCloudQwenEmbeddingTarget.QUERY: (
                    "Represent this question for retrieving relevant uploaded document passages."
                ),
            }
        },
    )


@lru_cache(maxsize=1)
def get_sparse_embedding_function() -> ChromaCloudSpladeEmbeddingFunction:
    return ChromaCloudSpladeEmbeddingFunction()


def create_cloud_client() -> chromadb.api.ClientAPI:
    api_key = os.getenv("CHROMA_API_KEY", "").strip()
    tenant = os.getenv("CHROMA_TENANT", "").strip()
    database = os.getenv("CHROMA_DATABASE", "").strip()
    host = os.getenv("CHROMA_HOST", "api.trychroma.com").strip()
    missing = [
        name
        for name, value in (
            ("CHROMA_API_KEY", api_key),
            ("CHROMA_TENANT", tenant),
            ("CHROMA_DATABASE", database),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Chroma Cloud is not configured. Set " + ", ".join(missing) + " in the environment or .env."
        )
    if host.startswith(("https://", "http://")):
        host = host.split("://", 1)[1].rstrip("/")
    return chromadb.CloudClient(
        tenant=tenant,
        database=database,
        api_key=api_key,
        cloud_host=host,
    )


def collection_name(conversation_id: str) -> str:
    return f"conversation_{conversation_id}"


def memory_collection_name(user_id: str | None) -> str:
    return f"memory_{user_id or 'legacy'}"


def _collection_schema() -> tuple[Schema, ChromaCloudQwenEmbeddingFunction]:
    dense_embedding = get_dense_embedding_function()
    sparse_embedding = get_sparse_embedding_function()
    schema = Schema()
    schema.create_index(
        config=VectorIndexConfig(
            space="cosine",
            source_key=K.DOCUMENT,
            embedding_function=dense_embedding,
        )
    )
    schema.create_index(
        key=SPARSE_VECTOR_KEY,
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT,
            embedding_function=sparse_embedding,
        ),
    )
    return schema, dense_embedding


def get_or_create_collection(
    client: chromadb.api.ClientAPI,
    name: str,
):
    schema, dense_embedding = _collection_schema()
    return client.get_or_create_collection(
        name=name,
        schema=schema,
        embedding_function=dense_embedding,
        metadata={"index_type": "dense_sparse_hybrid"},
    )


def delete_collection_if_exists(name: str) -> None:
    client = create_cloud_client()
    try:
        client.delete_collection(name=name)
    except chromadb.errors.NotFoundError:
        return


def delete_conversation_collection(conversation_id: str) -> None:
    delete_collection_if_exists(collection_name(str(conversation_id)))


def split_documents_for_cloud(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Document], list[str]]:
    safe_size = max(1, min(int(chunk_size), SAFE_CHUNK_BYTES))
    safe_overlap = min(max(0, int(chunk_overlap)), safe_size - 1)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=safe_size,
        chunk_overlap=safe_overlap,
        length_function=lambda text: len(text.encode("utf-8")),
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))
    ids: list[str] = []
    next_chunk_index: dict[str, int] = {}
    generated_ids: dict[str, str] = {}
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        source = str(metadata.get("source") or metadata.get("document_name") or "unknown")
        document_id = str(metadata.get("document_id") or generated_ids.setdefault(source, str(uuid4())))
        metadata["document_id"] = document_id
        metadata["source_document_id"] = document_id
        chunk_index = next_chunk_index.get(document_id, 0) + 1
        next_chunk_index[document_id] = chunk_index
        metadata["chunk_index"] = chunk_index
        metadata.setdefault("passage_id", f"{document_id}:{chunk_index}")
        chunk.metadata = metadata
        encoded_size = len(chunk.page_content.encode("utf-8"))
        if encoded_size > MAX_CHROMA_DOCUMENT_BYTES:
            raise ValueError(
                f"Document chunk is {encoded_size} bytes; Chroma Cloud allows at most "
                f"{MAX_CHROMA_DOCUMENT_BYTES} bytes per record."
            )
        ids.append(f"{document_id}:{chunk_index}")
    return chunks, ids


class _CloudRetriever:
    def __init__(self, vector_store: "ChromaCloudVectorStore", *, k: int) -> None:
        self.vector_store = vector_store
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        return self.vector_store.similarity_search(query, k=self.k)


class ChromaCloudVectorStore:
    def __init__(
        self,
        client: chromadb.api.ClientAPI,
        collection: Any,
        *,
        group_by_document: bool = True,
    ) -> None:
        self.client = client
        self.collection = collection
        self.group_by_document = group_by_document

    @property
    def name(self) -> str:
        return self.collection.name

    def count(self) -> int:
        return self.collection.count()

    def add_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
    ) -> None:
        self._write_documents(documents, ids=ids, upsert=False)

    def upsert_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str],
    ) -> None:
        self._write_documents(documents, ids=ids, upsert=True)

    def _write_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None,
        upsert: bool,
    ) -> None:
        if not documents:
            return
        texts = [document.page_content for document in documents]
        too_large = [
            len(text.encode("utf-8"))
            for text in texts
            if len(text.encode("utf-8")) > MAX_CHROMA_DOCUMENT_BYTES
        ]
        if too_large:
            raise ValueError(
                f"Document chunk is {max(too_large)} bytes; Chroma Cloud allows at most "
                f"{MAX_CHROMA_DOCUMENT_BYTES} bytes per record."
            )
        record_ids = ids or [str(uuid4()) for _ in documents]
        if len(record_ids) != len(documents):
            raise ValueError("Each Chroma Cloud document must have exactly one record ID.")
        write = self.collection.upsert if upsert else self.collection.add
        write(
            ids=record_ids,
            documents=texts,
            metadatas=[dict(document.metadata or {}) for document in documents],
        )

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        if k < 1:
            return []
        rank_limit = max(16, k * 4)
        rank = Rrf(
            ranks=[
                Knn(query=query, limit=rank_limit, return_rank=True),
                Knn(
                    query=query,
                    key=SPARSE_VECTOR_KEY,
                    limit=rank_limit,
                    return_rank=True,
                ),
            ]
        )
        group_by = None
        if self.group_by_document:
            group_by = GroupBy(
                keys=K("document_id"),
                aggregate=MinK(keys=K.SCORE, k=3),
            )
        search = Search(where=filter).rank(rank)
        if group_by is not None:
            search.group_by(group_by)
        result = self.collection.search(
            search.limit(k).select(K.DOCUMENT, K.METADATA, K.SCORE)
        )
        return [
            Document(
                page_content=str(row.get("document") or ""),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in result.rows()[0]
            if str(row.get("document") or "").strip()
        ]

    def as_retriever(self, *, search_type: str = "similarity", search_kwargs: dict[str, Any] | None = None):
        if search_type != "similarity":
            raise ValueError(f"Unsupported Chroma Cloud search type: {search_type}")
        return _CloudRetriever(self, k=int((search_kwargs or {}).get("k", 4)))

    def get(self, *, where: dict[str, Any] | None = None, include: list[str] | None = None) -> dict[str, Any]:
        return self.collection.get(where=where, include=include or ["documents", "metadatas"])

    def update(self, *, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        self.collection.update(ids=ids, metadatas=metadatas)

    def delete(
        self,
        *,
        where: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        self.collection.delete(where=where, ids=ids)

    def delete_collection(self) -> None:
        self.client.delete_collection(name=self.collection.name)
