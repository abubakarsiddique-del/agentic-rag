from __future__ import annotations

import re

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]{2,}", normalized):
            raise ValueError("Enter a valid email address")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_bcrypt_length(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be no more than 72 UTF-8 bytes")
        return value


class SigninRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class RequestPasswordReset(BaseModel):
    email: str = Field(min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]{2,}", normalized):
            raise ValueError("Enter a valid email address")
        return normalized


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=32, max_length=256)
    password: str = Field(min_length=12, max_length=256)

    @field_validator("password")
    @classmethod
    def validate_bcrypt_length(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be no more than 72 UTF-8 bytes")
        return value


class AddPasswordRequest(BaseModel):
    password: str = Field(min_length=12, max_length=256)

    @field_validator("password")
    @classmethod
    def validate_bcrypt_length(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be no more than 72 UTF-8 bytes")
        return value


class AuthUserResponse(BaseModel):
    id: str
    email: str
    created_at: str


class ConversationInfo(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    message_count: int = 0


class DocumentInfo(BaseModel):
    id: str
    conversation_id: str
    filename: str
    sha256: Optional[str] = None
    size_bytes: int = 0
    pages: int = 0
    chunks: int = 0
    status: Literal["queued", "processing", "ready", "failed"]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str


class RejectedUpload(BaseModel):
    filename: str
    error_code: str
    reason: str


class UploadBatchResponse(BaseModel):
    accepted: List[DocumentInfo]
    rejected: List[RejectedUpload]


class MessageInfo(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    sources: Optional[list | dict] = None
    trace: Optional[list | dict] = None
    rating: Optional[int] = None
    feedback: Optional[str] = None
    status: Literal["complete", "stopped", "error"] = "complete"
    created_at: str


class ConversationDetail(ConversationInfo):
    documents: List[DocumentInfo]
    messages: List[MessageInfo]


class ProcessDocumentsResponse(BaseModel):
    documents: int
    chunks: int
    embedding_dimension: int
    embedding_model: str
    llm_model: str


class QuestionRequest(BaseModel):
    question: str
    voice_output: bool = False
    answer_mode: Literal["agentic", "traditional"] = "agentic"
    max_retries: int = Field(default=2, ge=0, le=3)
    passages_per_search: int = Field(default=4, ge=1, le=12)
    show_reasoning_steps: bool = True
    document_ids: Optional[List[str]] = None
    rerank_enabled: bool = True
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    map_reduce_mode: Literal["auto", "off", "force"] = "auto"


class MemorySettingsRequest(BaseModel):
    enabled: bool


class SourceItem(BaseModel):
    filename: str
    page: int
    snippet: Optional[str]
    source: Optional[str] = None
    score: Optional[float]
    rerank_score: Optional[float] = None


class TraceEvent(BaseModel):
    step: str
    detail: Optional[str]


class TokenEvent(BaseModel):
    token: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    trace: List[TraceEvent]


class ErrorResponse(BaseModel):
    error: str
