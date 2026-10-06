from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import logging
import os
import re
import secrets
import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Request, UploadFile, File, Response, HTTPException, Body
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from groq import Groq

from persistence.models import Conversation, DocumentRecord, MessageRecord
from persistence.memory import ConversationMemory, format_memory_hints
from persistence.store import SQLiteConversationStore
from chroma_cloud import delete_conversation_collection
from app_helpers import format_conversation_export
from backend.schemas import (
    AddPasswordRequest,
    AuthUserResponse,
    ConversationDetail,
    ConversationInfo,
    DocumentInfo,
    MessageInfo,
    QuestionRequest,
    ErrorResponse,
    AnswerResponse,
    RejectedUpload,
    UploadBatchResponse,
    MemorySettingsRequest,
    SigninRequest,
    SignupRequest,
    RequestPasswordReset,
    ResetPasswordRequest,
)

from agentic_rag import RAGService, BaselineRAGService, tag_legacy_document_chunks
from backend.auth import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    SignInRateLimiter,
    hash_password,
    hash_session_token,
    new_session_token,
    normalize_email,
    signin_rate_limiter,
    verify_password,
)
from ingest import load_uploaded_documents, load_uploaded_documents_async
from guardrails.config import sensitive_content_enabled
from guardrails.sensitive_content import redact_persisted_value
from guardrails.scope import SCOPE_BLOCK_MESSAGE
from chitchat.classifier import classify_chitchat
from chitchat.responses import choose_response
from observability import (
    bind_context,
    flush_langfuse,
    get_tracer,
    langfuse_observation_context,
    record_audio_error,
)
from backend.voice_text_cleanup import clean_text_for_speech

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
if os.getenv("LANGFUSE_HOST") is None and os.getenv("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL")
if os.getenv("LANGFUSE_ENABLED") is None and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    os.environ["LANGFUSE_ENABLED"] = "true"
logger = logging.getLogger(__name__)
APP_ENV = os.getenv("APP_ENV", "development").strip().casefold()
COOKIE_SECURE = APP_ENV == "production" or os.getenv("RAG_COOKIE_SECURE", "0").strip().casefold() in {
    "1", "true", "yes", "on",
}
COOKIE_SAMESITE = os.getenv("RAG_COOKIE_SAMESITE", "lax").strip().casefold()
if COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    COOKIE_SAMESITE = "lax"
if COOKIE_SAMESITE == "none" and not COOKIE_SECURE:
    raise RuntimeError("SameSite=None cookies require RAG_COOKIE_SECURE=1 or APP_ENV=production")
CSRF_TOKEN_TTL_SECONDS = 15 * 60
PASSWORD_RESET_TTL_SECONDS = 45 * 60
GOOGLE_OAUTH_STATE_TTL_SECONDS = 10 * 60
GOOGLE_FLOW_COOKIE_NAME = "rag_google_flow"
GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"
GENERIC_RESET_RESPONSE = {"message": "If the account exists, reset instructions have been sent."}

app = FastAPI(title="Agentic RAG Backend")

FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").split(",")
    if origin.strip()
]
if not FRONTEND_ORIGINS or "*" in FRONTEND_ORIGINS:
    raise RuntimeError("FRONTEND_ORIGIN must contain one or more explicit origins, never '*'")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("FRONTEND_URL"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def bind_observability_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    conversation_id = request.path_params.get("conversation_id")
    request.state.request_id = request_id
    with bind_context(request_id=request_id, conversation_id=conversation_id):
        tracer = get_tracer("backend.http")
        with tracer.start_as_current_span("http.request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("request_id", request_id)
            if conversation_id:
                span.set_attribute("conversation.id", conversation_id)
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            return response


@app.middleware("http")
async def validate_conversation_uuid_paths(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if request.method == "POST" and request.url.path.endswith("/documents") and content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": {"code": "request_too_large", "message": "The selected files exceed the total upload limit."}},
                )
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": {"code": "invalid_content_length", "message": "The upload could not be read."}})
    parts = request.url.path.strip("/").split("/")
    if len(parts) >= 3 and parts[:2] == ["api", "conversations"]:
        try:
            UUID(parts[2])
        except (ValueError, TypeError, AttributeError):
            return JSONResponse(status_code=404, content={"detail": "Conversation not found"})
    return await call_next(request)

# In-process registry mapping conversation_id -> service instance
_SERVICE_REGISTRY: dict[str, RAGService | BaselineRAGService] = {}
# Active cancel events for long-running SSE requests per conversation
_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CONVERSATION_LOCKS: dict[str, threading.RLock] = {}
_CONVERSATION_LOCKS_GUARD = threading.Lock()
_MEMORY_MANAGER: ConversationMemory | None = None
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "10"))
MAX_FILES_PER_CONVERSATION = int(os.getenv("MAX_FILES_PER_CONVERSATION", "30"))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(50 * 1024 * 1024)))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


def derive_auto_title(question: str, answer: str | None = None, maximum: int = 50) -> str:
    source = (answer or question or '').strip()
    if not source:
        return 'New conversation'

    normalized = ' '.join(source.split())
    if len(normalized) <= maximum:
        return normalized or 'New conversation'

    prefix = normalized[: maximum + 1]
    boundary = prefix.rfind(' ')
    title = (prefix[:boundary] if boundary > 0 else prefix[:maximum]).rstrip()
    return title or 'New conversation'


def _require_uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Conversation not found")


_CSRF_EXEMPT_PATHS = {"/api/auth/signup", "/api/auth/signin"}
_PREAUTH_CSRF_PATHS = {"/api/auth/request-reset", "/api/auth/reset-password"}
_CSRF_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
reset_request_limiter = SignInRateLimiter(limit=3, window_seconds=60 * 60)
preauth_csrf_limiter = SignInRateLimiter(limit=60, window_seconds=CSRF_TOKEN_TTL_SECONDS)


def _csrf_required(request: Request) -> bool:
    if request.method in _CSRF_UNSAFE_METHODS:
        return True
    if request.method != "GET":
        return False
    parts = request.url.path.strip("/").split("/")
    if len(parts) == 3 and parts[:2] == ["api", "conversations"]:
        return True
    return len(parts) == 4 and parts[:2] == ["api", "conversations"] and parts[3] in {"documents", "status"}


@app.middleware("http")
async def enforce_csrf(request: Request, call_next):
    if not _csrf_required(request) or request.url.path in _CSRF_EXEMPT_PATHS:
        return await call_next(request)

    submitted_token = request.headers.get("x-csrf-token", "")
    if not submitted_token:
        return JSONResponse(status_code=403, content={"detail": "CSRF token missing or invalid"})

    now = _utc_iso()
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    session = None
    if session_token:
        session = _get_store().get_session_details(hash_session_token(session_token), now)
    if session is not None:
        expected_hash = str(session.get("csrf_token_hash") or "")
        if not expected_hash or not secrets.compare_digest(
            hash_session_token(submitted_token), expected_hash
        ):
            return JSONResponse(status_code=403, content={"detail": "CSRF token missing or invalid"})
        request.state.session_expires_at = session["expires_at"]
        request.state.session_token_hash = session["token_hash"]
        return await call_next(request)

    if request.url.path in _PREAUTH_CSRF_PATHS and _get_store().consume_preauth_csrf_token(
        hash_session_token(submitted_token), now
    ):
        return await call_next(request)
    if request.url.path in _PREAUTH_CSRF_PATHS:
        return JSONResponse(status_code=403, content={"detail": "CSRF token missing or invalid"})

    return await call_next(request)


def _conversation_lock(conversation_id: str) -> threading.RLock:
    with _CONVERSATION_LOCKS_GUARD:
        return _CONVERSATION_LOCKS.setdefault(conversation_id, threading.RLock())


def prune_default_titles(store: SQLiteConversationStore | None = None) -> int:
    store = store or _get_store()
    pruned = 0
    for conversation in store.list_conversations():
        current_title = (conversation.title or '').strip()
        if current_title not in {'', 'New conversation'}:
            continue
        question = next(
            (
                message.content
                for message in reversed(store.get_messages(conversation.id))
                if message.role == 'user'
            ),
            None,
        )
        if not question:
            continue
        store.rename_conversation(conversation.id, derive_auto_title(question, '', maximum=50))
        pruned += 1
    return pruned


def _schedule_conversation_title_refresh(conversation_id: str, question: str, answer: str | None) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _refresh_title() -> None:
        await asyncio.sleep(0)
        store = _get_store()
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            return

        fallback_title = derive_auto_title(question, '', maximum=50)
        current_title = (conversation.title or '').strip() or 'New conversation'
        if current_title not in {'New conversation', fallback_title} and current_title != '':
            return

        new_title = derive_auto_title(question, answer, maximum=50)
        if new_title == 'New conversation':
            new_title = fallback_title
        store.rename_conversation(conversation_id, new_title)

    loop.create_task(_refresh_title())


def _document_info(document) -> DocumentInfo:
    return DocumentInfo(**asdict(document))


def _conversation_info(store: SQLiteConversationStore, conversation_id: str) -> ConversationInfo:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationInfo(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        document_count=conversation.document_count,
        message_count=conversation.message_count,
    )


@app.on_event("startup")
def _startup():
    # ensure DB file path exists and clean up placeholder titles from older chats
    store = SQLiteConversationStore()
    try:
        prune_default_titles(store)
    except Exception:  # pragma: no cover - best-effort maintenance
        logger.exception("Could not prune stale default conversation titles")

    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on", "enabled"}
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://us.cloud.langfuse.com"
    logger.warning(
        "LANGFUSE_RUNTIME_CONFIG enabled=%s public_key=%s secret_key=%s host=%s",
        langfuse_enabled,
        bool(public_key),
        bool(secret_key),
        host,
    )


@app.on_event("shutdown")
def _shutdown():
    try:
        flush_langfuse()
    except Exception:  # pragma: no cover - best-effort shutdown cleanup
        logger.exception("Could not flush Langfuse on shutdown")


def _get_store() -> SQLiteConversationStore:
    return SQLiteConversationStore()
    

def _utc_iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _request_session_expired(request: Request) -> bool:
    expiry = getattr(request.state, "session_expires_at", None)
    if not expiry:
        return False
    try:
        expiry_time = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True
    return datetime.now(timezone.utc) >= expiry_time


def _create_auth_session(response: Response, user_id: str, store: SQLiteConversationStore) -> None:
    token = new_session_token()
    csrf_token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=SESSION_TTL_SECONDS)
    store.create_session(
        hash_session_token(token),
        user_id,
        _utc_iso(now),
        _utc_iso(expires_at),
        csrf_token_hash=hash_session_token(csrf_token),
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        expires=expires_at,
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite=COOKIE_SAMESITE,
    )


def _google_oauth_settings() -> tuple[str, str, str]:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError("Google OAuth is not configured")
    return client_id, client_secret, redirect_uri


async def _google_authorization_url(state: str) -> str:
    from authlib.integrations.httpx_client import AsyncOAuth2Client

    client_id, client_secret, redirect_uri = _google_oauth_settings()
    async with AsyncOAuth2Client(
        client_id=client_id,
        client_secret=client_secret,
        scope="openid email profile",
        redirect_uri=redirect_uri,
        token_endpoint_auth_method="client_secret_post",
    ) as client:
        logger.info("Google OAuth outgoing redirect_uri=%s", redirect_uri)
        authorization_url, _ = client.create_authorization_url(
            GOOGLE_AUTHORIZATION_ENDPOINT,
            state=state,
            redirect_uri=redirect_uri,
            scope="openid email profile",
        )
    return authorization_url


async def _fetch_google_userinfo(code: str) -> dict[str, Any]:
    from authlib.integrations.httpx_client import AsyncOAuth2Client

    client_id, client_secret, redirect_uri = _google_oauth_settings()
    async with AsyncOAuth2Client(
        client_id=client_id,
        client_secret=client_secret,
        scope="openid email profile",
        redirect_uri=redirect_uri,
        token_endpoint_auth_method="client_secret_post",
    ) as client:
        token = await client.fetch_token(
            GOOGLE_TOKEN_ENDPOINT,
            code=code,
            redirect_uri=redirect_uri,
            grant_type="authorization_code",
        )
        client.token = token
        response = await client.get(GOOGLE_USERINFO_ENDPOINT)
        response.raise_for_status()
        profile = response.json()
    return profile if isinstance(profile, dict) else {}


def _google_flow_cookie(
    response: Response,
    value: str = "",
    *,
    clear: bool = False,
) -> None:
    if clear:
        response.delete_cookie(
            key=GOOGLE_FLOW_COOKIE_NAME,
            path="/",
            secure=COOKIE_SECURE,
            httponly=True,
            samesite=COOKIE_SAMESITE,
        )
    else:
        response.set_cookie(
            key=GOOGLE_FLOW_COOKIE_NAME,
            value=value,
            max_age=GOOGLE_OAUTH_STATE_TTL_SECONDS,
            path="/",
            secure=COOKIE_SECURE,
            httponly=True,
            samesite=COOKIE_SAMESITE,
        )


def _google_failure_redirect() -> RedirectResponse:
    response = RedirectResponse(
        f"{FRONTEND_ORIGINS[0]}/login?auth_error=google_signin_failed",
        status_code=303,
    )
    _google_flow_cookie(response, clear=True)
    return response


def get_current_user(request: Request) -> dict[str, str]:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    logger.debug("Session cookie present: %s", bool(token))
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    session = _get_store().get_session_details(
        hash_session_token(token),
        _utc_iso(),
    )
    if session is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    request.state.session_expires_at = session["expires_at"]
    request.state.session_token_hash = session["token_hash"]
    return {
        "id": session["user_id"],
        "email": session["email"],
        "created_at": session["created_at"],
        "is_admin": session["is_admin"],
    }


def _require_owned_conversation(
    conversation_id: str,
    current_user: dict[str, str],
    store: SQLiteConversationStore | None = None,
):
    normalized_id = _require_uuid(conversation_id)
    conversation = (store or _get_store()).get_conversation(normalized_id)
    if conversation is None or conversation.user_id != current_user["id"]:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def get_owned_conversation(
    conversation_id: str,
    current_user: dict[str, str] = Depends(get_current_user),
) -> Conversation:
    return _require_owned_conversation(conversation_id, current_user)


def get_owned_document(
    conversation_id: str,
    document_id: str,
    conversation=Depends(get_owned_conversation),
    current_user: dict[str, str] = Depends(get_current_user),
) -> DocumentRecord:
    try:
        normalized_document_id = str(UUID(document_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Document not found") from None
    document = _get_store().get_document(normalized_document_id)
    if (
        document is None
        or document.conversation_id != conversation.id
        or document.user_id != current_user["id"]
    ):
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def _configured_emails(environment_name: str) -> set[str]:
    return {
        value.strip().casefold()
        for value in os.getenv(environment_name, "").split(",")
        if value.strip()
    }


@app.get("/api/auth/google/login")
async def google_login():
    state = secrets.token_urlsafe(32)
    flow_token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    store = _get_store()
    try:
        authorization_url = await _google_authorization_url(state)
        store.create_oauth_state(
            hash_session_token(state),
            hash_session_token(flow_token),
            _utc_iso(now),
            _utc_iso(now + timedelta(seconds=GOOGLE_OAUTH_STATE_TTL_SECONDS)),
        )
    except Exception as error:
        logger.warning("Google OAuth login could not be initiated (%s)", type(error).__name__)
        return _google_failure_redirect()

    response = RedirectResponse(authorization_url, status_code=302)
    _google_flow_cookie(response, flow_token)
    return response


@app.get("/api/auth/google/callback")
async def google_callback(request: Request):
    state = request.query_params.get("state", "")
    flow_token = request.cookies.get(GOOGLE_FLOW_COOKIE_NAME, "")
    if not state or not flow_token:
        return _google_failure_redirect()

    now = _utc_iso()
    store = _get_store()
    if not store.consume_oauth_state(
        hash_session_token(state),
        hash_session_token(flow_token),
        now,
    ):
        return _google_failure_redirect()

    code = request.query_params.get("code", "")
    if request.query_params.get("error") or not code:
        return _google_failure_redirect()

    try:
        profile = await _fetch_google_userinfo(code)
        email = normalize_email(str(profile.get("email") or ""))
        subject = str(profile.get("sub") or "").strip()
        if profile.get("email_verified") is not True or not email or not subject:
            return _google_failure_redirect()

        user = store.resolve_google_user(
            email,
            subject,
            is_admin=email in _configured_emails("RAG_ADMIN_EMAILS"),
        )
        if user is None:
            return _google_failure_redirect()

        response = RedirectResponse(f"{FRONTEND_ORIGINS[0]}/app", status_code=303)
        _create_auth_session(response, user["id"], store)
        _google_flow_cookie(response, clear=True)
        return response
    except Exception as error:
        logger.warning("Google OAuth callback failed (%s)", type(error).__name__)
        return _google_failure_redirect()


@app.post("/api/auth/signup", response_model=AuthUserResponse, status_code=201)
def signup(payload: SignupRequest, response: Response):
    store = _get_store()
    email = normalize_email(payload.email)
    user = store.create_user(
        email,
        hash_password(payload.password),
        is_admin=email in _configured_emails("RAG_ADMIN_EMAILS"),
    )
    if user is None:
        raise HTTPException(status_code=409, detail="Email is already registered")
    _create_auth_session(response, user["id"], store)
    return user


@app.post("/api/auth/signin", response_model=AuthUserResponse)
def signin(payload: SigninRequest, request: Request, response: Response):
    email = normalize_email(payload.email)
    client_ip = request.client.host if request.client else "unknown"
    limiter_key = (email, client_ip)
    if signin_rate_limiter.is_limited(limiter_key):
        raise HTTPException(status_code=429, detail="Too many sign-in attempts. Try again later.")

    store = _get_store()
    user = store.get_user_by_email(email)
    valid = verify_password(payload.password, user["password_hash"] if user else None)
    if not user or not valid:
        signin_rate_limiter.record_failure(limiter_key)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    signin_rate_limiter.clear(limiter_key)
    _create_auth_session(response, user["id"], store)
    return {key: user[key] for key in ("id", "email", "created_at")}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        _get_store().delete_session(hash_session_token(token))
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite=COOKIE_SAMESITE,
    )
    return {"logged_out": True}


@app.get("/api/auth/me", response_model=AuthUserResponse)
def auth_me(current_user: dict[str, str] = Depends(get_current_user)):
    return current_user


@app.get("/api/auth/password-status")
def password_status(current_user: dict[str, str] = Depends(get_current_user)):
    return {"has_password": _get_store().user_has_password(current_user["id"])}


@app.post("/api/auth/add-password")
def add_password(
    payload: AddPasswordRequest,
    current_user: dict[str, str] = Depends(get_current_user),
):
    if not _get_store().set_password_if_missing(
        current_user["id"],
        hash_password(payload.password),
    ):
        raise HTTPException(status_code=409, detail="A password is already set for this account.")
    return {"password_added": True}


@app.get("/api/auth/csrf")
def get_csrf_token(request: Request):
    store = _get_store()
    session_cookie = request.cookies.get(SESSION_COOKIE_NAME)
    session = (
        store.get_session_details(hash_session_token(session_cookie), _utc_iso())
        if session_cookie
        else None
    )
    now = datetime.now(timezone.utc)
    if session is not None:
        csrf_token = hash_session_token(f"csrf:{session['token_hash']}")
        store.set_session_csrf_token(session["token_hash"], hash_session_token(csrf_token))
        expires_at = session["expires_at"]
    else:
        csrf_token = secrets.token_urlsafe(32)
        client_ip = request.client.host if request.client else "unknown"
        limiter_key = (client_ip, "preauth-csrf")
        if preauth_csrf_limiter.is_limited(limiter_key):
            raise HTTPException(status_code=429, detail="Too many security-token requests. Try again later.")
        preauth_csrf_limiter.record_failure(limiter_key)
        store.add_preauth_csrf_token(
            hash_session_token(csrf_token),
            _utc_iso(now),
            _utc_iso(now + timedelta(seconds=CSRF_TOKEN_TTL_SECONDS)),
        )
        expires_at = _utc_iso(now + timedelta(seconds=CSRF_TOKEN_TTL_SECONDS))
    return {"csrf_token": csrf_token, "expires_at": expires_at}


@app.post("/api/voice/transcribe")
async def transcribe_voice(
    audio: UploadFile = File(...),
    current_user: dict[str, str] = Depends(get_current_user),
):
    _ = current_user
    content_type = (audio.content_type or "").split(";", 1)[0].strip().casefold()
    filename_by_type = {"audio/webm": "recording.webm", "audio/mp4": "recording.mp4"}
    filename = filename_by_type.get(content_type)
    if filename is None:
        raise HTTPException(status_code=415, detail="Use a WebM or MP4 audio recording.")

    audio_bytes = await audio.read(MAX_UPLOAD_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The recording is empty.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The recording exceeds the audio upload limit.")

    try:
        result = await run_in_threadpool(
            lambda: Groq().audio.transcriptions.create(
                file=(filename, audio_bytes, audio.content_type or content_type),
                model="whisper-large-v3-turbo",
                response_format="json",
            )
        )
    except Exception as error:  # noqa: BLE001
        record_audio_error("voice_transcription", type(error).__name__)
        logger.exception("Voice transcription failed")
        raise HTTPException(status_code=502, detail=f"Transcription failed: {error}") from error

    transcript = str(getattr(result, "text", "") or "").strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech was recognized. Try recording again.")
    return {"transcript": transcript}


@app.post("/api/auth/request-reset", status_code=202)
def request_password_reset(payload: RequestPasswordReset, request: Request):
    email = normalize_email(payload.email)
    client_ip = request.client.host if request.client else "unknown"
    limiter_key = (email, client_ip)
    if reset_request_limiter.is_limited(limiter_key):
        return GENERIC_RESET_RESPONSE
    reset_request_limiter.record_failure(limiter_key)

    user = _get_store().get_user_by_email(email)
    if user is not None:
        token = new_session_token()
        now = datetime.now(timezone.utc)
        _get_store().create_password_reset_token(
            hash_session_token(token),
            user["id"],
            _utc_iso(now),
            _utc_iso(now + timedelta(seconds=PASSWORD_RESET_TTL_SECONDS)),
        )
        if APP_ENV != "production":
            logger.warning(
                "DEV ONLY password reset URL for %s: %s/?reset_token=%s",
                email,
                FRONTEND_ORIGINS[0],
                token,
            )
    elif APP_ENV == "production":
        logger.error("Password reset delivery is not configured")
    return GENERIC_RESET_RESPONSE


@app.post("/api/auth/reset-password")
def reset_password(payload: ResetPasswordRequest, response: Response):
    user_id = _get_store().reset_password_with_token(
        hash_session_token(payload.token),
        hash_password(payload.password),
        _utc_iso(),
    )
    if user_id is None:
        raise HTTPException(status_code=400, detail="Reset token is invalid or expired")
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite=COOKIE_SAMESITE,
    )
    return {"password_reset": True}


def _get_memory_manager() -> ConversationMemory:
    global _MEMORY_MANAGER
    store = _get_store()
    if _MEMORY_MANAGER is None or _MEMORY_MANAGER.store.db_path != store.db_path:
        _MEMORY_MANAGER = ConversationMemory(store)
    return _MEMORY_MANAGER


def _should_sample_for_evaluation() -> bool:
    from eval.production.sampling import should_sample_request

    return should_sample_request()


def _run_post_response_evaluation(message: dict) -> None:
    from eval.production.sampling import run_sampled_message_evaluation

    run_sampled_message_evaluation(message)


def _persistence_safe(value):
    return redact_persisted_value(value) if sensitive_content_enabled() else value


def _speech_error_message(error: Exception) -> str:
    payload = getattr(error, "body", {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    error_body = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    code = error_body.get("code") if isinstance(error_body, dict) else None
    if code == "model_terms_required":
        model_slug = "canopylabs%2Forpheus-v1-english"
        return (
            "Audio output is unavailable because the Groq model terms must be accepted by an organization admin. "
            f"Ask your organization admin to review and accept the model terms for {model_slug}."
        )
    return str(error)


def _voice_segments_for_text(text: str) -> list[str]:
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip()]


def _generate_voice_chunk(text: str, sequence: int) -> dict | None:
    speech = Groq().audio.speech.create(
        input=text,
        model="canopylabs/orpheus-v1-english",
        voice="autumn",
        response_format="wav",
    )
    if speech is None:
        return None
    audio_bytes = speech.read()
    return {
        "sequence": sequence,
        "mime_type": "audio/wav",
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
    }


def _ensure_service(
    conversation_id: str,
    answer_mode: str = "agentic",
    passages_per_search: int = 4,
) -> RAGService | BaselineRAGService:
    svc = _SERVICE_REGISTRY.get(conversation_id)
    desired_type = BaselineRAGService if answer_mode == "traditional" else RAGService
    if isinstance(svc, desired_type):
        svc.top_k = passages_per_search
        if getattr(svc, "vector_store", None) is not None:
            svc.retriever = svc.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": passages_per_search},
            )
        return svc

    if desired_type is BaselineRAGService:
        svc = BaselineRAGService(
            conversation_id=conversation_id,
            top_k=passages_per_search,
        )
    else:
        svc = RAGService(
            conversation_id=conversation_id,
            top_k=passages_per_search,
        )
    _SERVICE_REGISTRY[conversation_id] = svc
    return svc


def _service_has_documents(service: object | None) -> bool:
    if service is None:
        return False
    has_documents = getattr(service, "has_documents", None)
    if callable(has_documents):
        return bool(has_documents())
    return bool(getattr(service, "documents", None) or getattr(service, "chunks", None))


class _ImmediateResponseService:
    def __init__(
        self,
        conversation_id: str,
        answer: str,
        route: str,
        *,
        category: str | None = None,
        confidence: float | None = None,
        guardrail: dict | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.last_reasoning = ""
        detail = (
            f"Handled {category.replace('_', ' ')} without searching documents."
            if category
            else "Returned a no-documents response without searching."
            if route == "no_documents_yet"
            else "Blocked by the scope guardrail before search."
        )
        self.last_trace = [{
            "step": "route",
            "route": route,
            **({"chitchat_category": category, "confidence": confidence} if category else {}),
            "retrieval_skipped": True,
            "detail": detail,
            **({"guardrail": guardrail} if guardrail else {}),
        }]
        if route == "scope_blocked":
            self.last_trace.append({
                "step": "abstain",
                "detail": "blocked by scope guardrail",
                "guardrail": guardrail or {},
            })
        self.answer = answer

    def ask_stream(self, question, cancel_event=None, on_trace=None):
        if on_trace:
            for item in self.last_trace:
                on_trace(dict(item))
        return iter([self.answer]), []


@app.post("/api/conversations", response_model=ConversationInfo)
def create_conversation(current_user: dict[str, str] = Depends(get_current_user)):
    store = _get_store()
    conv = store.create_conversation(user_id=current_user["id"])
    return ConversationInfo(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        document_count=0,
        message_count=0,
    )


@app.get("/api/conversations", response_model=list[ConversationInfo])
def list_conversations(current_user: dict[str, str] = Depends(get_current_user)):
    store = _get_store()
    items = store.list_conversations(user_id=current_user["id"])
    return [
        ConversationInfo(
            id=item.id,
            title=item.title,
            created_at=item.created_at,
            updated_at=item.updated_at,
            document_count=item.document_count,
            message_count=item.message_count,
        )
        for item in items
    ]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    conv = owned_conversation
    documents = store.list_document_records(conversation_id)
    ready_documents = [item for item in documents if item.status == "ready"]
    if ready_documents:
        service = _ensure_service(conversation_id)
        has_chroma_index = _service_has_documents(service)
        if has_chroma_index:
            for document in ready_documents:
                if document.sha256 is None and getattr(service, "vector_store", None) is not None:
                    tag_legacy_document_chunks(service.vector_store, document.id, document.filename)
    else:
        has_chroma_index = False
    if ready_documents and not has_chroma_index:
        for item in documents:
            if item.status == "ready":
                store.update_document(
                    item.id,
                    status="failed",
                    error_code="needs_reprocessing",
                    error_message="Upload these files again to continue.",
                )
        documents = store.list_document_records(conversation_id)
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        document_count=conv.document_count,
        message_count=conv.message_count,
        documents=[_document_info(item) for item in documents],
        messages=[MessageInfo(**asdict(message)) for message in store.get_messages(conversation_id)],
    )


@app.get("/api/conversations/{conversation_id}/memory")
def get_conversation_memory_settings(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    return {"enabled": store.get_conversation_memory_enabled(conversation_id)}


@app.put("/api/conversations/{conversation_id}/memory")
def set_conversation_memory_settings(
    conversation_id: str,
    payload: MemorySettingsRequest,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    store.set_conversation_memory_enabled(conversation_id, payload.enabled)
    if not payload.enabled:
        _get_memory_manager().delete_conversation(conversation_id)
    return {"enabled": payload.enabled}


@app.get("/api/memory/settings")
def get_memory_settings(current_user: dict[str, str] = Depends(get_current_user)):
    return {"enabled": _get_store().get_global_memory_enabled(current_user["id"])}


@app.put("/api/memory/settings")
def set_memory_settings(
    payload: MemorySettingsRequest,
    current_user: dict[str, str] = Depends(get_current_user),
):
    enabled = _get_store().set_global_memory_enabled(payload.enabled, current_user["id"])
    return {"enabled": enabled}


@app.delete("/api/memory")
def clear_all_memory(current_user: dict[str, str] = Depends(get_current_user)):
    deleted = _get_memory_manager().clear_all(owner_id=current_user["id"])
    return {"deleted_turns": deleted}


def _process_document(conversation_id: str, document_id: str, filename: str, content: bytes) -> None:
    store = _get_store()
    if store.get_document(document_id) is None:
        return
    store.update_document(document_id, status="processing", error_code=None, error_message=None)

    class UploadedBytes:
        name = filename

        def getvalue(self):
            return content

    with bind_context(request_id=f"doc:{document_id}", conversation_id=conversation_id, user_id="background"):
        try:
            docs = load_uploaded_documents([UploadedBytes()])
            with _conversation_lock(conversation_id):
                if store.get_document(document_id) is None:
                    return
                service = _ensure_service(conversation_id)
                stats = service.add_documents(docs, document_id=document_id, filename=filename)
                store.update_document(
                    document_id,
                    status="ready",
                    pages=stats["pages"],
                    chunks=stats["chunks"],
                    error_code=None,
                    error_message=None,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Document processing failed for document %s", document_id)
            message = str(exc)
            if "scanned" in message.lower() or "image-based" in message.lower():
                code = "image_only_pdf"
                friendly = "This PDF has no readable text. Try a text-based copy."
            elif "encrypted" in message.lower():
                code = "encrypted_pdf"
                friendly = "This PDF is protected. Upload an unprotected copy."
            else:
                code = "processing_failed"
                friendly = "We could not read this file. Check the file and try again."
            store.update_document(document_id, status="failed", error_code=code, error_message=friendly)
        finally:
            flush_langfuse()


@app.post("/api/conversations/{conversation_id}/documents", response_model=UploadBatchResponse)
async def upload_documents(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=413, detail={"code": "too_many_files", "message": f"Choose no more than {MAX_FILES_PER_UPLOAD} files at a time."})

    existing = store.list_document_records(conversation_id)
    available_slots = max(0, MAX_FILES_PER_CONVERSATION - len(existing))
    total_bytes = 0
    accepted = []
    rejected = []
    seen_hashes: set[str] = set()
    for upload in files:
        filename = Path(upload.filename or "document").name
        extension = Path(filename).suffix.lower()
        if extension not in {".pdf", ".txt"}:
            rejected.append(RejectedUpload(filename=filename, error_code="unsupported_type", reason="Choose a PDF or plain text file."))
            continue
        content = await upload.read()
        total_bytes += len(content)
        if len(content) > MAX_UPLOAD_BYTES:
            rejected.append(RejectedUpload(filename=filename, error_code="file_too_large", reason=f"This file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per-file limit."))
            continue
        if total_bytes > MAX_REQUEST_BYTES:
            rejected.append(RejectedUpload(filename=filename, error_code="request_too_large", reason="The selected files exceed the total upload limit."))
            continue
        if extension == ".pdf" and not content.startswith(b"%PDF-"):
            rejected.append(RejectedUpload(filename=filename, error_code="invalid_pdf", reason="This file does not appear to be a valid PDF."))
            continue
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen_hashes or store.find_document_by_sha(conversation_id, digest):
            rejected.append(RejectedUpload(filename=filename, error_code="duplicate_file", reason="This file is already in this chat."))
            continue
        seen_hashes.add(digest)
        if available_slots <= 0:
            rejected.append(RejectedUpload(filename=filename, error_code="conversation_file_limit", reason="This chat has reached its file limit."))
            continue

        document = DocumentRecord(
            id=str(uuid4()),
            conversation_id=conversation_id,
            filename=filename,
            sha256=digest,
            size_bytes=len(content),
            pages=0,
            chunks=0,
            status="queued",
            error_code=None,
            error_message=None,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        store.create_document(document)
        background_tasks.add_task(_process_document, conversation_id, document.id, filename, content)
        accepted.append(document)
        available_slots -= 1

    return UploadBatchResponse(
        accepted=[_document_info(item) for item in accepted],
        rejected=rejected,
    )


@app.get("/api/conversations/{conversation_id}/documents")
def list_documents(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    records = store.list_document_records(conversation_id)
    ready_records = [item for item in records if item.status == "ready"]
    has_chroma_index = _service_has_documents(_ensure_service(conversation_id)) if ready_records else True
    if ready_records and not has_chroma_index:
        for document in records:
            if document.status == "ready":
                store.update_document(
                    document.id,
                    status="failed",
                    error_code="needs_reprocessing",
                    error_message="Upload these files again to continue.",
                )
        records = store.list_document_records(conversation_id)
    return [_document_info(document) for document in records]

@app.delete("/api/conversations/{conversation_id}/documents/{document_id}")
def delete_document(
    conversation_id: str,
    document_id: str,
    document: DocumentRecord = Depends(get_owned_document),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    document_id = document.id
    service = _SERVICE_REGISTRY.get(conversation_id)
    if service is None and document.status == "ready":
        service = _ensure_service(conversation_id)
    with _conversation_lock(conversation_id):
        if service is not None and getattr(service, "vector_store", None) is not None:
            if document.sha256 is None:
                tag_legacy_document_chunks(service.vector_store, document.id, document.filename)
            service.delete_document(document_id, document.filename)
        store.delete_document(conversation_id, document_id)
    return {"deleted": True}


@app.get("/api/conversations/{conversation_id}/export")
def export_conversation(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    msgs = store.get_messages(conversation_id)
    # convert MessageRecord objects to simple namespaces expected by formatter
    from types import SimpleNamespace

    simple = [SimpleNamespace(role=m.role, content=m.content, reasoning=m.reasoning, trace=m.trace, sources=m.sources) for m in msgs]
    text = format_conversation_export(simple)
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="agentic-rag-chat.txt"'},
    )


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    msgs = store.get_messages(conversation_id)
    return [asdict(message) for message in msgs]


@app.post("/api/conversations/{conversation_id}/messages")
def post_message(
    conversation_id: str,
    message: dict = Body(...),
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    stored = store.append_message(
        conversation_id,
        _persistence_safe({**message, "conversation_id": conversation_id}),
    )
    return asdict(stored)


@app.post("/api/conversations/{conversation_id}/messages/{message_id}/feedback")
def message_feedback(
    conversation_id: str,
    message_id: str,
    payload: dict = Body(...),
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation

    rating = payload.get("rating")
    if rating is None:
        raise HTTPException(status_code=400, detail="rating is required")

    try:
        normalized = int(rating)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="rating must be an integer") from None

    if normalized not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="rating must be one of -1, 0, or 1")

    try:
        updated = store.set_message_rating(
            conversation_id,
            message_id,
            normalized,
            _persistence_safe(payload.get("comment")),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "conversation_id": conversation_id,
        "message_id": updated.id,
        "rating": updated.rating,
        "feedback": updated.feedback,
    }


async def _sse_event_publisher(
    question: str,
    service: RAGService | BaselineRAGService,
    cancel_event: threading.Event,
    request: Request,
    *,
    show_reasoning_steps: bool,
    memory_hits: list[dict],
    on_finish,
):
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    answer_parts: list[str] = []
    result: dict = {}
    persisted = False

    async def finish(content: str, sources: list, trace: list, status: str) -> None:
        nonlocal persisted
        if persisted:
            return
        await run_in_threadpool(on_finish, content, sources, trace, status)
        persisted = True

    def publish(kind: str, payload: dict | str | None = None) -> None:
        if loop.is_closed():
            return
        pending = events.put((kind, payload))
        try:
            asyncio.run_coroutine_threadsafe(pending, loop).result()
        except RuntimeError:
            pending.close()

    def run_agent():
        try:
            publish("memory_hits", {"hits": memory_hits})
            kwargs = {}
            parameters = inspect.signature(service.ask_stream).parameters
            if "cancel_event" in parameters:
                kwargs["cancel_event"] = cancel_event
            if "on_trace" in parameters:
                kwargs["on_trace"] = lambda item: publish("trace", item)
            if "document_ids" in parameters:
                kwargs["document_ids"] = request.state.document_ids
            if "document_sources" in parameters:
                kwargs["document_sources"] = request.state.document_sources
            if "history" in parameters:
                kwargs["history"] = request.state.history
            if "memory_hints" in parameters:
                kwargs["memory_hints"] = request.state.memory_hints
            for setting in ("rerank_enabled", "rerank_candidates", "rerank_top_n"):
                if setting in parameters:
                    kwargs[setting] = getattr(request.state, setting)
            if "map_reduce_mode" in parameters:
                kwargs["map_reduce_mode"] = request.state.map_reduce_mode
            if "on_map_progress" in parameters:
                kwargs["on_map_progress"] = lambda item: publish("map_progress", item)
            lock_id = getattr(service, "conversation_id", request.path_params.get("conversation_id", ""))
            with bind_context(
                request_id=request.state.request_id,
                conversation_id=lock_id,
                user_id=request.state.user_id,
            ), langfuse_observation_context(
                conversation_id=lock_id,
                user_id=request.state.user_id,
                tags=["rag", "ask"],
            ):
                with _conversation_lock(lock_id):
                    stream, docs = service.ask_stream(question, **kwargs)
                iterator = stream() if callable(stream) else iter(stream)
                for chunk in iterator:
                    if cancel_event.is_set():
                        break
                    text = str(chunk)
                    if not text:
                        continue
                    answer_parts.append(text)
                    publish("token", {"token": text})
                result["sources"] = [
                    {
                        "filename": (getattr(doc, "metadata", {}) or {}).get("source")
                        or (getattr(doc, "metadata", {}) or {}).get("document_name")
                        or "Unknown file",
                        "page": (getattr(doc, "metadata", {}) or {}).get("page", "?"),
                        "document_id": (getattr(doc, "metadata", {}) or {}).get("document_id"),
                        "snippet": (getattr(doc, "page_content", "") or "")[:400],
                        "source": getattr(doc, "page_content", "") or "",
                        "rerank_score": (getattr(doc, "metadata", {}) or {}).get("rerank_score"),
                    }
                    for doc in docs
                ]
                result["trace"] = list(getattr(service, "last_trace", []) or [])
                result["reasoning"] = getattr(service, "last_reasoning", "")
                flush_langfuse()
                if getattr(request.state, "voice_output", False):
                    answer_text = "".join(answer_parts)
                    for sequence, sentence in enumerate(_voice_segments_for_text(answer_text), start=1):
                        try:
                            chunk = _generate_voice_chunk(sentence, sequence)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Voice output skipped for sentence: %s", exc)
                            continue
                        if chunk is not None:
                            publish("audio_chunk", chunk)
            publish("complete", None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Answer streaming failed")
            publish("error", {"error": "The answer could not be completed. Please try again."})

    worker = threading.Thread(target=run_agent, daemon=True)
    worker.start()

    try:
        while True:
            if await request.is_disconnected():
                cancel_event.set()
                await finish(
                    "".join(answer_parts),
                    [],
                    list(getattr(service, "last_trace", []) or []),
                    "stopped",
                )
                return
            if _request_session_expired(request):
                cancel_event.set()
                await finish(
                    "".join(answer_parts),
                    [],
                    list(getattr(service, "last_trace", []) or []),
                    "stopped",
                )
                return
            if cancel_event.is_set():
                await finish(
                    "".join(answer_parts),
                    [],
                    list(getattr(service, "last_trace", []) or []),
                    "stopped",
                )
                yield "event: cancelled\ndata: {}\n\n"
                return
            try:
                kind, payload = await asyncio.wait_for(events.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            if kind == "trace":
                yield f"event: trace\ndata: {json.dumps(payload)}\n\n"
            elif kind == "token":
                if _request_session_expired(request):
                    cancel_event.set()
                    await finish(
                        "".join(answer_parts),
                        [],
                        list(getattr(service, "last_trace", []) or []),
                        "stopped",
                    )
                    return
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"
            elif kind == "memory_hits":
                yield f"event: memory_hits\ndata: {json.dumps(payload)}\n\n"
            elif kind == "audio_chunk":
                yield f"event: audio_chunk\ndata: {json.dumps(payload)}\n\n"
            elif kind == "map_progress":
                yield f"event: map_progress\ndata: {json.dumps(payload)}\n\n"
            elif kind == "error":
                await finish(
                    "".join(answer_parts) or payload.get("error", "The answer could not be completed."),
                    [],
                    list(getattr(service, "last_trace", []) or []),
                    "error",
                )
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return
            elif kind == "complete":
                final = {
                    "answer": "".join(answer_parts),
                    "sources": result.get("sources", []),
                    "trace": result.get("trace", []),
                    "reasoning": result.get("reasoning", "") if show_reasoning_steps else None,
                }
                await finish(final["answer"], final["sources"], final["trace"], "complete")
                yield f"event: answer\ndata: {json.dumps(final)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
    finally:
        if await request.is_disconnected():
            cancel_event.set()
            await finish(
                "".join(answer_parts),
                [],
                list(getattr(service, "last_trace", []) or []),
                "stopped",
            )


@app.post("/api/conversations/{conversation_id}/questions")
async def ask_question(
    conversation_id: str,
    request: Request,
    payload: QuestionRequest,
    current_user: dict[str, str] = Depends(get_current_user),
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation

    documents = store.list_document_records(conversation_id)
    ready = [document for document in documents if document.status == "ready"]
    fast_path_service = None
    if not ready:
        classification = classify_chitchat(payload.question)
        scope_result = classification.get("scope_result") or check_scope(
            payload.question,
            enabled=guardrail_enabled("scope"),
        )
        category = classification.get("category")
        if scope_result.get("outcome") == "blocked" and not category:
            fast_path_service = _ImmediateResponseService(
                conversation_id,
                SCOPE_BLOCK_MESSAGE,
                "scope_blocked",
                guardrail={"check": "scope", **scope_result},
            )
        elif category:
            fast_path_service = _ImmediateResponseService(
                conversation_id,
                choose_response(str(category), conversation_id),
                "chitchat",
                category=str(category),
                confidence=float(classification["confidence"]),
            )
        else:
            fast_path_service = _ImmediateResponseService(
                conversation_id,
                choose_response("no_documents_yet", conversation_id),
                "no_documents_yet",
                category="no_documents_yet",
                confidence=1.0,
            )
        selected = []
    elif payload.document_ids is None:
        selected = ready
    else:
        ready_by_id = {document.id: document for document in ready}
        if any(document_id not in ready_by_id for document_id in payload.document_ids):
            raise HTTPException(status_code=404, detail={"code": "document_not_found", "message": "One of the selected files is no longer available."})
        selected = [ready_by_id[document_id] for document_id in payload.document_ids]
        if not selected:
            raise HTTPException(status_code=409, detail={"code": "no_documents_selected", "message": "Select at least one ready file."})

    history = [
        {"role": message.role, "content": message.content}
        for message in store.get_messages(conversation_id)[-6:]
    ]
    memory_manager = _get_memory_manager()
    memory_hits = []
    memory_hints = "No related past-chat hints."
    if fast_path_service is None and (
        memory_manager.store.get_global_memory_enabled(current_user["id"])
        and memory_manager.store.get_conversation_memory_enabled(conversation_id)
    ):
        memory_hits = await run_in_threadpool(
            memory_manager.search,
            payload.question,
            exclude_conversation_id=conversation_id,
            limit=3,
            owner_id=current_user["id"],
        )
        memory_hints = format_memory_hints(memory_hits)
    stored_question = _persistence_safe(payload.question)
    store.append_message(
        conversation_id,
        MessageRecord(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=stored_question,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        ),
    )
    request.state.document_ids = [document.id for document in selected]
    request.state.document_sources = [document.filename for document in selected]
    request.state.history = history
    request.state.memory_hints = memory_hints
    request.state.voice_output = bool(payload.voice_output)
    request.state.rerank_enabled = payload.rerank_enabled
    request.state.rerank_candidates = payload.rerank_candidates
    request.state.rerank_top_n = payload.rerank_top_n
    request.state.map_reduce_mode = payload.map_reduce_mode
    request.state.user_id = current_user["id"]
    sample_for_evaluation = fast_path_service is None and _should_sample_for_evaluation()
    evaluation_context: dict = {}

    if fast_path_service is not None:
        service = fast_path_service
    else:
        service = await run_in_threadpool(
            _ensure_service,
            conversation_id,
            payload.answer_mode,
            payload.passages_per_search,
        )
        if isinstance(service, RAGService):
            service.max_retries = payload.max_retries
    if getattr(service, "vector_store", None) is not None:
        for document in selected:
            if document.sha256 is None:
                tag_legacy_document_chunks(service.vector_store, document.id, document.filename)

    def persist_assistant(content, sources, trace, status):
        message_id = str(uuid4())
        stored_content = _persistence_safe(content)
        stored_sources = _persistence_safe(sources)
        stored_trace = _persistence_safe(trace)
        store.append_message(
            conversation_id,
            MessageRecord(
                id=message_id,
                conversation_id=conversation_id,
                role="assistant",
                content=stored_content,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                sources=stored_sources,
                trace=stored_trace,
                status=status,
            ),
        )
        if sample_for_evaluation:
            evaluation_context.update({
                "message_id": message_id,
                "conversation_id": conversation_id,
                "question": payload.question,
                "answer": content,
                "sources": sources,
                "trace": trace,
                "status": status,
            })
        if status == "complete" and fast_path_service is None:
            _schedule_conversation_title_refresh(conversation_id, stored_question, stored_content)
            try:
                memory_manager.record_turn(
                    conversation_id,
                    stored_question,
                    stored_content,
                    [document.filename for document in selected],
                )
            except Exception:  # noqa: BLE001 - memory failures must not invalidate a completed answer
                logger.exception("Could not persist completed turn to cross-session memory")

    async def event_stream():
        ev = threading.Event()
        _CANCEL_EVENTS[conversation_id] = ev
        try:
            async for event in _sse_event_publisher(
                payload.question,
                service,
                ev,
                request,
                show_reasoning_steps=payload.show_reasoning_steps,
                memory_hits=memory_hits,
                on_finish=persist_assistant,
            ):
                yield event
        finally:
            ev.set()
            _CANCEL_EVENTS.pop(conversation_id, None)

    headers = {"Content-Type": "text/event-stream"}
    background = BackgroundTask(_run_post_response_evaluation, evaluation_context) if sample_for_evaluation else None
    return StreamingResponse(event_stream(), headers=headers, background=background)


@app.post("/api/conversations/{conversation_id}/cancel")
def cancel_conversation(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    _ = owned_conversation
    ev = _CANCEL_EVENTS.get(conversation_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="No active streaming request for this conversation")
    ev.set()
    return {"cancelled": True}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(
    conversation_id: str,
    payload: dict = Body(...),
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    title = payload.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    store = _get_store()
    _ = owned_conversation
    conv = store.rename_conversation(conversation_id, title)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
    }


@app.post("/api/maintenance/prune-titles")
def maintenance_prune_titles(current_user: dict[str, str] = Depends(get_current_user)):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=404, detail="Not found")
    return {"pruned": prune_default_titles()}


@app.get("/api/conversations/{conversation_id}/status")
def conversation_status(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation
    svc = _SERVICE_REGISTRY.get(conversation_id)
    # Attempt to reattach to existing persisted Chroma store if no live service
    if svc is None:
        svc = _ensure_service(conversation_id)
    has_service = svc is not None
    has_docs = _service_has_documents(svc)
    return {
        "conversation_id": conversation_id,
        "chroma_exists": has_docs,
        "service_loaded": has_service,
        "has_documents": has_docs,
    }


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    owned_conversation=Depends(get_owned_conversation),
):
    conversation_id = _require_uuid(conversation_id)
    store = _get_store()
    _ = owned_conversation

    _get_memory_manager().delete_conversation(conversation_id)
    svc = _SERVICE_REGISTRY.pop(conversation_id, None)
    if svc is not None and hasattr(svc, "cleanup_chroma_store"):
        svc.cleanup_chroma_store()
    else:
        delete_conversation_collection(conversation_id)

    deleted = store.delete_conversation(conversation_id)
    return {"deleted": bool(deleted)}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
