# Agentic RAG: Document Q&A

Agentic RAG answers questions using uploaded PDF and text files. This repository contains two user interfaces: a FastAPI API with a React/Vite chat frontend, and a separate Streamlit UI. They share the RAG engine and Chroma Cloud collections but are separate application paths. SQLite stores application records; Chroma Cloud stores and searches document vectors.

## Quick Start

### Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The React app has its own dependencies:

```bash
cd frontend
npm install
```

Copy `.env.example` to `.env` at the repository root and set your Groq key and Chroma Cloud credentials. Get the Chroma API key from your Chroma Cloud SDK page; do not commit or paste it into source code. Both app entry points load the root `.env` file.

```bash
cp .env.example .env
```

Set `CHROMA_HOST`, `CHROMA_TENANT`, and `CHROMA_DATABASE` to your Cloud SDK values. Set `CHROMA_API_KEY` locally; the key is required both to access Chroma Cloud and to call its Qwen/Splade embedding services.

### Run React and FastAPI

Start FastAPI from the repository root:

```bash
FRONTEND_ORIGIN=http://localhost:5191 ./.venv/bin/python -m uvicorn backend.app:app --reload --port 18005
```

In another terminal, start Vite:

```bash
cd frontend
VITE_API_PROXY_TARGET=http://localhost:18005 npm run dev -- --port 5191
```

Open <http://localhost:5191/>. The backend health endpoint is <http://localhost:18005/api/health>. Use `localhost` consistently for local development; do not switch to `127.0.0.1`, because Google treats those hosts as different redirect URIs. Vite proxies `/api` requests to `http://localhost:18005`; leave `VITE_API_BASE_URL` unset for this local setup. For Google OAuth, set `GOOGLE_REDIRECT_URI` to the exact URI registered in Google Cloud Console, for example `http://localhost:5191/api/auth/google/callback`.

### Run Streamlit

From the repository root:

```bash
streamlit run app.py
```

`app.py` is the unified Streamlit entry point. `agentic_app.py` remains as a legacy Streamlit UI. These are separate from the authenticated React/FastAPI application.

## Features

- PDF and UTF-8 text ingestion, with document status, duplicate detection, and per-conversation indexing.
- Agentic route/retrieve/rerank/grade/rewrite/generate-or-abstain flow, plus direct answers for supported app-use/small-talk cases.
- Optional CPU CrossEncoder reranking and token-budgeted map-reduce for broad questions.
- Citation-aware answers, source passages, reasoning traces, export, and answer feedback.
- Optional per-user cross-session memory. Past-chat hints can clarify references but are treated as untrusted and are not evidence.
- Account signup/signin, ownership-scoped conversations, cookie sessions, CSRF checks, and password reset flow.
- Optional voice input and output in the React UI. Voice output is off by default; see [Voice limitations](#voice-limitations).

## How Answers Run

The agentic engine is in `agentic_rag.py`. Its `ask()` method uses a LangGraph; the FastAPI question stream calls `ask_stream()`, which manually runs the corresponding path. The usual document path retrieves passages, optionally reranks them, grades sufficiency, may rewrite and retry, then generates or abstains. Scope and chitchat checks can short-circuit document retrieval. `answer_mode="traditional"` selects `BaselineRAGService`, which does not run the agentic route-LLM or grade/rewrite loop.

Answers are sent as Server-Sent Events (SSE) by `POST /api/conversations/{conversation_id}/questions`. The engine buffers generated text for output validation before yielding answer chunks. If model streaming is unavailable, it can fall back to one invoke result; do not assume every request is a single token/chunk.

## API Reference

Authenticated routes use the `rag_session` cookie. Unsafe methods require `X-CSRF-Token`; the React client obtains it from `GET /api/auth/csrf`. Conversation detail, document-list, and status GETs also require the CSRF header. The shared client is in `frontend/src/api.js`.

### Authentication

| Method and path | Purpose |
| --- | --- |
| `POST /api/auth/signup` | Create an account and session; JSON `{ "email", "password" }`. Password minimum is 12 characters and maximum is 72 UTF-8 bytes. |
| `POST /api/auth/signin` | Sign in with `{ "email", "password" }`. |
| `POST /api/auth/logout` | Revoke the session and clear the cookie. |
| `GET /api/auth/me` | Return the current account. |
| `GET /api/auth/csrf` | Return a session-bound CSRF token, or a one-use pre-auth token for password reset. |
| `POST /api/auth/request-reset` | Request reset instructions with `{ "email" }`; response is generic. |
| `POST /api/auth/reset-password` | Complete reset with `{ "token", "password" }`. |

Signup/signin are CSRF-exempt. Anonymous reset requests use a short-lived, one-use pre-auth CSRF token.

### Conversations and Documents

| Method and path | Purpose |
| --- | --- |
| `POST /api/conversations` | Create a conversation. |
| `GET /api/conversations` | List the signed-in user's conversations. |
| `GET /api/conversations/{id}` | Read conversation, documents, and messages. |
| `PATCH /api/conversations/{id}` | Rename with JSON `{ "title": "..." }`. |
| `DELETE /api/conversations/{id}` | Delete conversation and associated index/memory data. |
| `POST /api/conversations/{id}/documents` | Upload multipart `files` (PDF/TXT); returns accepted and rejected files. |
| `GET /api/conversations/{id}/documents` | Read document states. |
| `DELETE /api/conversations/{id}/documents/{document_id}` | Remove a document and its indexed chunks. |
| `GET /api/conversations/{id}/export` | Download a text export. |
| `GET /api/conversations/{id}/messages` | List persisted messages. |
| `POST /api/conversations/{id}/messages` | Append a message. |
| `POST /api/conversations/{id}/messages/{message_id}/feedback` | Set rating `-1`, `0`, or `1`, with optional comment. |
| `GET /api/conversations/{id}/status` | Report whether the Chroma index/service is present. |
| `POST /api/conversations/{id}/cancel` | Cancel the active answer stream. |

### Questions and SSE Events

Question JSON requires `question`. Optional settings: `answer_mode` (`agentic` or `traditional`, default `agentic`), `max_retries` (default 2), `passages_per_search` (default 4), `show_reasoning_steps` (default true), `document_ids`, `rerank_enabled` (default true), `rerank_candidates` (default 20), `rerank_top_n` (default 5), `map_reduce_mode` (`auto`, `off`, or `force`, default `auto`), and `voice_output` (default false).

| SSE event | Data |
| --- | --- |
| `trace` | A trace record; fields depend on the step. |
| `token` | `{ "token": "..." }` answer text. |
| `memory_hits` | `{ "hits": [...] }` related past-chat records. |
| `map_progress` | `{ "stage", "mapped", "total", "document" }`. |
| `audio_chunk` | `{ "sequence", "mime_type": "audio/wav", "audio_base64" }`; emitted only with voice output enabled. |
| `audio_error` | `{ "error": "..." }`; text answer remains available. |
| `error` | `{ "error": "..." }` stream failure. |
| `cancelled` | `{}` when the server observes cancellation. |
| `answer` | `{ "answer", "sources", "trace", "reasoning" }`; reasoning can be null. |
| `done` | `[DONE]` terminal marker. |

`complete` is an internal server queue signal, not an SSE event. Text event names and payloads are shared by both voice-off and voice-on requests; audio events are additive. Source page values may be strings such as `?` when page metadata is unavailable.

### Memory, Voice, and Operations

| Method and path | Purpose |
| --- | --- |
| `GET /api/memory/settings`, `PUT /api/memory/settings` | Read/update the signed-in user's global memory switch (`{ "enabled": true|false }`). |
| `DELETE /api/memory` | Clear the signed-in user's stored memory turns. |
| `GET /api/conversations/{id}/memory`, `PUT /api/conversations/{id}/memory` | Read/update per-conversation memory preference. |
| `POST /api/voice/transcribe` | Authenticated multipart `audio` upload (WebM or MP4); returns `{ "transcript": "..." }`. |
| `POST /api/maintenance/prune-titles` | Admin-only title maintenance. |
| `GET /api/health` | Public health response `{ "ok": true }`. |

## Configuration

| Variable | Purpose and default |
| --- | --- |
| `GROQ_API_KEY` | Required for Groq chat, Whisper, and PlayAI calls. |
| `GROQ_MODEL` | Chat model; default `openai/gpt-oss-120b`. |
| `CHROMA_HOST` | Chroma Cloud API host; default `api.trychroma.com`. |
| `CHROMA_API_KEY` | Required Chroma Cloud key for database access and hosted embedding APIs. |
| `CHROMA_TENANT`, `CHROMA_DATABASE` | Chroma Cloud tenant and database identifiers. |
| `FRONTEND_ORIGIN` | Comma-separated exact CORS origins; default `http://localhost:5173`. Set to `http://localhost:5191` for the local ports above. Wildcard is rejected. |
| `VITE_API_BASE_URL` | Optional API origin for direct cross-origin use; unset uses the Vite `/api` proxy. |
| `APP_ENV` | Defaults to `development`; `production` forces Secure session cookies. |
| `RAG_COOKIE_SECURE`, `RAG_COOKIE_SAMESITE` | Cookie overrides; defaults `0` and `lax`. `SameSite=None` requires Secure. |
| `RAG_ADMIN_EMAILS` | Comma-separated emails granted admin at signup. |
| `MAX_FILES_PER_UPLOAD` | Maximum files in one FastAPI upload; default 10. |
| `MAX_FILES_PER_CONVERSATION` | Maximum documents in a conversation; default 30. |
| `MAX_UPLOAD_BYTES` | Per-file and audio upload limit; default 10 MiB. |
| `MAX_REQUEST_BYTES` | Total document request limit; default 50 MiB. |
| `RERANKER_MODEL`, `RERANKER_BATCH_SIZE` | CrossEncoder model and batch size; defaults `cross-encoder/ms-marco-MiniLM-L-6-v2` and 16. |
| `MAP_REDUCE_TOKENIZER` | Tokenizer; default `openai/gpt-oss-120b`. |
| `MAP_REDUCE_TOKEN_BUDGET`, `MAP_REDUCE_MAX_CALLS`, `MAP_REDUCE_CONCURRENCY`, `MAP_REDUCE_MAX_RETRIES` | Defaults 5000, 30, 3, and 3. |
| `RAG_GUARDRAIL_<NAME>_ENABLED` | Guardrail toggles default on; recognized checks include scope, injection, groundedness, memory leakage, abstention, output schema, and harmful content. |
| `RAG_GUARDRAIL_GROUNDEDNESS_ACTION` | `abstain` by default; `regenerate_once` permits one retry. |
| `RAG_GUARDRAIL_MIN_<METRIC>` | Groundedness thresholds for faithfulness/citation metrics; default 1.0. |
| `RAG_GUARDRAIL_SENSITIVE_CONTENT_ENABLED` | Persistence masking; default off. |
| `RAG_EVAL_SAMPLE_RATE` | Post-response evaluation sampling rate; default 0 (disabled). |
| `PIPELINE_EVAL_<METRIC>`, `GOLDEN_EVAL_MIN_<METRIC>` | Offline evaluation threshold overrides. |

`config.py` also defines `MAX_UPLOAD_FILES`, but the FastAPI upload route does not use it; configure `MAX_FILES_PER_UPLOAD` for the API limit.

## Tests and Maintenance

```bash
# From repository root
./.venv/bin/python -m pytest -q

# From frontend/
npm test
npm run build
```

- `scripts/migrate_chroma_cloud.py` copies each local index into a temporary directory and dry-runs a migration into Chroma Cloud. It needs temporary disk space for the local indexes. Run `./.venv/bin/python scripts/migrate_chroma_cloud.py --apply` to copy and re-embed records; the original local indexes are preserved.
- `scripts/cleanup_conversations.py` lists what it would remove; pass `--apply` to delete conversations and their Chroma Cloud collections.
- `scripts/cleanup_sessions.py` removes expired sessions, reset tokens, and pre-auth CSRF tokens. Expiry cleanup is not scheduled automatically.
- `scripts/reassign_legacy_data.py --email <admin-email>` assigns ownerless legacy rows; back up `.rag_history.db` and `.chroma_store` first. It prompts unless `--yes` is supplied.
- Evaluation entry points live under `eval/`; see [eval/README.md](eval/README.md). Do not treat fixture/stub scores as production model-quality results.

## Limitations and Troubleshooting

- **Password reset:** No email provider is configured. In development, the reset URL is logged by the backend; production cannot deliver a reset link. Do not offer production password recovery until delivery is integrated.
- **Voice output:** The code calls Groq PlayAI model `canopylabs/orpheus-v1-english`. A real request on 2026-10-01 returned HTTP 400 `model_terms_required`; an organization admin must accept the model terms at [Groq's model page](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english). This access was not retested on 2026-10-02. Voice code is present, but real browser playback has not been verified. Mic/speaker preferences are not persisted, and Stop does not currently track already scheduled audio sources.
- **Evaluation:** `eval/data/golden_qa.json` is a synthetic evaluator fixture. Production variance/drift/shadow tools require reviewed local cases; no checked-in production baseline report was found. Production sampling defaults off, and its current scorer call uses the STUB judge by default.
- **Documents:** Only text-based PDF and TXT are supported; scanned/image-only PDFs need OCR elsewhere. No sample PDF/TXT corpus is bundled.
- **Upload limits:** API defaults are 10 files per request, 30 per conversation, 10 MiB per file/audio, and 50 MiB per document request. An error mentioning an unexpected file count may be due to setting `MAX_UPLOAD_FILES`, which does not control the FastAPI route.
- **CORS/CSRF:** For a custom frontend origin, set `FRONTEND_ORIGIN` to the exact scheme/host/port. For direct API calls, set `VITE_API_BASE_URL` too. A `403` on a mutation usually means the CSRF token is absent/stale; use the shared frontend API client rather than raw fetch.
- **401/404:** Sign in again after session expiry. Foreign or missing conversation/document IDs intentionally return 404. Legacy ownerless records are inaccessible until deliberately reassigned to an admin.
- **Missing Chroma index:** If SQLite says a document is ready but its conversation collection is empty in Chroma Cloud, the API marks it as needing re-upload/reprocessing. Confirm the Cloud credentials and run `scripts/migrate_chroma_cloud.py` to copy existing local indexes.
- **Live judge failures:** Groundedness is fail-closed when its live Groq judge errors. Check API logs and Groq availability; a passing STUB evaluation does not exercise this request path.

## Security Notes

Session cookies are `HttpOnly`, fixed-expiry after seven days, and host-only. Production must use HTTPS; configure the exact frontend origin. Each conversation has its own Cloud collection; cross-session memory is stored in user-sharded Cloud collections. `.env` is gitignored: keep credentials out of source control and do not print them into logs or documentation.
