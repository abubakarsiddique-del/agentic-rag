# Claude Context: Agentic RAG

Read the owning code before changing behavior. This is a two-frontend Python project: a FastAPI + React/Vite application and a separate Streamlit UI. Existing README/context claims have drifted before; treat them as leads, not ground truth. `PROJECT_CONTEXT.md` is an index; this file is the detailed agent handoff.

## Editing Constraints
- Preserve existing request shapes and text-path SSE event names/payloads unless the task explicitly authorizes a contract change.
- The API's question stream calls `ask_stream()`, not the LangGraph-backed `ask()`. Read both paths before changing pipeline behavior.
- Keep the two frontends and their persistence/indexing flows distinct; do not assume a change to one updates the other.
- Record discovered defects or unfinished work when the task is read-only. Do not repair adjacent issues without authorization.
- Use focused tests for the touched path. A prior context claim that all tests passed is not a current test result.

## Architecture Walk

### 1. Entry Points and Runtime
- FastAPI: from repository root, `./.venv/bin/python -m uvicorn backend.app:app --reload`. `backend/app.py` creates `app`; default API port is 8000.
- React/Vite: `cd frontend && npm run dev`; Vite proxies `/api` to `http://localhost:8000`. `npm test` runs Vitest; `npm run build` builds production assets. `package-lock.json` is checked in.
- Unified Streamlit UI: `streamlit run app.py` from repository root. `agentic_app.py` is a legacy Streamlit entry point.
- Python tests: `./.venv/bin/python -m pytest -q` from repository root. `requirements-dev.txt` declares `pytest` and `mocker`.
- Streamlit entry points call `load_dotenv`; FastAPI reads environment variables directly. Set `GROQ_API_KEY` in the API process environment.

### 2. Persistence
- `persistence/store.py` initializes `.rag_history.db` and applies schema migrations 1-7.
- Tables: `conversations`, `messages`, `indexed_documents`, `documents`, `schema_migrations`, `app_settings`, `conversation_memory_settings`, `memory_turns`, `users`, `sessions`, `user_settings`, `preauth_csrf_tokens`, and `password_reset_tokens`.
- Ownership fields are nullable on legacy conversation/document/index/memory rows; auth migrations add user-scoped settings and token tables. `persistence/models.py` defines the record dataclasses.
- Conversation document vectors live under `.chroma_store/<conversation_id>`; global memory uses `.chroma_store/_memory`.
- Production sampling uses a separate SQLite `evaluation_results` table (`eval/production/sampling_store.py`).

### 3. Core Engine
- `RAGService._build_graph()` defines `scope_guard` → `contextualize` → `route`; retrieval routes continue through `retrieve` → `rerank` → `grade` → `generate`, with rewrite/retry or abstention. Direct, chitchat, and map-reduce have separate graph paths.
- The FastAPI endpoint `POST /api/conversations/{conversation_id}/questions` uses `_sse_event_publisher()` and `service.ask_stream()`. `RAGService.ask_stream()` manually runs scope/chitchat, contextualization, route, optional map-reduce, retrieval/reranking/grading/rewrite, and generation. It does not invoke `self.graph`.
- Route errors fall back to retrieval. Generated retrieval answers are buffered for output validation before answer chunks are yielded. If streaming fails, the chain invoke fallback can produce one chunk.
- `answer_mode="traditional"` selects `BaselineRAGService`. It performs scope/chitchat, contextualization, retrieval, optional map-reduce/reranking, and output validation; it does not use the route LLM or grade/rewrite loop.
- With no ready documents, the API returns scope-block/chitchat/upload guidance before constructing the regular service. With documents, `classify_chitchat()` runs before normal routing.

### 4. Ingestion
- `ingest.py` is the shared parser: PDF pages become separate documents; TXT uses UTF-8 decoding with replacement fallback. Scanned/image-only and encrypted PDFs fail; there is no OCR.
- FastAPI `POST /api/conversations/{conversation_id}/documents` validates PDF/TXT, size/count/duplicates, stores queued metadata, and schedules `_process_document()` as a background task.
- Both engine classes split with `RecursiveCharacterTextSplitter` and index into per-conversation Chroma. Streamlit calls `build_index()` synchronously.
- `load_uploaded_documents_async()` is imported by the API module but has no current call site. Do not assume it is the active upload path.

### 5. Retrieval Enhancements
- `reranking.py` lazily loads the CPU CrossEncoder (`RERANKER_MODEL`, `RERANKER_BATCH_SIZE`). Failures retain vector order. Question defaults: enabled, 20 candidates, top 5.
- `map_reduce.py` classifies broad scope lexically. `auto` maps global questions, `off` disables, `force` requests it. It sorts passages, budgets tokenizer tokens, runs concurrent map calls, reduces partials, supports cancellation/progress, and can fall back to standard retrieval.
- `map_progress` payload: `{stage, mapped, total, document}`. Rerank trace fields include candidate/kept counts, score, latency, enabled, and fallback.

### 6. Cross-Session Memory
- `persistence/memory.py` stores a shortened answer summary and question in SQLite and indexes them in Chroma collection `global_memory`.
- Global memory is per-user and defaults off. Per-conversation memory defaults on but is only active with global memory enabled. Search excludes the current conversation and filters results by owner.
- Hints are explicitly untrusted and can clarify references only; they are not answer evidence/citations. Completed non-fast-path answers are recorded in the API finish callback.
- Routes: `GET/PUT /api/memory/settings`, `DELETE /api/memory`, and `GET/PUT /api/conversations/{conversation_id}/memory`.

### 7. Guardrails
- `guardrails/config.py` defaults most checks on; sensitive-content masking defaults off.
- `scope.py`: high-confidence input patterns. `chitchat/classifier.py` calls it and only uses an LLM fallback for a narrow ambiguous form.
- `injection.py`: pattern-based filtering of retrieved passages, used by both engine paths.
- `groundedness.py`: live Groq judge scoring, citation validation, and optional one-time regeneration. Judge errors fail closed to the no-information response.
- `harmful_content.py`: narrow regex checks for selected harmful operational output, including the direct path.
- `memory_leakage.py`: rejects judge evidence/citation IDs that do not match retrieved passages or appear to rely on memory hints.
- `output_schema.py`: validates rewritten query, trace shape, and recognized citations. `sensitive_content.py` optionally redacts selected email/phone/SSN/Luhn-valid card patterns at persistence boundaries.
- `abstention.py` has a tested helper, but no runtime caller was found; runtime abstention decisions live in engine/groundedness code. Scope/chitchat/direct paths do not all invoke the groundedness judge.

### 8. Evaluation
- `eval/pipeline_eval.py`, `eval/answer_eval.py`, and `eval/memory_eval.py` run against local fixtures. `golden_qa.json` is synthetic and validates the evaluator, not production model quality.
- Answer-eval and judge-calibration CLIs default to STUB; use explicit `--live` for Groq. Pipeline fixtures are trace snapshots, not end-to-end live evaluations.
- `eval/production/variance_eval.py`, `drift_check.py`, and `shadow_compare.py` need reviewed local case data. `judge_calibration.py` needs 20-30 cases plus human ratings. `feedback_correlation.py` scores rated turns with a live judge; `feedback_report.py` uses heuristic trace attribution.
- `sampling.py` is integrated after completed API responses, default sample rate 0. Its current scorer call reaches `score_answer()` without a judge argument, whose default is STUB. Do not describe sampled metrics as live-judge metrics without changing/verifying that path.
- No checked-in `eval/reports/` or `production_eval_results.db` was found during the documentation walk; there is no recorded production baseline in the workspace.

### 9. Auth and Authorization
- `backend/auth.py`: bcrypt password hashes; opaque session cookies (`rag_session`) with token hashes in SQLite; fixed seven-day expiry; process-local sign-in limiter.
- `backend/app.py`: `get_current_user`, `get_owned_conversation`, and `get_owned_document` enforce session and owner scope. Conversation/document/message/export/question routes use these dependencies; maintenance title pruning additionally checks `is_admin`.
- CORS uses explicit `FRONTEND_ORIGIN` values with credentials; wildcard is rejected. Cookie defaults are SameSite=Lax, not Secure in development; production forces Secure.
- CSRF middleware covers unsafe methods and selected conversation detail/document/status GETs. `/api/auth/csrf` provides session-bound or one-use pre-auth reset tokens. Signup/signin are exempt.
- Reset tokens are hashed, expire after 45 minutes, and are single-use. Development logs a reset URL; no email provider is configured, so production reset delivery is unavailable.
- Important routes: `/api/auth/{signup,signin,logout,me,csrf,request-reset,reset-password}`, `/api/conversations`, `/api/conversations/{id}` and its documents/messages/memory/export/status/questions/cancel routes, `/api/memory[/settings]`, `/api/maintenance/prune-titles`, `/api/health`, `/api/voice/transcribe`.

### 10. Voice Status
- Voice is implemented in the current code, not absent: authenticated `POST /api/voice/transcribe` accepts WebM/MP4 and calls `whisper-large-v3-turbo`; React `InputBar` uses shared `apiFetch`, then displays an editable transcript before ordinary question submission.
- TTS is opt-in via `QuestionRequest.voice_output` (default false). The question SSE path emits additive `audio_chunk` WAV/base64 payloads and `audio_error` events using Groq model `canopylabs/orpheus-v1-english`, voice `autumn`.
- Last real PlayAI request in this conversation (2026-10-01) returned HTTP 400 `model_terms_required`; current provider access was not retested on 2026-10-02. The unit test mocks Groq and uses fake audio bytes, so it does not verify real playback.
- Speaker preference is not persisted. Stop aborts the request but the UI does not retain active audio sources to stop already scheduled chunks. `voice/` is empty; implementation is in `backend/app.py`, `backend/voice_text_cleanup.py`, and React files.

### 11. Frontend and Stream Contract
- `frontend/src/App.jsx` owns session, conversation, message, settings, and SSE state. Components include `InputBar`, `Sidebar`, `MessageBubble`, `PipelineStepper`, `SourcesDrawer`, `AuthPanel`, and `UploadPanel`.
- `api.js` provides credentialed `apiFetch`/`apiUpload`, CSRF acquisition, and unauthorized-session notification. Do not add direct fetch calls for protected mutations.
- Wire event names: `trace`, `token` (`{token}`), `memory_hits` (`{hits}`), `map_progress`, `audio_chunk`, `audio_error`, `error`, `cancelled`, `answer` (`{answer,sources,trace,reasoning}`), and `done` (`[DONE]`). `complete` is an internal publisher queue type, not a wire event.
- `App.jsx` handles all listed events except `cancelled` explicitly; Stop aborts the fetch and posts the cancel route. Existing text event names/payloads must remain stable.
- `conversationUtils.mjs::parseCitationReferences()` recognizes `[Source n: Page p]` and maps numbered sources by array order. Backend map-reduce markers and `guardrails/output_schema.py` recognize the same marker family. `AnswerResponse` is not the active SSE response model; it declares integer page values although emitted source pages may be `"?"`.

### 12. Tests, Scripts, and Configuration
- Python tests: `tests/test_agentic_rag.py`, plus `tests/test_api.py`, `test_auth.py`, `test_chitchat.py`, `test_eval_regression.py`, `test_feedback_report.py`, guardrail, map-reduce, memory, production-eval, reattach, reranking, and voice cleanup tests.
- Frontend test files currently cover `api.js` and `conversationUtils.mjs`; there are no App/InputBar/browser playback tests.
- Ops: `scripts/cleanup_conversations.py` dry-runs unless `--apply`; `cleanup_sessions.py` deletes expired auth tokens; `reassign_legacy_data.py` requires an existing admin and confirmation unless `--yes`. Evaluation utilities live under `eval/`.
- Relevant env: `GROQ_API_KEY`, `GROQ_MODEL`, `EMBEDDING_MODEL`, `FRONTEND_ORIGIN`, `APP_ENV`, `RAG_COOKIE_SECURE`, `RAG_COOKIE_SAMESITE`, `RAG_ADMIN_EMAILS`, `MAX_FILES_PER_UPLOAD`, `MAX_FILES_PER_CONVERSATION`, `MAX_REQUEST_BYTES`, `MAX_UPLOAD_BYTES`, `RERANKER_MODEL`, `RERANKER_BATCH_SIZE`, `MAP_REDUCE_TOKENIZER`, `MAP_REDUCE_TOKEN_BUDGET`, `MAP_REDUCE_MAX_CALLS`, `MAP_REDUCE_CONCURRENCY`, `MAP_REDUCE_MAX_RETRIES`, `RAG_EVAL_SAMPLE_RATE`, `PIPELINE_EVAL_*`, `GOLDEN_EVAL_MIN_*`, `RAG_GUARDRAIL_*`, and frontend `VITE_API_BASE_URL`.
- `config.MAX_UPLOAD_FILES` is defined but unused; FastAPI upload enforcement uses `MAX_FILES_PER_UPLOAD` instead. Do not imply the former controls API uploads.
- `.env` is gitignored. Do not read or print secrets while inspecting the workspace.

## Current Verification Limits
- This documentation walk did not rerun the full Python or frontend test suites.
- A prior focused run passed 6 backend voice/cleanup checks and 22 frontend tests; a Vite build passed. These are not evidence that the full suite currently passes.
- No full production baseline or real browser TTS playback was verified.
