FROM python:3.12.14-slim-bookworm AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2" \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.12.14-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/home/app/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=.

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY backend/ ./backend/
COPY chitchat/ ./chitchat/
COPY eval/__init__.py ./eval/__init__.py
COPY eval/answer_eval.py ./eval/answer_eval.py
COPY eval/production/__init__.py ./eval/production/__init__.py
COPY eval/production/sampling.py ./eval/production/sampling.py
COPY eval/production/sampling_store.py ./eval/production/sampling_store.py
COPY guardrails/ ./guardrails/
COPY persistence/ ./persistence/
COPY agentic_rag.py app_helpers.py config.py ingest.py map_reduce.py observability.py reranking.py ./

RUN mkdir -p \
        /var/lib/agentic-rag/db \
        /var/lib/agentic-rag/chroma \
        /home/app/.cache/huggingface \
    && ln -s /var/lib/agentic-rag/db/.rag_history.db /app/.rag_history.db \
    && ln -s /var/lib/agentic-rag/chroma /app/.chroma_store \
    && chown -R app:app /app /var/lib/agentic-rag /home/app/.cache

USER app:app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"]

CMD sh -c "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-18005}"

