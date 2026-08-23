# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12


# ---- builder stage: has Java + Spark, runs the full pipeline ----
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    JAVA_HOME=/usr/lib/jvm/default-java \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra etl --no-install-project

COPY src/fineprint/__init__.py src/fineprint/__init__.py
COPY src/fineprint/data_generator src/fineprint/data_generator
COPY src/fineprint/etl src/fineprint/etl
COPY src/fineprint/models src/fineprint/models
COPY src/fineprint/rag src/fineprint/rag
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra etl --compile-bytecode

RUN uv run python -m fineprint.data_generator.generate
RUN uv run python -m fineprint.etl.run_local
RUN uv run python -m fineprint.models.train
RUN uv run python -m fineprint.rag.ingest_cli


# ---- final stage: no Java, no Spark, just the API and its baked-in state ----
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home /nonexistent \
    --shell /usr/sbin/nologin \
    --no-create-home \
    --uid ${UID} \
    appuser

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY --chown=appuser:appuser . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --compile-bytecode

COPY --from=builder --chown=appuser:appuser /app/data/contracts.json ./data/contracts.json
COPY --from=builder --chown=appuser:appuser /app/data/gold ./data/gold
COPY --from=builder --chown=appuser:appuser /app/data/chroma ./data/chroma
COPY --from=builder --chown=appuser:appuser /app/mlflow.db ./mlflow.db
COPY --from=builder --chown=appuser:appuser /app/mlruns ./mlruns
COPY --from=builder --chown=appuser:appuser /app/.cache/huggingface ./.cache/huggingface

USER appuser

EXPOSE 8000

CMD ["uvicorn", "fineprint.api.main:app", "--host", "0.0.0.0", "--port", "8000"]