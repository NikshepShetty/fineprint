# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

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

USER appuser

EXPOSE 8000

CMD ["uvicorn", "fineprint.api.main:app", "--host", "0.0.0.0", "--port", "8000"]