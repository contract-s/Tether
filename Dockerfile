# One image for the API and the Celery worker.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
# Docker CLI only (no daemon): the worker runs code_exec sandboxes on the host's Docker.
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1

WORKDIR /app
# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Workspace (shared files) and the embedding model cache live in volumes.
RUN useradd --create-home --home-dir /home/app --uid 1000 app \
    && mkdir -p /app/data/workspace /home/app/.cache/chroma \
    && chown -R app:app /app/data /home/app
ENV HOME=/home/app
USER app

EXPOSE 8000
CMD ["uvicorn", "tether.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
