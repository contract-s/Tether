.PHONY: install up stack down api worker test e2e e2e-live demo lint fmt

install:
	uv sync --group dev

# Infrastructure only, for running the API on your machine (make api)
up:
	docker compose up -d postgres redis chromadb jaeger

# Everything, in containers: API, Celery worker, infrastructure
stack:
	docker compose up -d --build

down:
	docker compose down

api:
	uv run uvicorn tether.api.main:app --reload

worker:
	TASK_RUNNER=celery uv run celery -A tether.workers worker --loglevel=info

test:
	uv run pytest tests/unit

# Against the running infrastructure (make up), with a scripted model: no API cost
e2e:
	uv run pytest tests/e2e -v

# Also the tests that call Claude (costs a little; needs ANTHROPIC_API_KEY)
e2e-live:
	TETHER_LIVE=1 uv run pytest tests/e2e -v

demo:
	uv run python scripts/demo.py

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
