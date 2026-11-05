.PHONY: install up down api test lint fmt

install:
	uv sync --group dev

up:
	docker compose up -d

down:
	docker compose down

api:
	uv run uvicorn agentorch.api.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
