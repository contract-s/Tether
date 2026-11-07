# AgentOrch

A multi-agent orchestration platform. A **supervisor agent** breaks complex tasks into subtasks and hands them to **specialist agents** that use tools. A **reviewer agent** checks their outputs. The system keeps **persistent memory** across tasks and **escalates to a human** when confidence is low or an action needs approval. Every decision is traced.

## Architecture

The full write-up, with diagrams, measurements and design decisions, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The demo video script is in
[docs/DEMO.md](docs/DEMO.md).

```
            ┌──────────────┐
 task ───▶  │  Supervisor  │ ◀── long-term memory (ChromaDB)
            │  (planning)  │
            └──────┬───────┘
                   │ execution plan (DAG of subtasks)
     ┌─────────────┼──────────────┬──────────────┐
     ▼             ▼              ▼              ▼
 Research      Analysis        Writing      Code Exec     ◀── tool registry (custom + MCP)
     └─────────────┴──────┬───────┴──────────────┘        ◀── working memory (Redis)
                          ▼
                   ┌──────────────┐   reject → back to specialist
                   │   Reviewer   │
                   └──────┬───────┘
                          │ low confidence / sensitive op
                          ▼
                   Human review queue ──▶ synthesis ──▶ delivery
```

## Tech stack

| Component        | Tool                     |
|------------------|--------------------------|
| Language         | Python 3.11+             |
| Orchestration    | LangGraph                |
| LLM providers    | Anthropic (Claude Opus 5.5 supervisor, Sonnet 5.5 everywhere else), behind a routing layer |
| Tools            | Custom registry: web search, HTTP, files, sandboxed code, DB query, notes |
| Memory           | Redis (working), PostgreSQL (state), ChromaDB (semantic) |
| Async execution  | Celery on Redis, with checkpoints in Postgres |
| Dashboards       | Static HTML pages served by the API (reviews, memory, traces, replay) |
| Observability    | OpenTelemetry → Postgres span store + Jaeger |
| Deployment       | Docker + docker-compose  |

## Layout

```
src/agentorch/
  agents/         supervisor, specialists, reviewer
  graph/          LangGraph state machine and edges
  tools/          tool registry and tool implementations
  memory/         working (Redis) and long-term (ChromaDB) memory
  hitl/           escalation triggers, approval queue, approval levels
  observability/  tracing, cost tracking, replay
  api/            FastAPI app; static/ holds the dashboards
  workers/        Celery worker
scripts/          demo and command-line runners
tests/            unit tests (fakes) and end-to-end tests (real infrastructure)
docs/             architecture notes and roadmap
```

## Getting started

```bash
cp .env.example .env    # add ANTHROPIC_API_KEY
make stack              # API, Celery worker, Postgres, Redis, ChromaDB, Jaeger in Docker
make demo               # the showcase task: research -> extraction -> analysis -> summary
```

Then open http://localhost:8000 for the review queue, with links to the memory, trace
and replay dashboards, and http://localhost:16686 for Jaeger.

For development, run the infrastructure in Docker and the API on your machine:

```bash
make install            # uv sync
make up                 # postgres, redis, chromadb, jaeger
make api                # http://localhost:8000 (runs tasks in-process)
```

## Tests

```bash
make test               # unit tests, fake model, no services needed
make e2e                # real Postgres/Redis/ChromaDB (make up), scripted model, no API cost
make e2e-live           # also checks agent behaviour with Claude (about $0.75)
```

## Reliability

- Runs are checkpointed in Postgres after every step. A task waiting on a human survives
  restarts, and a decision made after a restart resumes it.
- With Celery, messages are acknowledged only when a run step finishes, so a worker killed
  mid-run has its work redelivered and continued from the last checkpoint.
- A run that crashes (say, the model API is down) is marked failed; `POST /tasks/{id}/retry`
  continues it from its last checkpoint without redoing finished subtasks.
- Each specialist attempt has a model-spend cap (`SPECIALIST_BUDGET_USD`). Past it, its
  tools are switched off and it must answer with what it has.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).
