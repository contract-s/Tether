# AgentOrch

A multi-agent orchestration platform. A **supervisor agent** breaks complex tasks into subtasks and hands them to **specialist agents** that use tools. A **reviewer agent** checks their outputs. The system keeps **persistent memory** across tasks and **escalates to a human** when confidence is low or an action needs approval. Every decision is traced.

## Architecture

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
| LLM providers    | Anthropic + OpenAI       |
| Tools            | Custom registry + MCP    |
| Memory           | Redis (working), PostgreSQL (state), ChromaDB (semantic) |
| Async execution  | Redis + Celery           |
| Review UI        | React or Streamlit       |
| Observability    | OpenTelemetry            |
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
  api/            FastAPI app
  workers/        Celery tasks
ui/               review UI and trace explorer
tests/            unit and end-to-end tests
docs/             architecture notes and roadmap
```

## Getting started

```bash
cp .env.example .env    # add your API keys
make install            # uv sync
make up                 # postgres, redis, chromadb, jaeger
make api                # http://localhost:8000/health
make test
```

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).
