# Roadmap

## Phase 1: Agent architecture (Days 1-4)
- [x] Agent hierarchy: supervisor, specialists (research, analysis, writing, code), reviewer, each a LangGraph node with typed I/O schemas
- [x] Task decomposition: structured-output execution plan (subtask, specialist, inputs, output format, complexity, dependencies)
- [x] Tool registry: name, description, schemas, allowed agents, rate limits; web search, file I/O, sandboxed code exec, DB query, HTTP. Log every call
- [x] LangGraph state machine: intake → plan → execute → review → synthesize → deliver, with retry, reject-loop and escalation edges

## Phase 2: Memory (Days 4-7)
- [x] Short-term working memory in Redis, scoped to a single task
- [x] Long-term semantic memory in ChromaDB, extracted after each task
- [x] Memory retrieval injected into planning
- [x] Memory management: importance scoring, consolidation, decay, dashboard, delete endpoint

## Phase 3: Human-in-the-loop (Days 7-10)
- [x] Escalation triggers: low plan confidence, 2x specialist failure, sensitive ops, low review score, user request
- [x] Approval queue: pause, package context, notify, resume on decision
- [x] Approval levels: Notify / Approve action / Approve plan / Take over
- [x] Review UI with context, proposed action, memories, actions, and chat panel

## Phase 4: Observability (Days 10-12)
- [x] OpenTelemetry trace tree for every execution
- [x] Trace explorer UI (color-coded nodes, expandable prompts/responses)
- [x] Cost and performance tracking per task, agent, model
- [x] Replay system with step-through and input modification

## Phase 5: Integration and testing (Days 12-13)
- [x] Showcase demo scenario (research → extraction → analysis → summary)
- [x] Full docker-compose: API, Redis, Postgres, ChromaDB, Celery, both UIs
- [x] End-to-end tests (valid plans, tool use, reviewer catches bad output, memory helps, escalation fires, failure recovery)

## Phase 6: Polish (Days 13-14)
- [ ] Demo video (under 5 minutes): script in [DEMO.md](DEMO.md), recording to do
- [x] Architecture diagram and write-up ([ARCHITECTURE.md](ARCHITECTURE.md))
