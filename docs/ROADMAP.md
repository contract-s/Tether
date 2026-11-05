# Roadmap

## Phase 1: Agent architecture (Days 1-4)
- [x] Agent hierarchy: supervisor, specialists (research, analysis, writing, code), reviewer, each a LangGraph node with typed I/O schemas
- [x] Task decomposition: structured-output execution plan (subtask, specialist, inputs, output format, complexity, dependencies)
- [x] Tool registry: name, description, schemas, allowed agents, rate limits; web search, file I/O, sandboxed code exec, DB query, HTTP. Log every call
- [x] LangGraph state machine: intake → plan → execute → review → synthesize → deliver, with retry, reject-loop and escalation edges

## Phase 2: Memory (Days 4-7)
- [ ] Short-term working memory in Redis, scoped to a single task
- [ ] Long-term semantic memory in ChromaDB, extracted after each task
- [ ] Memory retrieval injected into planning
- [ ] Memory management: importance scoring, consolidation, decay, dashboard, delete endpoint

## Phase 3: Human-in-the-loop (Days 7-10)
- [ ] Escalation triggers: low plan confidence, 2x specialist failure, sensitive ops, low review score, user request
- [ ] Approval queue: pause, package context, notify, resume on decision
- [ ] Approval levels: Notify / Approve action / Approve plan / Take over
- [ ] Review UI with context, proposed action, memories, actions, and chat panel

## Phase 4: Observability (Days 10-12)
- [ ] OpenTelemetry trace tree for every execution
- [ ] Trace explorer UI (color-coded nodes, expandable prompts/responses)
- [ ] Cost and performance tracking per task, agent, model
- [ ] Replay system with step-through and input modification

## Phase 5: Integration and testing (Days 12-13)
- [ ] Showcase demo scenario (research → extraction → analysis → summary)
- [ ] Full docker-compose: API, Redis, Postgres, ChromaDB, Celery, both UIs
- [ ] End-to-end tests (valid plans, tool use, reviewer catches bad output, memory helps, escalation fires, failure recovery)

## Phase 6: Polish (Days 13-14)
- [ ] Demo video (under 5 minutes)
- [ ] Architecture diagram and write-up
