# Demo video script (under 5 minutes)

The showcase task takes about 4½ minutes to run, so the video doesn't wait for it live.
Start one run before recording, and cut between it and the dashboards. Times below are
for the edited video.

## Before recording

1. `make stack`, and wait until `docker compose ps` shows the API as healthy.
2. **Warm-up run.** `uv run python scripts/demo.py --auto --user video`. This gives you
   a finished task with a full trace, and it seeds long-term memory for user `video`.
   It costs about $1.
3. **Run to film.** Start a second one, in the review dashboard (New task) or with
   `uv run python scripts/demo.py --review --user video`. Use the same task, ticked to
   require review. It will stop for your plan approval, which you'll do on camera.
4. Open these tabs:
   - the review queue: http://localhost:8000/dashboard/reviews
   - the warm-up task's trace (the demo script prints the link)
   - the replay page for the warm-up task
   - the memory dashboard
5. Set the browser zoom to about 125%, so the text is readable at 1080p.

## Script

**0:00 – 0:30 · What it is.** Open on the architecture diagram in
`docs/ARCHITECTURE.md`.
> "Tether takes a task, has a supervisor agent plan it, runs the plan with specialist
> agents that use tools, checks every result with a reviewer, and asks a human when it
> should. It remembers what it learns, and every step is traced and replayable."

**0:30 – 1:15 · The plan waits for a human.** In the review queue, open the second run's
plan review.
- Show the plan: four subtasks (research → extract → compute → summary) with their
  dependencies, the confidence, and the memories the supervisor planned with. Those came
  from the warm-up run.
- Ask the chat panel "why is the CSV a separate step from the calculation?" and show the
  answer.
- Approve.

> "Nothing has run yet. The run is checkpointed in Postgres, so it could wait days."

**1:15 – 2:00 · Agents at work, and a sensitive action.** Cut to the task page as
subtasks complete.
- If the research subtask fails review and comes back for a human, show the reviewer's
  blocking issues and choose *accept* or *retry with guidance*.
- When the analysis specialist asks to write `storage_prices.csv`, show the approval
  card: the exact tool call, and the agent's reason. Approve it.

> "Sensitive tool calls never run without approval. The agent finishes what it can
> while it waits."

**2:00 – 3:00 · The trace.** Switch to the warm-up task in the trace explorer.
- Show the summary bar: cost, model calls, tokens, human wait.
- Expand one research specialist: its LLM calls, the web search tool span, and its
  input and output.
- Open one LLM span and expand the full prompt and response.
- Use the cost-by-agent table: specialists and web search are most of the cost.
- Mention: "A first version of this demo spent $2.71 on research and never finished.
  The trace showed why: the agent was paging through AWS's raw price list. So each
  specialist attempt now has a spending limit."

**3:00 – 3:50 · Replay.** On the replay page for the warm-up task:
- Step through to the checkpoint after the calculation subtask finished.
- Edit its output (for example, change one provider's total), then replay.
- Open the comparison: only the write-up and the final answer re-ran, it cost a fraction
  of the original, and the diff shows how the answer changed.

> "Replay forks from any checkpoint, with edits, without touching the original or
> writing memories."

**3:50 – 4:30 · Memory.** On the memory dashboard, pick user `video`.
- Show the memories the warm-up run saved: approaches, pitfalls and tool usage. The
  warm-up run should yield something like "the providers' pricing pages load their
  prices with script; use their price APIs".
- Point out the importance scores and the delete and consolidate controls.

> "The second run planned with these. You saw them in its plan review."

**4:30 – 4:55 · Close.** Back to the second run, now done, showing its final answer
table.
> "Docker Compose runs all of it: API, Celery workers, Postgres, Redis, ChromaDB and
> Jaeger. Runs survive restarts and worker crashes. The tests cover all of this with a
> scripted model, and with Claude."

## If something goes differently

- **No take-over review for research.** That's fine. Show the notify card from a
  rejected attempt instead, or skip that part.
- **No approval asked for the CSV write.** Only overwrites need approval. After the
  warm-up run, `storage_prices.csv` already exists in the workspace, so the second run
  has to ask to overwrite it. If it chose a new file name instead, show the approval from
  the warm-up run's trace (the `tool.file_write` span marked blocked).
- **Running long.** Cut the chat question at 0:45 and the memory controls at 4:15.
