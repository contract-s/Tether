"""Step through a finished (or paused) run and replay it from any step, with changes.

Every step of a run is a LangGraph checkpoint, so stepping through is reading the
checkpoint history. A replay forks the run: the state at the chosen step, plus any
changes, is copied onto a *new* task (new thread, new id, its own trace) as if the node
that produced that step had just written it. The graph then carries on from exactly
there, through the same routing, retries and human reviews as a normal run. The
original task is never touched.

What a replay may change:
  task           the request text the agents work from
  plan           a different plan; execution restarts from it (earlier results dropped)
  outputs        a subtask's output, as if a human had taken it over
  guidance       a note every later specialist and reviewer sees
  require_review pause for plan/final-answer approval
  context        extra context lines for planning (only matters before the plan step)
  reuse_memories plan with the original's memories (default) or retrieve afresh

Replays never write long-term memories, so experimenting doesn't change what later
tasks remember.

`compare` sets a replay against the part of the original that ran after the fork point:
cost, calls, time, escalations, per-subtask outcomes, and a diff of the final answers.
"""

from __future__ import annotations

import difflib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from agentorch.agents.planning import validate_plan
from agentorch.agents.schemas import ExecutionPlan, SubtaskResult
from agentorch.hitl.service import TaskService
from agentorch.hitl.store import ReplayRecord
from agentorch.memory.service import MemoryServices
from agentorch.observability import metrics
from agentorch.observability.store import SpanStore
from agentorch.observability.tracing import flush

# State that belongs to one run's bookkeeping, not to the work itself.
_RUN_ONLY = ("notifications", "memories_saved", "memory_errors")
# Execution state that a new plan makes meaningless.
_EXECUTION = ("results", "reviews", "attempts", "feedback", "accepted", "abandoned",
              "approvals", "preapproved", "escalations", "tool_calls", "final_output")
# Before this node has run there is no plan, so a fork starts the run over.
_PLAN_NODE = "plan"


class ReplayError(Exception):
    pass


class ReplayChanges(BaseModel):
    task: str | None = Field(default=None, min_length=1)
    plan: ExecutionPlan | None = None
    outputs: dict[str, str] = Field(default_factory=dict,
                                    description="subtask id -> output to use instead")
    guidance: str | None = None
    require_review: bool | None = None
    context: list[str] | None = None
    reuse_memories: bool = True

    def summary(self) -> dict[str, Any]:
        """What changed, for display (the plan is summarised, not repeated)."""
        out = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        if self.plan is not None:
            out["plan"] = [f"{st.id}: {st.description}" for st in self.plan.subtasks]
        if not self.reuse_memories:
            out["reuse_memories"] = False
        return out


@dataclass
class Step:
    checkpoint_id: str
    step: int
    ran: list[str]  # node(s) whose writes produced this checkpoint
    next: list[str]  # node(s) that run after it
    created_at: float
    changed: list[str]  # state keys that differ from the previous step
    paused: bool  # the run is waiting on a human at this step
    can_replay: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ReplayService:
    def __init__(self, tasks: TaskService, spans: SpanStore,
                 memory: MemoryServices | None = None) -> None:
        self.tasks = tasks
        self.graph = tasks.graph
        self.store = tasks.store
        self.spans = spans
        self.memory = memory

    # --- step-through ------------------------------------------------------------

    def steps(self, task_id: str) -> list[Step]:
        history = self._history(task_id)
        if not history:
            raise KeyError(task_id)
        steps: list[Step] = []
        previous: dict[str, Any] = {}
        for i, snap in enumerate(history):
            values = _jsonable(snap.values)
            changed = sorted(k for k in values.keys() | previous.keys()
                             if values.get(k) != previous.get(k))
            previous = values
            ran = _ran(history, i)
            steps.append(Step(
                checkpoint_id=snap.config["configurable"]["checkpoint_id"],
                step=snap.metadata.get("step", i),
                ran=ran, next=list(snap.next), created_at=_ts(snap.created_at),
                changed=changed,
                paused=any(t.interrupts for t in snap.tasks),
                can_replay=_can_replay(snap, ran),
            ))
        return steps

    def state_at(self, task_id: str, checkpoint_id: str) -> dict[str, Any]:
        history = self._history(task_id)
        for i, snap in enumerate(history):
            if snap.config["configurable"]["checkpoint_id"] == checkpoint_id:
                before = _jsonable(history[i - 1].values) if i else {}
                return {"values": _jsonable(snap.values), "previous": before}
        raise KeyError(checkpoint_id)

    # --- replay ------------------------------------------------------------------

    def replay(self, task_id: str, checkpoint_id: str, changes: ReplayChanges) -> ReplayRecord:
        source = self.store.get_task(task_id)
        if source is None:
            raise KeyError(task_id)
        history = self._history(task_id)
        index = next((i for i, s in enumerate(history)
                      if s.config["configurable"]["checkpoint_id"] == checkpoint_id), None)
        if index is None:
            raise KeyError(checkpoint_id)
        snap = history[index]
        ran = _ran(history, index)
        if not _can_replay(snap, ran):
            raise ReplayError("nothing runs after this step; pick an earlier one")

        new_id = uuid.uuid4().hex
        values, as_node = self._fork_state(dict(snap.values), ran[0], changes, new_id)
        if as_node == "intake" and changes.reuse_memories:
            # Forked before planning, so the memories the original planned with are
            # only in its later state.
            values["memories_used"] = list(
                self._final_values(task_id).get("memories_used", []))

        self.store.create_task(new_id, values["user_id"], values["task"],
                               bool(values.get("require_review")))
        record = self.store.add_replay(
            id=new_id, source_task_id=task_id, checkpoint_id=checkpoint_id,
            step=snap.metadata.get("step", index), resumed_after=as_node,
            fork_time=_ts(snap.created_at), changes=changes.summary())
        self.graph.update_state(TaskService._config(new_id), values, as_node=as_node)
        self._seed_working_memory(new_id, values)
        self.tasks.dispatch(new_id, {"kind": "continue"})
        return record

    def _fork_state(self, values: dict[str, Any], ran: str, changes: ReplayChanges,
                    new_id: str) -> tuple[dict[str, Any], str]:
        for key in _RUN_ONLY:
            values.pop(key, None)
        values.update(task_id=new_id, replay_of=values.get("task_id"), status="running",
                      error=None, reuse_memories=changes.reuse_memories)
        if changes.task is not None:
            values["task"] = changes.task
        if changes.require_review is not None:
            values["require_review"] = changes.require_review
        if changes.context is not None:
            values["memories"] = changes.context
        if changes.guidance:
            values["guidance"] = [*values.get("guidance", []), changes.guidance]

        as_node = ran
        if "plan" not in values:
            as_node = "intake"  # forked before planning: plan again from the start
            if changes.plan is not None:
                raise ReplayError("this step is before planning; replay from the plan step "
                                  "or later to supply a plan")
        if changes.plan is not None:
            try:
                batches = validate_plan(changes.plan)
            except ValueError as e:
                raise ReplayError(f"invalid plan: {e}") from e
            for key in _EXECUTION:
                values.pop(key, None)
            values.update(plan=changes.plan, batch_index=0,
                          batches=[[st.id for st in b] for b in batches])
            as_node = _PLAN_NODE  # route as if the planner had just produced it
        if changes.outputs:
            if "plan" not in values:
                raise ReplayError("there are no subtasks before the plan step")
            known = {st.id for st in values["plan"].subtasks}
            if unknown := set(changes.outputs) - known:
                raise ReplayError(f"unknown subtasks: {sorted(unknown)}")
            values["results"] = {**values.get("results", {}), **{
                sid: SubtaskResult(subtask_id=sid, output=text, succeeded=True)
                for sid, text in changes.outputs.items()}}
            values["accepted"] = {**values.get("accepted", {}),
                                  **dict.fromkeys(changes.outputs, "human")}
        if as_node in ("synthesize", "human_final", "deliver") and (
                changes.outputs or changes.task is not None or changes.guidance):
            # The answer was already written from the old inputs; write it again.
            values.pop("final_output", None)
            as_node = "schedule"
        return values, as_node

    def _seed_working_memory(self, task_id: str, values: dict[str, Any]) -> None:
        """Give the fork the shared results board the original had at that point."""
        wm = self.memory.working_for(task_id) if self.memory else None
        if wm is None or "plan" not in values:
            return
        try:
            wm.set_plan(values["plan"].model_dump(mode="json"))
            for sid in values.get("accepted", {}):
                if sid in values.get("results", {}):
                    wm.record_result(sid, values["results"][sid].output)
        except Exception:  # noqa: BLE001 - working memory is best-effort
            pass

    def replays_of(self, task_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self.store.replays_of(task_id):
            task = self.store.get_task(r.id)
            out.append({**_record(r), "status": task.status if task else "unknown"})
        return out

    # --- compare -----------------------------------------------------------------

    def compare(self, replay_id: str) -> dict[str, Any]:
        record = self.store.get_replay(replay_id)
        if record is None:
            raise KeyError(replay_id)
        flush()
        original_task = self.store.get_task(record.source_task_id)
        replay_task = self.store.get_task(replay_id)
        fork_ns = int(record.fork_time * 1e9)
        original_spans = self.spans.trace(record.source_task_id)
        replay_spans = self.spans.trace(replay_id)

        original_state = self._final_values(record.source_task_id)
        replay_state = self._final_values(replay_id)
        a = (original_task.final_output if original_task else None) or ""
        b = (replay_task.final_output if replay_task else None) or ""
        return {
            "replay": _record(record),
            "original": _side(original_task, original_spans, original_state, after_ns=fork_ns),
            "replayed": _side(replay_task, replay_spans, replay_state),
            "subtasks": _subtasks(original_state, replay_state),
            "final_output": {
                "similarity": round(difflib.SequenceMatcher(None, a, b).ratio(), 3),
                "diff": list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                                  "original", "replay", lineterm="", n=2)),
            },
        }

    # --- internals ---------------------------------------------------------------

    def _history(self, task_id: str) -> list[Any]:
        """Checkpoints oldest first."""
        return list(self.graph.get_state_history(TaskService._config(task_id)))[::-1]

    def _final_values(self, task_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(TaskService._config(task_id)).values)


def _ran(history: list[Any], index: int) -> list[str]:
    """Nodes whose writes produced checkpoint `index`: whatever was next at the one before.
    The first checkpoint of a fork comes from update_state rather than a node run."""
    snap = history[index]
    source = snap.metadata.get("source")
    if source == "input":
        return ["input"]
    if source == "update" or index == 0:
        writes = snap.metadata.get("writes") or {}
        return list(writes) or ["update"]
    return sorted(set(history[index - 1].next))


def _can_replay(snap: Any, ran: list[str]) -> bool:
    # Nothing left to run, or no node to resume after (the run's input, or the fork
    # point of a replay itself: replay that from the original instead).
    return bool(snap.next) and ran not in ([], ["input"], ["update"])


def _jsonable(values: Any) -> Any:
    return to_jsonable_python(values, fallback=str)


def _ts(created_at: str | None) -> float:
    return datetime.fromisoformat(created_at).timestamp() if created_at else 0.0


def _record(r: ReplayRecord) -> dict[str, Any]:
    return {"id": r.id, "source_task_id": r.source_task_id, "checkpoint_id": r.checkpoint_id,
            "step": r.step, "resumed_after": r.resumed_after, "fork_time": r.fork_time,
            "changes": r.changes, "created_at": r.created_at}


def _side(task: Any, all_spans: list[Any], state: dict[str, Any],
          after_ns: int = 0) -> dict[str, Any]:
    """One side of a comparison. For the original, only what ran after the fork point
    counts; a run segment that straddles it counts from the fork on."""
    spans = [s for s in all_spans if s.start_ns >= after_ns]
    cost = metrics.task_cost(spans)
    llm = [s for s in spans if s.kind == "llm"]
    agent_ns = sum(max(0.0, s.start_ns + s.duration_ms * 1e6 - max(s.start_ns, after_ns))
                   for s in all_spans if s.kind == "segment")
    return {
        "task_id": task.id if task else None,
        "task": task.task if task else None,
        "status": task.status if task else None,
        "final_output": task.final_output if task else None,
        "error": task.error if task else None,
        "cost_usd": cost["total_cost_usd"],
        "llm_calls": cost["llm_calls"],
        "input_tokens": cost["input_tokens"],
        "output_tokens": cost["output_tokens"],
        "tool_calls": cost["tool_calls"],
        "escalations": cost["escalations"],
        "agent_seconds": round(agent_ns / 1e9, 2),
        "model_seconds": round(sum(s.duration_ms for s in llm) / 1000, 2),
        "by_role": cost["by_role"],
        # Saving long-term memories; replays skip it, so it's shown apart.
        "memory_cost_usd": round(cost["by_role"].get("memory", {}).get("cost_usd", 0.0), 6),
        "plan": [f"{st.id}: {st.description}" for st in state["plan"].subtasks]
        if state.get("plan") else [],
    }


def _subtasks(original: dict[str, Any], replay: dict[str, Any]) -> list[dict[str, Any]]:
    def one(state: dict[str, Any], sid: str) -> dict[str, Any] | None:
        if not state.get("plan") or sid not in {st.id for st in state["plan"].subtasks}:
            return None
        result = state.get("results", {}).get(sid)
        review = state.get("reviews", {}).get(sid)
        return {
            "attempts": state.get("attempts", {}).get(sid, 0),
            "accepted_by": state.get("accepted", {}).get(sid),
            "abandoned": state.get("abandoned", {}).get(sid),
            "quality": review.quality_score if review else None,
            "output": result.output if result else None,
        }

    ids: list[str] = []
    for state in (original, replay):
        for st in (state["plan"].subtasks if state.get("plan") else []):
            if st.id not in ids:
                ids.append(st.id)
    rows = []
    for sid in ids:
        a, b = one(original, sid), one(replay, sid)
        same = None
        if a and b and a["output"] is not None and b["output"] is not None:
            same = round(difflib.SequenceMatcher(None, a["output"], b["output"]).ratio(), 3)
        rows.append({"id": sid, "original": a, "replay": b, "output_similarity": same})
    return rows
