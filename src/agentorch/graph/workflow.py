"""The orchestration graph.

    intake -> plan -> (low confidence? human_plan) -> schedule
    schedule --fan out--> work (one per runnable subtask in the current batch, in parallel)
    work -> schedule            (re-check: retry rejected subtasks, or move to next batch)
    schedule -> human_subtask   (a subtask failed review too many times)
    schedule -> synthesize -> deliver

`schedule` is the hub. It advances through the plan's batches, re-sends subtasks the
reviewer rejected (with its feedback), and escalates a subtask to a human once it has
used up its attempts. Human steps use LangGraph's `interrupt`, so the run pauses, is
checkpointed, and resumes with `Command(resume=...)`.

Memory: `plan` retrieves similar past memories for the user and gives them to the
supervisor; `work` shares a per-task Redis working memory (results, errors, notes board);
`deliver` distills the finished task into long-term memories and clears working memory.
Memory failures are recorded but never fail the task.
"""

from __future__ import annotations

import operator
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel

from agentorch.agents.planning import InvalidPlanError
from agentorch.agents.reviewer import Reviewer
from agentorch.agents.schemas import ExecutionPlan, ReviewVerdict, Specialist, SubtaskResult
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.config import settings
from agentorch.llm import LLMUsage
from agentorch.memory.long_term import StoredMemory
from agentorch.memory.service import MemoryServices
from agentorch.tools.registry import ToolCallRecord, ToolContext


def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class Escalation(BaseModel):
    id: str
    kind: Literal["plan_confidence", "subtask_failed"]
    subtask_id: str | None
    reason: str
    resolution: str


class PendingApproval(BaseModel):
    """A sensitive tool call a specialist wanted to make; it was not run."""

    subtask_id: str
    tool: str
    input: dict[str, Any]


class WorkflowState(TypedDict, total=False):
    task: str
    task_id: str
    user_id: str
    memories: list[str]  # extra context supplied by the caller
    memories_used: list[StoredMemory]  # retrieved from long-term memory for planning
    memories_saved: list[str]  # ids of memories written after the task
    memory_errors: Annotated[list[str], operator.add]
    status: Literal["running", "awaiting_human", "done", "failed", "rejected"]
    error: str | None

    plan: ExecutionPlan
    batches: list[list[str]]
    batch_index: int

    # Per-subtask bookkeeping. Parallel `work` branches each write their own key,
    # so these merge instead of overwrite.
    results: Annotated[dict[str, SubtaskResult], merge]
    reviews: Annotated[dict[str, ReviewVerdict], merge]
    attempts: Annotated[dict[str, int], merge]
    feedback: Annotated[dict[str, str], merge]
    accepted: Annotated[dict[str, str], merge]  # subtask id -> "reviewer" | "human"
    abandoned: Annotated[dict[str, str], merge]  # subtask id -> reason

    escalations: Annotated[list[Escalation], operator.add]
    pending_approvals: Annotated[list[PendingApproval], operator.add]
    usage: Annotated[list[LLMUsage], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    final_output: str


@dataclass
class Agents:
    supervisor: Supervisor
    reviewer: Reviewer
    make_specialist: Callable[[Specialist], SpecialistAgent]
    memory: MemoryServices | None = None


def build_workflow(
    agents: Agents,
    checkpointer: BaseCheckpointSaver | None = None,
    max_attempts: int = settings.max_specialist_retries,
    plan_confidence_threshold: float = settings.plan_confidence_threshold,
):
    """Compile the graph. A checkpointer is required for the human-in-the-loop pauses."""

    # --- nodes ---------------------------------------------------------------

    memory = agents.memory or MemoryServices()

    def intake(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
        task = state.get("task", "").strip()
        if not task:
            return {"status": "failed", "error": "empty task"}
        return {
            "task": task,
            "task_id": config["configurable"]["thread_id"],
            "user_id": state.get("user_id") or "default",
            "status": "running",
            "error": None,
        }

    def plan(state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {}
        retrieved: list[StoredMemory] = []
        try:
            retrieved = memory.retrieve(state["user_id"], state["task"])
        except Exception as e:  # noqa: BLE001 - memory is best-effort
            update["memory_errors"] = [f"retrieve: {type(e).__name__}: {e}"]
        context = [*state.get("memories", []), *(m.as_prompt_line() for m in retrieved)]
        try:
            result = agents.supervisor.plan(state["task"], context)
        except InvalidPlanError as e:
            return {**update, "status": "failed", "error": str(e)}
        wm = memory.working_for(state["task_id"])
        if wm:
            _best_effort(update, "working.set_plan",
                         lambda: wm.set_plan(result.plan.model_dump(mode="json")))
        return {
            **update,
            "memories_used": retrieved,
            "plan": result.plan,
            "batches": [[st.id for st in batch] for batch in result.batches],
            "batch_index": 0,
            "usage": result.usage,
        }

    def human_plan(state: WorkflowState) -> dict[str, Any]:
        p = state["plan"]
        reason = (f"Plan confidence {p.confidence:.2f} is below "
                  f"{plan_confidence_threshold:.2f}")
        decision = interrupt({
            "type": "approve_plan",
            "reason": reason,
            "task": state["task"],
            "plan": p.model_dump(mode="json"),
            "options": ["approve", "reject"],
        })
        action = _action(decision, {"approve", "reject"})
        escalation = Escalation(
            id=uuid.uuid4().hex, kind="plan_confidence", subtask_id=None,
            reason=reason, resolution=action,
        )
        if action == "approve":
            return {"escalations": [escalation], "status": "running"}
        return {"escalations": [escalation], "status": "rejected",
                "error": _note(decision) or "plan rejected by human"}

    def schedule(state: WorkflowState) -> dict[str, Any]:
        batches = state["batches"]
        idx = state.get("batch_index", 0)
        while idx < len(batches) and all(_resolved(state, sid) for sid in batches[idx]):
            idx += 1
        return {"batch_index": idx}

    def work(payload: dict[str, Any]) -> dict[str, Any]:
        """Run one attempt of one subtask: specialist, then reviewer."""
        subtask = payload["subtask"]
        wm = memory.working_for(payload["task_id"])
        specialist = agents.make_specialist(subtask.specialist)
        run = specialist.run(
            subtask,
            dependency_outputs=payload["dependency_outputs"],
            reviewer_feedback=payload["feedback"],
            context=ToolContext(caller=subtask.specialist, task_id=payload["task_id"],
                                subtask_id=subtask.id, working_memory=wm),
        )
        review = agents.reviewer.review(
            subtask, run.result, evidence=run.tool_calls,
            dependency_outputs=payload["dependency_outputs"],
        )
        update: dict[str, Any] = {
            "results": {subtask.id: run.result},
            "reviews": {subtask.id: review.verdict},
            "attempts": {subtask.id: payload["attempt"]},
            "feedback": {subtask.id: review.verdict.feedback},
            "usage": [*run.usage, review.usage],
            "tool_calls": run.tool_calls,
            "pending_approvals": [
                PendingApproval(subtask_id=subtask.id, tool=c.name, input=c.input)
                for c in run.pending_approvals
            ],
        }
        if review.passed:
            update["accepted"] = {subtask.id: "reviewer"}
        if wm:
            if review.passed:
                _best_effort(update, "working.record_result",
                             lambda: wm.record_result(subtask.id, run.result.output))
            else:
                _best_effort(update, "working.log_error", lambda: wm.log_error(
                    f"{subtask.id}#{payload['attempt']}", review.verdict.feedback))
        return update

    def human_subtask(state: WorkflowState) -> dict[str, Any]:
        sid = _exhausted(state, max_attempts)[0]
        result = state["results"].get(sid)
        verdict = state["reviews"].get(sid)
        reason = f"Subtask {sid} was rejected {state['attempts'].get(sid, 0)} times"
        decision = interrupt({
            "type": "subtask_failed",
            "reason": reason,
            "subtask": _subtask(state, sid).model_dump(mode="json"),
            "last_output": result.output if result else None,
            "last_error": result.error if result else None,
            "reviewer_feedback": verdict.feedback if verdict else None,
            "options": ["retry", "accept", "take_over", "abort"],
        })
        action = _action(decision, {"retry", "accept", "take_over", "abort"})
        escalation = Escalation(
            id=uuid.uuid4().hex, kind="subtask_failed", subtask_id=sid,
            reason=reason, resolution=action,
        )
        update: dict[str, Any] = {"escalations": [escalation], "status": "running"}
        if action == "retry":
            update["attempts"] = {sid: 0}
            if note := _note(decision):
                update["feedback"] = {sid: f"Guidance from a human reviewer: {note}"}
        elif action == "accept":
            update["accepted"] = {sid: "human"}
        elif action == "take_over":
            output = str(decision.get("output", "")) if isinstance(decision, dict) else ""
            update["results"] = {sid: SubtaskResult(subtask_id=sid, output=output,
                                                    succeeded=True)}
            update["accepted"] = {sid: "human"}
        else:
            update["abandoned"] = {sid: _note(decision) or "aborted by human"}
        return update

    def synthesize(state: WorkflowState) -> dict[str, Any]:
        outputs = {sid: state["results"][sid].output for sid in state.get("accepted", {})}
        final, usage = agents.supervisor.synthesize(state["task"], state["plan"], outputs)
        return {"final_output": final, "usage": [usage]}

    def deliver(state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {"status": "done", "memories_saved": []}
        try:
            ids, usage = memory.remember(state["user_id"], state["task_id"], _transcript(state))
            update["memories_saved"] = ids
            if usage:
                update["usage"] = [usage]
        except Exception as e:  # noqa: BLE001 - memory is best-effort
            update["memory_errors"] = [f"remember: {type(e).__name__}: {e}"]
        wm = memory.working_for(state["task_id"])
        if wm:
            _best_effort(update, "working.clear", wm.clear)
        return update

    # --- routing -------------------------------------------------------------

    def after_intake(state: WorkflowState) -> str:
        return END if state.get("status") == "failed" else "plan"

    def after_plan(state: WorkflowState) -> str:
        if state.get("status") == "failed":
            return END
        if state["plan"].confidence < plan_confidence_threshold:
            return "human_plan"
        return "schedule"

    def after_human_plan(state: WorkflowState) -> str:
        return END if state.get("status") == "rejected" else "schedule"

    def fan_out(state: WorkflowState) -> str | list[Send]:
        if state["batch_index"] >= len(state["batches"]):
            return "synthesize"
        unresolved = [s for s in state["batches"][state["batch_index"]]
                      if not _resolved(state, s)]
        runnable = [s for s in unresolved if state.get("attempts", {}).get(s, 0) < max_attempts]
        if not runnable:
            return "human_subtask"
        sends = []
        for sid in runnable:
            subtask = _subtask(state, sid)
            sends.append(Send("work", {
                "task_id": state["task_id"],
                "subtask": subtask,
                "attempt": state.get("attempts", {}).get(sid, 0) + 1,
                "feedback": state.get("feedback", {}).get(sid),
                "dependency_outputs": {
                    dep: state["results"][dep].output
                    for dep in subtask.depends_on
                    if dep in state.get("accepted", {})
                },
            }))
        return sends

    graph = StateGraph(WorkflowState)
    graph.add_node("intake", intake)
    graph.add_node("plan", plan)
    graph.add_node("human_plan", human_plan)
    graph.add_node("schedule", schedule)
    graph.add_node("work", work)
    graph.add_node("human_subtask", human_subtask)
    graph.add_node("synthesize", synthesize)
    graph.add_node("deliver", deliver)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", after_intake, ["plan", END])
    graph.add_conditional_edges("plan", after_plan, ["human_plan", "schedule", END])
    graph.add_conditional_edges("human_plan", after_human_plan, ["schedule", END])
    graph.add_conditional_edges("schedule", fan_out, ["work", "human_subtask", "synthesize"])
    graph.add_edge("work", "schedule")
    graph.add_edge("human_subtask", "schedule")
    graph.add_edge("synthesize", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# --- helpers -----------------------------------------------------------------

def _best_effort(update: dict[str, Any], label: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - memory is best-effort
        update.setdefault("memory_errors", []).append(f"{label}: {type(e).__name__}: {e}")


def _transcript(state: WorkflowState) -> str:
    """What the memory extractor sees about a finished task."""
    p = state["plan"]
    lines = [f"<request>\n{state['task']}\n</request>",
             f"<plan confidence=\"{p.confidence:.2f}\">{p.reasoning}</plan>", "<subtasks>"]
    for st in p.subtasks:
        if by := state.get("accepted", {}).get(st.id):
            status = f"accepted by {by}"
        elif st.id in state.get("abandoned", {}):
            status = "abandoned"
        else:
            status = "not completed"
        lines.append(f"- {st.id} [{st.specialist.value}] {st.description} | {status}, "
                     f"attempts {state.get('attempts', {}).get(st.id, 0)}")
        if (v := state.get("reviews", {}).get(st.id)) and v.blocking_issues:
            lines.append(f"  last review issues: {'; '.join(v.blocking_issues)[:600]}")
    lines.append("</subtasks>")
    used = sorted({r.tool for r in state.get("tool_calls", [])})
    lines.append(f"<tools_used>{', '.join(used) or 'none'}</tools_used>")
    for e in state.get("escalations", []):
        lines.append(f"<escalation kind=\"{e.kind}\">{e.reason} -> human chose "
                     f"{e.resolution}</escalation>")
    final = state.get("final_output", "")
    lines.append(f"<final_output>\n{final[:6000]}\n</final_output>")
    return "\n".join(lines)


def _resolved(state: WorkflowState, sid: str) -> bool:
    return sid in state.get("accepted", {}) or sid in state.get("abandoned", {})


def _exhausted(state: WorkflowState, max_attempts: int) -> list[str]:
    batch = state["batches"][state["batch_index"]]
    return [s for s in batch
            if not _resolved(state, s) and state.get("attempts", {}).get(s, 0) >= max_attempts]


def _subtask(state: WorkflowState, sid: str):
    return next(st for st in state["plan"].subtasks if st.id == sid)


def _action(decision: Any, allowed: set[str]) -> str:
    action = decision.get("action") if isinstance(decision, dict) else decision
    if action not in allowed:
        raise ValueError(f"resume with one of {sorted(allowed)}, got {action!r}")
    return action


def _note(decision: Any) -> str | None:
    return decision.get("note") if isinstance(decision, dict) else None
