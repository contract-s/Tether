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
"""

from __future__ import annotations

import operator
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

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
from agentorch.tools.registry import ToolCallRecord


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
    memories: list[str]
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


def build_workflow(
    agents: Agents,
    checkpointer: BaseCheckpointSaver | None = None,
    max_attempts: int = settings.max_specialist_retries,
    plan_confidence_threshold: float = settings.plan_confidence_threshold,
):
    """Compile the graph. A checkpointer is required for the human-in-the-loop pauses."""

    # --- nodes ---------------------------------------------------------------

    def intake(state: WorkflowState) -> dict[str, Any]:
        task = state.get("task", "").strip()
        if not task:
            return {"status": "failed", "error": "empty task"}
        return {"task": task, "status": "running", "error": None}

    def plan(state: WorkflowState) -> dict[str, Any]:
        try:
            result = agents.supervisor.plan(state["task"], state.get("memories", []))
        except InvalidPlanError as e:
            return {"status": "failed", "error": str(e)}
        return {
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
        specialist = agents.make_specialist(subtask.specialist)
        run = specialist.run(
            subtask,
            dependency_outputs=payload["dependency_outputs"],
            reviewer_feedback=payload["feedback"],
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
        # Phase 2: save lessons from this task to long-term memory here.
        return {"status": "done"}

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
