"""The orchestration graph.

    intake -> plan -> human_plan?  (low plan confidence, or the user asked to review)
           -> schedule
    schedule --fan out--> work (one per runnable subtask in the current batch, in parallel)
    work -> schedule            (re-check: retry rejected subtasks, or move to next batch)
    schedule -> human_action    (a specialist asked for a sensitive tool call)
    schedule -> human_subtask   (a subtask failed review too many times)
    schedule -> synthesize -> human_final? (the user asked to review) -> deliver

`schedule` is the hub. It advances through the plan's batches, re-sends subtasks the
reviewer rejected (with its feedback), and routes to a human step when one is needed.
Human steps use LangGraph's `interrupt`: the run pauses, is checkpointed, and resumes
with `Command(resume=Decision)`. See `hitl.escalation` for the triggers and the approval
level each maps to. Rejected attempts that will be retried produce non-blocking
`notifications` instead of a pause.

Sensitive tool calls never run without approval. The specialist finishes what it can
without the action and the subtask is held; once a human approves, the exact call is
pre-approved and the subtask re-runs so the specialist makes it and uses the result.

Memory: `plan` retrieves similar past memories for the user and gives them to the
supervisor; `work` shares a per-task Redis working memory (results, errors, notes board);
`deliver` distills the finished task into long-term memories and clears working memory.
Memory failures are recorded but never fail the task.
"""

from __future__ import annotations

import functools
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel

from agentorch.agents.planning import InvalidPlanError, validate_plan
from agentorch.agents.reviewer import Reviewer
from agentorch.agents.schemas import (
    Complexity,
    ExecutionPlan,
    ReviewVerdict,
    Specialist,
    Subtask,
    SubtaskResult,
)
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.config import settings
from agentorch.hitl.escalation import (
    OPTIONS,
    ApprovalLevel,
    Decision,
    EscalationRequest,
    ProposedAction,
    Trigger,
    action_key,
)
from agentorch.llm import LLMUsage, ToolCall
from agentorch.memory.long_term import StoredMemory
from agentorch.memory.long_term_types import MemoryKind
from agentorch.memory.service import MemoryServices
from agentorch.observability.tracing import mark_error, set_attributes, tracer
from agentorch.tools.registry import ToolCallRecord, ToolContext


def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class Escalation(BaseModel):
    """A resolved human decision, kept in the run's history."""

    id: str
    trigger: Trigger
    level: ApprovalLevel
    subtask_id: str | None
    reason: str
    resolution: str
    note: str | None = None
    reviewer: str | None = None


class PendingApproval(BaseModel):
    """A sensitive tool call a specialist asked for. It does not run until approved."""

    id: str
    subtask_id: str
    tool: str
    input: dict[str, Any]
    reasoning: str | None = None
    status: Literal["pending", "approved", "modified", "rejected"] = "pending"


class WorkflowState(TypedDict, total=False):
    task: str
    task_id: str
    user_id: str
    require_review: bool  # the user wants to approve the plan and the final answer
    memories: list[str]  # extra context supplied by the caller
    memories_used: list[StoredMemory]  # retrieved from long-term memory for planning
    memories_saved: list[str]  # ids of memories written after the task
    memory_errors: Annotated[list[str], operator.add]
    status: Literal["running", "done", "failed", "rejected"]
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

    approvals: Annotated[dict[str, PendingApproval], merge]
    preapproved: Annotated[list[str], operator.add]  # action_key()s a human approved
    guidance: Annotated[list[str], operator.add]  # notes from plan reviews, for every subtask
    escalations: Annotated[list[Escalation], operator.add]
    notifications: Annotated[list[EscalationRequest], operator.add]
    usage: Annotated[list[LLMUsage], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    final_output: str

    # Set on runs forked from another task's checkpoint (see `agentorch.replay`).
    replay_of: str | None  # source task id; replays never write long-term memories
    reuse_memories: bool  # plan with the source's memories instead of retrieving again


# Our types that live in checkpoints. LangGraph only restores types it is told about
# (unknown ones log a warning today and will be refused in a later version).
CHECKPOINT_TYPES: list[type] = [
    ExecutionPlan, Subtask, Specialist, Complexity, SubtaskResult, ReviewVerdict,
    Escalation, EscalationRequest, ProposedAction, PendingApproval, ApprovalLevel, Trigger,
    LLMUsage, ToolCall, ToolCallRecord, StoredMemory, MemoryKind,
]


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


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

    memory = agents.memory or MemoryServices()

    # --- nodes ---------------------------------------------------------------

    def intake(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
        task = state.get("task", "").strip()
        if not task:
            return {"status": "failed", "error": "empty task"}
        return {
            "task": task,
            "task_id": config["configurable"]["thread_id"],
            "user_id": state.get("user_id") or "default",
            "require_review": bool(state.get("require_review")),
            "status": "running",
            "error": None,
        }

    def plan(state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {}
        retrieved: list[StoredMemory] = []
        if state.get("reuse_memories"):
            # A replay compares like with like: same memories as the original run,
            # not ones the original (or later tasks) wrote since.
            retrieved = list(state.get("memories_used", []))
        else:
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
        low = p.confidence < plan_confidence_threshold
        trigger = Trigger.LOW_PLAN_CONFIDENCE if low else Trigger.USER_REQUESTED
        reason = (f"Plan confidence {p.confidence:.2f} is below "
                  f"{plan_confidence_threshold:.2f}" if low
                  else "The user asked to review the plan before work starts")
        request = _request(state, trigger, ApprovalLevel.APPROVE_PLAN, reason,
                           proposed=ProposedAction(summary="Run this plan",
                                                   reasoning=p.reasoning))
        decision = _decide(interrupt(request.model_dump(mode="json")), request.level)
        update: dict[str, Any] = {"escalations": [_resolved_as(request, decision)]}
        if decision.action == "reject":
            return {**update, "status": "rejected",
                    "error": decision.note or "plan rejected by human"}
        if decision.note:
            update["guidance"] = [decision.note]
        if decision.action == "modify":
            assert decision.plan is not None
            batches = validate_plan(decision.plan)  # checked again by the API before resuming
            update.update(plan=decision.plan, batch_index=0,
                          batches=[[st.id for st in b] for b in batches])
        return update

    def schedule(state: WorkflowState) -> dict[str, Any]:
        batches = state["batches"]
        idx = state.get("batch_index", 0)
        while idx < len(batches) and all(_resolved(state, sid) for sid in batches[idx]):
            idx += 1
        return {"batch_index": idx}

    def work(payload: dict[str, Any]) -> dict[str, Any]:
        """Run one attempt of one subtask: specialist, then reviewer."""
        subtask = payload["subtask"]
        task_id = payload["task_id"]
        wm = memory.working_for(task_id)
        specialist = agents.make_specialist(subtask.specialist)
        run = specialist.run(
            subtask,
            dependency_outputs=payload["dependency_outputs"],
            reviewer_feedback=payload["feedback"],
            guidance=payload["guidance"],
            context=ToolContext(caller=subtask.specialist, task_id=task_id,
                                subtask_id=subtask.id, working_memory=wm,
                                preapproved=frozenset(payload["preapproved"])),
        )
        review = agents.reviewer.review(
            subtask, run.result, evidence=run.tool_calls,
            dependency_outputs=payload["dependency_outputs"],
            guidance=payload["guidance"],
        )
        approvals = {
            f"{task_id}-{subtask.id}-{c.id}": PendingApproval(
                id=f"{task_id}-{subtask.id}-{c.id}", subtask_id=subtask.id, tool=c.name,
                input=c.input, reasoning=run.approval_reasons.get(c.id))
            for c in run.pending_approvals
        }
        update: dict[str, Any] = {
            "results": {subtask.id: run.result},
            "reviews": {subtask.id: review.verdict},
            "attempts": {subtask.id: payload["attempt"]},
            "feedback": {subtask.id: review.verdict.feedback},
            "usage": [*run.usage, review.usage],
            "tool_calls": run.tool_calls,
            "approvals": approvals,
        }
        if review.passed and not approvals:
            # Held while an approval is pending, even if the reviewer liked it.
            update["accepted"] = {subtask.id: "reviewer"}
        elif not review.passed and not approvals and payload["attempt"] < max_attempts:
            update["notifications"] = [EscalationRequest(
                id=f"{task_id}-{subtask.id}-attempt{payload['attempt']}",
                trigger=Trigger.LOW_QUALITY, level=ApprovalLevel.NOTIFY,
                reason=(f"Subtask {subtask.id} attempt {payload['attempt']} was rejected "
                        f"(quality {review.verdict.quality_score:.2f}); retrying "
                        "automatically"),
                task_id=task_id, user_id=payload["user_id"], task=payload["task"],
                subtask_id=subtask.id,
                proposed=ProposedAction(summary="Retry with the reviewer's feedback",
                                        reasoning=review.verdict.feedback),
            )]
        if wm:
            if subtask.id in update.get("accepted", {}):
                _best_effort(update, "working.record_result",
                             lambda: wm.record_result(subtask.id, run.result.output))
            elif not review.passed:
                _best_effort(update, "working.log_error", lambda: wm.log_error(
                    f"{subtask.id}#{payload['attempt']}", review.verdict.feedback))
        return update

    def human_action(state: WorkflowState) -> dict[str, Any]:
        approval = _pending_approvals(state)[0]
        sid = approval.subtask_id
        request = _request(
            state, Trigger.SENSITIVE_ACTION, ApprovalLevel.APPROVE_ACTION,
            f"Subtask {sid} wants to call {approval.tool}, which needs approval",
            subtask_id=sid,
            proposed=ProposedAction(
                summary=f"Call {approval.tool}", reasoning=approval.reasoning,
                tool=approval.tool, tool_input=approval.input),
        )
        decision = _decide(interrupt(request.model_dump(mode="json")), request.level)
        attempts = state.get("attempts", {}).get(sid, 1)
        update: dict[str, Any] = {"escalations": [_resolved_as(request, decision)]}
        call = f"{approval.tool} with input {json.dumps(approval.input)}"

        if decision.action in ("approve", "modify"):
            tool_input = decision.tool_input if decision.action == "modify" else approval.input
            assert tool_input is not None
            status = "approved" if decision.action == "approve" else "modified"
            update["approvals"] = {approval.id: approval.model_copy(update={"status": status})}
            update["preapproved"] = [action_key(approval.tool, tool_input)]
            extra = f" Note from the reviewer: {decision.note}" if decision.note else ""
            update["feedback"] = {sid: (
                f"A human approved {approval.tool} with input {json.dumps(tool_input)}. "
                "Call it now with exactly that input; it will run. Then finish the subtask "
                f"using its result.{extra}")}
            update["attempts"] = {sid: max(0, attempts - 1)}  # re-run without penalty
            return update

        update["approvals"] = {approval.id: approval.model_copy(update={"status": "rejected"})}
        verdict = state.get("reviews", {}).get(sid)
        others_pending = any(a.subtask_id == sid and a.id != approval.id
                             for a in _pending_approvals(state))
        if verdict and not verdict.blocking_issues and not others_pending:
            # The specialist already finished without the action, and it passed review.
            update["accepted"] = {sid: "reviewer"}
        else:
            reason = f" Reason: {decision.note}" if decision.note else ""
            update["feedback"] = {sid: (
                f"A human declined {call}.{reason} Do not attempt it again; complete the "
                "subtask without it and say what could not be done.")}
            update["attempts"] = {sid: max(0, attempts - 1)}
        return update

    def human_subtask(state: WorkflowState) -> dict[str, Any]:
        sid = _exhausted(state, max_attempts)[0]
        result = state["results"].get(sid)
        verdict = state["reviews"].get(sid)
        request = _request(
            state, Trigger.REPEATED_FAILURE, ApprovalLevel.TAKE_OVER,
            f"Subtask {sid} was rejected {state['attempts'].get(sid, 0)} times",
            subtask_id=sid,
            proposed=ProposedAction(
                summary="Last attempt (rejected by the reviewer)",
                reasoning=verdict.feedback if verdict else None,
                output=(result.output or result.error) if result else None),
        )
        decision = _decide(interrupt(request.model_dump(mode="json")), request.level)
        update: dict[str, Any] = {"escalations": [_resolved_as(request, decision)]}
        if decision.action == "retry":
            update["attempts"] = {sid: 0}
            if decision.note:
                update["feedback"] = {sid: f"Guidance from a human reviewer: {decision.note}"}
        elif decision.action == "accept":
            update["accepted"] = {sid: "human"}
        elif decision.action == "take_over":
            update["results"] = {sid: SubtaskResult(subtask_id=sid, output=decision.output or "",
                                                    succeeded=True)}
            update["accepted"] = {sid: "human"}
        else:
            update["abandoned"] = {sid: decision.note or "aborted by human"}
        return update

    def synthesize(state: WorkflowState) -> dict[str, Any]:
        outputs = {sid: state["results"][sid].output for sid in state.get("accepted", {})}
        final, usage = agents.supervisor.synthesize(state["task"], state["plan"], outputs)
        return {"final_output": final, "usage": [usage]}

    def human_final(state: WorkflowState) -> dict[str, Any]:
        request = _request(
            state, Trigger.USER_REQUESTED, ApprovalLevel.APPROVE_ACTION,
            "The user asked to approve the final answer before it is delivered",
            proposed=ProposedAction(summary="Deliver this answer",
                                    output=state["final_output"]),
        )
        decision = _decide(interrupt(request.model_dump(mode="json")), request.level)
        update: dict[str, Any] = {"escalations": [_resolved_as(request, decision)]}
        if decision.action == "modify":
            update["final_output"] = decision.output or state["final_output"]
        elif decision.action == "reject":
            update.update(status="rejected", error=decision.note or "answer rejected by human")
        return update

    def deliver(state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {"status": "done", "memories_saved": []}
        if not state.get("replay_of"):  # experiments shouldn't teach the user's memory
            try:
                ids, usage = memory.remember(state["user_id"], state["task_id"],
                                             _transcript(state))
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
        if state.get("require_review") or state["plan"].confidence < plan_confidence_threshold:
            return "human_plan"
        return "schedule"

    def after_human_plan(state: WorkflowState) -> str:
        return END if state.get("status") == "rejected" else "schedule"

    def fan_out(state: WorkflowState) -> str | list[Send]:
        if state["batch_index"] >= len(state["batches"]):
            return "synthesize"
        if _pending_approvals(state):
            return "human_action"
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
                "user_id": state["user_id"],
                "task": state["task"],
                "subtask": subtask,
                "attempt": state.get("attempts", {}).get(sid, 0) + 1,
                "feedback": state.get("feedback", {}).get(sid),
                "preapproved": list(state.get("preapproved", [])),
                "guidance": list(state.get("guidance", [])),
                "dependency_outputs": {
                    dep: state["results"][dep].output
                    for dep in subtask.depends_on
                    if dep in state.get("accepted", {})
                },
            }))
        return sends

    def after_synthesize(state: WorkflowState) -> str:
        return "human_final" if state.get("require_review") else "deliver"

    def after_human_final(state: WorkflowState) -> str:
        return END if state.get("status") == "rejected" else "deliver"

    graph = StateGraph(WorkflowState)
    for name, fn in [("intake", intake), ("plan", plan), ("human_plan", human_plan),
                     ("schedule", schedule), ("work", work), ("human_action", human_action),
                     ("human_subtask", human_subtask), ("synthesize", synthesize),
                     ("human_final", human_final), ("deliver", deliver)]:
        graph.add_node(name, _traced(name, fn))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", after_intake, ["plan", END])
    graph.add_conditional_edges("plan", after_plan, ["human_plan", "schedule", END])
    graph.add_conditional_edges("human_plan", after_human_plan, ["schedule", END])
    graph.add_conditional_edges(
        "schedule", fan_out, ["work", "human_action", "human_subtask", "synthesize"])
    graph.add_edge("work", "schedule")
    graph.add_edge("human_action", "schedule")
    graph.add_edge("human_subtask", "schedule")
    graph.add_conditional_edges("synthesize", after_synthesize, ["human_final", "deliver"])
    graph.add_conditional_edges("human_final", after_human_final, ["deliver", END])
    graph.add_edge("deliver", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=checkpoint_serde()))


# --- helpers -----------------------------------------------------------------

def _traced(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a node inside a `node.<name>` span. A human-review pause is recorded as
    `paused`, not as an error."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = args[0] if args and isinstance(args[0], dict) else {}
        attrs: dict[str, Any] = {"agentorch.node": name}
        if name == "work":
            attrs.update({"agentorch.subtask_id": payload["subtask"].id,
                          "agentorch.attempt": payload["attempt"]})
        with tracer().start_as_current_span(
            f"node.{name}", record_exception=False, set_status_on_exception=False,
        ) as s:
            s.set_attribute("agentorch.kind", "node")
            set_attributes(s, attrs)
            try:
                return fn(*args, **kwargs)
            except GraphInterrupt:
                s.set_attribute("agentorch.paused", True)
                raise
            except Exception as e:
                s.record_exception(e)
                mark_error(s, f"{type(e).__name__}: {e}")
                raise

    return wrapper


def _request(
    state: WorkflowState,
    trigger: Trigger,
    level: ApprovalLevel,
    reason: str,
    subtask_id: str | None = None,
    proposed: ProposedAction | None = None,
) -> EscalationRequest:
    """Package the context a reviewer needs. The id is deterministic because LangGraph
    re-runs the node from the top when the run resumes."""
    progress: dict[str, dict[str, Any]] = {}
    for st in state["plan"].subtasks:
        if st.id in state.get("accepted", {}):
            status = "done"
        elif st.id in state.get("abandoned", {}):
            status = "abandoned"
        elif st.id in state.get("results", {}):
            status = "in_progress"
        else:
            status = "not_started"
        result = state.get("results", {}).get(st.id)
        progress[st.id] = {"status": status, "specialist": st.specialist.value,
                           "description": st.description,
                           "output": result.output if result else None}
    return EscalationRequest(
        id=f"{state['task_id']}-esc{len(state.get('escalations', [])) + 1}",
        trigger=trigger, level=level, reason=reason,
        task_id=state["task_id"], user_id=state["user_id"], task=state["task"],
        subtask_id=subtask_id, plan=state["plan"].model_dump(mode="json"),
        progress=progress, proposed=proposed,
        memories=[m.as_prompt_line() for m in state.get("memories_used", [])],
        options=OPTIONS[level],
    )


def _decide(raw: Any, level: ApprovalLevel) -> Decision:
    decision = Decision.model_validate(raw if isinstance(raw, dict) else {"action": raw})
    if decision.action not in OPTIONS[level]:
        raise ValueError(f"{level.value} accepts {OPTIONS[level]}, got {decision.action!r}")
    return decision


def _resolved_as(request: EscalationRequest, decision: Decision) -> Escalation:
    return Escalation(id=request.id, trigger=request.trigger, level=request.level,
                      subtask_id=request.subtask_id, reason=request.reason,
                      resolution=decision.action, note=decision.note,
                      reviewer=decision.reviewer)


def _pending_approvals(state: WorkflowState) -> list[PendingApproval]:
    batch = set(state["batches"][state["batch_index"]])
    return [a for a in state.get("approvals", {}).values()
            if a.status == "pending" and a.subtask_id in batch]


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
        note = f" ({e.note})" if e.note else ""
        lines.append(f"<escalation trigger=\"{e.trigger.value}\">{e.reason} -> human chose "
                     f"{e.resolution}{note}</escalation>")
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
