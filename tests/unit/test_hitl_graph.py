"""Human-in-the-loop paths through the graph (fake model, no API calls)."""

import json
import re
import uuid
from typing import Any

import pytest
from langgraph.types import Command
from pydantic import BaseModel

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.schemas import ExecutionPlan, ReviewVerdict, Specialist
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.graph.workflow import Agents, build_workflow
from agentorch.llm import LLMUsage, ModelTurn, StructuredResult, ToolCall
from agentorch.tools.registry import Tool, ToolRegistry
from tests.unit.test_workflow import make_plan

USAGE = LLMUsage("fake", 1, 1)


class SendIn(BaseModel):
    to: str
    body: str


class SendOut(BaseModel):
    sent: bool


def one_subtask_plan(confidence: float = 0.9) -> ExecutionPlan:
    return ExecutionPlan.model_validate({
        "goal": "g", "reasoning": "r", "confidence": confidence,
        "subtasks": [{"id": "t1", "description": "email the summary", "specialist": "research",
                      "required_inputs": [], "depends_on": [], "expected_output": "text",
                      "complexity": "low"}],
    })


class ActionLLM:
    """A specialist that wants to send an email, which needs approval."""

    def __init__(self, plan: ExecutionPlan | None = None, approve_review=lambda n: True) -> None:
        self.plan = plan or one_subtask_plan()
        self.approve_review = approve_review
        self.reviews = 0
        self.specialist_prompts: list[str] = []

    def structured(self, role, system, prompt, schema):
        if role == "supervisor":
            return StructuredResult(self.plan, USAGE)
        if role == "memory":
            raise AssertionError("no memory configured")
        self.reviews += 1
        ok = self.approve_review(self.reviews)
        return StructuredResult(ReviewVerdict(
            subtask_id="t1", approved=ok, quality_score=0.9 if ok else 0.3,
            blocking_issues=[] if ok else ["wrong"], suggestions=[]), USAGE)

    def chat(self, role, system, messages, tools) -> ModelTurn:
        if role == "supervisor":
            return ModelTurn([], "FINAL ANSWER", [], "end_turn", USAGE)
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            result = last["content"][0]
            text = "skipped the email" if result.get("is_error") else "email sent"
            return ModelTurn([{"type": "text", "text": text}], text, [], "end_turn", USAGE)
        prompt = messages[0]["content"]
        self.specialist_prompts.append(prompt)
        if m := re.search(r"approved send_email with input (\{.*?\})\. Call", prompt):
            tool_input = json.loads(m.group(1))
        else:
            tool_input = {"to": "boss@example.com", "body": "summary"}
        call = ToolCall(f"c{len(self.specialist_prompts)}", "send_email", tool_input)
        return ModelTurn([{"type": "tool_use"}], "I need to email the summary to the boss.",
                         [call], "tool_use", USAGE)


def make_graph(llm, plan_threshold: float = 0.7):
    sent: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(Tool(
        name="send_email", description="Send an email", input_model=SendIn,
        output_model=SendOut,
        handler=lambda a: (sent.append(a.model_dump()), SendOut(sent=True))[1],
        allowed_specialists=frozenset({Specialist.RESEARCH}),
        is_sensitive=lambda a: True,
    ))
    agents = Agents(Supervisor(llm), Reviewer(llm, threshold=0.7),
                    lambda s: SpecialistAgent(s, llm, registry))
    graph = build_workflow(agents, plan_confidence_threshold=plan_threshold)
    return graph, {"configurable": {"thread_id": uuid.uuid4().hex}}, sent


def pause(graph, config) -> dict[str, Any] | None:
    state = graph.get_state(config)
    return state.tasks[0].interrupts[0].value if state.next else None


# --- sensitive actions ------------------------------------------------------

def test_sensitive_action_waits_and_runs_once_after_approval() -> None:
    llm = ActionLLM()
    graph, config, sent = make_graph(llm)
    graph.invoke({"task": "x"}, config)

    p = pause(graph, config)
    assert p["level"] == "approve_action" and p["trigger"] == "sensitive_action"
    assert p["proposed"]["tool"] == "send_email"
    assert p["proposed"]["tool_input"] == {"to": "boss@example.com", "body": "summary"}
    assert "email the summary" in p["proposed"]["reasoning"]
    assert p["options"] == ["approve", "modify", "reject"]
    assert sent == []  # nothing ran yet

    result = graph.invoke(Command(resume={"action": "approve", "reviewer": "sam"}), config)
    assert result["status"] == "done"
    assert sent == [{"to": "boss@example.com", "body": "summary"}]  # exactly once
    assert result["results"]["t1"].output == "email sent"
    assert result["escalations"][0].reviewer == "sam"
    assert result["attempts"]["t1"] == 1  # the held attempt didn't count as a failure


def test_modified_action_runs_with_the_humans_input() -> None:
    llm = ActionLLM()
    graph, config, sent = make_graph(llm)
    graph.invoke({"task": "x"}, config)
    graph.invoke(Command(resume={
        "action": "modify", "tool_input": {"to": "team@example.com", "body": "summary"}}), config)
    assert sent == [{"to": "team@example.com", "body": "summary"}]


def test_rejected_action_never_runs_and_passing_work_is_kept() -> None:
    llm = ActionLLM()
    graph, config, sent = make_graph(llm)
    graph.invoke({"task": "x"}, config)
    result = graph.invoke(Command(resume={"action": "reject", "note": "not now"}), config)

    assert result["status"] == "done" and sent == []
    assert result["results"]["t1"].output == "skipped the email"
    assert len(llm.specialist_prompts) == 1  # accepted as is, no re-run
    assert result["approvals"][next(iter(result["approvals"]))].status == "rejected"


def test_invalid_decision_is_refused() -> None:
    graph, config, _ = make_graph(ActionLLM())
    graph.invoke({"task": "x"}, config)
    with pytest.raises(ValueError, match="approve_action accepts"):
        graph.invoke(Command(resume={"action": "take_over"}), config)


# --- user-requested review ----------------------------------------------------

class PlainLLM(ActionLLM):
    """Specialists answer without tools."""

    def chat(self, role, system, messages, tools) -> ModelTurn:
        if role == "supervisor":
            return ModelTurn([], "FINAL ANSWER", [], "end_turn", USAGE)
        sid = re.search(r'<subtask id="(\w+)"', messages[0]["content"]).group(1)
        self.specialist_prompts.append(sid)
        return ModelTurn([], f"output for {sid}", [], "end_turn", USAGE)


def test_require_review_pauses_for_plan_and_final_answer() -> None:
    llm = PlainLLM()
    graph, config, _ = make_graph(llm)
    graph.invoke({"task": "x", "require_review": True}, config)

    first = pause(graph, config)
    assert first["level"] == "approve_plan" and first["trigger"] == "user_requested"
    graph.invoke(Command(resume={"action": "approve"}), config)

    final = pause(graph, config)
    assert final["level"] == "approve_action"
    assert final["proposed"]["output"] == "FINAL ANSWER"
    assert final["progress"]["t1"]["status"] == "done"

    result = graph.invoke(Command(resume={"action": "modify", "output": "EDITED"}), config)
    assert result["status"] == "done" and result["final_output"] == "EDITED"
    assert [e.level for e in result["escalations"]] == ["approve_plan", "approve_action"]


def test_rejecting_the_final_answer_ends_without_delivery() -> None:
    graph, config, _ = make_graph(PlainLLM())
    graph.invoke({"task": "x", "require_review": True}, config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    result = graph.invoke(Command(resume={"action": "reject", "note": "too vague"}), config)
    assert result["status"] == "rejected" and result["error"] == "too vague"


def test_modified_plan_replaces_the_original() -> None:
    llm = PlainLLM(plan=make_plan(confidence=0.4))  # t1, t2, t3
    graph, config, _ = make_graph(llm)
    graph.invoke({"task": "x"}, config)
    new_plan = one_subtask_plan().model_dump(mode="json")
    result = graph.invoke(Command(resume={"action": "modify", "plan": new_plan}), config)

    assert result["status"] == "done" and llm.specialist_prompts == ["t1"]
    assert result["batches"] == [["t1"]]


# --- notify -----------------------------------------------------------------

def test_rejected_attempt_notifies_without_pausing() -> None:
    llm = PlainLLM(approve_review=lambda n: n > 1)
    graph, config, _ = make_graph(llm)
    result = graph.invoke({"task": "x"}, config)

    assert result["status"] == "done" and pause(graph, config) is None
    [note] = result["notifications"]
    assert note.level == "notify" and note.trigger == "low_quality"
    assert "retrying automatically" in note.reason


def test_escalation_context_includes_plan_progress_and_memories() -> None:
    graph, config, _ = make_graph(PlainLLM(plan=make_plan(confidence=0.4)))
    graph.invoke({"task": "x", "memories": ["ignored: caller memories aren't stored"]}, config)
    p = pause(graph, config)
    assert p["plan"]["subtasks"][0]["id"] == "t1"
    assert set(p["progress"]) == {"t1", "t2", "t3"}
    assert p["id"].endswith("-esc1") and p["task"] == "x"
