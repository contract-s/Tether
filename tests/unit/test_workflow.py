"""End-to-end graph behaviour with a fake model (no API calls)."""

import re
import threading
import uuid
from collections import Counter
from collections.abc import Callable
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.schemas import ExecutionPlan, ReviewVerdict
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.graph.workflow import Agents, build_workflow
from agentorch.llm import LLMUsage, ModelTurn, Role, StructuredResult
from agentorch.tools.registry import ToolRegistry

USAGE = LLMUsage("fake", 1, 1)


def make_plan(confidence: float = 0.9) -> ExecutionPlan:
    def st(id: str, deps: list[str]) -> dict[str, Any]:
        return {"id": id, "description": f"do {id}", "specialist": "research",
                "required_inputs": [], "depends_on": deps, "expected_output": "text",
                "complexity": "low"}

    return ExecutionPlan.model_validate({
        "goal": "g", "reasoning": "r", "confidence": confidence,
        "subtasks": [st("t1", []), st("t2", []), st("t3", ["t1", "t2"])],
    })


class FakeLLM:
    """Thread-safe fake: specialists answer 'output for <id>', the reviewer follows `approve`."""

    def __init__(self, plan: ExecutionPlan, approve: Callable[[str, int], bool]) -> None:
        self.plan = plan
        self.approve = approve
        self.lock = threading.Lock()
        self.reviews: Counter[str] = Counter()
        self.specialist_prompts: dict[str, list[str]] = {}
        self.synthesis_prompt: str | None = None

    def structured(self, role: Role, system: str, prompt: str, schema: type[BaseModel]):
        if role == "supervisor":
            return StructuredResult(self.plan, USAGE)
        sid = re.search(r'<subtask id="(\w+)"', prompt).group(1)
        with self.lock:
            self.reviews[sid] += 1
            n = self.reviews[sid]
        ok = self.approve(sid, n)
        return StructuredResult(ReviewVerdict(
            subtask_id=sid, approved=ok, quality_score=0.9 if ok else 0.3,
            blocking_issues=[] if ok else [f"fix {sid} attempt {n}"], suggestions=[],
        ), USAGE)

    def chat(self, role: Role, system: str, messages, tools) -> ModelTurn:
        prompt = messages[0]["content"]
        if role == "supervisor":
            self.synthesis_prompt = prompt
            return ModelTurn([], "FINAL ANSWER", [], "end_turn", USAGE)
        sid = re.search(r'<subtask id="(\w+)"', prompt).group(1)
        with self.lock:
            self.specialist_prompts.setdefault(sid, []).append(prompt)
        text = f"output for {sid}"
        return ModelTurn([{"type": "text", "text": text}], text, [], "end_turn", USAGE)


def make_graph(llm: FakeLLM, max_attempts: int = 2):
    registry = ToolRegistry()
    agents = Agents(
        supervisor=Supervisor(llm),
        reviewer=Reviewer(llm, threshold=0.7),
        make_specialist=lambda s: SpecialistAgent(s, llm, registry),
    )
    graph = build_workflow(agents, max_attempts=max_attempts, plan_confidence_threshold=0.7)
    return graph, {"configurable": {"thread_id": uuid.uuid4().hex}}


def interrupt_of(result: dict[str, Any]) -> dict[str, Any] | None:
    pending = result.get("__interrupt__")
    return pending[0].value if pending else None


def test_happy_path_runs_batches_in_order_and_synthesizes() -> None:
    llm = FakeLLM(make_plan(), approve=lambda sid, n: True)
    graph, config = make_graph(llm)
    result = graph.invoke({"task": "research things"}, config)

    assert result["status"] == "done" and result["final_output"] == "FINAL ANSWER"
    assert set(result["accepted"]) == {"t1", "t2", "t3"}
    # t3 depends on t1 and t2, so it sees both outputs.
    t3_prompt = llm.specialist_prompts["t3"][0]
    assert "output for t1" in t3_prompt and "output for t2" in t3_prompt
    assert "output for t3" in llm.synthesis_prompt


def test_rejected_subtask_is_retried_with_feedback() -> None:
    llm = FakeLLM(make_plan(), approve=lambda sid, n: not (sid == "t1" and n == 1))
    graph, config = make_graph(llm)
    result = graph.invoke({"task": "x"}, config)

    assert result["status"] == "done"
    assert result["attempts"]["t1"] == 2 and result["attempts"]["t2"] == 1
    assert "fix t1 attempt 1" in llm.specialist_prompts["t1"][1]


def test_repeated_failure_escalates_then_human_takes_over() -> None:
    llm = FakeLLM(make_plan(), approve=lambda sid, n: sid != "t2")
    graph, config = make_graph(llm)
    result = graph.invoke({"task": "x"}, config)

    pause = interrupt_of(result)
    assert pause["type"] == "subtask_failed" and pause["subtask"]["id"] == "t2"
    assert len(llm.specialist_prompts["t2"]) == 2
    assert "t3" not in llm.specialist_prompts  # later batch waits for the human

    result = graph.invoke(Command(resume={"action": "take_over", "output": "human t2"}), config)
    assert result["status"] == "done"
    assert result["accepted"]["t2"] == "human"
    assert "human t2" in llm.specialist_prompts["t3"][0]
    assert [e.resolution for e in result["escalations"]] == ["take_over"]


def test_human_retry_resets_attempts_and_passes_guidance() -> None:
    llm = FakeLLM(make_plan(), approve=lambda sid, n: sid != "t2" or n >= 3)
    graph, config = make_graph(llm)
    graph.invoke({"task": "x"}, config)

    result = graph.invoke(Command(resume={"action": "retry", "note": "use the docs"}), config)
    assert result["status"] == "done"
    assert "use the docs" in llm.specialist_prompts["t2"][2]


def test_human_abort_skips_subtask_and_synthesis_reports_it_missing() -> None:
    llm = FakeLLM(make_plan(), approve=lambda sid, n: sid != "t2")
    graph, config = make_graph(llm)
    graph.invoke({"task": "x"}, config)

    result = graph.invoke(Command(resume={"action": "abort", "note": "not needed"}), config)
    assert result["status"] == "done" and result["abandoned"] == {"t2": "not needed"}
    assert "<missing_subtasks>t2</missing_subtasks>" in llm.synthesis_prompt


def test_low_confidence_plan_waits_for_human() -> None:
    llm = FakeLLM(make_plan(confidence=0.4), approve=lambda sid, n: True)
    graph, config = make_graph(llm)
    result = graph.invoke({"task": "x"}, config)

    assert interrupt_of(result)["type"] == "approve_plan"
    assert llm.specialist_prompts == {}  # nothing runs before approval
    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert result["status"] == "done"


def test_rejected_plan_ends_without_running_specialists() -> None:
    llm = FakeLLM(make_plan(confidence=0.4), approve=lambda sid, n: True)
    graph, config = make_graph(llm)
    graph.invoke({"task": "x"}, config)
    result = graph.invoke(Command(resume={"action": "reject", "note": "too vague"}), config)

    assert result["status"] == "rejected" and result["error"] == "too vague"
    assert llm.specialist_prompts == {}


def test_empty_task_fails_fast() -> None:
    llm = FakeLLM(make_plan(), approve=lambda sid, n: True)
    graph, config = make_graph(llm)
    assert graph.invoke({"task": "  "}, config)["status"] == "failed"


def test_reviewer_sees_dependency_outputs_as_evidence() -> None:
    class Recording(FakeLLM):
        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, **kw)
            self.review_prompts: dict[str, str] = {}

        def structured(self, role, system, prompt, schema):
            if role == "reviewer":
                sid = re.search(r'<subtask id="(\w+)"', prompt).group(1)
                self.review_prompts[sid] = prompt
            return super().structured(role, system, prompt, schema)

    llm = Recording(make_plan(), approve=lambda sid, n: True)
    graph, config = make_graph(llm)
    graph.invoke({"task": "x"}, config)
    assert '<accepted_input from_subtask="t1">' in llm.review_prompts["t3"]
    assert "accepted_input" not in llm.review_prompts["t1"]
