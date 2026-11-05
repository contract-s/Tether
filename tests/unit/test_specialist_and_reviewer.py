from typing import Any

from pydantic import BaseModel

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.schemas import (
    Complexity,
    ReviewVerdict,
    Specialist,
    Subtask,
    SubtaskResult,
)
from agentorch.agents.specialist import SpecialistAgent
from agentorch.llm import LLMUsage, ModelTurn, Role, StructuredResult, ToolCall
from agentorch.tools.registry import Tool, ToolRegistry

USAGE = LLMUsage("fake", 1, 1)


class AddIn(BaseModel):
    a: int
    b: int
    send: bool = False


class AddOut(BaseModel):
    total: int


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="add", description="Add numbers", input_model=AddIn, output_model=AddOut,
        handler=lambda x: AddOut(total=x.a + x.b),
        allowed_specialists=frozenset({Specialist.ANALYSIS}),
        is_sensitive=lambda x: x.send,
    ))
    return registry


SUBTASK = Subtask(
    id="t1", description="Add 2 and 3", specialist=Specialist.ANALYSIS, required_inputs=[],
    depends_on=[], expected_output="The sum", complexity=Complexity.LOW,
)


def tool_turn(*calls: ToolCall) -> ModelTurn:
    return ModelTurn(content=[{"type": "tool_use"}], text="", tool_calls=list(calls),
                     stop_reason="tool_use", usage=USAGE)


def final_turn(text: str) -> ModelTurn:
    return ModelTurn(content=[{"type": "text", "text": text}], text=text, tool_calls=[],
                     stop_reason="end_turn", usage=USAGE)


class ScriptedLLM:
    """Plays back scripted turns and keeps the messages it was sent."""

    def __init__(self, *turns: ModelTurn, verdict: ReviewVerdict | None = None) -> None:
        self.turns = list(turns)
        self.verdict = verdict
        self.sent: list[list[dict[str, Any]]] = []

    def chat(self, role: Role, system: str, messages, tools) -> ModelTurn:
        assert role == "specialist"
        assert [t["name"] for t in tools] == ["add"]
        self.sent.append(list(messages))
        return self.turns.pop(0)

    def structured(self, role: Role, system: str, prompt: str, schema: type[BaseModel]):
        assert role == "reviewer" and self.verdict is not None
        return StructuredResult(self.verdict, USAGE)


def test_specialist_runs_tools_then_returns_answer() -> None:
    llm = ScriptedLLM(tool_turn(ToolCall("c1", "add", {"a": 2, "b": 3})), final_turn("5"))
    run = SpecialistAgent(Specialist.ANALYSIS, llm, make_registry()).run(SUBTASK)

    assert run.result.succeeded and run.result.output == "5"
    assert run.turns == 2 and len(run.tool_calls) == 1
    tool_result = llm.sent[1][-1]["content"][0]
    assert tool_result["tool_use_id"] == "c1" and '"total": 5' in tool_result["content"]


def test_tool_failures_go_back_to_the_model_as_errors() -> None:
    llm = ScriptedLLM(tool_turn(ToolCall("c1", "add", {"a": "x"})), final_turn("could not add"))
    SpecialistAgent(Specialist.ANALYSIS, llm, make_registry()).run(SUBTASK)
    assert llm.sent[1][-1]["content"][0]["is_error"] is True


def test_sensitive_calls_are_held_for_approval() -> None:
    call = ToolCall("c1", "add", {"a": 1, "b": 1, "send": True})
    llm = ScriptedLLM(tool_turn(call), final_turn("skipped sending"))
    run = SpecialistAgent(Specialist.ANALYSIS, llm, make_registry()).run(SUBTASK)
    assert run.pending_approvals == [call]
    assert "approval" in llm.sent[1][-1]["content"][0]["content"]


def test_parallel_tool_calls_return_in_one_message() -> None:
    calls = [ToolCall("c1", "add", {"a": 1, "b": 1}), ToolCall("c2", "add", {"a": 2, "b": 2})]
    llm = ScriptedLLM(tool_turn(*calls), final_turn("2 and 4"))
    SpecialistAgent(Specialist.ANALYSIS, llm, make_registry()).run(SUBTASK)
    results = llm.sent[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["c1", "c2"]


def test_specialist_stops_after_max_turns() -> None:
    loop = [tool_turn(ToolCall(f"c{i}", "add", {"a": 1, "b": 1})) for i in range(3)]
    run = SpecialistAgent(Specialist.ANALYSIS, ScriptedLLM(*loop), make_registry(),
                          max_turns=3).run(SUBTASK)
    assert not run.result.succeeded and "3 turns" in run.result.error


def test_feedback_and_dependency_outputs_reach_the_prompt() -> None:
    llm = ScriptedLLM(final_turn("done"))
    SpecialistAgent(Specialist.ANALYSIS, llm, make_registry()).run(
        SUBTASK, dependency_outputs={"t0": "earlier result"}, reviewer_feedback="cite sources"
    )
    prompt = llm.sent[0][0]["content"]
    assert "earlier result" in prompt and "cite sources" in prompt


def verdict(approved: bool, score: float) -> ReviewVerdict:
    return ReviewVerdict(subtask_id="?", approved=approved, quality_score=score, feedback="f")


def test_reviewer_passes_only_above_threshold() -> None:
    ok = SubtaskResult(subtask_id="t1", output="5", succeeded=True)
    assert Reviewer(ScriptedLLM(verdict=verdict(True, 0.9)), 0.7).review(SUBTASK, ok).passed
    assert not Reviewer(ScriptedLLM(verdict=verdict(True, 0.5)), 0.7).review(SUBTASK, ok).passed
    out = Reviewer(ScriptedLLM(verdict=verdict(False, 0.9)), 0.7).review(SUBTASK, ok)
    assert not out.passed and out.verdict.subtask_id == "t1"


def test_reviewer_rejects_failed_attempts_without_calling_model() -> None:
    failed = SubtaskResult(subtask_id="t1", output="", succeeded=False, error="boom")
    out = Reviewer(ScriptedLLM(), 0.7).review(SUBTASK, failed)
    assert not out.passed and "boom" in out.verdict.feedback


def test_reviewer_sees_tool_evidence() -> None:
    class Recorder(ScriptedLLM):
        def structured(self, role, system, prompt, schema):
            self.prompt = prompt
            return super().structured(role, system, prompt, schema)

    llm = Recorder(tool_turn(ToolCall("c1", "add", {"a": 2, "b": 3})), final_turn("5"),
                   verdict=verdict(True, 0.9))
    run = SpecialistAgent(Specialist.ANALYSIS, llm, make_registry()).run(SUBTASK)
    Reviewer(llm, 0.7).review(SUBTASK, run.result, evidence=run.tool_calls)
    assert "<tool_evidence>" in llm.prompt and '"total": 5' in llm.prompt
