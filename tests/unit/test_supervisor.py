from typing import Any

import pytest
from pydantic import BaseModel

from tether.agents.planning import InvalidPlanError
from tether.agents.schemas import ExecutionPlan
from tether.agents.supervisor import Supervisor
from tether.llm import LLMUsage, Role, StructuredResult


def make_plan(subtasks: list[dict[str, Any]]) -> ExecutionPlan:
    return ExecutionPlan.model_validate(
        {"goal": "g", "reasoning": "r", "confidence": 0.8, "subtasks": subtasks}
    )


def subtask(id: str, deps: list[str]) -> dict[str, Any]:
    return {
        "id": id, "description": id, "specialist": "research", "required_inputs": [],
        "depends_on": deps, "expected_output": "text", "complexity": "low",
    }


class FakeLLM:
    """Returns canned plans in order and records the prompts it was given."""

    def __init__(self, *plans: ExecutionPlan) -> None:
        self.plans = list(plans)
        self.prompts: list[str] = []

    def structured(self, role: Role, system: str, prompt: str, schema: type[BaseModel]):
        assert role == "supervisor" and schema is ExecutionPlan
        self.prompts.append(prompt)
        return StructuredResult(self.plans.pop(0), LLMUsage("fake", 10, 20))


def test_returns_valid_plan_and_batches() -> None:
    llm = FakeLLM(make_plan([subtask("t1", []), subtask("t2", ["t1"])]))
    result = Supervisor(llm).plan("research X")
    assert result.attempts == 1
    assert [[s.id for s in b] for b in result.batches] == [["t1"], ["t2"]]


def test_retries_with_error_feedback_after_invalid_plan() -> None:
    bad = make_plan([subtask("t1", ["t2"]), subtask("t2", ["t1"])])
    good = make_plan([subtask("t1", [])])
    llm = FakeLLM(bad, good)
    result = Supervisor(llm).plan("research X")
    assert result.attempts == 2
    assert "previous plan was invalid" in llm.prompts[1]
    assert len(result.usage) == 2


def test_gives_up_after_max_attempts() -> None:
    bad = make_plan([subtask("t1", ["nope"])])
    with pytest.raises(InvalidPlanError, match="no valid plan"):
        Supervisor(FakeLLM(bad, bad), max_attempts=2).plan("x")


def test_memories_are_included_in_prompt() -> None:
    llm = FakeLLM(make_plan([subtask("t1", [])]))
    Supervisor(llm).plan("x", memories=["last time, searching news first worked"])
    assert "<past_tasks>" in llm.prompts[0]
