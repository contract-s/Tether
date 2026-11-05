import pytest

from agentorch.agents.planning import InvalidPlanError, validate_plan
from agentorch.agents.schemas import Complexity, ExecutionPlan, Specialist, Subtask


def st(id: str, *deps: str, specialist: Specialist = Specialist.RESEARCH) -> Subtask:
    return Subtask(
        id=id,
        description=f"do {id}",
        specialist=specialist,
        required_inputs=[],
        depends_on=list(deps),
        expected_output="text",
        complexity=Complexity.LOW,
    )


def plan(*subtasks: Subtask, confidence: float = 0.9) -> ExecutionPlan:
    return ExecutionPlan(goal="g", reasoning="r", subtasks=list(subtasks), confidence=confidence)


def ids(batches: list[list[Subtask]]) -> list[list[str]]:
    return [[s.id for s in b] for b in batches]


def test_independent_subtasks_share_a_batch() -> None:
    assert ids(validate_plan(plan(st("a"), st("b"), st("c")))) == [["a", "b", "c"]]


def test_dependencies_produce_ordered_batches() -> None:
    p = plan(st("a"), st("b"), st("c", "a", "b"), st("d", "c"))
    assert ids(validate_plan(p)) == [["a", "b"], ["c"], ["d"]]


@pytest.mark.parametrize(
    ("p", "message"),
    [
        (plan(), "no subtasks"),
        (plan(st("a"), st("a")), "duplicate"),
        (plan(st("a", "missing")), "unknown"),
        (plan(st("a", "a")), "itself"),
        (plan(st("a", "b"), st("b", "a")), "cycle"),
        (plan(st("a"), confidence=1.5), "confidence"),
    ],
)
def test_invalid_plans_are_rejected(p: ExecutionPlan, message: str) -> None:
    with pytest.raises(InvalidPlanError, match=message):
        validate_plan(p)
