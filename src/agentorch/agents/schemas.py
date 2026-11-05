"""Input/output schemas shared by every agent (the LangGraph node contracts)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Specialist(StrEnum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODE = "code"


class Complexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Subtask(BaseModel):
    id: str = Field(description="Short unique id, e.g. 't1'")
    description: str = Field(description="What this subtask must accomplish")
    specialist: Specialist
    required_inputs: list[str] = Field(
        description="Information this subtask needs (from the request or earlier subtasks)"
    )
    depends_on: list[str] = Field(
        description="Ids of subtasks whose output this one needs; empty if none"
    )
    expected_output: str = Field(description="Format and content of the deliverable")
    complexity: Complexity


class ExecutionPlan(BaseModel):
    goal: str = Field(description="One-sentence restatement of the user's request")
    reasoning: str = Field(description="Why the task is split this way")
    subtasks: list[Subtask]
    confidence: float = Field(
        description="0.0-1.0: how confident the supervisor is this plan will succeed"
    )


class SubtaskResult(BaseModel):
    subtask_id: str
    output: str
    succeeded: bool
    error: str | None = None


class ReviewVerdict(BaseModel):
    subtask_id: str
    blocking_issues: list[str] = Field(
        description="Problems that make the output unusable: wrong or unsupported facts, "
        "missing required parts, wrong format. Empty if none."
    )
    suggestions: list[str] = Field(
        description="Optional polish (style, brevity, extra detail). Never blocks approval."
    )
    quality_score: float = Field(description="0.0-1.0")
    approved: bool = Field(description="True exactly when blocking_issues is empty")

    @property
    def feedback(self) -> str:
        """What the specialist should fix on a retry: blocking issues only."""
        if self.blocking_issues:
            return "\n".join(f"- {i}" for i in self.blocking_issues)
        return "\n".join(f"- (optional) {s}" for s in self.suggestions)
