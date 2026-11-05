"""Reviewer agent: checks a specialist's output before it goes back to the supervisor."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentorch.agents.schemas import ReviewVerdict, Subtask, SubtaskResult
from agentorch.config import settings
from agentorch.llm import LLMClient, LLMUsage
from agentorch.tools.registry import ToolCallRecord

SYSTEM_PROMPT = """You review work produced by specialist agents. Judge the deliverable \
strictly against the subtask and its expected output.

Check:
- Completeness: does it do everything the subtask asked, in the requested format?
- Grounding: are factual claims supported by the tool evidence (when provided) or cited \
sources? Check numbers and facts against the evidence and flag anything that isn't there.
- Correctness: are there errors, contradictions, or calculations that don't add up?
- Honesty: if something could not be done, does the deliverable say so?

Score quality from 0.0 to 1.0. Approve only work you would hand to the user as-is. When \
you reject, give specific, actionable feedback the specialist can fix in one more attempt."""


@dataclass
class ReviewOutcome:
    verdict: ReviewVerdict
    usage: LLMUsage
    threshold: float

    @property
    def passed(self) -> bool:
        return self.verdict.approved and self.verdict.quality_score >= self.threshold


class Reviewer:
    def __init__(self, llm: LLMClient, threshold: float | None = None) -> None:
        self.llm = llm
        self.threshold = settings.review_quality_threshold if threshold is None else threshold

    def review(
        self,
        subtask: Subtask,
        result: SubtaskResult,
        evidence: list[ToolCallRecord] | None = None,
    ) -> ReviewOutcome:
        if not result.succeeded:
            # Nothing to review; send the failure straight back.
            verdict = ReviewVerdict(
                subtask_id=subtask.id, approved=False, quality_score=0.0,
                feedback=f"The attempt failed: {result.error}",
            )
            return ReviewOutcome(verdict, LLMUsage("none", 0, 0), self.threshold)

        prompt = (
            f"<subtask id=\"{subtask.id}\" specialist=\"{subtask.specialist.value}\">\n"
            f"{subtask.description}\n</subtask>\n\n"
            f"<expected_output>\n{subtask.expected_output}\n</expected_output>\n\n"
            f"<deliverable>\n{result.output}\n</deliverable>"
        )
        if evidence:
            prompt += "\n\n" + _format_evidence(evidence)
        out = self.llm.structured("reviewer", SYSTEM_PROMPT, prompt, ReviewVerdict)
        verdict = out.output.model_copy(update={"subtask_id": subtask.id})
        return ReviewOutcome(verdict, out.usage, self.threshold)


EVIDENCE_CHARS_PER_CALL = 4_000


def _format_evidence(records: list[ToolCallRecord]) -> str:
    """Tool calls the specialist made, so claims can be checked against real output."""
    lines = ["<tool_evidence>"]
    for r in records:
        body = json.dumps(r.output if r.success else {"error": r.error}, ensure_ascii=False)
        if len(body) > EVIDENCE_CHARS_PER_CALL:
            body = body[:EVIDENCE_CHARS_PER_CALL] + " [truncated]"
        lines.append(
            f"<call tool=\"{r.tool}\" success=\"{r.success}\">\n"
            f"input: {json.dumps(r.input, ensure_ascii=False)}\noutput: {body}\n</call>"
        )
    lines.append("</tool_evidence>")
    return "\n".join(lines)
