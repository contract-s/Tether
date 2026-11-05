"""Reviewer agent: checks a specialist's output before it goes back to the supervisor."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentorch.agents.schemas import ReviewVerdict, Subtask, SubtaskResult
from agentorch.agents.specialist import MAX_TOOL_RESULT_CHARS
from agentorch.config import settings
from agentorch.llm import LLMClient, LLMUsage
from agentorch.tools.registry import ToolCallRecord

SYSTEM_PROMPT = """You review work produced by specialist agents. Most deliverables \
are intermediate: they feed other agents and the supervisor, who writes the final answer. \
Your job is to stop bad work, not to perfect good work.

Blocking issues (each one means the work must be redone):
- A factual claim or number that contradicts the evidence, or that the evidence doesn't \
support and isn't labeled as an estimate or assumption. Evidence means tool calls and \
accepted inputs from earlier subtasks (already reviewed, so treat them as sourced).
- A calculation that doesn't add up.
- A required part of the subtask or expected output is missing, or the format is wrong.
- Something that couldn't be done is presented as done.

Everything else is a suggestion: brevity, style, structure, extra detail, scope beyond \
what was asked, or a better way to present the same correct content. Suggestions never \
block approval. Judge length from the text itself; don't guess.

Set approved=true exactly when there are no blocking issues. Score quality on this scale: \
0.9+ excellent, 0.7-0.9 solid with minor polish possible, 0.5-0.7 has a blocking issue, \
below 0.5 largely wrong or missing. Write each blocking issue as a specific fix the \
specialist can make in one more attempt."""


@dataclass
class ReviewOutcome:
    verdict: ReviewVerdict
    usage: LLMUsage
    threshold: float

    @property
    def passed(self) -> bool:
        # blocking_issues is the source of truth; `approved` is the model's own summary.
        return not self.verdict.blocking_issues and self.verdict.quality_score >= self.threshold


class Reviewer:
    def __init__(self, llm: LLMClient, threshold: float | None = None) -> None:
        self.llm = llm
        self.threshold = settings.review_quality_threshold if threshold is None else threshold

    def review(
        self,
        subtask: Subtask,
        result: SubtaskResult,
        evidence: list[ToolCallRecord] | None = None,
        dependency_outputs: dict[str, str] | None = None,
    ) -> ReviewOutcome:
        if not result.succeeded:
            # Nothing to review; send the failure straight back.
            verdict = ReviewVerdict(
                subtask_id=subtask.id, approved=False, quality_score=0.0,
                blocking_issues=[f"The attempt failed: {result.error}. Try a different approach."],
                suggestions=[],
            )
            return ReviewOutcome(verdict, LLMUsage("none", 0, 0), self.threshold)

        prompt = (
            f"<subtask id=\"{subtask.id}\" specialist=\"{subtask.specialist.value}\">\n"
            f"{subtask.description}\n</subtask>\n\n"
            f"<expected_output>\n{subtask.expected_output}\n</expected_output>\n\n"
            f"<deliverable>\n{result.output}\n</deliverable>"
        )
        for dep_id, output in (dependency_outputs or {}).items():
            # Already-reviewed outputs this subtask was built on count as evidence.
            prompt += (f"\n\n<accepted_input from_subtask=\"{dep_id}\">\n{output}\n"
                       "</accepted_input>")
        if evidence:
            prompt += "\n\n" + _format_evidence(evidence)
        out = self.llm.structured("reviewer", SYSTEM_PROMPT, prompt, ReviewVerdict)
        verdict = out.output.model_copy(update={"subtask_id": subtask.id})
        return ReviewOutcome(verdict, out.usage, self.threshold)


# Same per-call limit the specialist saw, so the reviewer judges against identical evidence.
EVIDENCE_CHARS_PER_CALL = MAX_TOOL_RESULT_CHARS
# Cap on the whole evidence block to bound review cost.
EVIDENCE_CHARS_TOTAL = 200_000

EVIDENCE_CUT_NOTE = (
    "[Evidence beyond this point was cut to limit review size. The specialist did see it, "
    "so claims that may come from the cut part are suggestions to verify, not blocking.]"
)


def _format_evidence(records: list[ToolCallRecord]) -> str:
    """Tool calls the specialist made, so claims can be checked against real output."""
    lines = ["<tool_evidence>"]
    used = 0
    for r in records:
        body = json.dumps(r.output if r.success else {"error": r.error}, ensure_ascii=False)
        if len(body) > EVIDENCE_CHARS_PER_CALL:
            # The specialist got this same cut-off, so anything past it was unseen by both.
            body = body[:EVIDENCE_CHARS_PER_CALL] + " [truncated: the specialist saw no more]"
        if used + len(body) > EVIDENCE_CHARS_TOTAL:
            lines.append(EVIDENCE_CUT_NOTE)
            break
        used += len(body)
        lines.append(
            f"<call tool=\"{r.tool}\" success=\"{r.success}\">\n"
            f"input: {json.dumps(r.input, ensure_ascii=False)}\noutput: {body}\n</call>"
        )
    lines.append("</tool_evidence>")
    return "\n".join(lines)
