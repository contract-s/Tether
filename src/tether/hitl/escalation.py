"""When agents hand control to a human, and how much control they hand over.

Triggers and the approval level each one maps to:

| Trigger             | When                                        | Level                 |
|---------------------|---------------------------------------------|-----------------------|
| low_plan_confidence | plan confidence < threshold                 | approve_plan          |
| user_requested      | task submitted with require_review=True     | approve_plan, then    |
|                     |                                             | approve_action on the |
|                     |                                             | final answer          |
| sensitive_action    | a tool call marked sensitive (overwrite,    | approve_action        |
|                     | delete, external side effects)              |                       |
| low_quality         | reviewer rejected an attempt (it retries)   | notify                |
| repeated_failure    | a subtask failed review max_attempts times  | take_over             |

Levels:
- notify: work continues; the human is informed and nothing waits on them.
- approve_action: one specific step waits for approve / modify / reject.
- approve_plan: nothing starts until the whole plan is approved, modified, or rejected.
- take_over: agents stand down on that subtask; the human retries it with guidance,
  accepts it as is, writes the output themselves, or drops it.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from tether.agents.schemas import ExecutionPlan


class Trigger(StrEnum):
    LOW_PLAN_CONFIDENCE = "low_plan_confidence"
    USER_REQUESTED = "user_requested"
    SENSITIVE_ACTION = "sensitive_action"
    LOW_QUALITY = "low_quality"
    REPEATED_FAILURE = "repeated_failure"


class ApprovalLevel(StrEnum):
    NOTIFY = "notify"
    APPROVE_ACTION = "approve_action"
    APPROVE_PLAN = "approve_plan"
    TAKE_OVER = "take_over"


OPTIONS: dict[ApprovalLevel, list[str]] = {
    ApprovalLevel.NOTIFY: [],
    ApprovalLevel.APPROVE_ACTION: ["approve", "modify", "reject"],
    ApprovalLevel.APPROVE_PLAN: ["approve", "modify", "reject"],
    ApprovalLevel.TAKE_OVER: ["retry", "accept", "take_over", "abort"],
}


class ProposedAction(BaseModel):
    """What the agent wants to do next, and why."""

    summary: str
    reasoning: str | None = None
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    output: str | None = None  # e.g. the draft final answer awaiting approval


class EscalationRequest(BaseModel):
    """Everything a reviewer needs to decide, packaged when the run pauses."""

    id: str
    trigger: Trigger
    level: ApprovalLevel
    reason: str
    task_id: str
    user_id: str
    task: str
    subtask_id: str | None = None
    plan: dict[str, Any] | None = None
    # Subtask id -> {"status", "output"} for everything done so far.
    progress: dict[str, dict[str, Any]] = Field(default_factory=dict)
    proposed: ProposedAction | None = None
    memories: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """A human's answer to an escalation."""

    action: str
    note: str | None = None
    output: str | None = None  # take_over, or modify on a final answer
    plan: ExecutionPlan | None = None  # modify on approve_plan
    tool_input: dict[str, Any] | None = None  # modify on approve_action for a tool call
    reviewer: str | None = None


def action_key(tool: str, tool_input: dict[str, Any]) -> str:
    """Stable identity for 'this tool with exactly this input' (used for pre-approval)."""
    return f"{tool}:{json.dumps(tool_input, sort_keys=True, separators=(',', ':'))}"
