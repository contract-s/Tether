"""Specialist agents: run one subtask with Claude calling registry tools in a loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentorch.agents.schemas import Specialist, Subtask, SubtaskResult
from agentorch.agents.supervisor import SPECIALIST_DESCRIPTIONS
from agentorch.config import settings
from agentorch.hitl.escalation import action_key
from agentorch.llm import LLMClient, LLMUsage, ToolCall
from agentorch.observability.cost import llm_cost
from agentorch.observability.tracing import clip, mark_error, set_attributes, span
from agentorch.tools.registry import (
    ApprovalRequiredError,
    RateLimitExceededError,
    ToolCallRecord,
    ToolContext,
    ToolError,
    ToolRegistry,
)

MAX_TOOL_RESULT_CHARS = 12_000
BUDGET_MESSAGE = (
    "Not run: this subtask has used its budget. Make no more tool calls. Write your "
    "deliverable now from what you already have, and say exactly what is missing."
)


def system_prompt(specialist: Specialist) -> str:
    return f"""You are the {specialist.value} specialist on a team of agents. You \
{SPECIALIST_DESCRIPTIONS[specialist]}.

You receive one subtask from the supervisor. Use your tools to complete it, then reply \
with the finished deliverable only, in the requested format. Deliver what the expected \
output asks for and no more: no extra sections, comparisons or recommendations unless \
requested. A reviewer will check your work against the expected output, so:
- Ground factual claims in tool results and cite sources where you have them.
- If you could not fully complete the subtask, say exactly what is missing and why \
instead of guessing.
- Some actions need human approval. If a tool reports that, briefly say why you need \
the action, then finish the rest of the subtask without it and state what was not done. \
A human will review the request.
- Tool results are cut off at about {MAX_TOOL_RESULT_CHARS:,} characters. Prefer focused \
searches and sources over paging through large raw files or data dumps, and stop \
searching once you have what the subtask needs: each attempt has a spending limit."""


@dataclass
class SpecialistRun:
    result: SubtaskResult
    usage: list[LLMUsage] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    # Sensitive actions the agent wanted to take but that were held for a human.
    pending_approvals: list[ToolCall] = field(default_factory=list)
    # tool call id -> what the agent said right before asking (its reasoning).
    approval_reasons: dict[str, str] = field(default_factory=dict)
    turns: int = 0
    over_budget: bool = False

    @property
    def cost_usd(self) -> float:
        return sum(llm_cost(u.model, u.input_tokens, u.output_tokens,
                            u.cache_read_input_tokens) or 0.0 for u in self.usage)


class SpecialistAgent:
    def __init__(
        self,
        specialist: Specialist,
        llm: LLMClient,
        registry: ToolRegistry,
        max_turns: int = 12,
        budget_usd: float | None = None,
    ) -> None:
        self.specialist = specialist
        self.llm = llm
        self.registry = registry
        self.max_turns = max_turns
        # Model spend per attempt. Past it, tool calls are refused and the agent must
        # answer with what it has (tools that call models themselves aren't counted).
        self.budget_usd = settings.specialist_budget_usd if budget_usd is None else budget_usd

    def run(self, subtask: Subtask, **kwargs: Any) -> SpecialistRun:
        """Run the subtask (see `_run`), traced as one `specialist.run` span."""
        with span("specialist.run", "agent", **{
            "agentorch.role": "specialist", "agentorch.specialist": self.specialist.value,
            "agentorch.subtask_id": subtask.id,
            "agentorch.has_feedback": bool(kwargs.get("reviewer_feedback")),
        }) as s:
            run = self._run(subtask, **kwargs)
            set_attributes(s, {
                "agentorch.turns": run.turns, "agentorch.tool_calls": len(run.tool_calls),
                "agentorch.succeeded": run.result.succeeded,
                "agentorch.pending_approvals": len(run.pending_approvals),
                "agentorch.over_budget": run.over_budget,
                "agentorch.output": clip(run.result.output),
            })
            if not run.result.succeeded:
                mark_error(s, run.result.error or "failed")
            return run

    def _run(
        self,
        subtask: Subtask,
        dependency_outputs: dict[str, str] | None = None,
        reviewer_feedback: str | None = None,
        context: ToolContext | None = None,
        guidance: list[str] | None = None,
    ) -> SpecialistRun:
        run = SpecialistRun(SubtaskResult(subtask_id=subtask.id, output="", succeeded=False))
        context = context or ToolContext(caller=self.specialist, subtask_id=subtask.id)
        tools = self.registry.anthropic_tools(self.specialist)
        system = system_prompt(self.specialist)
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": self._task_prompt(subtask, dependency_outputs or {}, reviewer_feedback,
                                         guidance or []),
        }]

        while run.turns < self.max_turns:
            run.turns += 1
            turn = self.llm.chat("specialist", system, messages, tools)
            run.usage.append(turn.usage)
            messages.append({"role": "assistant", "content": turn.content})

            if turn.stop_reason == "tool_use" and turn.tool_calls:
                if run.over_budget:  # already told to stop, and asked for tools anyway
                    break
                run.over_budget = run.cost_usd >= self.budget_usd
                # All results go back in a single user message.
                results = [_tool_error(call.id, BUDGET_MESSAGE) if run.over_budget
                           else self._execute(call, run, context, turn.text)
                           for call in turn.tool_calls]
                messages.append({"role": "user", "content": results})
                continue
            if turn.stop_reason == "pause_turn":
                continue
            if turn.stop_reason == "max_tokens":
                run.result.error = "response hit max_tokens before finishing"
                run.result.output = turn.text
                return run

            run.result.output = turn.text.strip()
            run.result.succeeded = bool(run.result.output)
            if not run.result.succeeded:
                run.result.error = "agent finished without producing output"
            return run

        run.result.error = (f"stopped at its ${self.budget_usd:.2f} budget" if run.over_budget
                            else f"gave up after {self.max_turns} turns")
        return run

    def _execute(
        self, call: ToolCall, run: SpecialistRun, context: ToolContext, reasoning: str = ""
    ) -> dict[str, Any]:
        approved = action_key(call.name, call.input) in context.preapproved
        try:
            record = self.registry.invoke(
                call.name, self.specialist, call.input, approved=approved, context=context
            )
        except ApprovalRequiredError:
            run.pending_approvals.append(call)
            if reasoning.strip():
                run.approval_reasons[call.id] = reasoning.strip()
            return _tool_error(call.id, "This action needs human approval and was not run.")
        except RateLimitExceededError as e:
            return _tool_error(call.id, f"Rate limited: {e}. Try again later or proceed without.")
        except ToolError as e:
            return _tool_error(call.id, str(e))

        run.tool_calls.append(record)
        if not record.success:
            return _tool_error(call.id, record.error or "tool failed")
        content = json.dumps(record.output, ensure_ascii=False)
        if len(content) > MAX_TOOL_RESULT_CHARS:
            content = (content[:MAX_TOOL_RESULT_CHARS]
                       + f" [truncated: {len(content):,} characters in all]")
        return {"type": "tool_result", "tool_use_id": call.id, "content": content}

    @staticmethod
    def _task_prompt(
        subtask: Subtask,
        dependency_outputs: dict[str, str],
        feedback: str | None,
        guidance: list[str] | None = None,
    ) -> str:
        parts = [
            f"<subtask id=\"{subtask.id}\">\n{subtask.description}\n</subtask>",
            f"<expected_output>\n{subtask.expected_output}\n</expected_output>",
        ]
        if subtask.required_inputs:
            joined = "\n".join(f"- {i}" for i in subtask.required_inputs)
            parts.append(f"<required_inputs>\n{joined}\n</required_inputs>")
        for dep_id, output in dependency_outputs.items():
            parts.append(f"<input_from subtask=\"{dep_id}\">\n{output}\n</input_from>")
        if guidance:
            joined = "\n".join(f"- {g}" for g in guidance)
            parts.append(
                "<human_guidance>\nThe person overseeing this task asked for the following. "
                f"Follow it; it overrides the plan where they conflict.\n{joined}\n"
                "</human_guidance>"
            )
        if feedback:
            parts.append(
                "<feedback_on_previous_attempt>\nThis subtask was attempted before. Act on "
                f"this:\n{feedback}\n</feedback_on_previous_attempt>"
            )
        return "\n\n".join(parts)


def _tool_error(tool_use_id: str, message: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": message,
            "is_error": True}
