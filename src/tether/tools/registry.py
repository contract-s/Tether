"""Tool registry: every tool an agent can call, who may call it, and a log of every call.

Agents never call tool functions directly. They go through `ToolRegistry.invoke`, which
validates input, enforces per-specialist access and rate limits, times the call, and
records the outcome. That single choke point is what later lets us trace tool use and
pause sensitive calls for human approval.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ValidationError

from tether.agents.schemas import Specialist
from tether.observability.tracing import clip, mark_error, set_attributes, span

if TYPE_CHECKING:
    from tether.memory.working import WorkingMemory


@dataclass(frozen=True)
class RateLimit:
    calls: int
    per_seconds: float


@dataclass
class ToolContext:
    """Who is calling a tool and for which task. Passed to tools that need it."""

    caller: Specialist
    task_id: str | None = None
    subtask_id: str | None = None
    working_memory: WorkingMemory | None = None
    # action_key(tool, input) of sensitive calls a human already approved for this task.
    preapproved: frozenset[str] = frozenset()


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[Any], BaseModel]
    allowed_specialists: frozenset[Specialist]
    rate_limit: RateLimit | None = None
    # Returns True when a particular call needs human approval (deletion, payments,
    # sending data out). Used by the human-in-the-loop layer in Phase 3.
    is_sensitive: Callable[[Any], bool] = field(default=lambda _input: False)
    # When True the handler is called as handler(args, ctx) with a ToolContext.
    takes_context: bool = False

    def anthropic_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": strict_schema(self.input_model),
            "strict": True,
        }


class ToolCallRecord(BaseModel):
    id: str
    tool: str
    caller: Specialist
    input: dict[str, Any]
    output: dict[str, Any] | None
    error: str | None
    success: bool
    sensitive: bool
    latency_ms: float
    started_at: datetime


class ToolCallLog(Protocol):
    def record(self, call: ToolCallRecord) -> None: ...


class InMemoryToolCallLog:
    def __init__(self) -> None:
        self.calls: list[ToolCallRecord] = []

    def record(self, call: ToolCallRecord) -> None:
        self.calls.append(call)


class ToolError(Exception):
    """Base class for errors raised before a tool runs."""


class UnknownToolError(ToolError):
    pass


class ToolPermissionError(ToolError):
    pass


class RateLimitExceededError(ToolError):
    pass


class ApprovalRequiredError(ToolError):
    """A sensitive call was attempted without human approval. Nothing was run."""


class ToolRegistry:
    def __init__(
        self,
        log: ToolCallLog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.log = log or InMemoryToolCallLog()
        self._clock = clock
        self._recent_calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(f"no tool named {name!r}") from None

    def for_specialist(self, specialist: Specialist) -> list[Tool]:
        return [t for t in self._tools.values() if specialist in t.allowed_specialists]

    def anthropic_tools(self, specialist: Specialist) -> list[dict[str, Any]]:
        """Tool definitions to pass in `tools=` for this specialist's Claude calls."""
        return [t.anthropic_definition() for t in self.for_specialist(specialist)]

    def invoke(
        self,
        name: str,
        caller: Specialist,
        raw_input: dict[str, Any],
        approved: bool = False,
        context: ToolContext | None = None,
    ) -> ToolCallRecord:
        """Run a tool and log the call. Tool failures are captured in the record, not raised.

        Raises ToolError only for calls that must not run: unknown tool, caller not
        allowed, rate limit hit, or a sensitive call without `approved=True`.
        """
        with span(f"tool.{name}", "tool", **{
            "tether.tool": name, "tether.caller": caller.value,
            "tether.subtask_id": context.subtask_id if context else None,
            "tool.input": raw_input,
        }) as s:
            try:
                record = self._invoke(name, caller, raw_input, approved, context)
            except ToolError as e:
                set_attributes(s, {"tool.blocked": type(e).__name__})
                mark_error(s, str(e))
                raise
            set_attributes(s, {
                "tool.success": record.success, "tool.sensitive": record.sensitive,
                "tool.output": clip(json.dumps(record.output, default=str))
                if record.output is not None else None,
            })
            if not record.success:
                mark_error(s, record.error or "tool failed")
            return record

    def _invoke(
        self,
        name: str,
        caller: Specialist,
        raw_input: dict[str, Any],
        approved: bool,
        context: ToolContext | None,
    ) -> ToolCallRecord:
        tool = self.get(name)
        if caller not in tool.allowed_specialists:
            raise ToolPermissionError(f"{caller.value} may not use {name!r}")

        started_at = datetime.now(UTC)
        start = self._clock()
        output: dict[str, Any] | None = None
        error: str | None = None
        sensitive = False
        blocked: ToolError | None = None
        try:
            parsed = tool.input_model.model_validate(raw_input)
            sensitive = tool.is_sensitive(parsed)
            if sensitive and not approved:
                raise ApprovalRequiredError(f"{name!r} call needs human approval")
            self._check_rate_limit(tool)
            if tool.takes_context:
                result = tool.handler(parsed, context or ToolContext(caller=caller))
            else:
                result = tool.handler(parsed)
            output = tool.output_model.model_validate(result).model_dump(mode="json")
        except ToolError as e:
            blocked = e
            error = f"blocked: {e}"
        except ValidationError as e:
            error = f"invalid input: {e.errors(include_url=False)}"
        except Exception as e:  # noqa: BLE001 - tool errors go back to the agent
            error = f"{type(e).__name__}: {e}"

        record = ToolCallRecord(
            id=uuid.uuid4().hex,
            tool=name,
            caller=caller,
            input=raw_input,
            output=output,
            error=error,
            success=error is None,
            sensitive=sensitive,
            latency_ms=(self._clock() - start) * 1000,
            started_at=started_at,
        )
        self.log.record(record)
        if blocked is not None:
            # Logged above so blocked attempts show up in traces, then surfaced to the caller.
            raise blocked
        return record

    def _check_rate_limit(self, tool: Tool) -> None:
        if tool.rate_limit is None:
            return
        # Specialists run in parallel threads, so the window check must be atomic.
        with self._lock:
            self._check_rate_limit_locked(tool)

    def _check_rate_limit_locked(self, tool: Tool) -> None:
        assert tool.rate_limit is not None
        now = self._clock()
        window = self._recent_calls[tool.name]
        while window and now - window[0] >= tool.rate_limit.per_seconds:
            window.popleft()
        if len(window) >= tool.rate_limit.calls:
            raise RateLimitExceededError(
                f"{tool.name!r} allows {tool.rate_limit.calls} calls per "
                f"{tool.rate_limit.per_seconds:g}s"
            )
        window.append(now)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema accepted by strict tool use: every object closed, every field required."""
    schema = model.model_json_schema()

    def close(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                close(item)
            return
        if not isinstance(node, dict):
            return
        if isinstance(node.get("additionalProperties"), dict):
            raise ValueError(
                f"{model.__name__}: free-form maps (dict fields) aren't allowed in strict "
                "tool schemas; use a list of objects instead"
            )
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        node.pop("title", None)
        node.pop("default", None)
        for key, value in node.items():
            if key in ("properties", "$defs"):
                # Maps of name -> schema: the names themselves aren't schema keywords.
                for sub in value.values():
                    close(sub)
            else:
                close(value)

    close(schema)
    return schema
