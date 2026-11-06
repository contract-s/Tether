"""Wraps any LLMClient so every model call becomes a span with tokens, cost, prompt and response."""

from __future__ import annotations

import json
from typing import Any

from agentorch.llm import LLMClient, ModelTurn, Role, StructuredResult, T
from agentorch.observability.tracing import clip, record_llm_usage, span


def _render(content: Any) -> str:
    """A readable version of a message's content (text, or tool results/calls)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "tool_result":
                parts.append(f"[tool_result {block.get('tool_use_id')}"
                             f"{' ERROR' if block.get('is_error') else ''}] {block.get('content')}")
            else:
                parts.append(json.dumps(block, default=str))
        else:
            parts.append(str(getattr(block, "text", block)))
    return "\n".join(parts)


class TracingLLM:
    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner

    def structured(
        self, role: Role, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        with span(f"llm.{role}", "llm", **{
            "agentorch.role": role, "llm.call": "structured", "llm.schema": schema.__name__,
            "llm.system": clip(system), "llm.prompt": clip(prompt),
        }) as s:
            result = self.inner.structured(role, system, prompt, schema)
            u = result.usage
            record_llm_usage(s, u.model, u.input_tokens, u.output_tokens,
                             u.cache_read_input_tokens)
            s.set_attribute("llm.response", clip(result.output.model_dump_json(indent=1)) or "")
            return result

    def chat(
        self, role: Role, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        last = messages[-1] if messages else {"content": ""}
        with span(f"llm.{role}", "llm", **{
            "agentorch.role": role, "llm.call": "chat", "llm.turn": len(messages),
            "llm.tools_offered": len(tools), "llm.system": clip(system),
            # The newest message is what this call responds to; earlier turns are
            # on the previous llm spans of the same agent.
            "llm.prompt": clip(_render(last.get("content", ""))),
        }) as s:
            turn = self.inner.chat(role, system, messages, tools)
            u = turn.usage
            record_llm_usage(s, u.model, u.input_tokens, u.output_tokens,
                             u.cache_read_input_tokens)
            response = turn.text
            if turn.tool_calls:
                response += "\n" + "\n".join(
                    f"[tool_use {c.name}] {json.dumps(c.input)}" for c in turn.tool_calls)
            s.set_attribute("llm.response", clip(response.strip()) or "")
            s.set_attribute("llm.stop_reason", turn.stop_reason)
            return turn
