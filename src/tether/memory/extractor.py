"""Distill a finished task into memories worth keeping."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tether.llm import LLMClient, LLMUsage
from tether.memory.long_term import MemoryItem

SYSTEM_PROMPT = """You write the long-term memory for a team of AI agents. Given a \
finished task, record only what will help plan similar tasks for this user in the future.

Kinds:
- request: what the user asked for, in one line.
- approach: a plan shape or method that worked well (e.g. which specialists, in what order).
- pitfall: something that failed or needed rework, and how it was resolved.
- tool_usage: how a tool was best used for this kind of task.
- fact: a durable domain fact that was verified, with its source and date. Skip facts \
that will go stale quickly unless the date makes that clear.
- preference: something the user prefers (format, length, depth), only if the request \
or a human decision shows it.

Each memory must make sense on its own, without the task. Prefer 3-8 high-value \
memories over many trivial ones. Never store secrets, credentials or personal data."""


class ExtractedMemories(BaseModel):
    memories: list[MemoryItem] = Field(description="3-8 memories worth keeping")


class MemoryExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, transcript: str) -> tuple[list[MemoryItem], LLMUsage]:
        out = self.llm.structured("memory", SYSTEM_PROMPT, transcript, ExtractedMemories)
        return out.output.memories, out.usage
