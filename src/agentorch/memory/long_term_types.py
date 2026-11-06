"""Data types for long-term memory (kept separate so scoring can import them)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryKind(StrEnum):
    REQUEST = "request"  # what the user asked for, in one line
    APPROACH = "approach"  # an approach or plan shape that worked
    PITFALL = "pitfall"  # an approach that failed, and why
    TOOL_USAGE = "tool_usage"  # how a tool was best used for this kind of task
    FACT = "fact"  # a domain fact discovered, with its source and date
    PREFERENCE = "preference"  # something the user prefers


class MemoryItem(BaseModel):
    kind: MemoryKind
    text: str = Field(description="Self-contained: readable without the original task")


class StoredMemory(BaseModel):
    id: str
    user_id: str
    kind: MemoryKind
    text: str
    task_id: str
    created_at: float
    access_count: int
    last_accessed_at: float
    consolidated_from: int = 1  # how many original memories were merged into this one
    distance: float | None = None  # cosine distance to the query, when retrieved
    importance: float | None = None  # filled in when listed or retrieved

    def as_prompt_line(self) -> str:
        return f"[{self.kind.value}] {self.text}"
