"""Bundles both memory layers for the workflow. Either layer can be absent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentorch.llm import LLMUsage
from agentorch.memory.extractor import MemoryExtractor
from agentorch.memory.long_term import LongTermMemory, StoredMemory
from agentorch.memory.working import WorkingMemory


@dataclass
class MemoryServices:
    long_term: LongTermMemory | None = None
    extractor: MemoryExtractor | None = None
    working: Callable[[str], WorkingMemory] | None = None  # task_id -> working memory

    def retrieve(self, user_id: str, task: str) -> list[StoredMemory]:
        return self.long_term.search(user_id, task) if self.long_term else []

    def remember(
        self, user_id: str, task_id: str, transcript: str
    ) -> tuple[list[str], LLMUsage | None]:
        if not (self.long_term and self.extractor):
            return [], None
        items, usage = self.extractor.extract(transcript)
        return self.long_term.add(user_id, task_id, items), usage

    def working_for(self, task_id: str) -> WorkingMemory | None:
        return self.working(task_id) if self.working else None
