"""Bundles both memory layers for the workflow. Either layer can be absent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tether.llm import LLMUsage
from tether.memory.extractor import MemoryExtractor
from tether.memory.long_term import LongTermMemory, StoredMemory
from tether.memory.working import WorkingMemory
from tether.observability.tracing import set_attributes, span


@dataclass
class MemoryServices:
    long_term: LongTermMemory | None = None
    extractor: MemoryExtractor | None = None
    working: Callable[[str], WorkingMemory] | None = None  # task_id -> working memory

    def retrieve(self, user_id: str, task: str) -> list[StoredMemory]:
        if not self.long_term:
            return []
        with span("memory.retrieve", "memory", **{"tether.user_id": user_id}) as s:
            found = self.long_term.search(user_id, task)
            set_attributes(s, {
                "memory.count": len(found),
                "memory.items": [{"kind": m.kind.value, "distance": round(m.distance or 0, 3),
                                  "importance": round(m.importance or 0, 3),
                                  "text": m.text[:300]} for m in found],
            })
            return found

    def remember(
        self, user_id: str, task_id: str, transcript: str
    ) -> tuple[list[str], LLMUsage | None]:
        if not (self.long_term and self.extractor):
            return [], None
        with span("memory.remember", "memory", **{"tether.user_id": user_id}) as s:
            items, usage = self.extractor.extract(transcript)
            ids = self.long_term.add(user_id, task_id, items)
            set_attributes(s, {"memory.count": len(ids), "memory.items": [
                {"kind": i.kind.value, "text": i.text[:300]} for i in items]})
            return ids, usage

    def working_for(self, task_id: str) -> WorkingMemory | None:
        return self.working(task_id) if self.working else None
