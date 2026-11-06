"""Merge near-duplicate memories into one higher-level memory.

Every task adds a handful of memories, so similar tasks pile up overlapping ones
("use the GitHub releases API for qdrant", "... for weaviate"). Consolidation groups
memories of the same kind that are very close in embedding space, asks Claude to write
one general memory per group, and replaces the group with it. The merged memory keeps
the group's combined usage, so importance isn't lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from agentorch.llm import LLMClient, LLMUsage
from agentorch.memory.long_term import LongTermMemory
from agentorch.memory.long_term_types import MemoryItem, MemoryKind, StoredMemory

# Same-kind memories closer than this are treated as overlapping. Measured on real
# memories with all-MiniLM-L6-v2: the same lesson worded twice lands at 0.24-0.32, while
# genuinely different memories start around 0.6 (e.g. Qdrant's vs Weaviate's version,
# 0.63, which must not merge). Preferences get reworded more freely by the extractor
# (two phrasings of "keep it to two sentences" measured 0.54), so they get a wider cutoff.
MERGE_DISTANCE = 0.35
MERGE_DISTANCE_BY_KIND: dict[MemoryKind, float] = {MemoryKind.PREFERENCE: 0.55}

SYSTEM_PROMPT = """You maintain the long-term memory of a team of AI agents. You get \
several overlapping memories of the same kind. Write one memory that replaces all of them.

- Generalize: if they're the same lesson applied to different cases, state the general \
lesson and mention the cases briefly.
- Keep every distinct, still-useful detail (sources, dates, specific settings). Drop \
repetition.
- If they contradict each other, keep the most recent information and say what changed.
- It must make sense on its own, in 1-3 sentences."""


class MergedMemory(BaseModel):
    text: str = Field(description="The single memory that replaces the group")


@dataclass
class ConsolidationReport:
    groups_merged: int = 0
    memories_before: int = 0
    memories_after: int = 0
    merged: list[tuple[list[str], str]] = field(default_factory=list)  # (old texts, new text)
    usage: list[LLMUsage] = field(default_factory=list)


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 1.0


def group_similar(
    memories: list[tuple[StoredMemory, list[float]]], max_distance: float | None = None
) -> list[list[StoredMemory]]:
    """Groups of 2+ same-kind memories close to the group's first member.

    `max_distance` overrides the per-kind cutoffs (MERGE_DISTANCE_BY_KIND, else
    MERGE_DISTANCE).

    Greedy: most important memory first, so it anchors its group. Fine for the hundreds of
    memories a user accumulates; a vector index would be needed for millions.
    """
    ordered = sorted(memories, key=lambda pair: pair[0].importance or 0, reverse=True)
    used: set[str] = set()
    groups = []
    for anchor, anchor_vec in ordered:
        if anchor.id in used:
            continue
        group = [anchor]
        cutoff = max_distance if max_distance is not None else MERGE_DISTANCE_BY_KIND.get(
            anchor.kind, MERGE_DISTANCE)
        for other, vec in ordered:
            if (other.id != anchor.id and other.id not in used and other.kind == anchor.kind
                    and cosine_distance(anchor_vec, vec) <= cutoff):
                group.append(other)
        if len(group) > 1:
            used.update(m.id for m in group)
            groups.append(group)
    return groups


class MemoryConsolidator:
    def __init__(self, llm: LLMClient, store: LongTermMemory) -> None:
        self.llm = llm
        self.store = store

    def consolidate(
        self, user_id: str, max_distance: float | None = None, dry_run: bool = False
    ) -> ConsolidationReport:
        memories = self.store.with_embeddings(user_id)
        report = ConsolidationReport(memories_before=len(memories))
        for group in group_similar(memories, max_distance):
            prompt = f"Kind: {group[0].kind.value}\n\n" + "\n".join(
                f"<memory used=\"{m.access_count}\">{m.text}</memory>" for m in group)
            out = self.llm.structured("memory", SYSTEM_PROMPT, prompt, MergedMemory)
            report.usage.append(out.usage)
            report.merged.append(([m.text for m in group], out.output.text))
            report.groups_merged += 1
            if dry_run:
                continue
            # Write the merged memory before deleting the originals, so a failure
            # in between can't lose anything.
            self.store.add(
                user_id, task_id=group[0].task_id,
                items=[MemoryItem(kind=group[0].kind, text=out.output.text)],
                overrides={
                    "access_count": sum(m.access_count for m in group),
                    "last_accessed_at": max(m.last_accessed_at for m in group),
                    "created_at": min(m.created_at for m in group),
                    "consolidated_from": sum(m.consolidated_from for m in group),
                },
            )
            self.store.delete([m.id for m in group])
        removed = sum(len(old) - 1 for old, _ in report.merged)
        report.memories_after = report.memories_before - (0 if dry_run else removed)
        return report
