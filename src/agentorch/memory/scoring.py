"""How much a memory matters: importance, decay, and retrieval ranking.

importance = (kind weight + how often it's been used) / 2, decayed by time since last use.

- Kind weight: preferences and pitfalls are the most valuable to plan with; one-line
  request summaries the least.
- Usage: each retrieval raises importance, with diminishing returns.
- Decay: halves every `half_life` days without use. Facts go stale fastest.

Retrieval ranks by a blend of similarity and importance, so a slightly less similar but
proven memory can beat a near-match nobody has ever needed. Memories whose importance
decays below the expiry threshold get deleted by `LongTermMemory.expire`.
"""

from __future__ import annotations

import math

from agentorch.memory.long_term_types import MemoryKind, StoredMemory

DAY = 86_400.0

KIND_WEIGHT: dict[MemoryKind, float] = {
    MemoryKind.PREFERENCE: 1.0,
    MemoryKind.PITFALL: 0.9,
    MemoryKind.APPROACH: 0.8,
    MemoryKind.TOOL_USAGE: 0.7,
    MemoryKind.FACT: 0.5,
    MemoryKind.REQUEST: 0.4,
}

HALF_LIFE_DAYS: dict[MemoryKind, float] = {
    MemoryKind.FACT: 30,
    MemoryKind.REQUEST: 60,
    MemoryKind.TOOL_USAGE: 120,
    MemoryKind.APPROACH: 120,
    MemoryKind.PITFALL: 120,
    MemoryKind.PREFERENCE: 180,
}

USAGE_SCALE = 5.0  # ~63% of the usage bonus after 5 retrievals
SIMILARITY_WEIGHT = 0.75
EXPIRE_BELOW = 0.1


def usage_score(access_count: int) -> float:
    return 1.0 - math.exp(-access_count / USAGE_SCALE)


def recency_factor(memory: StoredMemory, now: float) -> float:
    idle_days = max(0.0, now - memory.last_accessed_at) / DAY
    return 0.5 ** (idle_days / HALF_LIFE_DAYS[memory.kind])


def importance(memory: StoredMemory, now: float) -> float:
    """0.0-1.0."""
    base = (KIND_WEIGHT[memory.kind] + usage_score(memory.access_count)) / 2
    return base * recency_factor(memory, now)


def rank_score(distance: float, importance_value: float) -> float:
    """Higher is better. Cosine distance is 0 (identical) to 2 (opposite)."""
    similarity = 1.0 - min(distance, 1.0)
    return SIMILARITY_WEIGHT * similarity + (1 - SIMILARITY_WEIGHT) * importance_value
