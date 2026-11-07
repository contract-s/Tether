"""Long-term semantic memory in ChromaDB.

After a task finishes, the extractor distills what's worth keeping (what was asked, what
approach worked, useful facts, user preferences, pitfalls). Each item is embedded and
stored per user. Before planning, the supervisor retrieves the most relevant memories,
ranked by similarity and importance (see `scoring`).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Protocol

from agentorch.memory import scoring
from agentorch.memory.long_term_types import MemoryItem, MemoryKind, StoredMemory

__all__ = ["LongTermMemory", "MemoryItem", "MemoryKind", "StoredMemory", "Embedder",
           "ChromaDefaultEmbedder"]

COLLECTION = "agentorch-memories"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChromaDefaultEmbedder:
    """Chroma's built-in local model (all-MiniLM-L6-v2). Downloads ~80 MB on first use."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


# How far a preference may be from the task and still be used (see search()).
PREFERENCE_MAX_DISTANCE = 0.85


class LongTermMemory:
    def __init__(self, client: Any, embedder: Embedder, collection: str = COLLECTION) -> None:
        self._embedder = embedder
        # We pass embeddings ourselves, so Chroma doesn't need its own embedding function.
        self._col = client.get_or_create_collection(
            collection, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )

    # --- writes ----------------------------------------------------------------

    def add(
        self,
        user_id: str,
        task_id: str,
        items: list[MemoryItem],
        overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """Store memories. `overrides` sets metadata such as carried-over access counts."""
        items = [i for i in items if i.text.strip()]
        if not items:
            return []
        now = time.time()
        ids = [uuid.uuid4().hex for _ in items]
        self._col.add(
            ids=ids,
            embeddings=self.embed([i.text for i in items]),
            documents=[i.text for i in items],
            metadatas=[{
                "user_id": user_id, "kind": i.kind.value, "task_id": task_id,
                "created_at": now, "access_count": 0, "last_accessed_at": now,
                "consolidated_from": 1, **(overrides or {}),
            } for i in items],
        )
        return ids

    def delete(self, ids: list[str]) -> int:
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def delete_user(self, user_id: str) -> int:
        """Forget everything about a user (data deletion requests)."""
        return self.delete(self._col.get(where={"user_id": user_id}, include=[])["ids"])

    def expire(
        self, user_id: str | None = None, now: float | None = None,
        threshold: float = scoring.EXPIRE_BELOW,
    ) -> list[StoredMemory]:
        """Delete memories whose decayed importance fell below the threshold."""
        now = time.time() if now is None else now
        stale = [m for m in self.list(user_id, now=now) if (m.importance or 0) < threshold]
        self.delete([m.id for m in stale])
        return stale

    # --- reads -----------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def search(
        self,
        user_id: str,
        query: str,
        k: int = 8,
        max_distance: float = 0.7,
        now: float | None = None,
    ) -> list[StoredMemory]:
        """The k most relevant memories for this user. Retrieval counts toward importance.

        Candidates must be within `max_distance` (cosine). The 0.7 cutoff is calibrated
        for all-MiniLM-L6-v2: related memories land around 0.45-0.66, near-duplicates of
        the task under 0.2. Preferences ("start every summary with a TL;DR") apply across
        topics, so they sit further from any one task (around 0.65-0.8) and get a looser
        cutoff. Candidates are then ranked by similarity blended with importance.
        """
        if self._col.count() == 0:
            return []
        now = time.time() if now is None else now
        res = self._col.query(
            query_embeddings=self.embed([query]),
            n_results=k * 3,  # over-fetch so importance can reorder
            where={"user_id": user_id},
        )
        candidates = [
            _to_memory(id_, doc, meta, dist, now)
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                strict=True,
            )
            if dist <= (max(max_distance, PREFERENCE_MAX_DISTANCE)
                        if meta.get("kind") == MemoryKind.PREFERENCE.value else max_distance)
        ]
        candidates.sort(key=lambda m: scoring.rank_score(m.distance, m.importance), reverse=True)
        found = candidates[:k]
        self._touch(found, now)
        return found

    def list(self, user_id: str | None = None, now: float | None = None) -> list[StoredMemory]:
        """All memories (for one user, or everyone), most important first."""
        now = time.time() if now is None else now
        res = self._col.get(where={"user_id": user_id} if user_id else None)
        memories = [_to_memory(i, d, m, None, now)
                    for i, d, m in zip(res["ids"], res["documents"], res["metadatas"],
                                       strict=True)]
        return sorted(memories, key=lambda m: m.importance or 0, reverse=True)

    def with_embeddings(self, user_id: str) -> list[tuple[StoredMemory, list[float]]]:
        now = time.time()
        res = self._col.get(where={"user_id": user_id},
                            include=["documents", "metadatas", "embeddings"])
        return [(_to_memory(i, d, m, None, now), list(map(float, e)))
                for i, d, m, e in zip(res["ids"], res["documents"], res["metadatas"],
                                      res["embeddings"], strict=True)]

    def users(self) -> dict[str, int]:
        """user_id -> number of memories."""
        counts: dict[str, int] = {}
        for meta in self._col.get(include=["metadatas"])["metadatas"]:
            counts[meta["user_id"]] = counts.get(meta["user_id"], 0) + 1
        return counts

    # --- internals -------------------------------------------------------------

    def _touch(self, memories: list[StoredMemory], now: float) -> None:
        if not memories:
            return
        for m in memories:
            m.access_count += 1
            m.last_accessed_at = now
        self._col.update(
            ids=[m.id for m in memories],
            metadatas=[{"access_count": m.access_count, "last_accessed_at": now}
                       for m in memories],
        )


def _to_memory(
    id_: str, doc: str, meta: dict[str, Any], dist: float | None, now: float
) -> StoredMemory:
    memory = StoredMemory(
        id=id_, user_id=meta["user_id"], kind=MemoryKind(meta["kind"]), text=doc,
        task_id=meta["task_id"], created_at=meta["created_at"],
        access_count=int(meta["access_count"]), last_accessed_at=meta["last_accessed_at"],
        consolidated_from=int(meta.get("consolidated_from", 1)), distance=dist,
    )
    memory.importance = scoring.importance(memory, now)
    return memory
