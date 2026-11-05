"""Long-term semantic memory in ChromaDB.

After a task finishes, the extractor distills what's worth keeping (what was asked, what
approach worked, useful facts, user preferences, pitfalls). Each item is embedded and
stored per user. Before planning, the supervisor retrieves the most similar memories.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

COLLECTION = "agentorch-memories"


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
    distance: float | None = None  # cosine distance to the query, when retrieved

    def as_prompt_line(self) -> str:
        return f"[{self.kind.value}] {self.text}"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChromaDefaultEmbedder:
    """Chroma's built-in local model (all-MiniLM-L6-v2). Downloads ~80 MB on first use."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


class LongTermMemory:
    def __init__(self, client: Any, embedder: Embedder, collection: str = COLLECTION) -> None:
        self._embedder = embedder
        # We pass embeddings ourselves, so Chroma doesn't need its own embedding function.
        self._col = client.get_or_create_collection(
            collection, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )

    def add(self, user_id: str, task_id: str, items: list[MemoryItem]) -> list[str]:
        items = [i for i in items if i.text.strip()]
        if not items:
            return []
        now = time.time()
        ids = [uuid.uuid4().hex for _ in items]
        self._col.add(
            ids=ids,
            embeddings=self._embedder.embed([i.text for i in items]),
            documents=[i.text for i in items],
            metadatas=[{
                "user_id": user_id, "kind": i.kind.value, "task_id": task_id,
                "created_at": now, "access_count": 0, "last_accessed_at": now,
            } for i in items],
        )
        return ids

    def search(
        self, user_id: str, query: str, k: int = 8, max_distance: float = 0.7
    ) -> list[StoredMemory]:
        """Most similar memories for this user. Retrieval counts toward importance.

        The 0.7 cosine-distance cutoff is calibrated for all-MiniLM-L6-v2: related
        memories land around 0.45-0.66, near-duplicates of the task under 0.2.
        """
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=k,
            where={"user_id": user_id},
        )
        found = [
            _to_memory(id_, doc, meta, dist)
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                strict=True,
            )
            if dist <= max_distance
        ]
        self._touch(found)
        return found

    def list(self, user_id: str) -> list[StoredMemory]:
        res = self._col.get(where={"user_id": user_id})
        return [_to_memory(i, d, m, None)
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)]

    def delete_user(self, user_id: str) -> int:
        """Forget everything about a user (data deletion requests)."""
        ids = self._col.get(where={"user_id": user_id}, include=[])["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def _touch(self, memories: list[StoredMemory]) -> None:
        if not memories:
            return
        now = time.time()
        for m in memories:
            m.access_count += 1
            m.last_accessed_at = now
        self._col.update(
            ids=[m.id for m in memories],
            metadatas=[{"access_count": m.access_count, "last_accessed_at": now}
                       for m in memories],
        )


def _to_memory(id_: str, doc: str, meta: dict[str, Any], dist: float | None) -> StoredMemory:
    return StoredMemory(
        id=id_, user_id=meta["user_id"], kind=MemoryKind(meta["kind"]), text=doc,
        task_id=meta["task_id"], created_at=meta["created_at"],
        access_count=int(meta["access_count"]), last_accessed_at=meta["last_accessed_at"],
        distance=dist,
    )
