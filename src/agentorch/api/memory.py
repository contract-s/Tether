"""Memory management API: inspect, consolidate, expire, and delete what the system remembers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agentorch.llm import AnthropicClient, LLMClient
from agentorch.memory.consolidator import MemoryConsolidator
from agentorch.memory.long_term import LongTermMemory
from agentorch.memory.long_term_types import StoredMemory

router = APIRouter()
DASHBOARD = Path(__file__).parent / "static" / "memory_dashboard.html"


@lru_cache
def get_store() -> LongTermMemory:
    from agentorch.graph.factory import default_memory

    store = default_memory(llm=None).long_term  # type: ignore[arg-type]
    if store is None:
        raise HTTPException(503, "ChromaDB is not reachable; run `make up`")
    return store


@lru_cache
def get_llm() -> LLMClient:
    return AnthropicClient()


Store = Annotated[LongTermMemory, Depends(get_store)]
LLM = Annotated[LLMClient, Depends(get_llm)]


class UserSummary(BaseModel):
    user_id: str
    memories: int


class DeleteResult(BaseModel):
    deleted: int


class ExpireResult(BaseModel):
    expired: int
    texts: list[str]


class MergedGroup(BaseModel):
    replaced: list[str]
    merged_into: str


class ConsolidateResult(BaseModel):
    dry_run: bool
    memories_before: int
    memories_after: int
    groups: list[MergedGroup]


@router.get("/dashboard/memories", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD)


@router.get("/users", response_model=list[UserSummary])
def list_users(store: Store) -> list[UserSummary]:
    return [UserSummary(user_id=u, memories=n) for u, n in sorted(store.users().items())]


@router.get("/users/{user_id}/memories", response_model=list[StoredMemory])
def list_memories(user_id: str, store: Store):
    return store.list(user_id)


@router.delete("/users/{user_id}/memories", response_model=DeleteResult)
def delete_user_memories(user_id: str, store: Store):
    """Data deletion request: forget everything stored about this user."""
    return DeleteResult(deleted=store.delete_user(user_id))


@router.delete("/memories/{memory_id}", response_model=DeleteResult)
def delete_memory(memory_id: str, store: Store):
    return DeleteResult(deleted=store.delete([memory_id]))


@router.post("/users/{user_id}/memories/expire", response_model=ExpireResult)
def expire_memories(user_id: str, store: Store):
    stale = store.expire(user_id)
    return ExpireResult(expired=len(stale), texts=[m.text for m in stale])


@router.post("/users/{user_id}/memories/consolidate", response_model=ConsolidateResult)
def consolidate_memories(
    user_id: str,
    store: Store,
    llm: LLM,
    dry_run: bool = False,
):
    report = MemoryConsolidator(llm, store).consolidate(user_id, dry_run=dry_run)
    return ConsolidateResult(
        dry_run=dry_run,
        memories_before=report.memories_before,
        memories_after=report.memories_after,
        groups=[MergedGroup(replaced=old, merged_into=new) for old, new in report.merged],
    )
