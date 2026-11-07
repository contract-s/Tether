"""Step through a run's checkpoints, replay it from any step with changes, compare."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tether.api.observability import get_span_store
from tether.api.tasks import get_service
from tether.replay import ReplayChanges, ReplayError, ReplayService

router = APIRouter()
PAGE = Path(__file__).parent / "static" / "replay.html"


@lru_cache
def get_replay_service() -> ReplayService:
    from tether.graph.factory import default_working_memory

    return ReplayService(get_service(), get_span_store(), default_working_memory())


Replays = Annotated[ReplayService, Depends(get_replay_service)]


class ReplayIn(BaseModel):
    checkpoint_id: str
    changes: ReplayChanges = ReplayChanges()


@router.get("/tasks/{task_id}/steps")
def steps(task_id: str, replays: Replays) -> list[dict[str, Any]]:
    """Every checkpoint of the run, oldest first: what ran, what changed, what's next."""
    try:
        return [s.as_dict() for s in replays.steps(task_id)]
    except KeyError:
        raise HTTPException(404, "no run recorded for this task") from None


@router.get("/tasks/{task_id}/steps/{checkpoint_id}")
def step_state(task_id: str, checkpoint_id: str, replays: Replays) -> dict[str, Any]:
    """The full state at one step, and at the step before it."""
    try:
        return replays.state_at(task_id, checkpoint_id)
    except KeyError:
        raise HTTPException(404, "no such step") from None


@router.post("/tasks/{task_id}/replay", status_code=202)
def replay(task_id: str, body: ReplayIn, replays: Replays) -> dict[str, Any]:
    """Fork the run at a step, apply the changes, and run it as a new task."""
    try:
        record = replays.replay(task_id, body.checkpoint_id, body.changes)
    except KeyError:
        raise HTTPException(404, "no such task or step") from None
    except ReplayError as e:
        raise HTTPException(422, str(e)) from None
    return {"replay_id": record.id, "source_task_id": task_id, "step": record.step,
            "resumed_after": record.resumed_after}


@router.get("/tasks/{task_id}/replays")
def replays_of(task_id: str, replays: Replays) -> list[dict[str, Any]]:
    return replays.replays_of(task_id)


@router.get("/replays/{replay_id}/compare")
def compare(replay_id: str, replays: Replays) -> dict[str, Any]:
    """The replay against the part of the original that ran after the fork point."""
    try:
        return replays.compare(replay_id)
    except KeyError:
        raise HTTPException(404, "no such replay") from None


@router.get("/dashboard/replay", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)
