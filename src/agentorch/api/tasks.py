"""Task and review-queue API."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agentorch.hitl.escalation import Decision
from agentorch.hitl.service import InvalidDecision, ReviewNotPending, TaskService
from agentorch.hitl.store import ReviewRecord, TaskRecord

router = APIRouter()


@lru_cache
def get_service() -> TaskService:
    from agentorch.graph.factory import default_task_service

    return default_task_service()


Service = Annotated[TaskService, Depends(get_service)]


class TaskIn(BaseModel):
    task: str = Field(min_length=1)
    user_id: str = "default"
    require_review: bool = Field(
        default=False, description="Approve the plan and the final answer yourself")


class TaskOut(BaseModel):
    id: str
    user_id: str
    task: str
    require_review: bool
    status: str
    final_output: str | None
    error: str | None
    created_at: float
    updated_at: float


class TaskDetail(TaskOut):
    pending_review: str | None
    plan: dict[str, Any] | None
    progress: dict[str, dict[str, Any]]
    escalations: list[dict[str, Any]]
    notifications: list[dict[str, Any]]


class ReviewOut(BaseModel):
    id: str
    task_id: str
    user_id: str
    trigger: str
    level: str
    status: str
    reason: str
    request: dict[str, Any]
    decision: dict[str, Any] | None
    reviewer: str | None
    created_at: float
    resolved_at: float | None


class ReviewDetail(ReviewOut):
    past_decisions: list[ReviewOut]


def _task(r: TaskRecord) -> TaskOut:
    return TaskOut.model_validate(r, from_attributes=True)


def _review(r: ReviewRecord) -> ReviewOut:
    return ReviewOut.model_validate(r, from_attributes=True)


@router.post("/tasks", response_model=TaskOut, status_code=202)
def create_task(body: TaskIn, service: Service) -> TaskOut:
    return _task(service.start(body.task, body.user_id, body.require_review))


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(service: Service, user_id: str | None = None) -> list[TaskOut]:
    return [_task(t) for t in service.store.list_tasks(user_id)]


@router.get("/tasks/{task_id}", response_model=TaskDetail)
def get_task(task_id: str, service: Service) -> TaskDetail:
    record = service.store.get_task(task_id)
    if record is None:
        raise HTTPException(404, "no such task")
    state = service.state(task_id)
    pending = service.store.list_reviews(status="pending", task_id=task_id)
    plan = state.get("plan")
    progress = {}
    if plan is not None:
        accepted, results = state.get("accepted", {}), state.get("results", {})
        for st in plan.subtasks:
            progress[st.id] = {
                "specialist": st.specialist.value, "description": st.description,
                "depends_on": st.depends_on,
                "status": ("done" if st.id in accepted else
                           "abandoned" if st.id in state.get("abandoned", {}) else
                           "in_progress" if st.id in results else "not_started"),
                "attempts": state.get("attempts", {}).get(st.id, 0),
                "output": results[st.id].output if st.id in results else None,
            }
    return TaskDetail(
        **_task(record).model_dump(),
        pending_review=pending[0].id if pending else None,
        plan=plan.model_dump(mode="json") if plan is not None else None,
        progress=progress,
        escalations=[e.model_dump(mode="json") for e in state.get("escalations", [])],
        notifications=[n.model_dump(mode="json") for n in state.get("notifications", [])],
    )


@router.get("/reviews", response_model=list[ReviewOut])
def list_reviews(service: Service, status: str | None = "pending",
                 task_id: str | None = None) -> list[ReviewOut]:
    return [_review(r) for r in service.store.list_reviews(status=status, task_id=task_id)]


@router.get("/reviews/{review_id}", response_model=ReviewDetail)
def get_review(review_id: str, service: Service) -> ReviewDetail:
    record = service.store.get_review(review_id)
    if record is None:
        raise HTTPException(404, "no such review")
    past = service.store.past_decisions(record.user_id, record.trigger, exclude_id=record.id)
    return ReviewDetail(**_review(record).model_dump(), past_decisions=[_review(p) for p in past])


@router.post("/reviews/{review_id}/decision", response_model=ReviewOut)
def decide(review_id: str, decision: Decision, service: Service) -> ReviewOut:
    try:
        return _review(service.decide(review_id, decision))
    except KeyError:
        raise HTTPException(404, "no such review") from None
    except ReviewNotPending as e:
        raise HTTPException(409, str(e)) from None
    except InvalidDecision as e:
        raise HTTPException(422, str(e)) from None
