"""Runs tasks in the background and connects their pauses to the review queue.

start() -> the graph runs until it finishes or pauses at a human step.
On a pause, the escalation becomes a pending review, the reviewer is notified, and the
task waits (no thread is held; the state lives in the checkpointer).
decide() -> records the decision and resumes the run from the checkpoint.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.types import Command

from agentorch.agents.planning import validate_plan
from agentorch.hitl.escalation import OPTIONS, ApprovalLevel, Decision
from agentorch.hitl.notifier import LogNotifier, Notifier
from agentorch.hitl.store import HitlStore, ReviewRecord, TaskRecord

log = logging.getLogger(__name__)

Submit = Callable[[Callable[[], None]], Any]


class ReviewNotPending(Exception):
    pass


class InvalidDecision(Exception):
    pass


class TaskService:
    def __init__(
        self,
        graph: Any,
        store: HitlStore,
        notifier: Notifier | None = None,
        submit: Submit | None = None,
    ) -> None:
        self.graph = graph
        self.store = store
        self.notifier = notifier or LogNotifier()
        # Background by default; tests pass `lambda fn: fn()` to run inline.
        self._submit = submit or ThreadPoolExecutor(max_workers=4).submit
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # --- public ----------------------------------------------------------------

    def start(
        self, task: str, user_id: str = "default", require_review: bool = False
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex
        record = self.store.create_task(task_id, user_id, task, require_review)
        inp = {"task": task, "user_id": user_id, "require_review": require_review}
        self._submit(lambda: self._advance(task_id, inp))
        return record

    def decide(self, review_id: str, decision: Decision) -> ReviewRecord:
        review = self.store.get_review(review_id)
        if review is None:
            raise KeyError(review_id)
        if review.status != "pending":
            raise ReviewNotPending(f"review is {review.status}")
        self._validate(review, decision)
        current = self._current_interrupt(review.task_id)
        if current is None or current.get("id") != review_id:
            raise ReviewNotPending("the task is no longer waiting on this review")

        resolved = self.store.resolve_review(review_id, decision.model_dump(mode="json"))
        self.store.update_task(review.task_id, status="running")
        resume = Command(resume=decision.model_dump(mode="json", exclude_none=True))
        self._submit(lambda: self._advance(review.task_id, resume))
        return resolved

    def state(self, task_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(self._config(task_id)).values)

    # --- internals -------------------------------------------------------------

    def _advance(self, task_id: str, inp: Any) -> None:
        with self._lock(task_id):
            self.store.update_task(task_id, status="running")
            try:
                self.graph.invoke(inp, self._config(task_id))
            except Exception as e:  # noqa: BLE001 - surface any crash on the task
                log.exception("task %s crashed", task_id)
                self.store.update_task(task_id, status="failed", error=f"{type(e).__name__}: {e}")
                return
            self._sync(task_id)

    def _sync(self, task_id: str) -> None:
        """Mirror the checkpointed run into the task and review tables."""
        snapshot = self.graph.get_state(self._config(task_id))
        values = snapshot.values

        for note in values.get("notifications", []):
            record, created = self.store.add_review(note.model_dump(mode="json"), status="info")
            if created:
                self.notifier.notify(record)

        pause = self._current_interrupt(task_id, snapshot)
        if pause is not None:
            record, created = self.store.add_review(pause, status="pending")
            if created:
                self.notifier.notify(record)
            self.store.update_task(task_id, status="awaiting_review")
            return

        self.store.update_task(
            task_id,
            status=values.get("status", "failed"),
            final_output=values.get("final_output"),
            error=values.get("error"),
        )

    def _current_interrupt(self, task_id: str, snapshot: Any = None) -> dict[str, Any] | None:
        snapshot = snapshot or self.graph.get_state(self._config(task_id))
        for task in snapshot.tasks:
            for intr in task.interrupts:
                return intr.value
        return None

    def _validate(self, review: ReviewRecord, decision: Decision) -> None:
        level = ApprovalLevel(review.level)
        if decision.action not in OPTIONS[level]:
            raise InvalidDecision(f"{level.value} accepts {OPTIONS[level]}")
        if level == ApprovalLevel.APPROVE_PLAN and decision.action == "modify":
            if decision.plan is None:
                raise InvalidDecision("modify needs the edited plan")
            try:
                validate_plan(decision.plan)
            except ValueError as e:
                raise InvalidDecision(f"invalid plan: {e}") from e
        if level == ApprovalLevel.APPROVE_ACTION and decision.action == "modify":
            is_tool = bool(review.request.get("proposed", {}).get("tool"))
            if is_tool and decision.tool_input is None:
                raise InvalidDecision("modify needs the edited tool_input")
            if not is_tool and not decision.output:
                raise InvalidDecision("modify needs the edited output")
        if decision.action == "take_over" and not decision.output:
            raise InvalidDecision("take_over needs the output you want to use")

    def _lock(self, task_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(task_id, threading.Lock())

    @staticmethod
    def _config(task_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": task_id}}
