"""Runs tasks in the background and connects their pauses to the review queue.

start() -> the graph runs until it finishes or pauses at a human step.
On a pause, the escalation becomes a pending review, the reviewer is notified, and the
task waits (no thread is held; the state lives in the checkpointer).
decide() -> records the decision and resumes the run from the checkpoint.

Work is handed to a dispatcher as (task_id, payload), where the payload is plain JSON:
    {"kind": "start", "input": {...}}       first run
    {"kind": "resume", "review_id": ..., "value": {...}}   a human's decision
    {"kind": "continue"}                    carry on from the last checkpoint
By default a thread pool runs it in this process; with Celery a worker does (see
tether.workers). run() is safe to call twice for the same payload, so a worker that
dies mid-run can have its message redelivered: the run picks up from the last checkpoint.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.types import Command

from tether.agents.planning import validate_plan
from tether.hitl.assistant import ReviewAssistant
from tether.hitl.escalation import OPTIONS, ApprovalLevel, Decision
from tether.hitl.notifier import LogNotifier, Notifier
from tether.hitl.store import HitlStore, ReviewMessage, ReviewRecord, TaskRecord
from tether.observability.tracing import (
    flush,
    mark_error,
    set_attributes,
    task_context,
    tracer,
)

log = logging.getLogger(__name__)

Submit = Callable[[Callable[[], None]], Any]
Dispatch = Callable[[str, dict[str, Any]], Any]


class ReviewNotPending(Exception):
    pass


class InvalidDecision(Exception):
    pass


class NotRetryable(Exception):
    pass


class TaskService:
    def __init__(
        self,
        graph: Any,
        store: HitlStore,
        notifier: Notifier | None = None,
        submit: Submit | None = None,
        assistant: ReviewAssistant | None = None,
        dispatch: Dispatch | None = None,
    ) -> None:
        self.graph = graph
        self.store = store
        self.notifier = notifier or LogNotifier()
        self.assistant = assistant
        # Background by default; tests pass `lambda fn: fn()` to run inline.
        self._submit = submit or ThreadPoolExecutor(max_workers=4).submit
        # Where runs go: this process (via submit) unless another dispatcher is given.
        self.dispatch: Dispatch = dispatch or (
            lambda task_id, payload: self._submit(lambda: self.run(task_id, payload)))
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # --- public ----------------------------------------------------------------

    def start(
        self, task: str, user_id: str = "default", require_review: bool = False
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex
        record = self.store.create_task(task_id, user_id, task, require_review)
        inp = {"task": task, "user_id": user_id, "require_review": require_review}
        self.dispatch(task_id, {"kind": "start", "input": inp})
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
        self._trace_review(resolved)
        self.store.update_task(review.task_id, status="running")
        self.dispatch(review.task_id, {
            "kind": "resume", "review_id": review_id,
            "value": decision.model_dump(mode="json", exclude_none=True)})
        return resolved

    def retry(self, task_id: str) -> TaskRecord:
        """Carry on a failed or stalled task from its last checkpoint.

        The step that crashed runs again; finished work (including parallel subtasks
        that completed before the crash) is kept, not redone.
        """
        record = self.store.get_task(task_id)
        if record is None:
            raise KeyError(task_id)
        if record.status not in ("failed", "running", "queued"):
            raise NotRetryable(f"task is {record.status}")
        if not self.graph.get_state(self._config(task_id)).next:
            raise NotRetryable("the run has no step left to retry")
        self.store.update_task(task_id, status="running", error=None)
        self.dispatch(task_id, {"kind": "continue"})
        return self.store.get_task(task_id)  # type: ignore[return-value]

    def recover_stalled(self) -> list[str]:
        """Restart runs left 'running' by a crash (single-process mode, at startup)."""
        stalled = [t.id for t in self.store.list_tasks(limit=1000)
                   if t.status in ("running", "queued")]
        for task_id in stalled:
            if self.graph.get_state(self._config(task_id)).values:
                self.dispatch(task_id, {"kind": "continue"})
            else:  # never got going: nothing was checkpointed
                record = self.store.get_task(task_id)
                assert record is not None
                self.dispatch(task_id, {"kind": "start", "input": {
                    "task": record.task, "user_id": record.user_id,
                    "require_review": record.require_review}})
        if stalled:
            log.warning("recovering %d stalled task(s): %s", len(stalled), stalled)
        return stalled

    def run(self, task_id: str, payload: dict[str, Any]) -> None:
        """Do one dispatched piece of work. Repeating a payload is harmless."""
        kind = payload["kind"]
        snapshot = self.graph.get_state(self._config(task_id))
        paused = self._current_interrupt(task_id, snapshot)
        if kind not in ("start", "resume", "continue"):
            raise ValueError(f"unknown payload kind {kind!r}")
        inp: Any = None  # None = carry on from the last checkpoint
        if kind == "start" and not snapshot.values:
            inp = payload["input"]
        elif kind == "resume" and paused is not None and paused.get("id") == payload["review_id"]:
            inp = Command(resume=payload["value"])
        if inp is None and (paused is not None or (snapshot.values and not snapshot.next)):
            # Already finished, or waiting on a human (perhaps this very decision was
            # applied before a crash and the run paused again): just mirror it.
            self._sync(task_id)
            return
        self._advance(task_id, inp)

    def ask(self, review_id: str, question: str, author: str | None = None) -> ReviewMessage:
        """Answer a reviewer's question about a review, keeping the conversation."""
        if self.assistant is None:
            raise RuntimeError("no assistant configured")
        review = self.store.get_review(review_id)
        if review is None:
            raise KeyError(review_id)
        history = [(m.role, m.content) for m in self.store.messages(review_id)]
        self.store.add_message(review_id, "human", question, author)
        answer, _usage = self.assistant.answer(review.request, history, question)
        return self.store.add_message(review_id, "agent", answer)

    def state(self, task_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(self._config(task_id)).values)

    # --- internals -------------------------------------------------------------

    def _advance(self, task_id: str, inp: Any) -> None:
        try:
            self._advance_traced(task_id, inp)
        finally:
            flush()  # make the segment visible in the trace explorer right away

    def _advance_traced(self, task_id: str, inp: Any) -> None:
        name = ("task.resume" if isinstance(inp, Command) else
                "task.continue" if inp is None else "task.start")
        with self._lock(task_id), tracer().start_as_current_span(
            name, context=task_context(task_id),
            record_exception=False, set_status_on_exception=False,
        ) as s:
            s.set_attribute("tether.kind", "segment")
            s.set_attribute("tether.task_id", task_id)
            self.store.update_task(task_id, status="running")
            try:
                self.graph.invoke(inp, self._config(task_id))
            except Exception as e:  # noqa: BLE001 - surface any crash on the task
                log.exception("task %s crashed", task_id)
                s.record_exception(e)
                mark_error(s, f"{type(e).__name__}: {e}")
                self.store.update_task(task_id, status="failed", error=f"{type(e).__name__}: {e}")
                return
            self._sync(task_id)
            s.set_attribute("tether.task_status",
                            self.store.get_task(task_id).status)  # type: ignore[union-attr]

    def _trace_review(self, review: ReviewRecord) -> None:
        """A span covering the time the task waited on this human."""
        assert review.resolved_at is not None
        s = tracer().start_span(
            f"human.{review.level}", context=task_context(review.task_id),
            start_time=int(review.created_at * 1e9),
        )
        set_attributes(s, {
            "tether.kind": "human", "tether.review_id": review.id,
            "tether.trigger": review.trigger, "tether.level": review.level,
            "tether.subtask_id": review.request.get("subtask_id"),
            "human.decision": (review.decision or {}).get("action"),
            "human.note": (review.decision or {}).get("note"),
            "human.reviewer": review.reviewer,
            "human.wait_seconds": round(review.resolved_at - review.created_at, 3),
        })
        s.end(end_time=int(review.resolved_at * 1e9))

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
