"""Failure recovery: crashed runs carry on from their last checkpoint (fake model)."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from agentorch.agents.reviewer import Reviewer
from agentorch.agents.specialist import SpecialistAgent
from agentorch.agents.supervisor import Supervisor
from agentorch.api import tasks as tasks_api
from agentorch.api.main import app
from agentorch.graph.workflow import Agents, build_workflow
from agentorch.hitl.escalation import Decision
from agentorch.hitl.service import NotRetryable, TaskService
from agentorch.hitl.store import HitlStore, make_engine
from agentorch.tools.registry import ToolRegistry
from tests.unit.test_workflow import FakeLLM, make_plan


class WorkerDied(BaseException):
    """Stands in for the process being killed: nothing in the run gets to handle it."""


class CrashingLLM(FakeLLM):
    """The specialist for `crash_on` fails on its first call, once t1 has finished."""

    def __init__(self, crash_on: str, error: BaseException, confidence: float = 0.9) -> None:
        super().__init__(make_plan(confidence), approve=lambda sid, n: True)
        self.crash_on, self.error = crash_on, error
        self.t1_done = threading.Event()
        self.crashed = False

    def chat(self, role, system, messages, tools):
        prompt = messages[0]["content"]
        is_target = role == "specialist" and f'<subtask id="{self.crash_on}"' in prompt
        if is_target and not self.crashed:
            # t1 runs in parallel with t2; let it finish so its result is checkpointed.
            self.t1_done.wait(timeout=5)
            self.crashed = True
            raise self.error
        turn = super().chat(role, system, messages, tools)
        if role == "specialist" and '<subtask id="t1"' in prompt:
            self.t1_done.set()
        return turn

    def calls(self, sid: str) -> int:
        return len(self.specialist_prompts.get(sid, []))


def make_service(llm: FakeLLM, dispatched: list | None = None) -> TaskService:
    registry = ToolRegistry()
    agents = Agents(supervisor=Supervisor(llm), reviewer=Reviewer(llm, threshold=0.7),
                    make_specialist=lambda s: SpecialistAgent(s, llm, registry))
    graph = build_workflow(agents, max_attempts=2, plan_confidence_threshold=0.7)
    service = TaskService(graph, HitlStore(make_engine("sqlite://")), submit=lambda fn: fn())
    if dispatched is not None:
        run_inline = service.dispatch

        def recording(task_id, payload):
            dispatched.append(payload)
            run_inline(task_id, payload)

        service.dispatch = recording
    return service


def test_a_crashed_run_is_retried_from_its_checkpoint_keeping_parallel_work() -> None:
    llm = CrashingLLM("t2", RuntimeError("model API is down"))
    service = make_service(llm)
    task = service.start("x")

    record = service.store.get_task(task.id)
    assert record.status == "failed" and "model API is down" in record.error
    assert llm.calls("t1") == 1

    service.retry(task.id)
    record = service.store.get_task(task.id)
    assert record.status == "done" and record.final_output == "FINAL ANSWER" and not record.error
    # t1 finished alongside t2 before the crash: its result was kept, not redone.
    assert llm.calls("t1") == 1 and llm.calls("t2") == 1 and llm.calls("t3") == 1


def test_a_worker_that_dies_mid_run_has_its_message_redelivered() -> None:
    llm = CrashingLLM("t2", WorkerDied())
    dispatched: list = []
    service = make_service(llm, dispatched)
    with pytest.raises(WorkerDied):
        service.start("x")
    assert service.store.get_task(service.store.list_tasks()[0].id).status == "running"
    task_id = service.store.list_tasks()[0].id

    # The broker hands the same message to another worker.
    [payload] = dispatched
    assert payload["kind"] == "start" and json.loads(json.dumps(payload)) == payload
    service.run(task_id, payload)

    assert service.store.get_task(task_id).status == "done"
    assert llm.calls("t1") == 1 and llm.calls("t3") == 1
    # Delivered yet again after finishing: nothing runs.
    service.run(task_id, payload)
    assert llm.calls("t3") == 1 and service.store.get_task(task_id).status == "done"


def test_a_redelivered_decision_is_applied_once() -> None:
    llm = CrashingLLM("none", RuntimeError(), confidence=0.2)  # low confidence: plan review
    dispatched: list = []
    service = make_service(llm, dispatched)
    task = service.start("x")
    [review] = service.store.list_reviews(status="pending")
    service.decide(review.id, Decision(action="approve", reviewer="sam"))
    assert service.store.get_task(task.id).status == "done"
    calls = llm.calls("t1")

    service.run(task.id, dispatched[-1])  # the resume message again
    assert llm.calls("t1") == calls and service.store.get_task(task.id).status == "done"


def test_stalled_runs_are_recovered_at_startup() -> None:
    llm = CrashingLLM("t2", WorkerDied())
    service = make_service(llm)
    with pytest.raises(WorkerDied):
        service.start("x")
    [crashed] = service.store.list_tasks()
    queued = service.store.create_task("ab" * 16, "default", "y", False)

    assert set(service.recover_stalled()) == {crashed.id, queued.id}
    assert {t.status for t in service.store.list_tasks()} == {"done"}
    assert service.recover_stalled() == []


def test_only_unfinished_runs_can_be_retried() -> None:
    service = make_service(CrashingLLM("none", RuntimeError()))
    task = service.start("x")
    with pytest.raises(NotRetryable, match="done"):
        service.retry(task.id)
    with pytest.raises(KeyError):
        service.retry("nope")


def test_retry_api() -> None:
    service = make_service(CrashingLLM("t2", RuntimeError("boom")))
    task = service.start("x")
    app.dependency_overrides[tasks_api.get_service] = lambda: service
    try:
        client = TestClient(app)
        res = client.post(f"/tasks/{task.id}/retry")
        assert res.status_code == 202
        assert client.get(f"/tasks/{task.id}").json()["status"] == "done"
        assert client.post(f"/tasks/{task.id}/retry").status_code == 409
        assert client.post("/tasks/nope/retry").status_code == 404
    finally:
        app.dependency_overrides.clear()
