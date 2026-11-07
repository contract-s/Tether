"""The whole system on real Postgres, Redis and ChromaDB, driven through the HTTP API.

The model is scripted (no API cost), so these check the plumbing between the parts:
durable checkpoints, the review queue, long-term memory, recovery and the Celery path.
"""

from __future__ import annotations

import uuid

import pytest

from agentorch.memory.long_term_types import MemoryKind
from tests.unit.test_recovery import CrashingLLM
from tests.unit.test_workflow import make_plan
from tests.unit.test_workflow_memory import MemoryAwareLLM


def scripted(confidence: float = 0.9) -> MemoryAwareLLM:
    llm = MemoryAwareLLM()  # plans t1, t2 -> t3; saves one "approach" memory per task
    llm.plan = make_plan(confidence)
    return llm


def test_a_task_runs_end_to_end_and_its_record_survives_a_restart(make_stack, user) -> None:
    stack = make_stack(scripted())
    tid = stack.start("compare vector databases qdrant pgvector", user)
    task = stack.wait(tid)

    assert task["status"] == "done" and task["final_output"] == "FINAL ANSWER"
    assert {p["status"] for p in task["progress"].values()} == {"done"}
    # Working memory was used during the run and cleared after it.
    assert not list(stack.redis.scan_iter(f"agentorch:wm:{tid}:*"))

    # A new process reads the same run back from Postgres.
    stack.use(stack.new_service())
    again = stack.client.get(f"/tasks/{tid}").json()
    assert again["status"] == "done" and again["plan"]["subtasks"][0]["id"] == "t1"
    trace = stack.client.get(f"/tasks/{tid}/trace").json()
    assert trace["cost"]["llm_calls"] > 0


def test_escalation_waits_for_a_human_across_a_restart(make_stack, user) -> None:
    stack = make_stack(scripted(confidence=0.3))  # below the 0.7 threshold
    tid = stack.start("research things", user)
    task = stack.wait(tid, "awaiting_review")
    assert task["pending_review"] and not stack.llm.specialist_prompts  # nothing ran yet

    [review] = stack.client.get(f"/reviews?task_id={tid}").json()
    assert review["trigger"] == "low_plan_confidence" and review["level"] == "approve_plan"

    # The API restarts while the task waits; the decision still resumes it.
    stack.use(stack.new_service())
    res = stack.client.post(f"/reviews/{review['id']}/decision",
                            json={"action": "approve", "reviewer": "e2e"})
    assert res.status_code == 200, res.text
    assert stack.wait(tid)["status"] == "done"
    assert stack.client.post(f"/reviews/{review['id']}/decision",
                             json={"action": "approve"}).status_code == 409


def test_memory_from_one_task_informs_the_next(make_stack, user) -> None:
    stack = make_stack(scripted())
    first = stack.wait(stack.start("compare vector databases qdrant pgvector", user))
    assert first["status"] == "done"
    assert "<past_tasks>" not in stack.llm.plan_prompts[0]

    memories = stack.client.get(f"/users/{user}/memories").json()
    assert [m["kind"] for m in memories] == [MemoryKind.APPROACH.value]

    # A similar task for the same user plans with it (real embeddings, real ChromaDB).
    stack.wait(stack.start("compare vector databases weaviate milvus", user))
    assert "research each database in parallel" in stack.llm.plan_prompts[1]
    # Another user doesn't see it.
    stack.wait(stack.start("compare vector databases weaviate milvus", f"{user}-other"))
    assert "<past_tasks>" not in stack.llm.plan_prompts[2]
    stack.memory.long_term.delete_user(f"{user}-other")


def test_a_failed_task_is_retried_from_its_checkpoint(make_stack, user) -> None:
    llm = CrashingLLM("t2", RuntimeError("model API is down"))
    stack = make_stack(llm)
    tid = stack.start("research things", user)
    failed = stack.wait(tid)
    assert failed["status"] == "failed" and "model API is down" in failed["error"]

    assert stack.client.post(f"/tasks/{tid}/retry").status_code == 202
    assert stack.wait(tid)["status"] == "done"
    assert llm.calls("t1") == 1  # kept from before the crash


@pytest.fixture
def celery_worker(make_stack, monkeypatch):
    """A Celery worker in this process, on a queue of its own (so a running compose
    worker can't take these messages)."""
    from celery.contrib.testing.worker import start_worker

    import agentorch.workers as workers

    queue = f"e2e-{uuid.uuid4().hex[:8]}"
    stack = make_stack(scripted())
    monkeypatch.setattr(workers, "_service", lambda: stack.service)
    stack.service.dispatch = lambda tid, payload: workers.run.apply_async(
        (tid, payload), queue=queue)
    with start_worker(workers.app, pool="solo", perform_ping_check=False, queues=[queue],
                      shutdown_timeout=10):
        yield stack


def test_runs_go_through_the_celery_worker(celery_worker, user) -> None:
    stack = celery_worker
    stack.llm.plan = make_plan(0.3)  # pause for review, so the resume also goes via Celery
    tid = stack.start("research things", user)
    stack.wait(tid, "awaiting_review", timeout=30)
    review = stack.client.get(f"/tasks/{tid}").json()["pending_review"]
    stack.client.post(f"/reviews/{review}/decision", json={"action": "approve"})
    assert stack.wait(tid, timeout=30)["status"] == "done"
