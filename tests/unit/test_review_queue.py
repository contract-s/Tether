"""Task service, review queue and their API (SQLite, inline execution, fake model)."""

import pytest
from fastapi.testclient import TestClient

from agentorch.api import tasks as tasks_api
from agentorch.api.main import app
from agentorch.hitl.escalation import Decision
from agentorch.hitl.service import InvalidDecision, ReviewNotPending, TaskService
from agentorch.hitl.store import HitlStore, make_engine
from tests.unit.test_hitl_graph import ActionLLM, PlainLLM, make_graph
from tests.unit.test_workflow import make_plan


class RecordingNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, review) -> None:
        self.sent.append((review.status, review.id))


def make_service(llm):
    graph, _, sent = make_graph(llm)
    notifier = RecordingNotifier()
    service = TaskService(graph, HitlStore(make_engine("sqlite://")), notifier,
                          submit=lambda fn: fn())  # run inline
    return service, notifier, sent


def test_pause_becomes_a_pending_review_and_decision_resumes_the_task() -> None:
    service, notifier, sent = make_service(ActionLLM())
    task = service.start("email the summary", user_id="alice")

    assert service.store.get_task(task.id).status == "awaiting_review"
    [review] = service.store.list_reviews(status="pending")
    assert review.level == "approve_action" and review.user_id == "alice"
    assert review.request["proposed"]["tool"] == "send_email"
    assert notifier.sent == [("pending", review.id)]

    service.decide(review.id, Decision(action="approve", reviewer="sam"))
    done = service.store.get_task(task.id)
    assert done.status == "done" and done.final_output == "FINAL ANSWER"
    assert service.store.get_review(review.id).reviewer == "sam"
    assert len(sent) == 1
    assert notifier.sent == [("pending", review.id)]  # nothing re-notified


def test_a_review_can_only_be_decided_once() -> None:
    service, _, _ = make_service(ActionLLM())
    service.start("x")
    [review] = service.store.list_reviews(status="pending")
    service.decide(review.id, Decision(action="reject"))
    with pytest.raises(ReviewNotPending):
        service.decide(review.id, Decision(action="approve"))


@pytest.mark.parametrize("decision, message", [
    (Decision(action="take_over"), "accepts"),
    (Decision(action="modify"), "tool_input"),
])
def test_invalid_decisions_are_refused_before_resuming(decision, message) -> None:
    service, _, _ = make_service(ActionLLM())
    service.start("x")
    [review] = service.store.list_reviews(status="pending")
    with pytest.raises(InvalidDecision, match=message):
        service.decide(review.id, decision)
    assert service.store.get_review(review.id).status == "pending"


def test_modified_plan_must_be_valid() -> None:
    service, _, _ = make_service(PlainLLM(plan=make_plan(confidence=0.4)))
    service.start("x")
    [review] = service.store.list_reviews(status="pending")
    cyclic = make_plan().model_copy(deep=True)
    cyclic.subtasks[0].depends_on = ["t3"]
    with pytest.raises(InvalidDecision, match="cycle"):
        service.decide(review.id, Decision(action="modify", plan=cyclic))


def test_notify_level_messages_are_stored_as_info() -> None:
    service, notifier, _ = make_service(PlainLLM(approve_review=lambda n: n > 1))
    task = service.start("x")
    assert service.store.get_task(task.id).status == "done"
    [info] = service.store.list_reviews(status="info")
    assert info.level == "notify" and notifier.sent == [("info", info.id)]


def test_a_crash_marks_the_task_failed() -> None:
    class Exploding(PlainLLM):
        def structured(self, role, system, prompt, schema):
            raise RuntimeError("model down")

    service, _, _ = make_service(Exploding())
    task = service.start("x")
    record = service.store.get_task(task.id)
    assert record.status == "failed" and "model down" in record.error


# --- API --------------------------------------------------------------------

@pytest.fixture
def api():
    service, _, sent = make_service(ActionLLM())
    app.dependency_overrides[tasks_api.get_service] = lambda: service
    yield TestClient(app), service, sent
    app.dependency_overrides.clear()


def test_api_full_review_flow(api) -> None:
    client, _, sent = api
    task = client.post("/tasks", json={"task": "email the summary", "user_id": "alice"})
    assert task.status_code == 202
    task_id = task.json()["id"]

    detail = client.get(f"/tasks/{task_id}").json()
    assert detail["status"] == "awaiting_review" and detail["pending_review"]
    assert detail["progress"]["t1"]["status"] == "in_progress"

    [review] = client.get("/reviews").json()
    assert review["id"] == detail["pending_review"]
    assert client.get(f"/reviews/{review['id']}").json()["past_decisions"] == []

    r = client.post(f"/reviews/{review['id']}/decision",
                    json={"action": "approve", "reviewer": "sam"})
    assert r.status_code == 200 and r.json()["status"] == "resolved"
    assert client.post(f"/reviews/{review['id']}/decision",
                       json={"action": "approve"}).status_code == 409

    detail = client.get(f"/tasks/{task_id}").json()
    assert detail["status"] == "done" and detail["pending_review"] is None
    assert detail["escalations"][0]["resolution"] == "approve"
    assert sent == [{"to": "boss@example.com", "body": "summary"}]


def test_api_rejects_invalid_decisions_and_unknown_ids(api) -> None:
    client, _, _ = api
    client.post("/tasks", json={"task": "x"})
    [review] = client.get("/reviews").json()
    assert client.post(f"/reviews/{review['id']}/decision",
                       json={"action": "abort"}).status_code == 422
    assert client.post("/reviews/nope/decision", json={"action": "approve"}).status_code == 404
    assert client.get("/tasks/nope").status_code == 404


def test_past_decisions_give_reviewers_precedent(api) -> None:
    client, _, _ = api
    for note in ("first", "second"):
        client.post("/tasks", json={"task": "x", "user_id": "alice"})
        [review] = client.get("/reviews").json()
        client.post(f"/reviews/{review['id']}/decision", json={"action": "reject", "note": note})
    client.post("/tasks", json={"task": "x", "user_id": "alice"})
    [review] = client.get("/reviews").json()
    past = client.get(f"/reviews/{review['id']}").json()["past_decisions"]
    assert [p["decision"]["note"] for p in past] == ["second", "first"]
