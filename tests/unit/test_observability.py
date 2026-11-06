"""Tracing, cost and metrics (fake model, inline task execution)."""

import pytest
from fastapi.testclient import TestClient

import tests.unit.test_hitl_graph as hitl
from agentorch.api import observability as obs_api
from agentorch.api.main import app
from agentorch.hitl.escalation import Decision
from agentorch.hitl.service import TaskService
from agentorch.hitl.store import HitlStore, make_engine
from agentorch.llm import LLMUsage
from agentorch.observability import metrics
from agentorch.observability.cost import llm_cost
from agentorch.observability.llm_tracing import TracingLLM
from agentorch.observability.store import SpanStore, to_record


def test_llm_cost() -> None:
    assert llm_cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert llm_cost("claude-sonnet-5-5", 1_000_000, 0, 1_000_000) == pytest.approx(2.2)
    assert llm_cost("some-other-model", 10, 10) is None


@pytest.fixture
def traced_task(monkeypatch, finished_spans):
    """Run the email task with tracing: pause for approval, then approve."""
    monkeypatch.setattr(hitl, "USAGE", LLMUsage("claude-sonnet-5-5", 1000, 200))
    graph, _, sent = hitl.make_graph(TracingLLM(hitl.ActionLLM()))
    engine = make_engine("sqlite://")
    service = TaskService(graph, HitlStore(engine), submit=lambda fn: fn())
    task = service.start("email the summary", user_id="alice")
    [review] = service.store.list_reviews(status="pending")
    service.decide(review.id, Decision(action="approve", reviewer="sam"))
    store = SpanStore(engine)
    store.add([to_record(s) for s in finished_spans.get_finished_spans()])
    return task.id, store, sent


def test_every_span_of_a_task_shares_its_trace(traced_task) -> None:
    task_id, store, _ = traced_task
    spans = store.trace(task_id)
    names = [s.name for s in spans]
    assert {"task.start", "task.resume", "human.approve_action", "supervisor.plan",
            "specialist.run", "reviewer.review", "tool.send_email", "llm.specialist",
            "node.human_action"} <= set(names)
    assert all(s.trace_id == task_id for s in spans)


def test_parallel_work_and_tools_nest_under_their_segment(traced_task) -> None:
    task_id, store, _ = traced_task
    spans = {s.span_id: s for s in store.trace(task_id)}

    def ancestors(s):
        while s.parent_span_id in spans:
            s = spans[s.parent_span_id]
            yield s

    for tool in (s for s in spans.values() if s.name == "tool.send_email"):
        chain = [a.name for a in ancestors(tool)]
        assert chain[:2] == ["specialist.run", "node.work"]
        assert chain[-1] in ("task.start", "task.resume")


def test_pause_is_not_an_error_but_a_blocked_tool_is(traced_task) -> None:
    task_id, store, sent = traced_task
    spans = store.trace(task_id)
    paused = [s for s in spans if s.attributes.get("agentorch.paused")]
    assert paused and all(s.status == "ok" for s in paused)

    sends = [s for s in spans if s.name == "tool.send_email"]
    blocked, ran = sends  # first attempt held for approval, second ran after it
    assert blocked.status == "error"
    assert blocked.attributes["tool.blocked"] == "ApprovalRequiredError"
    assert ran.status == "ok" and ran.attributes["tool.success"] is True
    assert len(sent) == 1


def test_human_review_span_covers_the_wait(traced_task) -> None:
    task_id, store, _ = traced_task
    [human] = [s for s in store.trace(task_id) if s.kind == "human"]
    assert human.attributes["human.decision"] == "approve"
    assert human.attributes["human.reviewer"] == "sam"
    assert human.attributes["human.wait_seconds"] >= 0


def test_llm_spans_carry_tokens_cost_and_prompts(traced_task) -> None:
    task_id, store, _ = traced_task
    llm = [s for s in store.trace(task_id) if s.kind == "llm"]
    assert llm and all(s.model == "claude-sonnet-5-5" and s.input_tokens == 1000 for s in llm)
    assert all(s.cost_usd == pytest.approx(0.004) for s in llm)  # 1000 in + 200 out
    specialist = next(s for s in llm if s.role == "specialist")
    assert "email" in specialist.attributes["llm.prompt"]
    assert "[tool_use send_email]" in specialist.attributes["llm.response"]


def test_task_cost_and_tree(traced_task) -> None:
    task_id, store, _ = traced_task
    records = store.trace(task_id)
    cost = metrics.task_cost(records)
    llm_calls = sum(1 for s in records if s.kind == "llm")
    assert cost["llm_calls"] == llm_calls
    assert cost["total_cost_usd"] == pytest.approx(0.004 * llm_calls)
    assert cost["escalations"] == 1 and cost["task_type"] == "research"
    assert set(cost["by_role"]) == {"supervisor", "specialist", "reviewer"}
    assert cost["tools"]["send_email"]["calls"] == 2
    assert cost["tools"]["send_email"]["errors"] == 1

    roots = metrics.build_tree(records)
    assert {r["name"] for r in roots} == {"task.start", "task.resume", "human.approve_action"}
    assert sum(r["subtree_cost_usd"] for r in roots) == pytest.approx(cost["total_cost_usd"])


def test_api_trace_cost_and_metrics(traced_task) -> None:
    task_id, store, _ = traced_task
    app.dependency_overrides[obs_api.get_span_store] = lambda: store
    try:
        client = TestClient(app)
        tree = client.get(f"/tasks/{task_id}/trace").json()
        assert tree["cost"]["escalations"] == 1 and tree["spans"]
        assert client.get(f"/tasks/{task_id}/cost").json()["llm_calls"] > 0
        assert client.get("/tasks/nope/trace").status_code == 404

        m = client.get("/metrics?days=7").json()
        assert [t["task_id"] for t in m["tasks"]] == [task_id]
        assert m["cost_by_task_type"]["research"]["tasks"] == 1
        assert m["tools"]["send_email"]["error_rate"] == 0.5
        assert m["escalation_rate_by_day"][0]["rate"] == 1.0
        assert list(m["agents_by_cost"])[0] in {"specialist", "reviewer", "supervisor"}
    finally:
        app.dependency_overrides.clear()


def test_trace_explorer_page() -> None:
    client = TestClient(app)
    page = client.get("/dashboard/traces")
    assert page.status_code == 200 and "Trace explorer" in page.text
    # Span text comes from the web; the page must never insert it as HTML.
    assert "innerHTML" not in page.text
    assert "/dashboard/traces" in client.get("/dashboard/reviews").text
