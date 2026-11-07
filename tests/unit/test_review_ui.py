"""Review chat assistant, its API, and the review page."""

import pytest
from fastapi.testclient import TestClient

from tests.unit.test_hitl_graph import ActionLLM, make_graph
from tether.api import tasks as tasks_api
from tether.api.main import app
from tether.hitl.assistant import MAX_OUTPUT_CHARS, ReviewAssistant, format_context
from tether.hitl.service import TaskService
from tether.hitl.store import HitlStore, make_engine
from tether.llm import LLMUsage, ModelTurn


class ExplainerLLM(ActionLLM):
    """ActionLLM for the run, plus an explainer that echoes what it was shown."""

    def __init__(self) -> None:
        super().__init__()
        self.explainer_calls: list[list[dict]] = []

    def chat(self, role, system, messages, tools):
        if role != "explainer":
            return super().chat(role, system, messages, tools)
        self.explainer_calls.append(messages)
        n = len(self.explainer_calls)
        return ModelTurn([], f"answer {n}", [], "end_turn", LLMUsage("fake", 1, 1))


@pytest.fixture
def setup():
    llm = ExplainerLLM()
    graph, _, _ = make_graph(llm)
    service = TaskService(graph, HitlStore(make_engine("sqlite://")),
                          submit=lambda fn: fn(), assistant=ReviewAssistant(llm))
    app.dependency_overrides[tasks_api.get_service] = lambda: service
    yield TestClient(app), service, llm
    app.dependency_overrides.clear()


def test_context_is_trimmed_and_marked_as_data() -> None:
    request = {"progress": {"t1": {"output": "x" * (MAX_OUTPUT_CHARS + 500)}}}
    ctx = format_context(request)
    assert "[truncated]" in ctx and ctx.count("x") < MAX_OUTPUT_CHARS + 100


def test_chat_answers_with_review_context_and_keeps_history(setup) -> None:
    client, _, llm = setup
    client.post("/tasks", json={"task": "email the summary"})
    [review] = client.get("/reviews").json()

    first = client.post(f"/reviews/{review['id']}/ask",
                        json={"question": "why email?", "author": "sam"}).json()
    second = client.post(f"/reviews/{review['id']}/ask", json={"question": "risks?"}).json()
    assert (first["content"], second["content"]) == ("answer 1", "answer 2")

    sent = llm.explainer_calls[1]
    assert "send_email" in sent[0]["content"]  # the review context
    assert [m["content"] for m in sent[2:]] == ["why email?", "answer 1", "risks?"]

    log = client.get(f"/reviews/{review['id']}/messages").json()
    assert [(m["role"], m["content"]) for m in log] == [
        ("human", "why email?"), ("agent", "answer 1"), ("human", "risks?"), ("agent", "answer 2")]
    assert log[0]["author"] == "sam"


def test_chat_on_unknown_review_is_404(setup) -> None:
    client, _, _ = setup
    assert client.post("/reviews/nope/ask", json={"question": "hi"}).status_code == 404


def test_review_page_is_served_and_never_renders_html_from_data(setup) -> None:
    client, _, _ = setup
    assert client.get("/", follow_redirects=False).headers["location"] == "/dashboard/reviews"
    page = client.get("/dashboard/reviews")
    assert page.status_code == 200 and "Review queue" in page.text
    assert "innerHTML" not in page.text and "insertAdjacentHTML" not in page.text
