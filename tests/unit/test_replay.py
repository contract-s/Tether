"""Step-through and replay: forking a run from a checkpoint, with changes (fake model)."""

import time

import pytest
from fastapi.testclient import TestClient

from tests.unit.test_workflow import FakeLLM, make_plan
from tether.agents.reviewer import Reviewer
from tether.agents.schemas import ExecutionPlan
from tether.agents.specialist import SpecialistAgent
from tether.agents.supervisor import Supervisor
from tether.api import observability as obs_api
from tether.api import replay as replay_api
from tether.api import tasks as tasks_api
from tether.api.main import app
from tether.graph.workflow import Agents, build_workflow
from tether.hitl.service import TaskService
from tether.hitl.store import HitlStore, make_engine
from tether.memory.long_term_types import MemoryKind, StoredMemory
from tether.memory.service import MemoryServices
from tether.observability.llm_tracing import TracingLLM
from tether.observability.store import SpanStore, to_record
from tether.replay import ReplayChanges, ReplayError, ReplayService
from tether.tools.registry import ToolRegistry

MEMORY = StoredMemory(id="m1", user_id="u", kind=MemoryKind.PITFALL, text="check dates",
                      task_id="old", created_at=0, access_count=0, last_accessed_at=0)


class SpyMemory(MemoryServices):
    """Long-term memory that returns one memory and records what it's asked."""

    def __init__(self) -> None:
        super().__init__()
        self.retrieved: list[str] = []
        self.remembered: list[str] = []

    def retrieve(self, user_id, task):
        self.retrieved.append(task)
        return [MEMORY]

    def remember(self, user_id, task_id, transcript):
        self.remembered.append(task_id)
        return [], None


class Setup:
    def __init__(self, finished_spans) -> None:
        # t1 fails review once, so the run has a retry in it.
        self.llm = FakeLLM(make_plan(), approve=lambda sid, n: not (sid == "t1" and n == 1))
        self.memory = SpyMemory()
        llm = TracingLLM(self.llm)
        agents = Agents(supervisor=Supervisor(llm), reviewer=Reviewer(llm, threshold=0.7),
                        make_specialist=lambda s: SpecialistAgent(s, llm, ToolRegistry()),
                        memory=self.memory)
        graph = build_workflow(agents, max_attempts=2, plan_confidence_threshold=0.7)
        engine = make_engine("sqlite://")
        self.tasks = TaskService(graph, HitlStore(engine), submit=lambda fn: fn())
        self.spans = SpanStore(engine)
        self.finished = finished_spans
        self.service = ReplayService(self.tasks, self.spans)
        self.task = self.tasks.start("research things", user_id="u")
        self.save_spans()

    def save_spans(self) -> None:
        self.spans.add([to_record(s) for s in self.finished.get_finished_spans()])
        self.finished.clear()

    def step(self, ran: str, nth: int = 0):
        return [s for s in self.service.steps(self.task.id) if s.ran == [ran]][nth]

    def replay(self, step, **changes):
        record = self.service.replay(self.task.id, step.checkpoint_id, ReplayChanges(**changes))
        self.save_spans()
        return record, self.tasks.state(record.id)


@pytest.fixture
def s(finished_spans):
    return Setup(finished_spans)


def test_steps_walk_the_run_in_order(s) -> None:
    steps = s.service.steps(s.task.id)
    ran = [x.ran for x in steps]
    assert ran[0] == ["input"] and ran[1] == ["__start__"] and ran[2] == ["intake"]
    assert ["work"] in ran and ran[-1] == ["deliver"]
    assert [x.step for x in steps] == sorted(x.step for x in steps)
    # t1 failed once, so there are three rounds of work: t1+t2, t1 again, t3.
    assert sum(1 for r in ran if r == ["work"]) == 3
    plan_step = s.step("plan")
    assert {"plan", "batches", "memories_used"} <= set(plan_step.changed)
    assert not steps[0].can_replay and not steps[-1].can_replay
    assert plan_step.can_replay and not any(x.paused for x in steps)

    state = s.service.state_at(s.task.id, plan_step.checkpoint_id)
    assert state["values"]["plan"]["subtasks"][0]["id"] == "t1"
    assert "plan" not in state["previous"]


def test_unchanged_replay_reaches_the_same_answer_without_touching_the_original(s) -> None:
    before = s.tasks.state(s.task.id)
    record, state = s.replay(s.step("work", 1))  # after t1's retry

    assert state["status"] == "done" and state["final_output"] == "FINAL ANSWER"
    assert state["replay_of"] == s.task.id and state["task_id"] == record.id
    assert s.tasks.store.get_task(record.id).status == "done"
    assert s.tasks.state(s.task.id) == before
    # Only t3 was left to run; earlier results were carried over, not redone.
    assert len(s.llm.specialist_prompts["t3"]) == 2
    assert len(s.llm.specialist_prompts["t1"]) == 2
    # Replays never teach long-term memory.
    assert s.memory.remembered == [s.task.id]


def test_replay_with_an_edited_subtask_output_flows_downstream(s) -> None:
    record, state = s.replay(s.step("work", 1), outputs={"t1": "EDITED t1"})
    assert state["accepted"]["t1"] == "human"
    assert "EDITED t1" in s.llm.specialist_prompts["t3"][-1]
    assert record.changes == {"outputs": {"t1": "EDITED t1"}}


def test_replay_after_synthesis_with_new_output_writes_the_answer_again(s) -> None:
    s.llm.synthesis_prompt = None
    _, state = s.replay(s.step("synthesize"), outputs={"t3": "NEW t3"})
    assert state["status"] == "done" and "NEW t3" in s.llm.synthesis_prompt


def test_replay_with_a_new_plan_restarts_execution(s) -> None:
    plan = ExecutionPlan.model_validate({
        "goal": "g", "reasoning": "r", "confidence": 0.9,
        "subtasks": [{"id": "only", "description": "do only", "specialist": "research",
                      "required_inputs": [], "depends_on": [], "expected_output": "text",
                      "complexity": "low"}]})
    record, state = s.replay(s.step("work", 2), plan=plan)
    assert record.resumed_after == "plan"
    assert set(state["accepted"]) == {"only"} and state["status"] == "done"
    assert "t1" not in state.get("results", {})


def test_replay_with_guidance_reaches_later_specialists(s) -> None:
    s.replay(s.step("work", 1), guidance="cite sources")
    assert "cite sources" in s.llm.specialist_prompts["t3"][-1]


def test_replay_from_the_start_with_a_new_task_reuses_the_original_memories(s) -> None:
    assert s.memory.retrieved == ["research things"]
    _, state = s.replay(s.step("intake"), task="research other things")
    assert state["task"] == "research other things" and state["status"] == "done"
    assert s.memory.retrieved == ["research things"]  # not retrieved again
    assert state["memories_used"] == [MEMORY]

    _, state = s.replay(s.step("intake"), reuse_memories=False)
    assert s.memory.retrieved[-1] == "research things" and len(s.memory.retrieved) == 2


def test_replay_can_ask_for_human_review(s) -> None:
    record, _ = s.replay(s.step("plan"), require_review=True)
    assert s.tasks.store.get_task(record.id).status == "awaiting_review"
    [review] = s.tasks.store.list_reviews(status="pending", task_id=record.id)
    assert review.id.startswith(record.id) and review.level == "approve_plan"


def test_invalid_replays_are_refused(s) -> None:
    with pytest.raises(ReplayError, match="before planning"):
        s.replay(s.step("intake"), plan=make_plan())
    with pytest.raises(ReplayError, match="unknown subtasks"):
        s.replay(s.step("work"), outputs={"nope": "x"})
    with pytest.raises(ReplayError, match="nothing runs after"):
        s.replay(s.step("deliver"))
    with pytest.raises(KeyError):
        s.service.replay(s.task.id, "no-such-checkpoint", ReplayChanges())


def test_compare_counts_only_what_ran_after_the_fork(s) -> None:
    record, _ = s.replay(s.step("work", 1), outputs={"t1": "EDITED t1"})
    c = s.service.compare(record.id)
    original_calls = sum(1 for x in s.spans.trace(s.task.id) if x.kind == "llm")
    assert c["replay"]["changes"]["outputs"] == {"t1": "EDITED t1"}
    # After the fork only t3 and synthesis ran, on both sides.
    assert c["original"]["llm_calls"] == c["replayed"]["llm_calls"] == 3
    assert original_calls > 3  # planning and the earlier work aren't counted
    assert c["original"]["agent_seconds"] >= 0 and c["replayed"]["status"] == "done"
    assert c["final_output"]["similarity"] == 1.0 and c["final_output"]["diff"] == []
    t1 = next(r for r in c["subtasks"] if r["id"] == "t1")
    assert t1["original"]["accepted_by"] == "reviewer" and t1["replay"]["accepted_by"] == "human"
    assert t1["output_similarity"] < 1.0


@pytest.fixture
def client(s):
    app.dependency_overrides[tasks_api.get_service] = lambda: s.tasks
    app.dependency_overrides[obs_api.get_span_store] = lambda: s.spans
    app.dependency_overrides[replay_api.get_replay_service] = lambda: s.service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_replay_api(s, client) -> None:
    steps = client.get(f"/tasks/{s.task.id}/steps").json()
    assert steps[0]["ran"] == ["input"]
    work = next(x for x in steps if x["ran"] == ["work"])
    assert client.get(f"/tasks/{s.task.id}/steps/{work['checkpoint_id']}").json()["values"]
    assert client.get("/tasks/nope/steps").status_code == 404
    assert client.get(f"/tasks/{s.task.id}/steps/nope").status_code == 404

    res = client.post(f"/tasks/{s.task.id}/replay", json={
        "checkpoint_id": work["checkpoint_id"], "changes": {"guidance": "be brief"}})
    assert res.status_code == 202
    replay_id = res.json()["replay_id"]
    s.save_spans()
    [listed] = client.get(f"/tasks/{s.task.id}/replays").json()
    assert listed["id"] == replay_id and listed["status"] == "done"
    assert client.get(f"/replays/{replay_id}/compare").json()["replayed"]["status"] == "done"
    assert client.get("/replays/nope/compare").status_code == 404

    bad = client.post(f"/tasks/{s.task.id}/replay", json={
        "checkpoint_id": steps[-1]["checkpoint_id"], "changes": {}})
    assert bad.status_code == 422

    page = client.get("/dashboard/replay")
    assert page.status_code == 200 and "innerHTML" not in page.text


def test_fork_time_comes_from_the_checkpoint(s) -> None:
    step = s.step("work", 1)
    record, _ = s.replay(step)
    assert record.fork_time == step.created_at <= time.time()
