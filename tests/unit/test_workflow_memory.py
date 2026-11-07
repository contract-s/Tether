"""The workflow's use of working and long-term memory (fake model, fake Redis)."""

import uuid

import chromadb
import fakeredis

from tests.unit.test_memory import BagOfWordsEmbedder
from tests.unit.test_workflow import USAGE, FakeLLM, make_plan
from tether.agents.reviewer import Reviewer
from tether.agents.specialist import SpecialistAgent
from tether.agents.supervisor import Supervisor
from tether.graph.workflow import Agents, build_workflow
from tether.llm import StructuredResult
from tether.memory.extractor import ExtractedMemories, MemoryExtractor
from tether.memory.long_term import LongTermMemory, MemoryItem, MemoryKind
from tether.memory.service import MemoryServices
from tether.memory.working import WorkingMemory
from tether.tools.registry import ToolRegistry


class MemoryAwareLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(make_plan(), approve=lambda sid, n: True)
        self.plan_prompts: list[str] = []
        self.transcripts: list[str] = []

    def structured(self, role, system, prompt, schema):
        if role == "memory":
            self.transcripts.append(prompt)
            return StructuredResult(ExtractedMemories(memories=[MemoryItem(
                kind=MemoryKind.APPROACH,
                text="compare vector databases: research each database in parallel",
            )]), USAGE)
        if role == "supervisor":
            self.plan_prompts.append(prompt)
        return super().structured(role, system, prompt, schema)


def make(llm, long_term=None, redis_client=None):
    redis_client = redis_client or fakeredis.FakeRedis()
    long_term = long_term or LongTermMemory(
        chromadb.EphemeralClient(), BagOfWordsEmbedder(), collection=f"t-{uuid.uuid4().hex}")
    registry = ToolRegistry()
    memory = MemoryServices(
        long_term=long_term, extractor=MemoryExtractor(llm),
        working=lambda tid: WorkingMemory(redis_client, tid),
    )
    agents = Agents(Supervisor(llm), Reviewer(llm, threshold=0.7),
                    lambda s: SpecialistAgent(s, llm, registry), memory=memory)
    return build_workflow(agents, plan_confidence_threshold=0.7), redis_client


def run(graph, task, user="alice"):
    tid = uuid.uuid4().hex
    return tid, graph.invoke({"task": task, "user_id": user},
                             {"configurable": {"thread_id": tid}})


def test_second_task_plans_with_memories_from_the_first() -> None:
    llm = MemoryAwareLLM()
    graph, _ = make(llm)
    _, first = run(graph, "compare vector databases qdrant pgvector")
    assert first["status"] == "done" and len(first["memories_saved"]) == 1
    assert "<past_tasks>" not in llm.plan_prompts[0]

    _, second = run(graph, "compare vector databases weaviate milvus")
    assert "research each database in parallel" in llm.plan_prompts[1]
    assert second["memories_used"][0].kind == MemoryKind.APPROACH


def test_memories_are_per_user() -> None:
    llm = MemoryAwareLLM()
    graph, _ = make(llm)
    run(graph, "compare vector databases", user="alice")
    run(graph, "compare vector databases", user="bob")
    assert "<past_tasks>" not in llm.plan_prompts[1]


def test_transcript_covers_outcomes_and_final_output() -> None:
    llm = MemoryAwareLLM()
    graph, _ = make(llm)
    run(graph, "compare vector databases")
    t = llm.transcripts[0]
    assert "<request>" in t and "t3 [research]" in t and "accepted by reviewer" in t
    assert "FINAL ANSWER" in t


def test_working_memory_is_filled_during_the_task_and_cleared_after() -> None:
    llm = MemoryAwareLLM()
    graph, redis_client = make(llm)
    seen: dict[str, dict] = {}
    original = Supervisor.synthesize

    def spy(self, task, plan, outputs):
        # Just before delivery, working memory should hold the plan and all results.
        seen["snap"] = WorkingMemory(redis_client, tid).snapshot()
        return original(self, task, plan, outputs)

    tid = uuid.uuid4().hex
    Supervisor.synthesize = spy
    try:
        graph.invoke({"task": "x"}, {"configurable": {"thread_id": tid}})
    finally:
        Supervisor.synthesize = original
    assert seen["snap"]["plan"]["goal"] == "g"
    assert set(seen["snap"]["results"]) == {"t1", "t2", "t3"}
    assert redis_client.keys(f"tether:wm:{tid}:*") == []


def test_memory_failures_never_fail_the_task() -> None:
    class BrokenMemory(LongTermMemory):
        def __init__(self) -> None:
            pass

        def search(self, *a, **kw):
            raise ConnectionError("chroma down")

        def add(self, *a, **kw):
            raise ConnectionError("chroma down")

    llm = MemoryAwareLLM()
    graph, _ = make(llm, long_term=BrokenMemory())
    _, result = run(graph, "x")
    assert result["status"] == "done"
    assert [e.split(":")[0] for e in result["memory_errors"]] == ["retrieve", "remember"]
