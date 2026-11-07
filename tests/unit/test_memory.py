import hashlib
import math
import uuid

import chromadb
import fakeredis
import pytest

from agentorch.agents.schemas import Specialist
from agentorch.memory.long_term import LongTermMemory, MemoryItem, MemoryKind
from agentorch.memory.working import WorkingMemory
from agentorch.tools.defaults import build_default_registry
from agentorch.tools.registry import ToolContext


class BagOfWordsEmbedder:
    """Deterministic stand-in for a real embedding model: hashed word counts."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * 64
            for word in text.lower().split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


@pytest.fixture
def ltm() -> LongTermMemory:
    # Ephemeral clients share one in-process store, so give each test its own collection.
    name = f"test-{uuid.uuid4().hex}"
    return LongTermMemory(chromadb.EphemeralClient(), BagOfWordsEmbedder(), collection=name)


# --- working memory -------------------------------------------------------

def test_working_memory_round_trip_and_clear(redis_client) -> None:
    wm = WorkingMemory(redis_client, "task-1")
    wm.set_plan({"goal": "g"})
    wm.record_result("t1", "out 1")
    wm.log_error("t2#1", "bad source")
    assert wm.add_note("research:t1", "Qdrant is at v1.20.0") == 1

    snap = wm.snapshot()
    assert snap["plan"] == {"goal": "g"} and snap["results"] == {"t1": "out 1"}
    assert snap["notes"][0]["text"] == "Qdrant is at v1.20.0"
    assert snap["errors"][0]["source"] == "t2#1"
    assert 0 < redis_client.ttl("agentorch:wm:task-1:notes") <= 24 * 3600

    wm.clear()
    assert WorkingMemory(redis_client, "task-1").snapshot() == {
        "plan": None, "results": {}, "notes": [], "errors": []}


def test_working_memory_is_scoped_per_task(redis_client) -> None:
    WorkingMemory(redis_client, "a").add_note("x", "only in a")
    assert WorkingMemory(redis_client, "b").notes() == []


def test_notes_tools_share_findings_between_specialists(redis_client) -> None:
    registry = build_default_registry()
    wm = WorkingMemory(redis_client, "task-1")
    research = ToolContext(caller=Specialist.RESEARCH, task_id="task-1", subtask_id="t1",
                           working_memory=wm)
    analysis = ToolContext(caller=Specialist.ANALYSIS, task_id="task-1", subtask_id="t2",
                           working_memory=wm)

    registry.invoke("notes_write", Specialist.RESEARCH, {"text": "pgvector is v0.8.7"},
                    context=research)
    rec = registry.invoke("notes_read", Specialist.ANALYSIS, {}, context=analysis)
    assert rec.output["notes"] == [{"author": "research:t1", "text": "pgvector is v0.8.7"}]


def test_notes_tool_fails_cleanly_without_working_memory() -> None:
    rec = build_default_registry().invoke("notes_read", Specialist.RESEARCH, {})
    assert not rec.success and "no working memory" in rec.error


# --- long-term memory -----------------------------------------------------

def test_search_returns_similar_memories_for_that_user_only(ltm) -> None:
    ltm.add("alice", "t1", [
        MemoryItem(kind=MemoryKind.APPROACH,
                   text="compare vector databases research then analysis"),
        MemoryItem(kind=MemoryKind.PREFERENCE, text="keep answers short with a table"),
    ])
    ltm.add("bob", "t2", [MemoryItem(kind=MemoryKind.FACT, text="compare vector databases bob")])

    found = ltm.search("alice", "compare vector databases", max_distance=0.9)
    assert found and all(m.user_id == "alice" for m in found)
    assert found[0].kind == MemoryKind.APPROACH


def test_retrieval_counts_toward_importance(ltm) -> None:
    ltm.add("alice", "t1", [MemoryItem(kind=MemoryKind.FACT, text="qdrant uses hnsw indexes")])
    ltm.search("alice", "qdrant hnsw", max_distance=0.9)
    ltm.search("alice", "qdrant hnsw", max_distance=0.9)
    assert ltm.list("alice")[0].access_count == 2


def test_unrelated_queries_are_filtered_by_distance(ltm) -> None:
    ltm.add("alice", "t1", [MemoryItem(kind=MemoryKind.FACT, text="qdrant uses hnsw indexes")])
    assert ltm.search("alice", "banana bread recipe", max_distance=0.5) == []


def test_delete_user_forgets_everything(ltm) -> None:
    ltm.add("alice", "t1", [MemoryItem(kind=MemoryKind.FACT, text="a"),
                            MemoryItem(kind=MemoryKind.FACT, text="b")])
    ltm.add("bob", "t2", [MemoryItem(kind=MemoryKind.FACT, text="c")])
    assert ltm.delete_user("alice") == 2
    assert ltm.list("alice") == [] and len(ltm.list("bob")) == 1


def test_preferences_are_found_from_further_away_than_task_memories() -> None:
    class Fixed:
        """'task' points one way; the memories sit at cosine distance 0.78 from it."""

        def embed(self, texts):
            far = [0.22, math.sqrt(1 - 0.22**2)]
            return [[1.0, 0.0] if t == "task" else far for t in texts]

    store = LongTermMemory(chromadb.EphemeralClient(), Fixed(),
                           collection=f"t-{uuid.uuid4().hex}")
    store.add("u", "old", [MemoryItem(kind=MemoryKind.PREFERENCE, text="TL;DR first"),
                           MemoryItem(kind=MemoryKind.APPROACH, text="research in parallel")])
    found = store.search("u", "task")
    assert [m.kind for m in found] == [MemoryKind.PREFERENCE]
    assert abs(found[0].distance - 0.78) < 0.01
