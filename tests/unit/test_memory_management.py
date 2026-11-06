import time
import uuid

import chromadb
import pytest
from fastapi.testclient import TestClient

from agentorch.api import memory as memory_api
from agentorch.api.main import app
from agentorch.llm import StructuredResult
from agentorch.memory import scoring
from agentorch.memory.consolidator import MemoryConsolidator, MergedMemory, group_similar
from agentorch.memory.long_term import LongTermMemory
from agentorch.memory.long_term_types import MemoryItem, MemoryKind, StoredMemory
from tests.unit.test_memory import BagOfWordsEmbedder
from tests.unit.test_workflow import USAGE

DAY = scoring.DAY


def memory(kind=MemoryKind.FACT, uses=0, idle_days=0.0, now=1e9) -> StoredMemory:
    return StoredMemory(id="m", user_id="u", kind=kind, text="t", task_id="x",
                        created_at=now - idle_days * DAY, access_count=uses,
                        last_accessed_at=now - idle_days * DAY)


@pytest.fixture
def store() -> LongTermMemory:
    return LongTermMemory(chromadb.EphemeralClient(), BagOfWordsEmbedder(),
                          collection=f"t-{uuid.uuid4().hex}")


class MergeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def structured(self, role, system, prompt, schema):
        assert role == "memory" and schema is MergedMemory
        self.prompts.append(prompt)
        return StructuredResult(MergedMemory(text="use the github releases api for any repo"),
                                USAGE)


# --- scoring --------------------------------------------------------------

def test_usage_raises_importance_with_diminishing_returns() -> None:
    now = 1e9
    i0, i5, i50 = (scoring.importance(memory(uses=n, now=now), now) for n in (0, 5, 50))
    assert i0 < i5 < i50 <= 1.0
    assert (i5 - i0) > (i50 - i5) / 9  # most of the gain comes early


def test_importance_halves_per_half_life_and_facts_decay_fastest() -> None:
    now = 1e9
    fresh = scoring.importance(memory(MemoryKind.FACT, now=now), now)
    month = scoring.importance(memory(MemoryKind.FACT, idle_days=30, now=now), now)
    assert month == pytest.approx(fresh / 2)
    pref = scoring.importance(memory(MemoryKind.PREFERENCE, idle_days=30, now=now), now)
    assert pref > month


def test_ranking_lets_a_proven_memory_beat_a_slightly_closer_one() -> None:
    assert scoring.rank_score(0.50, 0.9) > scoring.rank_score(0.45, 0.2)
    assert scoring.rank_score(0.10, 0.2) > scoring.rank_score(0.50, 0.9)  # but not a far one


# --- store: ranking and expiry ----------------------------------------------

def test_search_prefers_important_memories_among_similar_ones(store) -> None:
    now = time.time()
    store.add("u", "t1", [MemoryItem(kind=MemoryKind.REQUEST,
                                     text="qdrant release version lookup")])
    store.add("u", "t2", [MemoryItem(kind=MemoryKind.PITFALL,
                                     text="qdrant release version lookup pitfall")],
              overrides={"access_count": 20})
    found = store.search("u", "qdrant release version lookup", max_distance=0.9, now=now)
    assert found[0].kind == MemoryKind.PITFALL


def test_expire_removes_only_decayed_memories(store) -> None:
    long_ago = time.time() - 400 * DAY
    store.add("u", "t1", [MemoryItem(kind=MemoryKind.FACT, text="stale fact")],
              overrides={"last_accessed_at": long_ago, "created_at": long_ago})
    store.add("u", "t2", [MemoryItem(kind=MemoryKind.PREFERENCE, text="likes tables")])
    expired = store.expire("u")
    assert [m.text for m in expired] == ["stale fact"]
    assert [m.text for m in store.list("u")] == ["likes tables"]


# --- consolidation ----------------------------------------------------------

def test_group_similar_only_groups_same_kind_neighbours(store) -> None:
    store.add("u", "t1", [
        MemoryItem(kind=MemoryKind.TOOL_USAGE, text="github releases api works for qdrant"),
        MemoryItem(kind=MemoryKind.TOOL_USAGE, text="github releases api works for weaviate"),
        MemoryItem(kind=MemoryKind.FACT, text="github releases api works for qdrant"),
        MemoryItem(kind=MemoryKind.PREFERENCE, text="short answers with sources"),
    ])
    groups = group_similar(store.with_embeddings("u"), max_distance=0.3)
    assert len(groups) == 1
    assert {m.kind for m in groups[0]} == {MemoryKind.TOOL_USAGE} and len(groups[0]) == 2


def test_consolidate_merges_and_keeps_usage(store) -> None:
    store.add("u", "t1", [MemoryItem(kind=MemoryKind.TOOL_USAGE,
                                     text="github releases api works for qdrant")],
              overrides={"access_count": 3})
    store.add("u", "t2", [MemoryItem(kind=MemoryKind.TOOL_USAGE,
                                     text="github releases api works for weaviate")],
              overrides={"access_count": 4})
    store.add("u", "t3", [MemoryItem(kind=MemoryKind.PREFERENCE, text="short answers")])

    llm = MergeLLM()
    report = MemoryConsolidator(llm, store).consolidate("u")
    assert report.groups_merged == 1 and (report.memories_before, report.memories_after) == (3, 2)
    merged = next(m for m in store.list("u") if m.kind == MemoryKind.TOOL_USAGE)
    assert merged.text == "use the github releases api for any repo"
    assert merged.access_count == 7 and merged.consolidated_from == 2
    assert 'used="4"' in llm.prompts[0]


def test_dry_run_changes_nothing(store) -> None:
    store.add("u", "t1", [MemoryItem(kind=MemoryKind.FACT, text="a b c d"),
                          MemoryItem(kind=MemoryKind.FACT, text="a b c d e")])
    report = MemoryConsolidator(MergeLLM(), store).consolidate("u", dry_run=True)
    assert report.groups_merged == 1 and len(store.list("u")) == 2


# --- API --------------------------------------------------------------------

@pytest.fixture
def client(store):
    app.dependency_overrides[memory_api.get_store] = lambda: store
    app.dependency_overrides[memory_api.get_llm] = MergeLLM
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_lists_users_and_memories(client, store) -> None:
    store.add("alice", "t1", [MemoryItem(kind=MemoryKind.FACT, text="one"),
                              MemoryItem(kind=MemoryKind.PREFERENCE, text="two")])
    assert client.get("/users").json() == [{"user_id": "alice", "memories": 2}]
    memories = client.get("/users/alice/memories").json()
    assert [m["kind"] for m in memories] == ["preference", "fact"]  # most important first
    assert all(0 < m["importance"] <= 1 for m in memories)


def test_api_deletes_one_memory_and_a_whole_user(client, store) -> None:
    ids = store.add("alice", "t1", [MemoryItem(kind=MemoryKind.FACT, text="one"),
                                    MemoryItem(kind=MemoryKind.FACT, text="two")])
    store.add("bob", "t2", [MemoryItem(kind=MemoryKind.FACT, text="three")])
    assert client.delete(f"/memories/{ids[0]}").json() == {"deleted": 1}
    assert client.delete("/users/alice/memories").json() == {"deleted": 1}
    assert client.get("/users").json() == [{"user_id": "bob", "memories": 1}]


def test_api_consolidate_and_expire(client, store) -> None:
    store.add("u", "t1", [MemoryItem(kind=MemoryKind.FACT, text="a b c d"),
                          MemoryItem(kind=MemoryKind.FACT, text="a b c d e")])
    r = client.post("/users/u/memories/consolidate").json()
    assert (r["memories_before"], r["memories_after"]) == (2, 1)
    assert client.post("/users/u/memories/expire").json() == {"expired": 0, "texts": []}


def test_dashboard_is_served(client) -> None:
    page = client.get("/dashboard/memories")
    assert page.status_code == 200 and "What AgentOrch remembers" in page.text
    # Memory text must never be inserted as HTML.
    assert "innerHTML" not in page.text
