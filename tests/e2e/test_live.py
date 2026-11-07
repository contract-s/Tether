"""With the real model: do the agents behave? (TETHER_LIVE=1, costs roughly $0.75.)

Assertions are about behaviour that should hold on any good run, not exact wording.
"""

from __future__ import annotations

import pytest

from scripts.demo import TASK as DEMO_TASK
from tether.agents.planning import validate_plan
from tether.agents.reviewer import Reviewer
from tether.agents.schemas import Specialist, Subtask, SubtaskResult
from tether.agents.specialist import SpecialistAgent
from tether.agents.supervisor import Supervisor
from tether.config import settings
from tether.llm import AnthropicClient
from tether.memory.extractor import MemoryExtractor
from tether.memory.long_term_types import MemoryKind
from tether.tools.defaults import build_default_registry

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def llm() -> AnthropicClient:
    return AnthropicClient()


def subtask(id: str, description: str, expected: str, specialist=Specialist.ANALYSIS) -> Subtask:
    return Subtask(id=id, description=description, specialist=specialist, required_inputs=[],
                   depends_on=[], expected_output=expected, complexity="low")


def test_the_showcase_task_gets_a_valid_plan_in_the_right_order(llm) -> None:
    plan = Supervisor(llm).plan(DEMO_TASK).plan
    validate_plan(plan)  # ids unique, dependencies exist, no cycles

    by_id = {st.id: st for st in plan.subtasks}
    kinds = {st.specialist for st in plan.subtasks}
    assert Specialist.RESEARCH in kinds and Specialist.WRITING in kinds
    assert kinds & {Specialist.ANALYSIS, Specialist.CODE}

    def upstream(sid: str) -> set[str]:
        deps = set(by_id[sid].depends_on)
        return deps.union(*(upstream(d) for d in deps)) if deps else set()

    # The write-up comes after the analysis, which comes after the research.
    writing = [st for st in plan.subtasks if st.specialist == Specialist.WRITING]
    analysis = [st for st in plan.subtasks
                if st.specialist in (Specialist.ANALYSIS, Specialist.CODE)]
    research = {st.id for st in plan.subtasks if st.specialist == Specialist.RESEARCH}
    assert any({a.id for a in analysis} & upstream(w.id) for w in writing)
    assert all(research & upstream(a.id) for a in analysis)


def test_a_specialist_uses_the_code_sandbox_for_a_calculation(llm) -> None:
    registry = build_default_registry()
    run = SpecialistAgent(Specialist.CODE, llm, registry).run(subtask(
        "t1", "Compute the sum of the squares of the primes below 10,000 by running code.",
        "The number, and the code you ran", Specialist.CODE))

    calls = [c for c in run.tool_calls if c.tool == "code_exec"]
    assert calls and any(c.success and c.output["exit_code"] == 0 for c in calls)
    assert "37546387960" in run.result.output.replace(",", "")


def test_the_reviewer_catches_a_wrong_answer_and_passes_a_right_one(llm) -> None:
    reviewer = Reviewer(llm)
    st = subtask("t1", "What is 1,234 x 5,678? Show the working.",
                 "The product, with a line of working")
    bad = reviewer.review(st, SubtaskResult(
        subtask_id="t1", succeeded=True,
        output="1,234 x 5,678 = 1,234 x 5,000 + 1,234 x 678 = 6,170,000 + 836,652 = 7,106,652"))
    good = reviewer.review(st, SubtaskResult(
        subtask_id="t1", succeeded=True,
        output="1,234 x 5,678 = 1,234 x 5,000 + 1,234 x 678 = 6,170,000 + 836,652 = 7,006,652"))

    assert not bad.passed and bad.verdict.blocking_issues
    assert good.passed


def test_a_sensitive_action_escalates_to_a_human(make_stack, user, llm, tmp_path,
                                                 monkeypatch) -> None:
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path))
    (tmp_path / "greeting.txt").write_text("hi\n")
    stack = make_stack(llm, registry=build_default_registry())

    tid = stack.start("Replace the contents of the existing workspace file greeting.txt "
                      "with the single line 'hello, world'.", user)
    while True:
        task = stack.wait(tid, "awaiting_review", "done", "failed", timeout=240)
        assert task["status"] == "awaiting_review", task
        review = stack.client.get(f"/reviews/{task['pending_review']}").json()
        if review["level"] != "approve_plan":
            break
        # A cautious plan review may come first; that's not what this test is about.
        stack.client.post(f"/reviews/{review['id']}/decision", json={"action": "approve"})
        stack.wait(tid, "running", "awaiting_review", "done", timeout=30)

    assert review["trigger"] == "sensitive_action" and review["level"] == "approve_action"
    assert review["request"]["proposed"]["tool"] == "file_write"
    assert (tmp_path / "greeting.txt").read_text() == "hi\n"  # nothing written yet

    stack.client.post(f"/reviews/{review['id']}/decision",
                      json={"action": "approve", "reviewer": "e2e"}).raise_for_status()
    assert stack.wait(tid, timeout=240)["status"] == "done"
    assert "hello, world" in (tmp_path / "greeting.txt").read_text()


def test_a_remembered_preference_shapes_the_next_plan(llm, long_term, user) -> None:
    transcript = """<request>Summarise last week's incident reports for the team</request>
<outcome>
t1 [research]: gathered the three incident reports. accepted by reviewer
t2 [writing]: wrote the summary. accepted by human after edits
Human note on the final answer: "Always start my summaries with a one-line TL;DR, then
the details. I skim everything on my phone first."
</outcome>
<final_output>TL;DR: three incidents, all resolved, one needs a follow-up ...</final_output>"""
    items, _ = MemoryExtractor(llm).extract(transcript)
    assert any(i.kind == MemoryKind.PREFERENCE and "tl;dr" in i.text.lower() for i in items)
    long_term.add(user, "past-task", items)

    task = "Summarise this month's customer feedback themes for the team"
    found = long_term.search(user, task)
    assert any("tl;dr" in m.text.lower() for m in found)

    with_memory = Supervisor(llm).plan(task, [m.as_prompt_line() for m in found]).plan
    without = Supervisor(llm).plan(task).plan
    assert "tl;dr" in with_memory.model_dump_json().lower()
    assert "tl;dr" not in without.model_dump_json().lower()
