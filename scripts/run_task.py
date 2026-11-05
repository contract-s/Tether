"""Run a task through the full workflow, pausing for you when it escalates.

    uv run python scripts/run_task.py "Compare Qdrant and Weaviate for a small RAG app"
    uv run python scripts/run_task.py --auto "..."   # approve/accept escalations automatically
"""

import argparse
import json
import uuid

from langgraph.types import Command

from agentorch.graph.factory import default_workflow

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="+")
parser.add_argument("--auto", action="store_true", help="resolve escalations without asking")
parser.add_argument("--user", default="default", help="whose long-term memory to use")
args = parser.parse_args()

graph = default_workflow()
config = {"configurable": {"thread_id": uuid.uuid4().hex}}


def stream(inp) -> None:
    for update in graph.stream(inp, config, stream_mode="updates"):
        for node, change in update.items():
            if node == "__interrupt__" or not change:
                continue
            if node == "plan" and change.get("memories_used"):
                print(f"[memory] using {len(change['memories_used'])} past memories:")
                for m in change["memories_used"]:
                    print(f"   ({m.distance:.2f}) {m.as_prompt_line()[:110]}")
            if node == "plan" and "plan" in change:
                p = change["plan"]
                print(f"[plan] confidence={p.confidence:.2f}, batches={change['batches']}")
                for st in p.subtasks:
                    print(f"   {st.id} ({st.specialist.value}): {st.description[:90]}")
            elif node == "work":
                sid = next(iter(change["results"]))
                v = change["reviews"][sid]
                verdict = "accepted" if sid in change.get("accepted", {}) else "rejected"
                print(f"[work] {sid} attempt {change['attempts'][sid]}: {verdict} "
                      f"(quality {v.quality_score:.2f}, {len(change['tool_calls'])} tool calls)")
            elif node in ("human_plan", "human_subtask"):
                print(f"[{node}] -> {change['escalations'][0].resolution}")
            else:
                print(f"[{node}]")


def ask(pause: dict) -> dict:
    print(f"\n*** Human needed: {pause['reason']}")
    if pause["type"] == "approve_plan":
        print(json.dumps(pause["plan"], indent=2)[:3000])
        default = "approve"
    else:
        print(f"Reviewer feedback: {pause['reviewer_feedback']}")
        default = "accept"
    if args.auto:
        print(f"--auto: {default}")
        return {"action": default}
    action = input(f"Action {pause['options']} [{default}]: ").strip() or default
    decision = {"action": action}
    if action == "take_over":
        decision["output"] = input("Your output: ")
    elif action in ("retry", "abort", "reject"):
        decision["note"] = input("Note (optional): ")
    return decision


stream({"task": " ".join(args.task), "user_id": args.user})
while (state := graph.get_state(config)).next:
    pause = state.tasks[0].interrupts[0].value
    stream(Command(resume=ask(pause)))

final = graph.get_state(config).values
print(f"\n=== status: {final['status']}")
if final.get("error"):
    print(f"error: {final['error']}")
if final.get("final_output"):
    print(f"\n{final['final_output']}")
print(f"\nMemories saved: {len(final.get('memories_saved', []))}"
      + (f", memory errors: {final['memory_errors']}" if final.get("memory_errors") else ""))
usage = final.get("usage", [])
tokens_in = sum(u.input_tokens for u in usage)
tokens_out = sum(u.output_tokens for u in usage)
print(f"\nLLM calls: {len(usage)}, tokens in/out: {tokens_in}/{tokens_out}, "
      f"tool calls: {len(final.get('tool_calls', []))}, "
      f"pending approvals: {len(final.get('pending_approvals', []))}")
