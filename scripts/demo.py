"""Showcase run: research -> extraction -> analysis -> summary, through the HTTP API.

    make stack                         # or: make up && make api
    uv run python scripts/demo.py      # watch it run; approve reviews in the dashboard
    uv run python scripts/demo.py --review   # you approve the plan and the final answer
    uv run python scripts/demo.py --auto     # approve anything that asks, from here

Run it twice with the same --user: the second plan is made with what the first taught
long-term memory (shown under "memory").
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from typing import Any

import httpx

TASK = (
    "Compare what it costs to keep data in object storage on AWS S3 (Standard), Google "
    "Cloud Storage (Standard) and Azure Blob Storage (Hot, LRS). "
    "1) Use web search to find each provider's current list price for the first storage "
    "tier, per GB-month, in a US East region, and their internet egress price per GB for "
    "the first paid tier, citing the providers' pricing pages as sources. "
    "2) Extract the figures, with their source URLs, into a CSV file named "
    "storage_prices.csv in the workspace. "
    "3) With code, compute each provider's monthly bill for 50 TB stored and 5 TB of "
    "egress per month (1 TB = 1024 GB), and how the bills compare. "
    "4) Write a short summary for an engineering manager: a table of the results, the "
    "cheapest option, caveats (free tiers, request and retrieval fees not included), and "
    "the sources."
)

print = functools.partial(print, flush=True)  # noqa: A001 - show progress when piped too

STATUS_ICON = {"not_started": "·", "in_progress": "…", "done": "✓", "abandoned": "✗"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--user", default="demo")
    parser.add_argument("--task", default=TASK, help="run something else instead")
    parser.add_argument("--review", action="store_true",
                        help="ask for your approval of the plan and the final answer")
    parser.add_argument("--auto", action="store_true",
                        help="approve/accept every review automatically")
    args = parser.parse_args()

    api = httpx.Client(base_url=args.api, timeout=30)
    try:
        api.get("/health").raise_for_status()
    except httpx.HTTPError as e:
        print(f"API not reachable at {args.api} ({e}). Start it with `make stack`.")
        return 1

    task = api.post("/tasks", json={"task": args.task, "user_id": args.user,
                                     "require_review": args.review}).json()
    tid = task["id"]
    print(f"task {tid}\n  {args.api}/dashboard/reviews\n")

    shown: dict[str, Any] = {"plan": False, "progress": {}, "reviews": set(), "status": None}
    started = time.time()
    while True:
        t = api.get(f"/tasks/{tid}").json()
        show(api, t, shown, args)
        if t["status"] in ("done", "failed", "rejected"):
            break
        time.sleep(2)

    print(f"\n=== {t['status']} in {time.time() - started:.0f}s")
    if t["error"]:
        print(f"error: {t['error']}")
    if t["final_output"]:
        print(f"\n{t['final_output']}\n")
    report(api, tid, args)
    return 0 if t["status"] == "done" else 1


def show(api: httpx.Client, t: dict[str, Any], shown: dict[str, Any], args) -> None:
    if t["plan"] and not shown["plan"]:
        shown["plan"] = True
        plan = t["plan"]
        print(f"[plan] confidence {plan['confidence']:.2f}: {plan['goal']}")
        for st in plan["subtasks"]:
            deps = f" (after {', '.join(st['depends_on'])})" if st["depends_on"] else ""
            print(f"   {st['id']} {st['specialist']}: {st['description'][:100]}{deps}")
        print()

    for sid, p in t["progress"].items():
        key = (p["status"], p["attempts"])
        if shown["progress"].get(sid) != key:
            shown["progress"][sid] = key
            if p["status"] != "not_started":
                attempt = f" attempt {p['attempts']}" if p["attempts"] > 1 else ""
                print(f"[{STATUS_ICON[p['status']]}] {sid} {p['status']}{attempt}")

    for n in t["notifications"]:
        key = ("note", n.get("id"))
        if key not in shown["reviews"]:
            shown["reviews"].add(key)
            print(f"[notify] {n.get('reason', '')[:140]}")

    if t["pending_review"] and t["pending_review"] not in shown["reviews"]:
        shown["reviews"].add(t["pending_review"])
        review = api.get(f"/reviews/{t['pending_review']}").json()
        print(f"\n*** human needed ({review['level']}): {review['reason']}")
        if args.auto:
            options = review["request"].get("options", [])
            action = next(a for a in ("approve", "accept") if a in options)
            api.post(f"/reviews/{review['id']}/decision",
                     json={"action": action, "reviewer": "demo --auto"}).raise_for_status()
            print(f"    --auto: {action}\n")
        else:
            print(f"    decide at {args.api}/dashboard/reviews\n")


def report(api: httpx.Client, tid: str, args) -> None:
    cost = api.get(f"/tasks/{tid}/cost").json()
    print(f"cost ${cost['total_cost_usd']:.3f}: {cost['llm_calls']} model calls, "
          f"{cost['input_tokens']:,} in / {cost['output_tokens']:,} out tokens, "
          f"{cost['tool_calls']} tool calls, {cost['escalations']} human reviews")
    for role, r in sorted(cost["by_role"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        print(f"   {role:<11} ${r['cost_usd']:.3f}  ({r['calls']} calls)")
    if cost["tools"]:
        print("   tools: " + ", ".join(
            f"{name} ×{t['calls']}" + (f" ({t['errors']} errors)" if t["errors"] else "")
            for name, t in cost["tools"].items()))

    memories = api.get(f"/users/{args.user}/memories").json()
    print(f"\nmemory: {len(memories)} long-term memories for user {args.user!r}")
    for m in memories[:5]:
        print(f"   [{m['kind']}] {m['text'][:110]}")

    print(f"\ntrace   {args.api}/dashboard/traces?task={tid}")
    print(f"replay  {args.api}/dashboard/replay?task={tid}")
    print(f"memory  {args.api}/dashboard/memories")


if __name__ == "__main__":
    sys.exit(main())
