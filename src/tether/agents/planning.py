"""Plan validation: turns a supervisor plan into ordered batches of runnable subtasks."""

from __future__ import annotations

from tether.agents.schemas import ExecutionPlan, Subtask


class InvalidPlanError(ValueError):
    pass


def validate_plan(plan: ExecutionPlan) -> list[list[Subtask]]:
    """Check the plan is a valid DAG and return it as execution batches.

    Each batch holds subtasks whose dependencies are all in earlier batches,
    so subtasks within a batch can run in parallel.
    """
    if not plan.subtasks:
        raise InvalidPlanError("plan has no subtasks")
    if not 0.0 <= plan.confidence <= 1.0:
        raise InvalidPlanError(f"confidence {plan.confidence} is outside 0.0-1.0")

    by_id: dict[str, Subtask] = {}
    for st in plan.subtasks:
        if st.id in by_id:
            raise InvalidPlanError(f"duplicate subtask id {st.id!r}")
        by_id[st.id] = st

    for st in plan.subtasks:
        for dep in st.depends_on:
            if dep not in by_id:
                raise InvalidPlanError(f"subtask {st.id!r} depends on unknown {dep!r}")
            if dep == st.id:
                raise InvalidPlanError(f"subtask {st.id!r} depends on itself")

    # Kahn's algorithm, level by level.
    remaining = {st.id: set(st.depends_on) for st in plan.subtasks}
    batches: list[list[Subtask]] = []
    while remaining:
        ready = [sid for sid, deps in remaining.items() if not deps]
        if not ready:
            raise InvalidPlanError(f"dependency cycle among {sorted(remaining)}")
        # Keep the supervisor's ordering within a batch.
        order = {st.id: i for i, st in enumerate(plan.subtasks)}
        ready.sort(key=order.__getitem__)
        batches.append([by_id[sid] for sid in ready])
        for sid in ready:
            del remaining[sid]
        for deps in remaining.values():
            deps.difference_update(ready)
    return batches
