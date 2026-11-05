"""Short-term working memory: one Redis namespace per task, cleared when the task ends.

Holds the current plan, completed subtask outputs, an error log, and a shared notes
board that specialists running in parallel use to pass findings to each other.
Everything also carries a TTL so an abandoned task can't leak memory forever.
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis

DEFAULT_TTL_SECONDS = 24 * 3600
MAX_NOTES_RETURNED = 50


class WorkingMemory:
    def __init__(
        self, client: redis.Redis, task_id: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
    ) -> None:
        self._r = client
        self.task_id = task_id
        self._ttl = ttl_seconds
        prefix = f"agentorch:wm:{task_id}"
        self._state_key = f"{prefix}:state"
        self._results_key = f"{prefix}:results"
        self._notes_key = f"{prefix}:notes"
        self._errors_key = f"{prefix}:errors"
        self._keys = [self._state_key, self._results_key, self._notes_key, self._errors_key]

    # --- writes ----------------------------------------------------------------

    def set_plan(self, plan: dict[str, Any]) -> None:
        self._write(lambda p: p.hset(self._state_key, "plan", json.dumps(plan)))

    def record_result(self, subtask_id: str, output: str) -> None:
        self._write(lambda p: p.hset(self._results_key, subtask_id, output))

    def log_error(self, source: str, message: str) -> None:
        entry = json.dumps({"source": source, "message": message, "at": time.time()})
        self._write(lambda p: p.rpush(self._errors_key, entry))

    def add_note(self, author: str, text: str) -> int:
        """Append a note; returns how many notes the task now has."""
        entry = json.dumps({"author": author, "text": text, "at": time.time()})
        return int(self._write(lambda p: p.rpush(self._notes_key, entry))[0])

    # --- reads -----------------------------------------------------------------

    def plan(self) -> dict[str, Any] | None:
        raw = self._r.hget(self._state_key, "plan")
        return json.loads(raw) if raw else None

    def results(self) -> dict[str, str]:
        return {_s(k): _s(v) for k, v in self._r.hgetall(self._results_key).items()}

    def notes(self, limit: int = MAX_NOTES_RETURNED) -> list[dict[str, Any]]:
        return [json.loads(n) for n in self._r.lrange(self._notes_key, -limit, -1)]

    def errors(self) -> list[dict[str, Any]]:
        return [json.loads(e) for e in self._r.lrange(self._errors_key, 0, -1)]

    def snapshot(self) -> dict[str, Any]:
        return {"plan": self.plan(), "results": self.results(),
                "notes": self.notes(), "errors": self.errors()}

    def clear(self) -> None:
        self._r.delete(*self._keys)

    # --- internals -------------------------------------------------------------

    def _write(self, op) -> list[Any]:
        with self._r.pipeline() as p:
            op(p)
            for key in self._keys:
                p.expire(key, self._ttl)
            return p.execute()


def _s(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
