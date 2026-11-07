"""Celery worker that runs tasks outside the API process.

    celery -A agentorch.workers worker --loglevel=info

With TASK_RUNNER=celery the API only records tasks and decisions; it sends each piece of
work here as (task_id, payload) over Redis. Messages are acknowledged only after the work
finishes, so if a worker dies mid-run, the message goes back on the queue and another
worker picks the run up from its last checkpoint (TaskService.run is safe to repeat).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from celery import Celery

from agentorch.config import settings

app = Celery("agentorch", broker=settings.redis_url)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_acks_late=True,  # ack after the run, not on receipt
    task_reject_on_worker_lost=True,  # a killed worker's message is redelivered
    worker_prefetch_multiplier=1,  # runs are long; don't hoard messages
    broker_connection_retry_on_startup=True,
    # How long an unacknowledged run may take before Redis hands it to another worker.
    broker_transport_options={"visibility_timeout": settings.task_visibility_timeout_seconds},
)


@lru_cache
def _service():
    from agentorch.graph.factory import default_task_service

    return default_task_service()


@app.task(name="agentorch.run")
def run(task_id: str, payload: dict[str, Any]) -> None:
    _service().run(task_id, payload)


def celery_dispatch(task_id: str, payload: dict[str, Any]) -> None:
    run.delay(task_id, payload)
