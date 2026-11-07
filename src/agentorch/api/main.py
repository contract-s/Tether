import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from agentorch.api.memory import router as memory_router
from agentorch.api.observability import router as observability_router
from agentorch.api.replay import router as replay_router
from agentorch.api.tasks import router as tasks_router
from agentorch.config import settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the services once before serving, so concurrent first requests don't race."""
    from agentorch.api.observability import get_span_store
    from agentorch.api.replay import get_replay_service
    from agentorch.api.tasks import get_service

    try:
        service = get_service()
        get_span_store()
        get_replay_service()
    except Exception:  # noqa: BLE001 - e.g. Postgres not up yet; endpoints retry lazily
        log.exception("could not start services; will retry on first request")
    else:
        # In-process runs die with the process, so pick up any a restart cut short.
        # (With Celery, the broker redelivers them instead.)
        if settings.task_runner == "thread":
            service.recover_stalled()
    yield


app = FastAPI(title="AgentOrch", version="0.1.0", lifespan=lifespan)
app.include_router(memory_router)
app.include_router(tasks_router)
app.include_router(observability_router)
app.include_router(replay_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/dashboard/reviews")
