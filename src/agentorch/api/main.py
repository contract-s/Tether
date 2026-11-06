from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from agentorch.api.memory import router as memory_router
from agentorch.api.tasks import router as tasks_router

app = FastAPI(title="AgentOrch", version="0.1.0")
app.include_router(memory_router)
app.include_router(tasks_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/dashboard/memories")
