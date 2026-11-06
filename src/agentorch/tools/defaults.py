"""The standard tool set and which specialists may use each tool."""

from __future__ import annotations

from agentorch.agents.schemas import Specialist as S
from agentorch.config import settings
from agentorch.tools import code_exec, db_query, files, http, notes, web_search
from agentorch.tools.registry import RateLimit, Tool, ToolCallLog, ToolRegistry


def build_default_registry(log: ToolCallLog | None = None) -> ToolRegistry:
    registry = ToolRegistry(log=log)
    workspace = files.Workspace(settings.workspace_dir)
    sandbox = code_exec.DockerSandbox(settings.sandbox_image, settings.sandbox_timeout_seconds)
    db = db_query.ReadOnlyDatabase(url=settings.database_url)
    http_client = http.HttpClient()
    searcher = web_search.ClaudeWebSearch()

    registry.register(Tool(
        name="web_search",
        description="Search the web. Returns a factual summary and the source URLs.",
        input_model=web_search.WebSearchInput,
        output_model=web_search.WebSearchOutput,
        handler=searcher.search,
        allowed_specialists=frozenset({S.RESEARCH}),
        rate_limit=RateLimit(calls=20, per_seconds=60),
    ))
    registry.register(Tool(
        name="http_request",
        description="Make an HTTP request to a public URL. Non-GET requests need approval.",
        input_model=http.HttpRequestInput,
        output_model=http.HttpRequestOutput,
        handler=http_client.request,
        allowed_specialists=frozenset({S.RESEARCH, S.CODE}),
        rate_limit=RateLimit(calls=30, per_seconds=60),
        is_sensitive=http.is_sensitive,
    ))
    registry.register(Tool(
        name="file_read",
        description="Read a UTF-8 text file from the shared task workspace.",
        input_model=files.FileReadInput,
        output_model=files.FileReadOutput,
        handler=workspace.read,
        allowed_specialists=frozenset(S),
    ))
    registry.register(Tool(
        name="file_write",
        description="Write a UTF-8 text file to the shared task workspace.",
        input_model=files.FileWriteInput,
        output_model=files.FileWriteOutput,
        handler=workspace.write,
        allowed_specialists=frozenset({S.ANALYSIS, S.WRITING, S.CODE}),
        # Overwriting destroys the existing file, so it needs a human's approval.
        is_sensitive=lambda args: args.overwrite,
    ))
    registry.register(Tool(
        name="code_exec",
        description=(
            "Run Python 3.11 in an isolated sandbox with no network or file access. "
            "Returns stdout, stderr and the exit code."
        ),
        input_model=code_exec.CodeExecInput,
        output_model=code_exec.CodeExecOutput,
        handler=sandbox.run,
        allowed_specialists=frozenset({S.ANALYSIS, S.CODE}),
        rate_limit=RateLimit(calls=10, per_seconds=60),
    ))
    registry.register(Tool(
        name="notes_write",
        description=(
            "Post a finding to the shared notes board for this task, so other specialists "
            "(including ones running in parallel) can use it."
        ),
        input_model=notes.NotesWriteInput,
        output_model=notes.NotesWriteOutput,
        handler=notes.write_note,
        allowed_specialists=frozenset(S),
        takes_context=True,
    ))
    registry.register(Tool(
        name="notes_read",
        description="Read the shared notes board for this task. Check it before searching.",
        input_model=notes.NotesReadInput,
        output_model=notes.NotesReadOutput,
        handler=notes.read_notes,
        allowed_specialists=frozenset(S),
        takes_context=True,
    ))
    registry.register(Tool(
        name="db_query",
        description="Run a read-only SQL SELECT against the application database.",
        input_model=db_query.DbQueryInput,
        output_model=db_query.DbQueryOutput,
        handler=db.query,
        allowed_specialists=frozenset({S.ANALYSIS}),
        rate_limit=RateLimit(calls=30, per_seconds=60),
    ))
    return registry
