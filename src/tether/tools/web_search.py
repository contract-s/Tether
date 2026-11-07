"""Web search backed by Claude's server-side web search tool (no extra API key needed)."""

from __future__ import annotations

from typing import Any

import anthropic
from pydantic import BaseModel, Field

from tether.config import settings
from tether.observability.cost import WEB_SEARCH_PER_REQUEST
from tether.observability.tracing import record_llm_usage, span


class WebSearchInput(BaseModel):
    query: str = Field(description="What to search for")
    max_results: int = Field(description="Number of sources to return, 1-10")


class Source(BaseModel):
    title: str
    url: str


class WebSearchOutput(BaseModel):
    summary: str = Field(description="Findings, with facts tied to sources")
    sources: list[Source]


SYSTEM = (
    "You are a search tool. Search the web for the query and report what you found: "
    "concise factual findings with dates where relevant, no opinions or recommendations. "
    "If results conflict or are thin, say so."
)


class ClaudeWebSearch:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model or settings.specialist_model

    @property
    def client(self) -> anthropic.Anthropic:
        # Created on first use so building the registry doesn't require an API key.
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        return self._client

    def search(self, args: WebSearchInput) -> WebSearchOutput:
        limit = max(1, min(args.max_results, 10))
        messages: list[Any] = [{"role": "user", "content": args.query}]
        # Server tools can pause a long turn; resend to let the server continue.
        for _ in range(3):
            with span("llm.web_search", "llm", **{
                "tether.role": "web_search", "llm.prompt": args.query,
            }) as s:
                response = self._create(messages)
                searches = getattr(getattr(response.usage, "server_tool_use", None),
                                   "web_search_requests", 0) or 0
                record_llm_usage(
                    s, response.model, response.usage.input_tokens,
                    response.usage.output_tokens,
                    response.usage.cache_read_input_tokens or 0,
                    extra_cost=searches * WEB_SEARCH_PER_REQUEST,
                )
                s.set_attribute("llm.web_searches", searches)
            if response.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": response.content}]
        return self._parse(response, limit)

    def _create(self, messages: list[Any]) -> Any:
        return self.client.messages.create(
            model=self._model,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "low"},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
            messages=messages,
        )

    @staticmethod
    def _parse(response: Any, limit: int) -> WebSearchOutput:
        if response.stop_reason == "refusal":
            raise RuntimeError("search request was refused")

        summary_parts: list[str] = []
        sources: dict[str, Source] = {}
        for block in response.content:
            if block.type == "text":
                summary_parts.append(block.text)
            elif block.type == "web_search_tool_result":
                # A list means results; an object means a search error.
                if isinstance(block.content, list):
                    for r in block.content:
                        sources.setdefault(r.url, Source(title=r.title, url=r.url))
                else:
                    code = getattr(block.content, "error_code", "unknown")
                    summary_parts.append(f"[search error: {code}]")
        return WebSearchOutput(
            summary="".join(summary_parts).strip(),
            sources=list(sources.values())[:limit],
        )
