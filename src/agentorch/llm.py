"""Model routing layer.

Every agent talks to an `LLMClient` instead of a provider SDK directly, so adding
OpenAI later means writing one more client class and a routing rule, not touching agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from agentorch.config import settings

Role = Literal["supervisor", "specialist", "reviewer"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ModelRoute:
    provider: Literal["anthropic"]
    model: str
    effort: Effort


@dataclass
class LLMUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0


@dataclass
class StructuredResult(Generic[T]):  # noqa: UP046 - PEP 695 syntax needs 3.12
    output: T
    usage: LLMUsage


class LLMRefusalError(RuntimeError):
    """The model (and any fallback) declined the request."""


class LLMClient(Protocol):
    def structured(
        self, role: Role, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]: ...


def route_for(role: Role) -> ModelRoute:
    """Which model and effort each agent role uses. Claude-only for now."""
    routes: dict[Role, ModelRoute] = {
        "supervisor": ModelRoute("anthropic", settings.supervisor_model, "high"),
        "specialist": ModelRoute("anthropic", settings.specialist_model, "medium"),
        "reviewer": ModelRoute("anthropic", settings.reviewer_model, "medium"),
    }
    return routes[role]


class AnthropicClient:
    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        # Zero-arg client reads ANTHROPIC_API_KEY from the environment.
        self._client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key or None
        )

    def structured(
        self, role: Role, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        route = route_for(role)
        response = self._client.beta.messages.parse(
            model=route.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": route.effort},
            # On a safety refusal, the API retries on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"{route.model} refused the request")
        if response.parsed_output is None:
            raise ValueError(f"{route.model} returned no parsable output "
                             f"(stop_reason={response.stop_reason})")
        usage = LLMUsage(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
        )
        return StructuredResult(output=response.parsed_output, usage=usage)
