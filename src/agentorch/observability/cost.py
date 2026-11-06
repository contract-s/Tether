"""What an LLM call costs, in US dollars.

Prices are per million tokens on the Claude API (first-party). Anthropic reports
`input_tokens` excluding cache reads, so cache reads are priced separately.
"""

from __future__ import annotations

from dataclasses import dataclass

PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.00, 50.00, 0.25),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20),
    "claude-opus-5": Price(5.00, 25.00, 0.50),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20),
    "claude-sonnet-5": Price(2.00, 10.00, 0.20),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01),
}

WEB_SEARCH_PER_REQUEST = 10.00 / 1000  # $10 per 1,000 searches


def llm_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
) -> float | None:
    """Cost of one call, or None for a model we don't have a price for."""
    price = PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price.input + output_tokens * price.output
            + cache_read_tokens * price.cache_read) / PER_MILLION
