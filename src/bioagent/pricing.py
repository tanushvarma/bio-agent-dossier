"""Token prices for cost estimates (USD per 1M tokens: input, output)."""
from __future__ import annotations

PRICES: dict[tuple[str, str], tuple[float, float]] = {
    ("gemini", "gemini-3.5-flash-lite"): (0.30, 2.50),
}


def cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int,
             prices: dict[tuple[str, str], tuple[float, float]] | None = None) -> float | None:
    """Estimated cost, or None when the model has no price entry."""
    p = (prices or PRICES).get((provider, model))
    return None if p is None else (input_tokens * p[0] + output_tokens * p[1]) / 1e6
