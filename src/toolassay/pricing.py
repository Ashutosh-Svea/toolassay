"""Cost estimation from token counts.

Prices are USD per million tokens. The built-in table is a snapshot; pass a price file to
override or extend it. Unknown models get no cost estimate rather than a wrong one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from toolassay.core import ConfigError, Usage

PRICES_SNAPSHOT_DATE = "2026-06-24"


class ModelPrice(BaseModel):
    """Per-million-token prices for one model."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_write_multiplier: float = Field(default=1.25, ge=0)
    cache_read_multiplier: float = Field(default=0.1, ge=0)


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(input=10.0, output=50.0),
    "claude-fable-5": ModelPrice(input=10.0, output=50.0),
    "claude-opus-5": ModelPrice(input=5.0, output=25.0),
    "claude-opus-4-8": ModelPrice(input=5.0, output=25.0),
    "claude-opus-4-7": ModelPrice(input=5.0, output=25.0),
    "claude-opus-4-6": ModelPrice(input=5.0, output=25.0),
    "claude-sonnet-5": ModelPrice(input=2.0, output=10.0),
    "claude-sonnet-4-6": ModelPrice(input=3.0, output=15.0),
    "claude-haiku-4-5": ModelPrice(input=1.0, output=5.0),
}


class PriceBook:
    """Looks up prices by model id, falling back to the longest matching prefix."""

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None) -> None:
        self._prices: dict[str, ModelPrice] = dict(DEFAULT_PRICES)
        if prices:
            self._prices.update(prices)

    @classmethod
    def from_file(cls, path: Path) -> PriceBook:
        """Load a YAML mapping of model id to {input, output} and merge it over the defaults."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read price file {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"price file {path} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"price file {path} must map model ids to prices")
        parsed: dict[str, ModelPrice] = {}
        for model, entry in raw.items():
            try:
                parsed[str(model)] = ModelPrice.model_validate(entry)
            except ValidationError as exc:
                raise ConfigError(f"price file {path}, model {model!r}: {exc}") from exc
        return cls(parsed)

    def lookup(self, model: str) -> ModelPrice | None:
        if model in self._prices:
            return self._prices[model]
        candidates = [known for known in self._prices if model.startswith(known + "-")]
        if not candidates:
            return None
        return self._prices[max(candidates, key=len)]

    def estimate(self, model: str, usage: Usage) -> float | None:
        """Estimated USD cost for the usage, or None when the model has no price."""
        price = self.lookup(model)
        if price is None:
            return None
        per_token_in = price.input / 1_000_000
        per_token_out = price.output / 1_000_000
        cost = usage.input_tokens * per_token_in
        cost += usage.output_tokens * per_token_out
        cost += usage.cache_creation_input_tokens * per_token_in * price.cache_write_multiplier
        cost += usage.cache_read_input_tokens * per_token_in * price.cache_read_multiplier
        return cost
