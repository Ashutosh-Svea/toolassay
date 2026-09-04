from __future__ import annotations

from pathlib import Path

import pytest

from toolassay.core import ConfigError, Usage
from toolassay.pricing import ModelPrice, PriceBook


def test_default_prices_for_opus_5() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert PriceBook().estimate("claude-opus-5", usage) == pytest.approx(30.0)


def test_cache_tokens_use_multipliers() -> None:
    usage = Usage(cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    assert PriceBook().estimate("claude-opus-5", usage) == pytest.approx(5.0 * 1.25 + 5.0 * 0.1)


def test_prefix_match_and_unknown_model() -> None:
    book = PriceBook()
    assert book.lookup("claude-opus-5-20260101") is book.lookup("claude-opus-5")
    assert book.lookup("claude-opus-50") is None
    assert book.estimate("gpt-x", Usage(input_tokens=5)) is None


def test_price_file_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text("claude-opus-5: {input: 1, output: 2}\nmy-model: {input: 0.5, output: 1}\n")
    book = PriceBook.from_file(path)
    assert book.lookup("claude-opus-5") == ModelPrice(input=1, output=2)
    assert book.estimate("my-model", Usage(input_tokens=2_000_000)) == pytest.approx(1.0)


def test_price_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text("- not a mapping\n")
    with pytest.raises(ConfigError, match="must map"):
        PriceBook.from_file(path)
    path.write_text("m: {input: -1, output: 2}\n")
    with pytest.raises(ConfigError, match="model 'm'"):
        PriceBook.from_file(path)
