from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "bookshop"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES
