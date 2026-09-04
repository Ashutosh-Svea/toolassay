from __future__ import annotations

import pytest

from toolassay.judge import parse_verdict


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"pass": true, "reason": "fine"}', (True, "fine")),
        ('Sure. {"pass": false, "reason": "vague"} That is my view.', (False, "vague")),
        ('{"pass": true}', (True, "")),
    ],
)
def test_parse_verdict_accepts_booleans(text: str, expected: tuple[bool, str]) -> None:
    assert parse_verdict(text) == expected


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('{"pass": "false", "reason": "x"}', "not a JSON boolean"),
        ('{"pass": "true"}', "not a JSON boolean"),
        ('{"pass": 1}', "not a JSON boolean"),
        ('{"verdict": true}', "lacked a 'pass' field"),
        ("looks good to me", "not JSON"),
        ('{"pass": tru', "not JSON"),
        ("[true]", "not JSON"),
        ('{"pass": tru}', "not valid JSON"),
    ],
)
def test_parse_verdict_fails_closed(text: str, fragment: str) -> None:
    passed, reason = parse_verdict(text)
    assert passed is False
    assert fragment in reason
