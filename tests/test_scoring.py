from __future__ import annotations

from toolassay.cases import Case, ToolMatch
from toolassay.scoring import check_args, check_substrings, match_tools, score_case

from .support import call


def test_exact_match_requires_same_sequence() -> None:
    assert match_tools(["a", "b"], ["a", "b"], ToolMatch.EXACT)
    assert not match_tools(["a", "b"], ["a", "x", "b"], ToolMatch.EXACT)
    assert not match_tools(["a"], ["a", "a"], ToolMatch.EXACT)
    assert match_tools([], [], ToolMatch.EXACT)


def test_contains_match_allows_extra_calls() -> None:
    assert match_tools(["a", "b"], ["x", "a", "y", "b", "z"], ToolMatch.CONTAINS)
    assert not match_tools(["b", "a"], ["a", "b"], ToolMatch.CONTAINS)
    assert match_tools([], ["a"], ToolMatch.CONTAINS)


def test_args_checked_against_first_call_to_each_tool() -> None:
    calls = [
        call("reserve_book", "c1", book_id="BK-1", pickup_date="12 Sep"),
        call("reserve_book", "c2", book_id="BK-1", pickup_date="2026-09-12"),
    ]
    ok, failures = check_args({"reserve_book": {"pickup_date": "2026-09-12"}}, calls)
    assert ok is False
    assert failures == ["reserve_book: argument 'pickup_date' was '12 Sep', expected '2026-09-12'"]


def test_args_missing_call_and_missing_key() -> None:
    ok, failures = check_args({"get_book": {"book_id": "BK-1"}}, [])
    assert ok is False
    assert "never called" in failures[0]
    ok, failures = check_args({"get_book": {"book_id": "BK-1"}}, [call("get_book", other=1)])
    assert ok is False
    assert "missing" in failures[0]


def test_args_compare_after_json_normalisation() -> None:
    ok, _ = check_args({"t": {"n": 1.0, "items": (1, 2)}}, [call("t", n=1, items=[1, 2])])
    assert ok is True
    assert check_args({}, []) == (None, [])


def test_substrings_are_case_insensitive() -> None:
    assert check_substrings(["salt AND ember"], "The cheapest is Salt and Ember.") == (True, [])
    ok, failures = check_substrings(["P3"], "Shelf unknown")
    assert ok is False
    assert failures == ["final answer does not contain 'P3'"]
    assert check_substrings([], "anything") == (None, [])


def test_score_case_combines_checks() -> None:
    case = Case(
        id="c",
        prompt="p",
        expected_tools=["find_book"],
        expected_args={"find_book": {"query": "x"}},
        expected_substrings=["found"],
    )
    score = score_case(case, [call("find_book", query="x")], "Found it")
    assert score.tool_selection_correct is True
    assert score.args_correct is True
    assert score.answer_correct is True
    assert score.failures == []

    score = score_case(case, [call("list_inventory")], "nothing")
    assert score.tool_selection_correct is False
    assert score.args_correct is False
    assert score.answer_correct is False
    assert len(score.failures) == 3


def test_empty_expected_tools_means_no_tool_calls() -> None:
    case = Case(id="c", prompt="p", expected_tools=[])
    assert score_case(case, [], "answer").tool_selection_correct is True
    assert score_case(case, [call("x")], "answer").tool_selection_correct is False
    unchecked = Case(id="c", prompt="p")
    assert score_case(unchecked, [call("x")], "answer").tool_selection_correct is None
