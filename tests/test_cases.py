from __future__ import annotations

from pathlib import Path

import pytest

from toolassay.cases import CaseFile, ToolMatch, load_cases
from toolassay.core import ConfigError


def test_example_case_file_loads(examples_dir: Path) -> None:
    case_file = load_cases(examples_dir / "cases.yaml")
    assert len(case_file.cases) == 7
    assert case_file.system is not None
    assert case_file.max_turns == 8
    by_id = {case.id: case for case in case_file.cases}
    assert by_id["shelf-location"].expected_tools == ["find_book", "get_book"]
    assert by_id["stock-count"].tool_match is ToolMatch.CONTAINS
    assert by_id["unknown-title"].judge is not None


def test_single_string_fields_become_lists() -> None:
    case_file = CaseFile.model_validate(
        {"cases": [{"id": "a", "prompt": "p", "expected_tools": "x", "expected_substrings": "y"}]}
    )
    case = case_file.cases[0]
    assert case.expected_tools == ["x"]
    assert case.expected_substrings == ["y"]


def test_top_level_list_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text("- id: a\n  prompt: hello\n", encoding="utf-8")
    assert load_cases(path).cases[0].id == "a"


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text("cases:\n- id: a\n  prompt: x\n- id: a\n  prompt: y\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate case id"):
        load_cases(path)


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text("cases:\n- id: a\n  prompt: x\n  expected_tool: y\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected_tool"):
        load_cases(path)


def test_invalid_yaml_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text("cases: [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_cases(path)
    with pytest.raises(ConfigError, match="cannot read"):
        load_cases(tmp_path / "missing.yaml")


def test_select_filters_and_validates_ids() -> None:
    case_file = CaseFile.model_validate(
        {"cases": [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}]}
    )
    assert [c.id for c in case_file.select(["b"]).cases] == ["b"]
    assert case_file.select([]) is case_file
    with pytest.raises(ConfigError, match="unknown case id"):
        case_file.select(["zzz"])
