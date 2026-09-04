"""Case files: the YAML that says what to ask and what a correct run looks like."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from toolassay.core import ConfigError


class ToolMatch(StrEnum):
    """How the expected tool list is compared with the tools the model called."""

    EXACT = "exact"
    """The called tools must equal the expected list, in order, with nothing extra."""

    CONTAINS = "contains"
    """The expected tools must appear in order; extra calls anywhere are allowed."""


class Case(BaseModel):
    """One evaluation case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_tools: list[str] | None = None
    """Tool names the model should call, in order. Omit to skip the tool-selection check.

    An empty list means the model should answer without calling any tool.
    """
    tool_match: ToolMatch = ToolMatch.EXACT
    expected_args: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per tool name, argument values the first call to that tool must contain."""
    expected_substrings: list[str] = Field(default_factory=list)
    """Strings that must appear in the final answer (case-insensitive)."""
    judge: str | None = None
    """Optional rubric for the LLM judge. Only used when a run enables the judge."""
    system: str | None = None
    max_turns: int | None = Field(default=None, ge=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("expected_tools", mode="before")
    @classmethod
    def _single_tool_as_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("expected_substrings", "tags", mode="before")
    @classmethod
    def _single_string_as_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value


class CaseFile(BaseModel):
    """A case file: shared defaults plus the list of cases."""

    model_config = ConfigDict(extra="forbid")

    system: str | None = None
    max_turns: int = Field(default=10, ge=1)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> CaseFile:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id {case.id!r}")
            seen.add(case.id)
        return self

    def select(self, ids: list[str]) -> CaseFile:
        """Return a copy containing only the cases whose id is listed."""
        if not ids:
            return self
        wanted = set(ids)
        unknown = wanted - {case.id for case in self.cases}
        if unknown:
            raise ConfigError(f"unknown case id(s): {', '.join(sorted(unknown))}")
        return self.model_copy(update={"cases": [c for c in self.cases if c.id in wanted]})


def load_cases(path: Path) -> CaseFile:
    """Load and validate a YAML case file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read case file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"case file {path} is not valid YAML: {exc}") from exc
    if isinstance(raw, list):
        raw = {"cases": raw}
    if not isinstance(raw, dict):
        raise ConfigError(f"case file {path} must be a mapping with a 'cases' list")
    try:
        return CaseFile.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"case file {path} is invalid:\n{exc}") from exc
