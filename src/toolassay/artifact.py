"""The JSON artifact a run produces, and helpers to read and write it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from toolassay import __version__
from toolassay.core import ConfigError, ToolDefinition, Usage
from toolassay.judge import JudgeVerdict

SCHEMA_VERSION = 1


class ToolCallRecord(BaseModel):
    turn: int
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str
    is_error: bool = False
    latency_ms: float
    result_truncated: bool = False


class CaseResult(BaseModel):
    id: str
    prompt: str
    tags: list[str] = Field(default_factory=list)
    expected_tools: list[str] | None = None
    called_tools: list[str] = Field(default_factory=list)
    passed: bool
    completed: bool
    tool_selection_correct: bool | None = None
    args_correct: bool | None = None
    answer_correct: bool | None = None
    turns: int
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float | None = None
    latency_ms: float
    stop_reason: str | None = None
    final_answer: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    judge: JudgeVerdict | None = None
    error: str | None = None


class RunSummary(BaseModel):
    cases: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    turns: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float | None
    latency_ms: float
    judge_cost_usd: float | None = None


class RunArtifact(BaseModel):
    schema_version: int = SCHEMA_VERSION
    toolassay_version: str = __version__
    created_at: datetime
    label: str | None = None
    provider: str
    model: str
    effort: str | None = None
    max_tokens: int
    strict_tools: bool
    server: dict[str, Any]
    cases_path: str
    judge_model: str | None = None
    tools: list[ToolDefinition]
    summary: RunSummary
    cases: list[CaseResult]


def summarize(cases: list[CaseResult], judge_cost_usd: float | None) -> RunSummary:
    usage = Usage()
    cost: float | None = 0.0
    for case in cases:
        usage = usage + case.usage
        if case.cost_usd is None:
            cost = None
        elif cost is not None:
            cost += case.cost_usd
    passed = sum(1 for c in cases if c.passed)
    errored = sum(1 for c in cases if c.error is not None)
    return RunSummary(
        cases=len(cases),
        passed=passed,
        failed=len(cases) - passed,
        errored=errored,
        pass_rate=(passed / len(cases)) if cases else 0.0,
        turns=sum(c.turns for c in cases),
        input_tokens=usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=cost,
        latency_ms=sum(c.latency_ms for c in cases),
        judge_cost_usd=judge_cost_usd,
    )


def now() -> datetime:
    return datetime.now(tz=UTC)


def write_artifact(artifact: RunArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_artifact(path: Path) -> RunArtifact:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read run artifact {path}: {exc}") from exc
    try:
        artifact = RunArtifact.model_validate_json(text)
    except ValidationError as exc:
        raise ConfigError(f"{path} is not a toolassay run artifact:\n{exc}") from exc
    if artifact.schema_version != SCHEMA_VERSION:
        raise ConfigError(
            f"{path} has artifact schema version {artifact.schema_version}; "
            f"this toolassay reads version {SCHEMA_VERSION}"
        )
    return artifact
