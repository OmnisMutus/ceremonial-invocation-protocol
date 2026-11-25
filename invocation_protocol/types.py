"""Core types, enums, validated configuration, and exceptions for invocation_protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


class Arm(str, Enum):
    """Evaluation matrix arms comparing plain baseline, naive summoning, and full protocol."""

    PLAIN_BASELINE = "arm_0_baseline"
    NAIVE_SUMMONING = "arm_1_naive_summoning"
    PROTOCOL_NEUTRAL = "arm_2_protocol_neutral"
    PROTOCOL_FORM = "arm_3_protocol_form"
    CORRUPTED_CORRESPONDENCE = "arm_3b_corrupted_correspondence"


class LifecycleStage(str, Enum):
    """Fixed 7-step runtime lifecycle stages."""

    BANISH = "banish"
    OPEN_HIERARCHY = "open_hierarchy"
    ASSUME_FORM = "assume_form"
    RUN_CHARGE = "run_charge"
    TEST_SPIRIT = "test_spirit"
    DEPART = "depart"
    RECORD = "record"


class GateStatus(str, Enum):
    """Deterministic outcome of an individual test_spirit() gate."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class Verdict(str, Enum):
    """Fail-closed aggregate verdict from test_spirit()."""

    PASSED = "PASSED"
    BANISHED = "BANISHED"


class EpistemicTag(str, Enum):
    """Allowed epistemic grounding tags for claims."""

    A = "A"
    B = "B"
    C = "C"


class InvocationError(Exception):
    """Base exception for protocol, backend, and verification failures."""


class ValidationError(InvocationError):
    """Raised when configuration, form specs, or task inputs violate contracts."""


class LifecycleOrderError(InvocationError):
    """Raised when lifecycle methods are invoked out of order or without depart()."""


class ToolBudgetError(InvocationError):
    """Raised when a run exceeds max_tool_calls."""


class ToolScopeError(InvocationError):
    """Raised when a model requests a tool outside allowed_tools."""


class SpiritBanishedError(InvocationError):
    """Raised fail-closed when test_spirit() produces Verdict.BANISHED."""

    def __init__(self, report: VerificationReport) -> None:
        self.report = report
        failed = [
            f"{gate.gate_name}={gate.status.value}"
            for gate in report.gates
            if gate.status != GateStatus.PASS
        ]
        summary = ", ".join(failed) if failed else "unknown failure"
        super().__init__(f"Output BANISHED by test_spirit(): {summary}")


def require_int(
    value: Any,
    name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Validate that value is a non-bool int within optional inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer (not bool), got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")
    return value


def require_float(
    value: Any,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
) -> float:
    """Validate that value is a finite non-bool float/int within bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number (not bool), got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    if min_value is not None:
        if strict_min and numeric <= min_value:
            raise ValidationError(f"{name} must be > {min_value}, got {numeric}")
        if not strict_min and numeric < min_value:
            raise ValidationError(f"{name} must be >= {min_value}, got {numeric}")
    if max_value is not None and numeric > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {numeric}")
    return numeric


def require_non_empty_str(value: Any, name: str) -> str:
    """Validate that value is a string containing non-whitespace characters."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value


def require_str_tuple(value: Any, name: str, *, allow_empty_items: bool = False) -> tuple[str, ...]:
    """Validate a sequence (not str/bytes) of strings and return an immutable tuple."""
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{name} must be a list or tuple of strings")
    items: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValidationError(f"{name}[{idx}] must be a string")
        if not allow_empty_items and not item.strip():
            raise ValidationError(f"{name}[{idx}] must be non-empty")
        items.append(item)
    return tuple(items)


@dataclass(frozen=True)
class CorrespondenceSlice:
    """Declarative correspondence mapping for a FormSpec."""

    domain: str
    element: str
    planet_or_sphere: str
    symbols: tuple[str, ...]
    operational_focus: str


@dataclass(frozen=True)
class FormExemplar:
    """Input/output exemplar attached to a FormSpec."""

    input: str
    output: str


@dataclass(frozen=True)
class FormSpec:
    """Declarative symbolic FormSpec loaded from forms/*.json."""

    schema_version: str
    form_id: str
    canonical_id: str | None
    arm_type: str
    display_name: str
    tradition: str
    epithets: tuple[str, ...]
    correspondence_slice: CorrespondenceSlice
    forbidden_attributions: tuple[str, ...]
    banned_cliches: tuple[str, ...]
    exemplars: tuple[FormExemplar, ...]


@dataclass(frozen=True)
class TaskSpec:
    """Fixed evaluation task definition with immutable sources and tool fixtures."""

    task_id: str
    category: str
    prompt: str
    sources: dict[str, str]
    allowed_tools: tuple[str, ...]
    tool_fixtures: dict[str, dict[str, str]]
    expected_answer_substrings: tuple[str, ...]
    forbidden_answer_substrings: tuple[str, ...]
    expected_answer_any: tuple[tuple[str, ...], ...] = ()
    """Groups of acceptable alternatives; each group needs at least one match (e.g. refusal wordings)."""


@dataclass(frozen=True)
class RunConfig:
    """Deterministic runtime hyperparameters and hard execution budgets."""

    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512
    context_length: int = 4096
    max_turns: int = 3
    max_tool_calls: int = 2
    timeout_seconds: float = 30.0
    fail_closed: bool = True

    def __post_init__(self) -> None:
        temp = require_float(self.temperature, "temperature", min_value=0.0, max_value=2.0)
        object.__setattr__(self, "temperature", temp)
        require_int(self.seed, "seed", min_value=0)
        require_int(self.max_tokens, "max_tokens", min_value=1)
        require_int(self.context_length, "context_length", min_value=256)
        require_int(self.max_turns, "max_turns", min_value=1)
        require_int(self.max_tool_calls, "max_tool_calls", min_value=0)
        timeout = require_float(self.timeout_seconds, "timeout_seconds", min_value=0.0, strict_min=True)
        object.__setattr__(self, "timeout_seconds", timeout)
        if not isinstance(self.fail_closed, bool):
            raise ValidationError("fail_closed must be a bool")


@dataclass(frozen=True)
class ToolCallRecord:
    """Immutable log record of a permitted or denied sandbox tool call."""

    tool_name: str
    args: dict[str, Any]
    allowed: bool
    output: str | None
    error: str | None


@dataclass(frozen=True)
class ModelToolRequest:
    """Tool invocation requested by a model backend during run_charge()."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response returned by any ModelBackend."""

    content: str
    tool_requests: tuple[ModelToolRequest, ...] = ()
    model_name: str = "unknown"
    raw_payload: dict[str, Any] | None = None
    thinking: str = ""


@dataclass(frozen=True)
class GateResult:
    """Result from a single deterministic gate inside test_spirit()."""

    gate_name: str
    status: GateStatus
    reasons: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    """Aggregate report produced by test_spirit()."""

    verdict: Verdict
    gates: tuple[GateResult, ...]
    parsed_output: dict[str, Any] | None

    def raise_for_status(self) -> None:
        """Raise SpiritBanishedError when verdict is BANISHED."""
        if self.verdict != Verdict.PASSED:
            raise SpiritBanishedError(self)


@dataclass(frozen=True)
class DepartureReceipt:
    """Proof of zero-persistence ephemeral teardown from depart()."""

    cleared_messages: int
    cleared_form_id: str | None
    persisted_persona_bytes: int
    clean: bool


@dataclass(frozen=True)
class RunResult:
    """Complete result object returned by InvocationRunner.invoke()."""

    trace_id: str
    arm: Arm
    form_id: str | None
    task_id: str
    lifecycle_stages: tuple[LifecycleStage, ...]
    verdict: Verdict
    accepted_output: dict[str, Any] | None
    raw_output: str
    tool_log: tuple[ToolCallRecord, ...]
    report: VerificationReport
    departure: DepartureReceipt
    trace_hash: str
    thinking_chars: int = 0

    def raise_for_status(self) -> None:
        """Fail closed if verification banished the response."""
        self.report.raise_for_status()
