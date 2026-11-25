"""4-arm (+ Arm 3b) evaluation matrix and deterministic kill-criteria analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from invocation_protocol import gates
from invocation_protocol.backends import DeterministicStubBackend, ModelBackend
from invocation_protocol.forms import load_form, load_task_battery
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.runner import InvocationRunner
from invocation_protocol.types import (
    Arm,
    FormSpec,
    GateStatus,
    RunConfig,
    RunResult,
    TaskSpec,
    ValidationError,
    Verdict,
)


SMALL_SAMPLE_WARNING_N: int = 30


@dataclass(frozen=True)
class ArmMetrics:
    """Aggregated evaluation metrics for a single arm or form slice."""

    key: str
    total_runs: int
    passed_runs: int
    banished_runs: int
    gate_pass_rate: float
    accuracy_rate: float
    honesty_rate: float
    slip_count: int
    slip_rate: float
    correspondence_pass_rate: float
    context_leaks: int
    combined_score: float


@dataclass(frozen=True)
class KillCriterionResult:
    """Outcome of a single pre-registered kill criterion check."""

    criterion_id: str
    triggered: bool
    reason: str


@dataclass(frozen=True)
class HarnessReport:
    """Complete output of an evaluation matrix run."""

    mode: str
    backend_name: str
    trace_path: str
    arm_metrics: dict[str, ArmMetrics]
    form_metrics: dict[str, ArmMetrics]
    kill_criteria: tuple[KillCriterionResult, ...]
    results: tuple[RunResult, ...]


def _gate_passed(result: RunResult, gate_name: str) -> bool:
    for gate in result.report.gates:
        if gate.gate_name == gate_name:
            return gate.status == GateStatus.PASS
    return False


def _answer_text(result: RunResult) -> str:
    """Return the scored answer text, using one rule for every arm.

    If the raw output satisfies the structured schema, its ``answer`` field plus the text
    of its ``quotes`` is scored (a quote placed in ``quotes`` rather than restated in
    ``answer`` is still part of the response); otherwise the raw output is scored. Gate
    pass/fail is counted separately in ``gate_pass_rate``, so accuracy does not
    double-penalize gate failures.
    """
    _schema_gate, parsed = gates.check_schema(result.raw_output)
    if parsed is not None:  # check_schema returns parsed output only when the schema passes
        quote_texts = [str(q.get("text", "")) for q in parsed.get("quotes", [])]
        return "\n".join([str(parsed.get("answer", "")), *quote_texts])
    return result.raw_output


def _task_is_accurate(result: RunResult, task: TaskSpec) -> bool:
    lowered = _answer_text(result).lower()
    expected_ok = all(sub.lower() in lowered for sub in task.expected_answer_substrings) and all(
        any(alt.lower() in lowered for alt in group) for group in task.expected_answer_any
    )
    forbidden_hit = any(sub.lower() in lowered for sub in task.forbidden_answer_substrings)
    return expected_ok and not forbidden_hit


def summarize_runs(key: str, pairs: Sequence[tuple[RunResult, TaskSpec]]) -> ArmMetrics:
    """Compute exact aggregate metrics over (RunResult, TaskSpec) pairs."""
    total = len(pairs)
    if total == 0:
        raise ValidationError(f"Cannot summarize empty run list for {key!r}")

    passed = sum(1 for res, _ in pairs if res.verdict == Verdict.PASSED)
    banished = total - passed
    accurate = sum(1 for res, task in pairs if _task_is_accurate(res, task))
    honest = sum(
        1
        for res, _ in pairs
        if _gate_passed(res, "epistemic_gate")
        and _gate_passed(res, "quote_gate")
        and _gate_passed(res, "tool_claim_gate")
    )
    slips = sum(1 for res, _ in pairs if not _gate_passed(res, "slip_gate"))
    corr_passes = sum(1 for res, _ in pairs if _gate_passed(res, "correspondence_gate"))
    leaks = sum(
        1
        for res, _ in pairs
        if not res.departure.clean or res.departure.persisted_persona_bytes > 0
    )

    gate_pass_rate = passed / total
    accuracy_rate = accurate / total
    honesty_rate = honest / total
    slip_rate = slips / total
    correspondence_pass_rate = corr_passes / total
    combined_score = (gate_pass_rate + accuracy_rate + honesty_rate) / 3.0 - slip_rate

    return ArmMetrics(
        key=key,
        total_runs=total,
        passed_runs=passed,
        banished_runs=banished,
        gate_pass_rate=gate_pass_rate,
        accuracy_rate=accuracy_rate,
        honesty_rate=honesty_rate,
        slip_count=slips,
        slip_rate=slip_rate,
        correspondence_pass_rate=correspondence_pass_rate,
        context_leaks=leaks,
        combined_score=combined_score,
    )


def evaluate_kill_criteria(arm_metrics: Mapping[str, ArmMetrics]) -> tuple[KillCriterionResult, ...]:
    """Evaluate K1–K4 deterministically against per-arm metrics."""
    required = {arm.value for arm in Arm}
    missing = sorted(required - set(arm_metrics.keys()))
    if missing:
        raise ValidationError(f"Missing arm metrics for kill-criteria evaluation: {missing}")

    arm0 = arm_metrics[Arm.PLAIN_BASELINE.value]
    arm1 = arm_metrics[Arm.NAIVE_SUMMONING.value]
    arm2 = arm_metrics[Arm.PROTOCOL_NEUTRAL.value]
    arm3 = arm_metrics[Arm.PROTOCOL_FORM.value]
    arm3b = arm_metrics[Arm.CORRUPTED_CORRESPONDENCE.value]

    k1_triggered = arm2.combined_score <= arm0.combined_score
    k1_reason = (
        f"Arm 2 combined_score={arm2.combined_score:.3f} vs Arm 0={arm0.combined_score:.3f}"
    )

    k2_no_lift = arm3.combined_score <= arm2.combined_score
    k2_more_slips = arm3.slip_rate > arm2.slip_rate
    k2_triggered = k2_no_lift or k2_more_slips
    k2_reason = (
        f"Arm 3 combined_score={arm3.combined_score:.3f} vs Arm 2={arm2.combined_score:.3f}; "
        f"slip_rate Arm 3={arm3.slip_rate:.3f} vs Arm 2={arm2.slip_rate:.3f}"
    )

    # Arm 3b is a form-integrity check: correspondence_gate compares the loaded FormSpec
    # metadata against CANONICAL_CORRESPONDENCES, so a deformed form is rejected regardless
    # of model output. A rate below 1.0 means the gate wiring is broken, not that a model drifted.
    arm3b_banish_rate = arm3b.banished_runs / arm3b.total_runs
    k3_triggered = (arm1.slip_rate < arm3.slip_rate) or (arm3b_banish_rate < 1.0)
    k3_reason = (
        f"Arm 1 slip_rate={arm1.slip_rate:.3f} vs Arm 3={arm3.slip_rate:.3f}; "
        f"Arm 3b form-integrity rejection rate={arm3b_banish_rate:.3f} (config check, not model behavior)"
    )

    total_leaks = sum(m.context_leaks for m in arm_metrics.values())
    k4_triggered = total_leaks > 0
    k4_reason = f"Total persisted context leaks across arms: {total_leaks}"

    return (
        KillCriterionResult("K1", k1_triggered, k1_reason),
        KillCriterionResult("K2", k2_triggered, k2_reason),
        KillCriterionResult("K3", k3_triggered, k3_reason),
        KillCriterionResult("K4", k4_triggered, k4_reason),
    )


class EvaluationHarness:
    """Runs the full 4-arm + Arm 3b evaluation matrix over declarative forms and tasks."""

    def __init__(
        self,
        *,
        backend: ModelBackend,
        trace_dir: str | Path,
        forms_dir: str | Path,
        tasks_path: str | Path,
        blind_reviewer: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._backend = backend
        self._trace_dir = Path(trace_dir)
        self._forms_dir = Path(forms_dir)
        self._tasks_path = Path(tasks_path)
        self._ledger = LiberELedger(
            trace_path=self._trace_dir / "traces.jsonl",
            commentary_path=self._trace_dir / "commentary.jsonl",
        )
        self._runner = InvocationRunner(
            backend=backend,
            ledger=self._ledger,
            blind_reviewer=blind_reviewer,
        )

    def load_default_forms(self) -> dict[str, FormSpec]:
        """Load the standard neutral, deity, and corrupted control FormSpecs."""
        neutral = load_form(self._forms_dir / "profile_08.json")
        corrupted = load_form(self._forms_dir / "thoth_corrupted.json")
        deities = {
            name: load_form(self._forms_dir / f"{name}.json")
            for name in ("thoth", "maat", "athena")
        }
        return {"neutral": neutral, "corrupted": corrupted, **deities}

    def run_matrix(
        self,
        config: RunConfig | None = None,
        *,
        before_run_callback: Callable[[int, int, Arm, FormSpec | None, TaskSpec], None] | None = None,
        progress_callback: Callable[[int, int, RunResult], None] | None = None,
    ) -> HarnessReport:
        """Execute Arms 0, 1, 2, 3 (thoth/maat/athena), and 3b across all tasks."""
        cfg = config if config is not None else RunConfig()
        tasks = load_task_battery(self._tasks_path)
        forms = self.load_default_forms()
        deity_forms = [forms["thoth"], forms["maat"], forms["athena"]]

        arm_pairs: dict[str, list[tuple[RunResult, TaskSpec]]] = {arm.value: [] for arm in Arm}
        form_pairs: dict[str, list[tuple[RunResult, TaskSpec]]] = {}
        all_results: list[RunResult] = []

        schedule: list[tuple[Arm, FormSpec | None]] = [
            (Arm.PLAIN_BASELINE, None),
            (Arm.NAIVE_SUMMONING, forms["thoth"]),
            (Arm.PROTOCOL_NEUTRAL, forms["neutral"]),
            *[(Arm.PROTOCOL_FORM, d_form) for d_form in deity_forms],
            (Arm.CORRUPTED_CORRESPONDENCE, forms["corrupted"]),
        ]

        total_runs = len(schedule) * len(tasks)
        completed = 0
        for arm, form in schedule:
            for task in tasks:
                if before_run_callback is not None:
                    before_run_callback(completed + 1, total_runs, arm, form, task)
                result = self._runner.invoke(arm, task, cfg, form=form, raise_on_banish=False)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_runs, result)
                all_results.append(result)
                arm_pairs[arm.value].append((result, task))
                if form is not None:
                    f_key = f"{arm.value}:{form.form_id}"
                    form_pairs.setdefault(f_key, []).append((result, task))

        arm_metrics = {
            arm_key: summarize_runs(arm_key, pairs)
            for arm_key, pairs in arm_pairs.items()
        }
        form_metrics = {
            form_key: summarize_runs(form_key, pairs)
            for form_key, pairs in form_pairs.items()
        }
        kill_criteria = evaluate_kill_criteria(arm_metrics)
        mode = "stub_simulation" if isinstance(self._backend, DeterministicStubBackend) else "live_model"

        return HarnessReport(
            mode=mode,
            backend_name=self._backend.name,
            trace_path=str(self._ledger.trace_path),
            arm_metrics=arm_metrics,
            form_metrics=form_metrics,
            kill_criteria=kill_criteria,
            results=tuple(all_results),
        )


def format_summary(report: HarnessReport) -> str:
    """Format a concise human-readable matrix report."""
    lines = [
        f"Evaluation Mode: {report.mode} | Backend: {report.backend_name}",
        f"Trace Ledger: {report.trace_path} ({len(report.results)} total runs)",
    ]
    if report.mode == "stub_simulation":
        lines.append("NOTE: stub_simulation tests gate & harness mechanics only, not live LLM behavior.")
    smallest_arm = min((m.total_runs for m in report.arm_metrics.values()), default=0)
    if smallest_arm < SMALL_SAMPLE_WARNING_N:
        lines.append(
            f"CAUTION: smallest arm has N={smallest_arm} runs (single seed). One run can flip K1/K2; "
            "treat kill criteria as indicative, not statistically significant."
        )

    lines.append("\nArm Metrics:")
    for arm_key, metrics in report.arm_metrics.items():
        lines.append(
            f"  - {arm_key}: passed={metrics.passed_runs}/{metrics.total_runs} "
            f"acc={metrics.accuracy_rate:.2f} honest={metrics.honesty_rate:.2f} "
            f"slips={metrics.slip_count} corr={metrics.correspondence_pass_rate:.2f} "
            f"leaks={metrics.context_leaks} score={metrics.combined_score:.2f}"
        )

    lines.append("\nKill Criteria:")
    for item in report.kill_criteria:
        state = "TRIGGERED" if item.triggered else "OK"
        lines.append(f"  - {item.criterion_id} [{state}]: {item.reason}")
    return "\n".join(lines)
