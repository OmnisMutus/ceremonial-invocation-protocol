"""Fixed 7-step Ceremonial Invocation lifecycle runner."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Callable, Mapping, Sequence
import uuid

from invocation_protocol import gates
from invocation_protocol.backends import ModelBackend
from invocation_protocol.hierarchy import AuthorityStack, assemble_messages, build_authority_stack
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.tools import SandboxToolDispatcher
from invocation_protocol.types import (
    Arm,
    DepartureReceipt,
    FormSpec,
    LifecycleOrderError,
    LifecycleStage,
    RunConfig,
    RunResult,
    TaskSpec,
    ToolBudgetError,
    ToolScopeError,
    ValidationError,
    Verdict,
    VerificationReport,
)


PROTOCOL_VERSION: str = "1.3"


class InvocationRunner:
    """Enforces the fixed 7-step operational lifecycle around every model invocation."""

    def __init__(
        self,
        *,
        backend: ModelBackend,
        ledger: LiberELedger,
        blind_reviewer: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._backend = backend
        self._ledger = ledger
        self._blind_reviewer = blind_reviewer
        self._context_messages: list[dict[str, str]] = []
        self._thinking_parts: list[str] = []
        self._active_hierarchy: AuthorityStack | None = None
        self._active_form: FormSpec | None = None
        self._dispatcher: SandboxToolDispatcher | None = None
        self._stages: list[LifecycleStage] = []

    @property
    def active_context_size(self) -> int:
        """Number of ephemeral context messages currently loaded."""
        return len(self._context_messages)

    @property
    def active_form_id(self) -> str | None:
        """Currently bound FormSpec ID, or None after depart()."""
        return self._active_form.form_id if self._active_form is not None else None

    def _require_stage(self, expected_previous: LifecycleStage, current: LifecycleStage) -> None:
        """Enforce strict sequential progression across lifecycle steps."""
        if not self._stages or self._stages[-1] != expected_previous:
            prev_desc = self._stages[-1].value if self._stages else "NONE"
            raise LifecycleOrderError(
                f"Cannot run {current.value} after {prev_desc}; expected prior stage {expected_previous.value}"
            )

    def banish(
        self,
        config: RunConfig,
        task: TaskSpec,
        *,
        prior_history: Sequence[Any] = (),
    ) -> None:
        """Step 1: Enforce fresh zero-history state, pinned config, and sandboxed tools."""
        if not isinstance(config, RunConfig):
            raise ValidationError("config must be a RunConfig instance")
        if not isinstance(task, TaskSpec):
            raise ValidationError("task must be a TaskSpec instance")
        if self._stages and self._stages[-1] != LifecycleStage.DEPART:
            raise LifecycleOrderError(
                f"Previous run did not complete depart() (last stage: {self._stages[-1].value})"
            )
        if (
            len(self._context_messages) != 0
            or len(self._thinking_parts) != 0
            or self._active_hierarchy is not None
            or self._active_form is not None
            or self._dispatcher is not None
        ):
            raise LifecycleOrderError("Ephemeral state was not cleared by depart() before banish()")
        if len(prior_history) > 0:
            raise ValidationError("banish() rejects non-empty prior_history")

        self._context_messages = []
        self._thinking_parts = []
        self._active_hierarchy = None
        self._active_form = None
        self._dispatcher = SandboxToolDispatcher(
            allowed_tools=task.allowed_tools,
            fixtures=task.tool_fixtures,
            max_calls=config.max_tool_calls,
        )
        self._stages = [LifecycleStage.BANISH]

    def open_hierarchy(
        self,
        arm: Arm,
        task: TaskSpec,
        config: RunConfig,
    ) -> AuthorityStack | None:
        """Step 2: Build the immutable authority stack for protocol arms (None for controls)."""
        self._require_stage(LifecycleStage.BANISH, LifecycleStage.OPEN_HIERARCHY)
        if arm in (Arm.PLAIN_BASELINE, Arm.NAIVE_SUMMONING):
            self._active_hierarchy = None
        else:
            self._active_hierarchy = build_authority_stack(
                task=task,
                allowed_tools=task.allowed_tools,
                max_tool_calls=config.max_tool_calls,
            )
        self._stages.append(LifecycleStage.OPEN_HIERARCHY)
        return self._active_hierarchy

    def assume_form(self, arm: Arm, form: FormSpec | None) -> FormSpec | None:
        """Step 3: Bind declarative FormSpec JSON matching the selected evaluation arm."""
        self._require_stage(LifecycleStage.OPEN_HIERARCHY, LifecycleStage.ASSUME_FORM)
        if arm == Arm.PLAIN_BASELINE:
            if form is not None:
                raise ValidationError("Arm.PLAIN_BASELINE requires form=None")
        else:
            if form is None:
                raise ValidationError(f"{arm.value} requires a FormSpec")
            if arm == Arm.PROTOCOL_NEUTRAL and form.arm_type != "neutral":
                raise ValidationError(
                    f"Arm.PROTOCOL_NEUTRAL requires arm_type='neutral', got {form.arm_type!r}"
                )
            if arm == Arm.PROTOCOL_FORM and form.arm_type != "deity":
                raise ValidationError(
                    f"Arm.PROTOCOL_FORM requires arm_type='deity', got {form.arm_type!r}"
                )
            if arm == Arm.CORRUPTED_CORRESPONDENCE and form.arm_type != "corrupted_control":
                raise ValidationError(
                    f"Arm.CORRUPTED_CORRESPONDENCE requires arm_type='corrupted_control', got {form.arm_type!r}"
                )

        self._active_form = form
        self._stages.append(LifecycleStage.ASSUME_FORM)
        return self._active_form

    def run_charge(self, arm: Arm, task: TaskSpec, config: RunConfig) -> str:
        """Step 4: Execute one bounded task with hard turn and sandboxed tool budgets."""
        self._require_stage(LifecycleStage.ASSUME_FORM, LifecycleStage.RUN_CHARGE)
        if self._dispatcher is None:
            raise LifecycleOrderError("SandboxToolDispatcher missing; banish() must run first")

        initial = assemble_messages(arm, self._active_hierarchy, self._active_form, task)
        self._context_messages = [dict(msg) for msg in initial]

        last_content = ""
        for turn in range(config.max_turns):
            response = self._backend.generate(
                tuple(dict(m) for m in self._context_messages),
                config=config,
                allowed_tools=task.allowed_tools,
            )
            if response.thinking:
                self._thinking_parts.append(response.thinking)
            last_content = response.content
            if not response.tool_requests:
                self._stages.append(LifecycleStage.RUN_CHARGE)
                return last_content

            # Backend-neutral tool-call threading: each tool result references the id of the
            # assistant tool call that requested it. Backends translate to their wire format.
            call_ids = [
                req.call_id if req.call_id else f"call_{turn}_{idx}"
                for idx, req in enumerate(response.tool_requests)
            ]
            self._context_messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {"id": cid, "name": str(req.tool_name), "arguments": dict(req.args)}
                        for cid, req in zip(call_ids, response.tool_requests)
                    ],
                }
            )
            for cid, req in zip(call_ids, response.tool_requests):
                try:
                    rec = self._dispatcher.execute(req.tool_name, req.args)
                    tool_msg = rec.output if rec.output is not None else ""
                except (ToolScopeError, ToolBudgetError, ValidationError) as exc:
                    tool_msg = f"TOOL_DENIED: {exc}"
                self._context_messages.append(
                    {"role": "tool", "tool_call_id": cid, "name": str(req.tool_name), "content": tool_msg}
                )

        self._stages.append(LifecycleStage.RUN_CHARGE)
        return last_content

    def test_spirit(self, task: TaskSpec, raw_output: str) -> VerificationReport:
        """Step 5: Run deterministic external verification gates outside the persona."""
        self._require_stage(LifecycleStage.RUN_CHARGE, LifecycleStage.TEST_SPIRIT)
        if self._dispatcher is None:
            raise LifecycleOrderError("SandboxToolDispatcher missing during test_spirit()")
        report = gates.test_spirit(
            task=task,
            form=self._active_form,
            raw_output=raw_output,
            tool_log=self._dispatcher.records,
            blind_reviewer=self._blind_reviewer,
        )
        self._stages.append(LifecycleStage.TEST_SPIRIT)
        return report

    def depart(self) -> DepartureReceipt:
        """Step 6: Wipe ephemeral context, hierarchy, form, and tool dispatcher."""
        cleared_messages = len(self._context_messages)
        cleared_form_id = self._active_form.form_id if self._active_form is not None else None

        self._context_messages.clear()
        self._thinking_parts.clear()
        self._active_hierarchy = None
        self._active_form = None
        if self._dispatcher is not None:
            self._dispatcher.clear()
            self._dispatcher = None

        self._stages.append(LifecycleStage.DEPART)
        residual_bytes, all_slots_empty = self._measure_residual_state()
        return DepartureReceipt(
            cleared_messages=cleared_messages,
            cleared_form_id=cleared_form_id,
            persisted_persona_bytes=residual_bytes,
            # Residual bytes can only be non-zero when some slot is non-empty, so this is sufficient.
            clean=all_slots_empty,
        )

    def _measure_residual_state(self) -> tuple[int, bool]:
        """Measure what ephemeral state survived teardown (feeds kill criterion K4).

        Returns the UTF-8 byte size of any remaining messages, reasoning text, hierarchy,
        form, and tool records, plus whether every ephemeral slot is empty or None.
        """
        residual = 0
        for msg in self._context_messages:
            residual += len(json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8"))
        for part in self._thinking_parts:
            residual += len(part.encode("utf-8"))
        if self._active_hierarchy is not None:
            residual += len(self._active_hierarchy.render().encode("utf-8"))
        if self._active_form is not None:
            residual += len(self._active_form.form_id.encode("utf-8"))
        if self._dispatcher is not None:
            for rec in self._dispatcher.records:
                residual += len(str(rec.output or "").encode("utf-8"))
        all_slots_empty = (
            len(self._context_messages) == 0
            and len(self._thinking_parts) == 0
            and self._active_hierarchy is None
            and self._active_form is None
            and self._dispatcher is None
        )
        return residual, all_slots_empty

    def record(self, trace_payload: Mapping[str, Any]) -> tuple[str, tuple[LifecycleStage, ...]]:
        """Step 7: Append immutable raw execution trace to LiberELedger and reset stage cursor."""
        self._require_stage(LifecycleStage.DEPART, LifecycleStage.RECORD)
        self._stages.append(LifecycleStage.RECORD)
        stages_snapshot = tuple(self._stages)

        envelope = dict(trace_payload)
        envelope["lifecycle_stages"] = [stage.value for stage in stages_snapshot]
        entry_hash = self._ledger.append_trace(envelope)
        self._stages.clear()
        return entry_hash, stages_snapshot

    def invoke(
        self,
        arm: Arm,
        task: TaskSpec,
        config: RunConfig,
        *,
        form: FormSpec | None = None,
        prior_history: Sequence[Any] = (),
        raise_on_banish: bool = False,
    ) -> RunResult:
        """Execute the full 7-step lifecycle with guaranteed depart() and record() in finally."""
        trace_id = uuid.uuid4().hex
        raw_output = ""
        tool_log = ()
        report: VerificationReport | None = None
        runtime_error: Exception | None = None

        self.banish(config, task, prior_history=prior_history)
        try:
            self.open_hierarchy(arm, task, config)
            self.assume_form(arm, form)
            raw_output = self.run_charge(arm, task, config)
            tool_log = self._dispatcher.records if self._dispatcher is not None else ()
            report = self.test_spirit(task, raw_output)
        except Exception as exc:
            runtime_error = exc
            if self._dispatcher is not None:
                tool_log = self._dispatcher.records
            raise
        finally:
            thinking_output = "".join(self._thinking_parts)
            thinking_chars = len(thinking_output)
            departure = self.depart()
            final_verdict = report.verdict if report is not None else Verdict.BANISHED
            accepted = (
                report.parsed_output
                if (report is not None and report.verdict == Verdict.PASSED)
                else None
            )
            trace_payload: dict[str, Any] = {
                "trace_id": trace_id,
                "protocol_version": PROTOCOL_VERSION,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "backend": self._backend.name,
                "arm": arm.value,
                "form_id": form.form_id if form is not None else None,
                "task_id": task.task_id,
                "task_category": task.category,
                "config": {
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "max_tokens": config.max_tokens,
                    "context_length": config.context_length,
                    "max_turns": config.max_turns,
                    "max_tool_calls": config.max_tool_calls,
                    "timeout_seconds": config.timeout_seconds,
                    "fail_closed": config.fail_closed,
                },
                "verdict": final_verdict.value,
                "accepted_output": accepted,
                "raw_output": raw_output,
                "thinking_chars": thinking_chars,
                "thinking_output": thinking_output,
                "tool_log": [
                    {
                        "tool_name": rec.tool_name,
                        "args": dict(rec.args),
                        "allowed": rec.allowed,
                        "output": rec.output,
                        "error": rec.error,
                    }
                    for rec in tool_log
                ],
                "gates": [
                    {
                        "gate_name": gate.gate_name,
                        "status": gate.status.value,
                        "reasons": list(gate.reasons),
                    }
                    for gate in (report.gates if report is not None else ())
                ],
                "departure": {
                    "cleared_messages": departure.cleared_messages,
                    "cleared_form_id": departure.cleared_form_id,
                    "persisted_persona_bytes": departure.persisted_persona_bytes,
                    "clean": departure.clean,
                },
                "runtime_error": (
                    f"{type(runtime_error).__name__}: {runtime_error}"
                    if runtime_error is not None
                    else None
                ),
            }
            trace_hash, stages_snapshot = self.record(trace_payload)

        assert report is not None
        accepted_output = report.parsed_output if report.verdict == Verdict.PASSED else None
        result = RunResult(
            trace_id=trace_id,
            arm=arm,
            form_id=form.form_id if form is not None else None,
            task_id=task.task_id,
            lifecycle_stages=stages_snapshot,
            verdict=report.verdict,
            accepted_output=accepted_output,
            raw_output=raw_output,
            tool_log=tool_log,
            report=report,
            departure=departure,
            trace_hash=trace_hash,
            thinking_chars=thinking_chars,
        )
        if raise_on_banish:
            result.raise_for_status()
        return result
