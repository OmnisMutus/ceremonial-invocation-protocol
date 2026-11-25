"""Tests for InvocationRunner 7-step lifecycle, fail-closed gates, and zero-persistence teardown."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from invocation_protocol.backends import DeterministicStubBackend, FAILURE_MODES
from invocation_protocol.forms import load_form, load_task_battery
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.runner import InvocationRunner
from invocation_protocol.types import (
    Arm,
    GateStatus,
    InvocationError,
    LifecycleOrderError,
    LifecycleStage,
    ModelResponse,
    ModelToolRequest,
    RunConfig,
    SpiritBanishedError,
    ValidationError,
    Verdict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMS_DIR = REPO_ROOT / "forms"
TASKS_PATH = REPO_ROOT / "tasks" / "battery.json"


class TestInvocationRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = {t.task_id: t for t in load_task_battery(TASKS_PATH)}
        self.neutral = load_form(FORMS_DIR / "profile_08.json")
        self.thoth = load_form(FORMS_DIR / "thoth.json")
        self.corrupted = load_form(FORMS_DIR / "thoth_corrupted.json")

    def test_full_7_step_order_and_zero_leak_across_sequential_runs(self) -> None:
        captured_messages: list[tuple[dict[str, str], ...]] = []
        base_stub = DeterministicStubBackend()

        def capturing_responder(messages, config, allowed_tools):
            captured_messages.append(tuple(messages))
            return base_stub.generate(messages, config=config, allowed_tools=allowed_tools)

        with tempfile.TemporaryDirectory() as tmp:
            ledger = LiberELedger(Path(tmp) / "traces.jsonl")
            runner = InvocationRunner(
                backend=DeterministicStubBackend(responder=capturing_responder),
                ledger=ledger,
            )
            cfg = RunConfig()

            # Run 1: Thoth
            r1 = runner.invoke(Arm.PROTOCOL_FORM, self.tasks["task_accuracy_01"], cfg, form=self.thoth)
            self.assertEqual(r1.verdict, Verdict.PASSED)
            self.assertIsNotNone(r1.accepted_output)
            self.assertIn("36", r1.accepted_output["answer"])  # type: ignore[index]
            self.assertEqual(
                r1.lifecycle_stages,
                (
                    LifecycleStage.BANISH,
                    LifecycleStage.OPEN_HIERARCHY,
                    LifecycleStage.ASSUME_FORM,
                    LifecycleStage.RUN_CHARGE,
                    LifecycleStage.TEST_SPIRIT,
                    LifecycleStage.DEPART,
                    LifecycleStage.RECORD,
                ),
            )
            self.assertTrue(r1.departure.clean)
            self.assertEqual(r1.departure.persisted_persona_bytes, 0)
            self.assertEqual(runner.active_context_size, 0)
            self.assertIsNone(runner.active_form_id)

            # Run 2: Neutral profile_08 must contain ZERO Thoth prose from Run 1
            r2 = runner.invoke(Arm.PROTOCOL_NEUTRAL, self.tasks["task_sycophancy_01"], cfg, form=self.neutral)
            self.assertEqual(r2.verdict, Verdict.PASSED)
            self.assertIsNotNone(r2.accepted_output)
            run2_serialized = "\n".join(m["content"] for m in captured_messages[1])
            self.assertNotIn("Thoth", run2_serialized)
            self.assertNotIn("Scribe of the Ennead", run2_serialized)
            self.assertNotIn("hod_mercury_scribe", run2_serialized)

            ok, errs = ledger.verify_chain()
            self.assertTrue(ok, msg=str(errs))
            traces = ledger.read_traces()
            self.assertEqual(len(traces), 2)
            self.assertIsNotNone(traces[0]["accepted_output"])
            self.assertIsNotNone(traces[1]["accepted_output"])

    def test_lifecycle_order_and_prior_history_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = LiberELedger(Path(tmp) / "traces.jsonl")
            runner = InvocationRunner(backend=DeterministicStubBackend(), ledger=ledger)
            cfg = RunConfig()
            task = self.tasks["task_accuracy_01"]

            # Out of order before banish()
            with self.assertRaises(LifecycleOrderError):
                runner.open_hierarchy(Arm.PROTOCOL_NEUTRAL, task, cfg)
            with self.assertRaises(LifecycleOrderError):
                runner.assume_form(Arm.PROTOCOL_NEUTRAL, self.neutral)
            with self.assertRaises(LifecycleOrderError):
                runner.run_charge(Arm.PROTOCOL_NEUTRAL, task, cfg)
            with self.assertRaises(LifecycleOrderError):
                runner.test_spirit(task, "{}")
            with self.assertRaises(LifecycleOrderError):
                runner.record({"trace_id": "x"})

            # Non-empty prior_history rejected
            with self.assertRaises(ValidationError):
                runner.banish(cfg, task, prior_history=[{"role": "user", "content": "old"}])

            # Skipping depart() before next banish() rejected
            runner.banish(cfg, task)
            runner.open_hierarchy(Arm.PROTOCOL_NEUTRAL, task, cfg)
            with self.assertRaises(LifecycleOrderError):
                runner.banish(cfg, task)

            # Arm/form mismatch rejected
            with self.assertRaises(ValidationError):
                runner.assume_form(Arm.PLAIN_BASELINE, self.neutral)
            with self.assertRaises(ValidationError):
                runner.assume_form(Arm.PROTOCOL_NEUTRAL, None)
            with self.assertRaises(ValidationError):
                runner.assume_form(Arm.PROTOCOL_NEUTRAL, self.thoth)
            with self.assertRaises(ValidationError):
                runner.assume_form(Arm.PROTOCOL_FORM, self.neutral)
            with self.assertRaises(ValidationError):
                runner.assume_form(Arm.CORRUPTED_CORRESPONDENCE, self.thoth)

            receipt = runner.depart()
            self.assertTrue(receipt.clean)

            # After depart(), a fresh banish() without record() is allowed cleanly
            runner.banish(cfg, task)
            runner.depart()

            # Any single residual ephemeral field after depart() triggers LifecycleOrderError on banish()
            for attr, dirty_val, clean_val in (
                ("_context_messages", [{"role": "user", "content": "stale"}], []),
                ("_thinking_parts", ["stale_thought"], []),
                ("_active_hierarchy", object(), None),
                ("_active_form", self.thoth, None),
                ("_dispatcher", object(), None),
            ):
                setattr(runner, attr, dirty_val)
                try:
                    with self.assertRaises(LifecycleOrderError, msg=attr):
                        runner.banish(cfg, task)
                finally:
                    setattr(runner, attr, clean_val)

    def test_all_stub_failure_injections_banish_fail_closed_and_still_depart_and_record(self) -> None:
        task_map = {
            "bad_json": (self.tasks["task_accuracy_01"], "schema_gate"),
            "missing_epistemic_tag": (self.tasks["task_accuracy_01"], "schema_gate"),
            "unbacked_a_claim": (self.tasks["task_accuracy_01"], "epistemic_gate"),
            "fabricated_quote": (self.tasks["task_accuracy_01"], "quote_gate"),
            "misattributed_quote": (self.tasks["task_quote_01"], "quote_gate"),
            "phantom_tool_claim": (self.tasks["task_tool_01"], "tool_claim_gate"),
            "mismatched_tool_result": (self.tasks["task_tool_01"], "tool_claim_gate"),
            "ontological_slip": (self.tasks["task_boundary_01"], "slip_gate"),
            "directive_slip": (self.tasks["task_boundary_01"], "slip_gate"),
            "cliche_violation": (self.tasks["task_accuracy_01"], "correspondence_gate"),
            "correspondence_drift": (self.tasks["task_accuracy_01"], "correspondence_gate"),
            "tool_budget_overflow": (self.tasks["task_tool_01"], "schema_gate"),
            "unscoped_tool_call": (self.tasks["task_tool_01"], "tool_claim_gate"),
        }
        self.assertEqual(set(task_map.keys()), set(FAILURE_MODES))

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"
            ledger = LiberELedger(trace_path)

            for mode, (task, expected_gate) in task_map.items():
                with self.subTest(mode=mode):
                    runner = InvocationRunner(
                        backend=DeterministicStubBackend(inject_failure=mode),
                        ledger=ledger,
                    )
                    cfg = RunConfig(max_turns=3, max_tool_calls=2)
                    if mode == "tool_budget_overflow":
                        # Force budget overflow inside max_turns=3 by capping tool calls at 1
                        cfg = RunConfig(max_turns=3, max_tool_calls=1)

                    if mode == "tool_budget_overflow":
                        # 3 turns calling lookup_fact with max_tool_calls=1 -> denied records logged, empty final content -> BANISHED
                        res = runner.invoke(Arm.PROTOCOL_FORM, task, cfg, form=self.thoth, raise_on_banish=False)
                        self.assertEqual(res.verdict, Verdict.BANISHED)
                        self.assertIsNone(res.accepted_output)
                        self.assertTrue(any(not rec.allowed for rec in res.tool_log))
                        self.assertTrue(res.departure.clean)
                        failed_gates = {g.gate_name for g in res.report.gates if g.status != GateStatus.PASS}
                        self.assertIn(expected_gate, failed_gates)
                    else:
                        with self.assertRaises(SpiritBanishedError) as ctx:
                            runner.invoke(Arm.PROTOCOL_FORM, task, cfg, form=self.thoth, raise_on_banish=True)
                        self.assertEqual(ctx.exception.report.verdict, Verdict.BANISHED)
                        self.assertIsNone(ctx.exception.report.parsed_output)
                        self.assertEqual(runner.active_context_size, 0)
                        self.assertIsNone(runner.active_form_id)
                        failed_gates = {
                            g.gate_name for g in ctx.exception.report.gates if g.status != GateStatus.PASS
                        }
                        self.assertIn(expected_gate, failed_gates)
                        if mode == "mismatched_tool_result":
                            latest_trace = ledger.read_traces()[-1]
                            self.assertEqual(len(latest_trace["tool_log"]), 1)
                            self.assertTrue(latest_trace["tool_log"][0]["allowed"])
                        if mode == "unscoped_tool_call":
                            latest_trace = ledger.read_traces()[-1]
                            self.assertEqual(len(latest_trace["tool_log"]), 1)
                            self.assertFalse(latest_trace["tool_log"][0]["allowed"])

            # Verify exact 5-request boundary for tool_budget_overflow with sufficient turns
            overflow_runner = InvocationRunner(
                backend=DeterministicStubBackend(inject_failure="tool_budget_overflow"),
                ledger=ledger,
            )
            overflow_res = overflow_runner.invoke(
                Arm.PROTOCOL_FORM,
                self.tasks["task_tool_01"],
                RunConfig(max_turns=6, max_tool_calls=2),
                form=self.thoth,
            )
            self.assertEqual(overflow_res.verdict, Verdict.PASSED)
            self.assertEqual(len(overflow_res.tool_log), 5)
            self.assertEqual(sum(1 for rec in overflow_res.tool_log if rec.allowed), 2)
            self.assertEqual(sum(1 for rec in overflow_res.tool_log if not rec.allowed), 3)

            traces = ledger.read_traces()
            self.assertEqual(len(traces), len(FAILURE_MODES) + 1)
            self.assertTrue(all(t["verdict"] == "BANISHED" for t in traces[: len(FAILURE_MODES)]))
            self.assertTrue(all(t["accepted_output"] is None for t in traces[: len(FAILURE_MODES)]))
            ok, errs = ledger.verify_chain()
            self.assertTrue(ok, msg=str(errs))

    def test_backend_exception_still_executes_depart_and_record(self) -> None:
        def failing_responder(_messages, _config, _allowed_tools) -> ModelResponse:
            raise InvocationError("simulated HTTP 500")

        with tempfile.TemporaryDirectory() as tmp:
            ledger = LiberELedger(Path(tmp) / "traces.jsonl")
            runner = InvocationRunner(
                backend=DeterministicStubBackend(responder=failing_responder),
                ledger=ledger,
            )
            with self.assertRaises(InvocationError):
                runner.invoke(Arm.PROTOCOL_NEUTRAL, self.tasks["task_accuracy_01"], RunConfig(), form=self.neutral)

            self.assertEqual(runner.active_context_size, 0)
            self.assertIsNone(runner.active_form_id)
            traces = ledger.read_traces()
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["verdict"], "BANISHED")
            self.assertIn("simulated HTTP 500", traces[0]["runtime_error"])
            self.assertTrue(traces[0]["departure"]["clean"])

    def test_tool_execution_succeeds_within_budget_on_task_tool_01(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = LiberELedger(Path(tmp) / "traces.jsonl")
            runner = InvocationRunner(backend=DeterministicStubBackend(), ledger=ledger)
            res = runner.invoke(
                Arm.PROTOCOL_FORM,
                self.tasks["task_tool_01"],
                RunConfig(max_turns=3, max_tool_calls=2),
                form=self.thoth,
            )
            self.assertEqual(res.verdict, Verdict.PASSED)
            self.assertEqual(len(res.tool_log), 1)
            self.assertTrue(res.tool_log[0].allowed)
            self.assertEqual(res.tool_log[0].tool_name, "lookup_fact")
            self.assertIn("7 fixed lifecycle steps", res.tool_log[0].output or "")

            # Also test max_turns=1 cutoff when tool request consumes only turn
            def endless_tool(_m, _c, _a) -> ModelResponse:
                return ModelResponse(content="", tool_requests=(ModelToolRequest("lookup_fact", {"key": "protocol_steps"}),))

            runner2 = InvocationRunner(backend=DeterministicStubBackend(responder=endless_tool), ledger=ledger)
            res2 = runner2.invoke(
                Arm.PROTOCOL_FORM,
                self.tasks["task_tool_01"],
                RunConfig(max_turns=1, max_tool_calls=2),
                form=self.thoth,
            )
            self.assertEqual(res2.verdict, Verdict.BANISHED)
            self.assertEqual(len(res2.tool_log), 1)

    def test_thinking_recorded_and_cleared_across_runs(self) -> None:
        base_stub = DeterministicStubBackend()
        call_num = 0

        def responder(messages, config, allowed_tools) -> ModelResponse:
            nonlocal call_num
            call_num += 1
            base = base_stub.generate(messages, config=config, allowed_tools=allowed_tools)
            think = "reasoning_run_1" if call_num == 1 else ""
            return ModelResponse(content=base.content, model_name=base.model_name, thinking=think)

        with tempfile.TemporaryDirectory() as tmp:
            ledger = LiberELedger(Path(tmp) / "traces.jsonl")
            runner = InvocationRunner(backend=DeterministicStubBackend(responder=responder), ledger=ledger)
            r1 = runner.invoke(Arm.PROTOCOL_NEUTRAL, self.tasks["task_accuracy_01"], RunConfig(), form=self.neutral)
            r2 = runner.invoke(Arm.PROTOCOL_NEUTRAL, self.tasks["task_accuracy_01"], RunConfig(), form=self.neutral)

            self.assertEqual(r1.thinking_chars, len("reasoning_run_1"))
            self.assertEqual(r2.thinking_chars, 0)
            traces = ledger.read_traces()
            self.assertEqual(traces[0]["thinking_chars"], len("reasoning_run_1"))
            self.assertEqual(traces[0]["thinking_output"], "reasoning_run_1")
            self.assertEqual(traces[1]["thinking_chars"], 0)
            self.assertEqual(traces[1]["thinking_output"], "")


if __name__ == "__main__":
    unittest.main()
