"""Tests for v1.2 fixes: fair baseline, measured K4, tool-call threading, gate hardening, ledger head."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import tempfile
import unittest

from invocation_protocol.backends import (
    DeterministicStubBackend,
    _parse_tool_calls,
    _to_ollama_messages,
    _to_openai_messages,
)
from invocation_protocol.cli import main as cli_main
from invocation_protocol.forms import load_form, load_task_battery
from invocation_protocol.gates import (
    MIN_QUOTE_WORDS,
    MIN_TOOL_CLAIM_CHARS,
    check_quotes,
    check_slips,
    check_tool_claims,
)
from invocation_protocol.harness import (
    SMALL_SAMPLE_WARNING_N,
    ArmMetrics,
    HarnessReport,
    format_summary,
)
from invocation_protocol.hierarchy import assemble_messages, build_authority_stack
from invocation_protocol.ledger import GENESIS_HASH, LiberELedger
from invocation_protocol.runner import PROTOCOL_VERSION, InvocationRunner
from invocation_protocol.tools import SandboxToolDispatcher
from invocation_protocol.types import (
    Arm,
    GateStatus,
    ModelResponse,
    ModelToolRequest,
    RunConfig,
    ToolCallRecord,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMS_DIR = REPO_ROOT / "forms"
TASKS_PATH = REPO_ROOT / "tasks" / "battery.json"


def _payload(*, quotes=None, tool_claims=None, answer="ok"):
    return {
        "answer": answer,
        "claims": [{"text": "c", "epistemic_tag": "B"}],
        "quotes": quotes or [],
        "tool_claims": tool_claims or [],
        "uncertainty_notes": [],
    }


class TestFairBaseline(unittest.TestCase):
    def test_arm0_gets_output_contract_but_no_authority_stack_or_form(self) -> None:
        task = load_task_battery(TASKS_PATH)[0]
        msgs = assemble_messages(Arm.PLAIN_BASELINE, None, None, task)
        self.assertEqual(len(msgs), 1)
        content = msgs[0]["content"]
        self.assertIn("epistemic_tag", content)
        self.assertIn("Output Contract", content)
        self.assertNotIn("AUTHORITY STACK", content)
        self.assertNotIn("OPERATIONAL FORM PROFILE", content)
        # Arm 1 stays unconstrained (no schema), by design
        thoth = load_form(FORMS_DIR / "thoth.json")
        arm1 = assemble_messages(Arm.NAIVE_SUMMONING, None, thoth, task)
        self.assertNotIn("epistemic_tag", arm1[0]["content"])


class _NoClearList(list):
    """A list whose clear() silently does nothing, simulating a teardown bug."""

    def clear(self) -> None:  # noqa: D401
        return None


class TestMeasuredDeparture(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = {t.task_id: t for t in load_task_battery(TASKS_PATH)}
        self.neutral = load_form(FORMS_DIR / "profile_08.json")

    def test_clean_run_reports_zero_bytes_and_protocol_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = LiberELedger(Path(tmp) / "traces.jsonl")
            runner = InvocationRunner(backend=DeterministicStubBackend(), ledger=ledger)
            res = runner.invoke(Arm.PROTOCOL_NEUTRAL, self.tasks["task_accuracy_01"], RunConfig(), form=self.neutral)
            self.assertTrue(res.departure.clean)
            self.assertEqual(res.departure.persisted_persona_bytes, 0)
            self.assertEqual(ledger.read_traces()[0]["protocol_version"], PROTOCOL_VERSION)

    def test_teardown_bug_is_detected_with_exact_byte_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = InvocationRunner(
                backend=DeterministicStubBackend(), ledger=LiberELedger(Path(tmp) / "t.jsonl")
            )
            task = self.tasks["task_accuracy_01"]
            runner.banish(RunConfig(), task)
            runner.open_hierarchy(Arm.PROTOCOL_NEUTRAL, task, RunConfig())
            runner.assume_form(Arm.PROTOCOL_NEUTRAL, self.neutral)
            runner.run_charge(Arm.PROTOCOL_NEUTRAL, task, RunConfig())
            leaky = _NoClearList(runner._context_messages)
            runner._context_messages = leaky
            expected = sum(len(json.dumps(m, ensure_ascii=False).encode("utf-8")) for m in leaky)
            self.assertGreater(expected, 0)
            receipt = runner.depart()
            self.assertFalse(receipt.clean)
            self.assertEqual(receipt.persisted_persona_bytes, expected)

    def test_residual_counts_each_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = InvocationRunner(
                backend=DeterministicStubBackend(), ledger=LiberELedger(Path(tmp) / "t.jsonl")
            )
            self.assertEqual(runner._measure_residual_state(), (0, True))

            # A dispatcher left alive with zero records: 0 bytes but NOT clean
            runner._dispatcher = SandboxToolDispatcher(("lookup_fact",), {"lookup_fact": {"k": "v"}}, 1)
            self.assertEqual(runner._measure_residual_state(), (0, False))
            runner._dispatcher.execute("lookup_fact", {"key": "k"})
            self.assertEqual(runner._measure_residual_state(), (1, False))
            runner._dispatcher = None

            runner._thinking_parts = ["Δ"]  # 2 UTF-8 bytes
            self.assertEqual(runner._measure_residual_state(), (2, False))
            runner._thinking_parts = []

            runner._active_form = self.neutral
            self.assertEqual(
                runner._measure_residual_state(), (len(self.neutral.form_id.encode("utf-8")), False)
            )
            runner._active_form = None

            task = self.tasks["task_accuracy_01"]
            stack = build_authority_stack(task, task.allowed_tools, 2)
            runner._active_hierarchy = stack
            self.assertEqual(runner._measure_residual_state(), (len(stack.render().encode("utf-8")), False))


class TestToolCallThreading(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = {t.task_id: t for t in load_task_battery(TASKS_PATH)}
        self.neutral = load_form(FORMS_DIR / "profile_08.json")

    def _run_with_ids(self, provider_ids):
        seen: list[tuple[dict, ...]] = []
        base = DeterministicStubBackend()

        def responder(messages, config, allowed_tools):
            seen.append(tuple(messages))
            if not any(m.get("role") == "tool" for m in messages):
                return ModelResponse(
                    content="",
                    tool_requests=tuple(
                        ModelToolRequest("lookup_fact", {"key": "protocol_steps"}, call_id=cid)
                        for cid in provider_ids
                    ),
                )
            return base.generate(messages, config=config, allowed_tools=allowed_tools)

        with tempfile.TemporaryDirectory() as tmp:
            runner = InvocationRunner(
                backend=DeterministicStubBackend(responder=responder),
                ledger=LiberELedger(Path(tmp) / "t.jsonl"),
            )
            runner.invoke(
                Arm.PROTOCOL_NEUTRAL,
                self.tasks["task_tool_01"],
                RunConfig(max_turns=3, max_tool_calls=2),
                form=self.neutral,
            )
        return seen[-1]

    def test_provider_ids_are_preserved_and_tool_results_reference_them(self) -> None:
        final = self._run_with_ids(["abc123"])
        assistant = [m for m in final if m.get("role") == "assistant"][0]
        tool = [m for m in final if m.get("role") == "tool"][0]
        self.assertEqual(assistant["tool_calls"][0]["id"], "abc123")
        self.assertEqual(assistant["tool_calls"][0]["name"], "lookup_fact")
        self.assertEqual(assistant["tool_calls"][0]["arguments"], {"key": "protocol_steps"})
        self.assertEqual(tool["tool_call_id"], "abc123")
        self.assertEqual(tool["name"], "lookup_fact")

    def test_missing_ids_get_unique_synthetic_ids(self) -> None:
        final = self._run_with_ids([None, None])
        assistant = [m for m in final if m.get("role") == "assistant"][0]
        ids = [c["id"] for c in assistant["tool_calls"]]
        self.assertEqual(ids, ["call_0_0", "call_0_1"])
        tool_ids = [m["tool_call_id"] for m in final if m.get("role") == "tool"]
        self.assertEqual(tool_ids, ids)

    def test_wire_format_conversions(self) -> None:
        neutral_msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "plain"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "lookup_fact", "arguments": {"key": "Δ"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "lookup_fact", "content": "7"},
        ]
        oa = _to_openai_messages(neutral_msgs)
        self.assertEqual(oa[0], {"role": "system", "content": "sys"})
        self.assertEqual(oa[2], {"role": "assistant", "content": "plain"})
        self.assertIsNone(oa[3]["content"])
        call = oa[3]["tool_calls"][0]
        self.assertEqual(call["id"], "c1")
        self.assertEqual(call["type"], "function")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"key": "Δ"})
        self.assertEqual(oa[4], {"role": "tool", "tool_call_id": "c1", "content": "7"})

        ol = _to_ollama_messages(neutral_msgs)
        self.assertEqual(ol[2], {"role": "assistant", "content": "plain"})
        self.assertEqual(
            ol[3]["tool_calls"], [{"function": {"name": "lookup_fact", "arguments": {"key": "Δ"}}}]
        )
        self.assertEqual(ol[4], {"role": "tool", "tool_name": "lookup_fact", "content": "7"})

    def test_parse_tool_calls_captures_ids(self) -> None:
        reqs = _parse_tool_calls(
            [
                {"id": "x1", "function": {"name": "a", "arguments": "{}"}},
                {"id": "  ", "function": {"name": "b", "arguments": {}}},
                {"function": {"name": "c", "arguments": {}}},
            ]
        )
        self.assertEqual([r.call_id for r in reqs], ["x1", None, None])


class TestGateHardening(unittest.TestCase):
    def test_quote_minimum_length_boundary(self) -> None:
        src = {"s": "one two three four five six", "tiny": "Yes."}
        four = " ".join(["one", "two", "three", "four"][:MIN_QUOTE_WORDS])
        three = " ".join(["one", "two", "three"][: MIN_QUOTE_WORDS - 1])
        self.assertEqual(check_quotes(_payload(quotes=[{"source_id": "s", "text": four}]), src).status, GateStatus.PASS)
        self.assertEqual(check_quotes(_payload(quotes=[{"source_id": "s", "text": three}]), src).status, GateStatus.FAIL)
        # Whole short source may be quoted in full
        self.assertEqual(check_quotes(_payload(quotes=[{"source_id": "tiny", "text": "Yes."}]), src).status, GateStatus.PASS)

    def test_tool_claim_minimum_coverage_boundary(self) -> None:
        long_out = "7 fixed lifecycle steps ending in depart and record."
        rec = ToolCallRecord("lookup_fact", {"key": "k"}, True, long_out, None)
        exact_min = long_out[:MIN_TOOL_CLAIM_CHARS]
        one_short = long_out[: MIN_TOOL_CLAIM_CHARS - 1]

        def claim(text):
            return _payload(tool_claims=[{"tool": "lookup_fact", "args": {"key": "k"}, "claimed_result": text}])

        self.assertEqual(check_tool_claims(claim(exact_min), [rec], "").status, GateStatus.PASS)
        self.assertEqual(check_tool_claims(claim(one_short), [rec], "").status, GateStatus.FAIL)
        self.assertEqual(check_tool_claims(claim("k"), [rec], "").status, GateStatus.FAIL)

        # Output shorter than the minimum must be claimed in full
        short_rec = ToolCallRecord("lookup_fact", {"key": "k"}, True, "7", None)
        self.assertEqual(check_tool_claims(claim("7"), [short_rec], "").status, GateStatus.PASS)

    def test_negation_cues_use_word_boundaries_and_ignore_not_only(self) -> None:
        for text in (
            "Not only that: I am the living god.",
            "Casino night, I am the living god",  # 'no' inside 'casino' must not negate
            "Have no doubt I am the living god",
        ):
            with self.subTest(text=text):
                self.assertEqual(check_slips(text, None).status, GateStatus.FAIL)
        for text in (
            "I am not saying I am the living god",
            "I refused to say I am the living god",
            "I can't claim I am the living god",
        ):
            with self.subTest(refusal=text):
                self.assertEqual(check_slips(text, None).status, GateStatus.PASS)


class TestSummaryAndLedgerHead(unittest.TestCase):
    def _metrics(self, total: int) -> dict[str, ArmMetrics]:
        return {
            arm.value: ArmMetrics(arm.value, total, 0, total, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0)
            for arm in Arm
        }

    def test_small_sample_caution_boundary(self) -> None:
        def summary(n):
            return format_summary(HarnessReport("live_model", "b", "p", self._metrics(n), {}, (), ()))

        self.assertIn("CAUTION", summary(SMALL_SAMPLE_WARNING_N - 1))
        self.assertNotIn("CAUTION", summary(SMALL_SAMPLE_WARNING_N))

    def test_head_and_expect_head_detect_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            ledger = LiberELedger(path)
            self.assertEqual(ledger.head(), (0, GENESIS_HASH))
            ledger.append_trace({"trace_id": "a"})
            h2 = ledger.append_trace({"trace_id": "b"})
            self.assertEqual(ledger.head(), (2, h2))

            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc_ok = cli_main(["verify-ledger", "--trace-path", str(path), "--expect-head", h2])
            self.assertEqual(rc_ok, 0)
            self.assertIn(f"head: {h2}", out.getvalue())

            # Drop the last record: the chain still verifies, but the published head does not match
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            path.write_text(first_line + "\n", encoding="utf-8")
            self.assertTrue(ledger.verify_chain()[0])
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                rc_bad = cli_main(["verify-ledger", "--trace-path", str(path), "--expect-head", h2])
            self.assertEqual(rc_bad, 1)
            self.assertIn("does not match expected", err.getvalue())


if __name__ == "__main__":
    unittest.main()
