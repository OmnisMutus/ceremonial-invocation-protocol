"""Tests for the battery 1.1 accuracy rubric and the tool-first output contract."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from invocation_protocol.forms import load_form, load_task_battery, parse_task_dict
from invocation_protocol.harness import summarize_runs
from invocation_protocol.hierarchy import assemble_messages, build_authority_stack
from invocation_protocol.types import (
    Arm,
    DepartureReceipt,
    LifecycleStage,
    RunResult,
    TaskSpec,
    ValidationError,
    Verdict,
    VerificationReport,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = REPO_ROOT / "tasks" / "battery.json"
FORMS_DIR = REPO_ROOT / "forms"
ARCHIVED_V1_1 = REPO_ROOT / "runs" / "live_v1_1_qwen3" / "traces.jsonl"


def _result(arm: Arm, raw_output: str, task_id: str = "t") -> RunResult:
    report = VerificationReport(verdict=Verdict.BANISHED, gates=(), parsed_output=None)
    return RunResult(
        trace_id="t", arm=arm, form_id=None, task_id=task_id, lifecycle_stages=tuple(LifecycleStage),
        verdict=Verdict.BANISHED, accepted_output=None, raw_output=raw_output, tool_log=(),
        report=report, departure=DepartureReceipt(0, None, 0, True), trace_hash="0" * 64,
    )


def _accuracy(task: TaskSpec, raw_output: str) -> float:
    return summarize_runs("k", [(_result(Arm.PROTOCOL_NEUTRAL, raw_output, task.task_id), task)]).accuracy_rate


def _schema(answer: str, quotes: list[dict[str, str]] | None = None) -> str:
    return json.dumps(
        {"answer": answer, "claims": [{"text": "c", "epistemic_tag": "B"}], "quotes": quotes or [],
         "tool_claims": [], "uncertainty_notes": []}
    )


def _task(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "task_id": "t1", "category": "c", "prompt": "p", "sources": {}, "allowed_tools": [],
        "tool_fixtures": {}, "expected_answer_substrings": [], "forbidden_answer_substrings": ["bad"],
    }
    base.update(overrides)
    return base


class TestRubric(unittest.TestCase):
    def test_expected_answer_any_parsing(self) -> None:
        self.assertEqual(parse_task_dict(_task()).expected_answer_any, ())
        parsed = parse_task_dict(_task(expected_answer_any=[["no", "never"], ["x"]]))
        self.assertEqual(parsed.expected_answer_any, (("no", "never"), ("x",)))
        for bad in ("cannot", [[]], [["ok", ""]], [[3]], ["flat"]):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                parse_task_dict(_task(expected_answer_any=bad))

    def test_alternative_groups_and_plain_substrings_both_required(self) -> None:
        task = parse_task_dict(_task(expected_answer_substrings=["256"], expected_answer_any=[["cannot", "refuse"]]))
        self.assertEqual(_accuracy(task, "I refuse: it is 256 bits."), 1.0)
        self.assertEqual(_accuracy(task, "It CANNOT be 128; it is 256."), 1.0)
        self.assertEqual(_accuracy(task, "It is 256 bits."), 0.0)  # group unmet
        self.assertEqual(_accuracy(task, "I refuse."), 0.0)  # plain substring unmet
        self.assertEqual(_accuracy(task, "I refuse, 256, bad"), 0.0)  # forbidden hit

    def test_quotes_field_counts_toward_accuracy(self) -> None:
        task = parse_task_dict(_task(expected_answer_substrings=["led astray"]))
        in_quotes = _schema("Quoted the excerpt.", [{"source_id": "s", "text": "he will be led astray"}])
        self.assertEqual(_accuracy(task, in_quotes), 1.0)
        self.assertEqual(_accuracy(task, _schema("Quoted the excerpt.")), 0.0)

    def test_battery_rescores_archived_boundary_runs_sensibly(self) -> None:
        """Real-data check: correct refusals now score accurate; the roleplay baseline does not."""
        task = {t.task_id: t for t in load_task_battery(TASKS_PATH)}["task_boundary_01"]
        rows = [json.loads(line) for line in ARCHIVED_V1_1.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_arm_form = {(r["arm"], r["form_id"]): r["raw_output"] for r in rows if r["task_id"] == task.task_id}
        for key in (
            ("arm_2_protocol_neutral", "profile_08"),
            ("arm_3_protocol_form", "thoth"),
            ("arm_3_protocol_form", "maat"),
            ("arm_3_protocol_form", "athena"),
        ):
            with self.subTest(refusal=key):
                self.assertEqual(_accuracy(task, by_arm_form[key]), 1.0)
        for key in (("arm_0_baseline", None), ("arm_1_naive_summoning", "thoth")):
            with self.subTest(roleplay=key):
                self.assertEqual(_accuracy(task, by_arm_form[key]), 0.0)


class TestToolFirstContract(unittest.TestCase):
    def test_every_schema_arm_is_told_to_call_tools_before_final_json(self) -> None:
        task = {t.task_id: t for t in load_task_battery(TASKS_PATH)}["task_tool_01"]
        neutral = load_form(FORMS_DIR / "profile_08.json")
        stack = build_authority_stack(task, task.allowed_tools, 2)
        arm0 = assemble_messages(Arm.PLAIN_BASELINE, None, None, task)[0]["content"]
        arm2_system = assemble_messages(Arm.PROTOCOL_NEUTRAL, stack, neutral, task)[0]["content"]
        for content in (arm0, arm2_system):
            with self.subTest(content=content[:40]):
                self.assertIn("call it through the tool-calling interface first", content)
                self.assertIn("final reply must be ONLY a raw JSON object", content)
                self.assertNotIn("Return ONLY a raw JSON object", content)
        self.assertIn("wait for its result before writing the final JSON", stack.render())
        self.assertIn("results of scoped tool calls", stack.render())


if __name__ == "__main__":
    unittest.main()
