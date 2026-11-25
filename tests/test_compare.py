"""Tests for multi-ledger trace comparison, verification, and CLI reporting."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
from statistics import mean
import tempfile
import unittest

import invocation_protocol
from invocation_protocol import (
    Arm,
    DeterministicStubBackend,
    EvaluationHarness,
    LedgerArmStats,
    LedgerSummary,
    LiberELedger,
    ValidationError,
    compare_ledgers,
    format_comparison,
    summarize_ledger,
)
from invocation_protocol.cli import main as cli_main
from invocation_protocol.compare import _summarize_group

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMS_DIR = REPO_ROOT / "forms"
TASKS_PATH = REPO_ROOT / "tasks" / "battery.json"


def _independent_group_reference(raw_lines: list[str], *, by_form: bool) -> dict[str, dict[str, object]]:
    """Compute reference stats directly from raw JSONL lines via a different method."""
    buckets: dict[str, list[dict[str, object]]] = {}
    for line in raw_lines:
        row = json.loads(line)
        key = row["arm"] if not by_form else f"{row['arm']}|{row.get('form_id') or 'none'}"
        buckets.setdefault(key, []).append(row)

    out: dict[str, dict[str, object]] = {}
    for key, rows in buckets.items():
        gate_maps = [
            {g["gate_name"]: g["status"] for g in row.get("gates", [])}
            for row in rows
        ]
        total = len(rows)
        passed = sum(1 for row in rows if row.get("verdict") == "PASSED")
        honest = sum(
            1
            for gm in gate_maps
            if gm.get("epistemic_gate") == "PASS"
            and gm.get("quote_gate") == "PASS"
            and gm.get("tool_claim_gate") == "PASS"
        )
        slips = sum(1 for gm in gate_maps if gm.get("slip_gate") != "PASS")
        corr_fail = sum(1 for gm in gate_maps if gm.get("correspondence_gate") != "PASS")
        leaks = sum(
            1
            for row in rows
            if (not row["departure"]["clean"]) or row["departure"]["persisted_persona_bytes"] > 0
        )
        thinking = [row.get("thinking_chars", 0) for row in rows]
        out[key] = {
            "total": total,
            "passed": passed,
            "banished": total - passed,
            "honest": honest,
            "slips": slips,
            "correspondence_failures": corr_fail,
            "leaks": leaks,
            "mean_thinking_chars": round(float(mean(thinking)), 1),
        }
    return out


def _make_trace_entry(
    trace_id: str,
    *,
    arm: str,
    form_id: str | None = None,
    backend: str = "stub",
    verdict: str = "PASSED",
    epistemic: str = "PASS",
    quote: str = "PASS",
    tool_claim: str = "PASS",
    slip: str = "PASS",
    correspondence: str = "PASS",
    clean: bool = True,
    persisted_bytes: int = 0,
    thinking_chars: int | None = 0,
) -> dict[str, object]:
    record: dict[str, object] = {
        "trace_id": trace_id,
        "backend": backend,
        "arm": arm,
        "form_id": form_id,
        "task_id": f"task_{trace_id}",
        "verdict": verdict,
        "gates": [
            {"gate_name": "schema_gate", "status": "PASS", "reasons": []},
            {"gate_name": "epistemic_gate", "status": epistemic, "reasons": []},
            {"gate_name": "quote_gate", "status": quote, "reasons": []},
            {"gate_name": "tool_claim_gate", "status": tool_claim, "reasons": []},
            {"gate_name": "slip_gate", "status": slip, "reasons": []},
            {"gate_name": "correspondence_gate", "status": correspondence, "reasons": []},
        ],
        "departure": {
            "clean": clean,
            "persisted_persona_bytes": persisted_bytes,
        },
    }
    if thinking_chars is not None:
        record["thinking_chars"] = thinking_chars
    return record


class TestCompareLedgers(unittest.TestCase):
    def test_ground_truth_parity_against_evaluation_harness_and_raw_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_a_dir = tmp_path / "run_a"
            run_b_dir = tmp_path / "run_b"

            for out_dir in (run_a_dir, run_b_dir):
                EvaluationHarness(
                    backend=DeterministicStubBackend(),
                    trace_dir=out_dir,
                    forms_dir=FORMS_DIR,
                    tasks_path=TASKS_PATH,
                ).run_matrix()

            custom_b = tmp_path / "subdir" / "custom_b.jsonl"
            custom_b.parent.mkdir(parents=True, exist_ok=True)
            custom_b.write_bytes((run_b_dir / "traces.jsonl").read_bytes())

            trace_a = run_a_dir / "traces.jsonl"
            for by_form in (False, True):
                summaries = compare_ledgers([trace_a, custom_b], by_form=by_form)
                self.assertEqual([s.label for s in summaries], ["run_a", "custom_b"])
                self.assertEqual(summaries[0].total, 35)
                self.assertEqual(summaries[1].total, 35)

                raw_lines = trace_a.read_text(encoding="utf-8").splitlines()
                expected_groups = _independent_group_reference(raw_lines, by_form=by_form)
                self.assertEqual(set(summaries[0].groups.keys()), set(expected_groups.keys()))
                for key, exp in expected_groups.items():
                    got = summaries[0].groups[key]
                    self.assertEqual(got.total, exp["total"])
                    self.assertEqual(got.passed, exp["passed"])
                    self.assertEqual(got.banished, exp["banished"])
                    self.assertEqual(got.honest, exp["honest"])
                    self.assertEqual(got.slips, exp["slips"])
                    self.assertEqual(got.correspondence_failures, exp["correspondence_failures"])
                    self.assertEqual(got.leaks, exp["leaks"])
                    self.assertEqual(got.mean_thinking_chars, exp["mean_thinking_chars"])

    def test_honesty_gates_leak_boundaries_mean_thinking_and_arm_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "mix_run" / "traces.jsonl"
            ledger = LiberELedger(trace_path)

            # 1 clean pass, 3 single-honesty-gate failures, 1 slip + uncleaned depart,
            # 1 corr fail + persisted bytes, plus a custom non-enum arm missing thinking_chars
            records = [
                _make_trace_entry(
                    "t1",
                    arm=Arm.PROTOCOL_FORM.value,
                    form_id="thoth",
                    verdict="PASSED",
                    thinking_chars=100,
                ),
                _make_trace_entry(
                    "t2",
                    arm=Arm.PROTOCOL_FORM.value,
                    form_id="thoth",
                    verdict="BANISHED",
                    epistemic="FAIL",
                    thinking_chars=101,
                ),
                _make_trace_entry(
                    "t3",
                    arm=Arm.PROTOCOL_FORM.value,
                    form_id="athena",
                    verdict="BANISHED",
                    quote="FAIL",
                    thinking_chars=0,
                ),
                _make_trace_entry(
                    "t4",
                    arm=Arm.PROTOCOL_FORM.value,
                    form_id="maat",
                    verdict="BANISHED",
                    tool_claim="FAIL",
                    thinking_chars=None,
                ),
                _make_trace_entry(
                    "t5",
                    arm=Arm.NAIVE_SUMMONING.value,
                    form_id="thoth",
                    verdict="BANISHED",
                    slip="FAIL",
                    clean=False,
                    persisted_bytes=0,
                    thinking_chars=50,
                ),
                _make_trace_entry(
                    "t6",
                    arm=Arm.CORRUPTED_CORRESPONDENCE.value,
                    form_id="thoth_corrupted",
                    verdict="BANISHED",
                    correspondence="FAIL",
                    clean=True,
                    persisted_bytes=7,
                    thinking_chars=50,
                ),
                _make_trace_entry(
                    "t7",
                    arm="z_custom_arm",
                    form_id=None,
                    verdict="PASSED",
                    thinking_chars=None,
                ),
            ]
            for rec in records:
                ledger.append_trace(rec)

            summary_arm = summarize_ledger(trace_path, by_form=False)
            self.assertEqual(summary_arm.label, "mix_run")
            self.assertEqual(summary_arm.backend, "stub")
            self.assertEqual(summary_arm.total, 7)

            # Overall check across all 7 runs: mean([100, 101, 0, 0, 50, 50, 0]) = 301 / 7 = 43.0
            overall = summary_arm.overall
            self.assertEqual(overall.total, 7)
            self.assertEqual(overall.passed, 2)
            self.assertEqual(overall.banished, 5)
            self.assertEqual(overall.honest, 4)  # t1, t5, t6, t7 (epistemic+quote+tool all PASS)
            self.assertEqual(overall.slips, 1)
            self.assertEqual(overall.correspondence_failures, 1)
            self.assertEqual(overall.leaks, 2)  # both (clean=False, bytes=0) and (clean=True, bytes=7)
            self.assertEqual(overall.mean_thinking_chars, 43.0)

            thoth_summary = summarize_ledger(trace_path, by_form=True)
            thoth_stats = thoth_summary.groups["arm_3_protocol_form|thoth"]
            self.assertEqual(thoth_stats.passed, 1)
            self.assertEqual(thoth_stats.banished, 1)
            self.assertEqual(thoth_stats.honest, 1)
            self.assertEqual(thoth_stats.mean_thinking_chars, 100.5)

            rendered = format_comparison([thoth_summary], markdown=False)
            idx_naive = rendered.index("arm_1_naive_summoning|thoth")
            idx_athena = rendered.index("arm_3_protocol_form|athena")
            idx_maat = rendered.index("arm_3_protocol_form|maat")
            idx_thoth = rendered.index("arm_3_protocol_form|thoth")
            idx_corr = rendered.index("arm_3b_corrupted_correspondence|thoth_corrupted")
            idx_custom = rendered.index("z_custom_arm|none")
            self.assertLess(idx_naive, idx_athena)
            self.assertLess(idx_athena, idx_maat)
            self.assertLess(idx_maat, idx_thoth)
            self.assertLess(idx_thoth, idx_corr)
            self.assertLess(idx_corr, idx_custom)

    def test_label_edge_cases_and_fail_closed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bare_path = tmp_path / "traces.jsonl"
            ledger = LiberELedger(bare_path)
            ledger.append_trace(_make_trace_entry("t1", arm=Arm.PLAIN_BASELINE.value))

            # Relative bare "traces.jsonl" has empty parent.name -> stem "traces"
            prev_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                bare_summary = summarize_ledger(Path("traces.jsonl"))
                self.assertEqual(bare_summary.label, "traces")
            finally:
                os.chdir(prev_cwd)

            # Non-ASCII UTF-8 label override
            unicode_summary = compare_ledgers([bare_path], labels=["Δίκη_run"])[0]
            self.assertEqual(unicode_summary.label, "Δίκη_run")

            # Fail-closed empty / whitespace / mismatched / missing / directory / empty file / tampered
            with self.assertRaises(ValidationError):
                compare_ledgers([])
            with self.assertRaises(ValidationError):
                format_comparison([])
            with self.assertRaises(ValidationError):
                _summarize_group("empty", [])
            with self.assertRaises(ValidationError):
                summarize_ledger(bare_path, label="   ")
            with self.assertRaises(ValidationError):
                compare_ledgers([bare_path, bare_path], labels=["only_one"])
            with self.assertRaises(ValidationError):
                summarize_ledger(tmp_path / "missing.jsonl")
            with self.assertRaises(ValidationError):
                summarize_ledger(tmp_path)

            empty_file = tmp_path / "empty.jsonl"
            empty_file.write_text("", encoding="utf-8")
            with self.assertRaises(ValidationError):
                summarize_ledger(empty_file)

            # 1-byte tamper rejection
            tampered_file = tmp_path / "tampered.jsonl"
            original_text = bare_path.read_text(encoding="utf-8")
            obj = json.loads(original_text.splitlines()[0])
            obj["verdict"] = "TAMPERED"
            tampered_file.write_text(
                json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError) as ctx:
                summarize_ledger(tampered_file)
            self.assertIn("chain verification failed", str(ctx.exception))

    def test_markdown_and_asymmetric_groups_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p1 = tmp_path / "alpha" / "traces.jsonl"
            p2 = tmp_path / "beta" / "traces.jsonl"

            l1 = LiberELedger(p1)
            l1.append_trace(_make_trace_entry("a1", arm=Arm.PLAIN_BASELINE.value, verdict="BANISHED"))
            l1.append_trace(_make_trace_entry("a2", arm=Arm.PROTOCOL_NEUTRAL.value, verdict="PASSED"))

            l2 = LiberELedger(p2)
            l2.append_trace(_make_trace_entry("b1", arm=Arm.PROTOCOL_NEUTRAL.value, verdict="PASSED"))

            summaries = compare_ledgers([p1, p2], by_form=False)
            plain = format_comparison(summaries, markdown=False)
            md = format_comparison(summaries, markdown=True)

            self.assertIn("=== Ledger Totals ===", plain)
            self.assertIn("=== By Group ===", plain)
            self.assertIn("1/2", plain)
            self.assertIn("1/1", plain)

            self.assertIn("## Ledger Totals", md)
            self.assertIn("## By Group", md)
            md_lines = [line for line in md.splitlines() if line.startswith("|")]
            self.assertGreaterEqual(len(md_lines), 8)
            for line in md_lines:
                self.assertTrue(line.endswith(" |"))

    def test_cli_compare_exit_codes_and_package_exports(self) -> None:
        self.assertIs(invocation_protocol.LedgerArmStats, LedgerArmStats)
        self.assertIs(invocation_protocol.LedgerSummary, LedgerSummary)
        self.assertIs(invocation_protocol.compare_ledgers, compare_ledgers)
        self.assertIs(invocation_protocol.format_comparison, format_comparison)
        self.assertIs(invocation_protocol.summarize_ledger, summarize_ledger)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p1 = tmp_path / "one" / "traces.jsonl"
            p2 = tmp_path / "two" / "traces.jsonl"
            LiberELedger(p1).append_trace(_make_trace_entry("1", arm=Arm.PROTOCOL_NEUTRAL.value))
            LiberELedger(p2).append_trace(_make_trace_entry("2", arm=Arm.PROTOCOL_NEUTRAL.value))

            out_single = io.StringIO()
            with redirect_stdout(out_single):
                rc_single = cli_main(["compare", str(p1)])
            self.assertEqual(rc_single, 0)
            self.assertIn("one", out_single.getvalue())

            out_multi = io.StringIO()
            with redirect_stdout(out_multi):
                rc_multi = cli_main(["compare", str(p1), str(p2), "--by-form", "--markdown"])
            self.assertEqual(rc_multi, 0)
            self.assertIn("## Ledger Totals", out_multi.getvalue())
            self.assertIn("arm_2_protocol_neutral|none", out_multi.getvalue())

            err_missing = io.StringIO()
            with redirect_stderr(err_missing):
                rc_missing = cli_main(["compare", str(tmp_path / "nope.jsonl")])
            self.assertEqual(rc_missing, 1)
            self.assertIn("ERROR", err_missing.getvalue())

            tampered = tmp_path / "bad.jsonl"
            tampered.write_text("{corrupt\n", encoding="utf-8")
            err_tampered = io.StringIO()
            with redirect_stderr(err_tampered):
                rc_tampered = cli_main(["compare", str(tampered)])
            self.assertEqual(rc_tampered, 1)
            self.assertIn("ERROR", err_tampered.getvalue())


if __name__ == "__main__":
    unittest.main()
