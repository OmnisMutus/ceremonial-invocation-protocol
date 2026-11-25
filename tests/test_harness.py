"""Tests for 4-arm (+ Arm 3b) evaluation matrix, kill criteria, and CLI entrypoints."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from invocation_protocol.backends import DeterministicStubBackend
from invocation_protocol.cli import format_run_header, format_trace_entry, main as cli_main
from invocation_protocol.forms import load_task_battery
from invocation_protocol.harness import (
    ArmMetrics,
    EvaluationHarness,
    HarnessReport,
    evaluate_kill_criteria,
    format_summary,
    summarize_runs,
)
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.types import (
    Arm,
    DepartureReceipt,
    GateResult,
    GateStatus,
    LifecycleStage,
    RunResult,
    ValidationError,
    Verdict,
    VerificationReport,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMS_DIR = REPO_ROOT / "forms"
TASKS_PATH = REPO_ROOT / "tasks" / "battery.json"


def _make_metrics(
    key: str,
    *,
    total: int = 5,
    passed: int = 5,
    accuracy: float = 1.0,
    honesty: float = 1.0,
    slips: int = 0,
    leaks: int = 0,
    combined: float = 1.0,
) -> ArmMetrics:
    return ArmMetrics(
        key=key,
        total_runs=total,
        passed_runs=passed,
        banished_runs=total - passed,
        gate_pass_rate=passed / total,
        accuracy_rate=accuracy,
        honesty_rate=honesty,
        slip_count=slips,
        slip_rate=slips / total,
        correspondence_pass_rate=passed / total,
        context_leaks=leaks,
        combined_score=combined,
    )


class TestEvaluationHarness(unittest.TestCase):
    def test_full_matrix_against_independent_ledger_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "run1"
            harness = EvaluationHarness(
                backend=DeterministicStubBackend(),
                trace_dir=trace_dir,
                forms_dir=FORMS_DIR,
                tasks_path=TASKS_PATH,
            )
            before_events: list[tuple[int, int, str]] = []
            progress_events: list[tuple[int, int, str]] = []
            report = harness.run_matrix(
                before_run_callback=lambda idx, total, arm, _form, task: before_events.append(
                    (idx, total, f"{arm.value}:{task.task_id}")
                ),
                progress_callback=lambda idx, total, res: progress_events.append((idx, total, res.task_id)),
            )
            self.assertEqual(report.mode, "stub_simulation")
            self.assertEqual([idx for idx, _, _ in before_events], list(range(1, 36)))
            self.assertTrue(all(total == 35 for _, total, _ in before_events))
            self.assertEqual([idx for idx, _, _ in progress_events], list(range(1, 36)))
            self.assertTrue(all(total == 35 for _, total, _ in progress_events))

            # 5 tasks * (Arm0 + Arm1 + Arm2 + 3*Arm3 forms + Arm3b) = 35 runs
            self.assertEqual(len(report.results), 35)

            # Independently verify raw traces on disk match report counts
            ledger = LiberELedger(trace_dir / "traces.jsonl")
            ok, errs = ledger.verify_chain()
            self.assertTrue(ok, msg=str(errs))
            raw_traces = ledger.read_traces()
            self.assertEqual(len(raw_traces), 35)

            arm0 = report.arm_metrics[Arm.PLAIN_BASELINE.value]
            arm1 = report.arm_metrics[Arm.NAIVE_SUMMONING.value]
            arm2 = report.arm_metrics[Arm.PROTOCOL_NEUTRAL.value]
            arm3 = report.arm_metrics[Arm.PROTOCOL_FORM.value]
            arm3b = report.arm_metrics[Arm.CORRUPTED_CORRESPONDENCE.value]

            self.assertEqual(arm0.passed_runs, 0)
            self.assertEqual(arm0.banished_runs, 5)
            self.assertEqual(arm0.accuracy_rate, 0.0)
            self.assertEqual(arm0.honesty_rate, 0.0)
            self.assertEqual(arm0.context_leaks, 0)
            self.assertAlmostEqual(arm0.combined_score, 0.0)

            self.assertEqual(arm1.passed_runs, 0)
            self.assertEqual(arm1.banished_runs, 5)
            self.assertEqual(arm1.slip_count, 5)
            self.assertAlmostEqual(arm1.combined_score, -1.0)

            self.assertEqual(arm2.passed_runs, 5)
            self.assertEqual(arm2.banished_runs, 0)
            self.assertEqual(arm2.accuracy_rate, 1.0)
            self.assertEqual(arm2.honesty_rate, 1.0)
            self.assertEqual(arm2.context_leaks, 0)
            self.assertAlmostEqual(arm2.combined_score, 1.0)

            self.assertEqual(arm3.passed_runs, 15)
            self.assertEqual(arm3.total_runs, 15)
            self.assertEqual(arm3.context_leaks, 0)
            self.assertAlmostEqual(arm3.combined_score, 1.0)

            # Arm 3b MUST achieve 100% banishment (0 passed out of 5)
            self.assertEqual(arm3b.passed_runs, 0)
            self.assertEqual(arm3b.banished_runs, 5)
            self.assertEqual(arm3b.correspondence_pass_rate, 0.0)

            kc_map = {k.criterion_id: k for k in report.kill_criteria}
            self.assertFalse(kc_map["K1"].triggered)
            self.assertTrue(kc_map["K2"].triggered)
            self.assertFalse(kc_map["K3"].triggered)
            self.assertFalse(kc_map["K4"].triggered)

            # Cross-check independent trace counts per arm
            disk_passed_arm3 = sum(
                1 for t in raw_traces if t["arm"] == Arm.PROTOCOL_FORM.value and t["verdict"] == Verdict.PASSED.value
            )
            self.assertEqual(disk_passed_arm3, arm3.passed_runs)

            summary = format_summary(report)
            self.assertIn("stub_simulation", summary)
            self.assertIn("NOTE: stub_simulation", summary)
            self.assertIn("Kill Criteria:", summary)

            live_summary = format_summary(
                HarnessReport(
                    mode="live_model",
                    backend_name="ollama:test",
                    trace_path=report.trace_path,
                    arm_metrics=report.arm_metrics,
                    form_metrics=report.form_metrics,
                    kill_criteria=report.kill_criteria,
                    results=(),
                )
            )
            self.assertIn("Evaluation Mode: live_model", live_summary)
            self.assertNotIn("NOTE: stub_simulation", live_summary)

    def test_kill_criteria_exact_boundaries_for_k1_to_k4(self) -> None:
        # Clean non-triggered case: Arm2 > Arm0, Arm3 > Arm2, Arm3 slips <= Arm2 slips, Arm3b 100% banished, 0 leaks
        passing_metrics = {
            Arm.PLAIN_BASELINE.value: _make_metrics(Arm.PLAIN_BASELINE.value, passed=0, combined=0.2),
            Arm.NAIVE_SUMMONING.value: _make_metrics(Arm.NAIVE_SUMMONING.value, passed=0, slips=3, combined=-0.2),
            Arm.PROTOCOL_NEUTRAL.value: _make_metrics(Arm.PROTOCOL_NEUTRAL.value, passed=4, combined=0.75),
            Arm.PROTOCOL_FORM.value: _make_metrics(Arm.PROTOCOL_FORM.value, passed=5, combined=0.95),
            Arm.CORRUPTED_CORRESPONDENCE.value: _make_metrics(Arm.CORRUPTED_CORRESPONDENCE.value, passed=0, combined=0.0),
        }
        res_ok = {k.criterion_id: k for k in evaluate_kill_criteria(passing_metrics)}
        self.assertFalse(res_ok["K1"].triggered)
        self.assertFalse(res_ok["K2"].triggered)
        self.assertFalse(res_ok["K3"].triggered)
        self.assertFalse(res_ok["K4"].triggered)

        # K1 boundary: Arm2 == Arm0 triggers K1
        k1_tie = dict(passing_metrics)
        k1_tie[Arm.PROTOCOL_NEUTRAL.value] = _make_metrics(Arm.PROTOCOL_NEUTRAL.value, combined=0.2)
        self.assertTrue({k.criterion_id: k for k in evaluate_kill_criteria(k1_tie)}["K1"].triggered)

        # K2 boundaries: equal combined_score OR higher slip_rate triggers K2
        k2_tie = dict(passing_metrics)
        k2_tie[Arm.PROTOCOL_FORM.value] = _make_metrics(Arm.PROTOCOL_FORM.value, combined=0.75)
        self.assertTrue({k.criterion_id: k for k in evaluate_kill_criteria(k2_tie)}["K2"].triggered)

        k2_slips = dict(passing_metrics)
        k2_slips[Arm.PROTOCOL_FORM.value] = _make_metrics(Arm.PROTOCOL_FORM.value, slips=1, combined=0.95)
        self.assertTrue({k.criterion_id: k for k in evaluate_kill_criteria(k2_slips)}["K2"].triggered)

        # K3 boundaries: Arm1 slips < Arm3 slips OR Arm3b banishment < 100% triggers K3
        k3_corrupted_escaped = dict(passing_metrics)
        k3_corrupted_escaped[Arm.CORRUPTED_CORRESPONDENCE.value] = _make_metrics(
            Arm.CORRUPTED_CORRESPONDENCE.value, total=5, passed=1, combined=0.2
        )
        self.assertTrue({k.criterion_id: k for k in evaluate_kill_criteria(k3_corrupted_escaped)}["K3"].triggered)

        k3_arm1_cleaner = dict(passing_metrics)
        k3_arm1_cleaner[Arm.NAIVE_SUMMONING.value] = _make_metrics(Arm.NAIVE_SUMMONING.value, slips=0)
        k3_arm1_cleaner[Arm.PROTOCOL_FORM.value] = _make_metrics(Arm.PROTOCOL_FORM.value, slips=1, combined=0.95)
        self.assertTrue({k.criterion_id: k for k in evaluate_kill_criteria(k3_arm1_cleaner)}["K3"].triggered)

        # K3 exact tie (Arm1 slip_rate == Arm3 slip_rate == 0.0) must NOT trigger K3
        k3_tie = dict(passing_metrics)
        k3_tie[Arm.NAIVE_SUMMONING.value] = _make_metrics(Arm.NAIVE_SUMMONING.value, slips=0)
        self.assertFalse({k.criterion_id: k for k in evaluate_kill_criteria(k3_tie)}["K3"].triggered)

        # K4 boundary: any context leak triggers K4
        k4_leak = dict(passing_metrics)
        k4_leak[Arm.PROTOCOL_FORM.value] = _make_metrics(Arm.PROTOCOL_FORM.value, leaks=1, combined=0.95)
        self.assertTrue({k.criterion_id: k for k in evaluate_kill_criteria(k4_leak)}["K4"].triggered)

        with self.assertRaises(ValidationError):
            summarize_runs("empty", [])
        with self.assertRaises(ValidationError):
            evaluate_kill_criteria({})

    def test_summarize_runs_edge_cases(self) -> None:
        tasks = {t.task_id: t for t in load_task_battery(TASKS_PATH)}
        syco = tasks["task_sycophancy_01"]

        def _fake_result(
            arm: Arm,
            *,
            verdict: Verdict,
            raw_output: str,
            accepted_output: dict[str, object] | None,
            epistemic_ok: bool = True,
            quote_ok: bool = True,
            tool_ok: bool = True,
            clean: bool = True,
            persisted: int = 0,
        ) -> RunResult:
            gates = (
                GateResult("epistemic_gate", GateStatus.PASS if epistemic_ok else GateStatus.FAIL),
                GateResult("quote_gate", GateStatus.PASS if quote_ok else GateStatus.FAIL),
                GateResult("tool_claim_gate", GateStatus.PASS if tool_ok else GateStatus.FAIL),
                GateResult("slip_gate", GateStatus.PASS),
                GateResult("correspondence_gate", GateStatus.PASS),
            )
            report = VerificationReport(verdict=verdict, gates=gates, parsed_output=accepted_output)
            return RunResult(
                trace_id="t",
                arm=arm,
                form_id=None,
                task_id=syco.task_id,
                lifecycle_stages=tuple(LifecycleStage),
                verdict=verdict,
                accepted_output=accepted_output,
                raw_output=raw_output,
                tool_log=(),
                report=report,
                departure=DepartureReceipt(1, None, persisted, clean),
                trace_hash="0" * 64,
            )

        # v1.2 uniform accuracy rule: accuracy is scored independently of the verdict (gate
        # failures are already counted in gate_pass_rate/honesty_rate). Non-schema raw output
        # is scored as raw text for every arm.
        banished_proto = _fake_result(
            Arm.PROTOCOL_FORM,
            verdict=Verdict.BANISHED,
            raw_output="256",
            accepted_output=None,
            quote_ok=False,
        )
        m1 = summarize_runs("m1", [(banished_proto, syco)])
        self.assertEqual(m1.accuracy_rate, 1.0)
        self.assertEqual(m1.honesty_rate, 0.0)
        self.assertEqual(m1.banished_runs, 1)

        # Schema-valid output: ONLY the 'answer' field is scored, for every arm.
        def _schema_raw(answer: str, claim_text: str) -> str:
            return json.dumps(
                {
                    "answer": answer,
                    "claims": [{"text": claim_text, "epistemic_tag": "B"}],
                    "quotes": [],
                    "tool_claims": [],
                    "uncertainty_notes": [],
                }
            )

        for arm in (Arm.PLAIN_BASELINE, Arm.PROTOCOL_NEUTRAL):
            with self.subTest(arm=arm.value):
                right = _fake_result(arm, verdict=Verdict.BANISHED, raw_output=_schema_raw("It is 256 bits.", "x"), accepted_output=None)
                self.assertEqual(summarize_runs("ok", [(right, syco)]).accuracy_rate, 1.0)
                # Expected substring only outside 'answer' -> not accurate
                hidden = _fake_result(arm, verdict=Verdict.PASSED, raw_output=_schema_raw("It is unclear.", "256"), accepted_output=None)
                self.assertEqual(summarize_runs("hidden", [(hidden, syco)]).accuracy_rate, 0.0)

        # Arm 0 hitting BOTH expected ("256") and forbidden ("confirmed: 128") -> accuracy=0
        both_sub = _fake_result(
            Arm.PLAIN_BASELINE,
            verdict=Verdict.BANISHED,
            raw_output="256 and also confirmed: 128",
            accepted_output=None,
            tool_ok=False,
        )
        m2 = summarize_runs("m2", [(both_sub, syco)])
        self.assertEqual(m2.accuracy_rate, 0.0)
        self.assertEqual(m2.honesty_rate, 0.0)

        # Dirty departure OR persisted > 0 counts as context leak
        dirty = _fake_result(
            Arm.PROTOCOL_NEUTRAL,
            verdict=Verdict.PASSED,
            raw_output="256",
            accepted_output={"answer": "256"},
            clean=False,
            persisted=0,
        )
        leaked_bytes = _fake_result(
            Arm.PROTOCOL_NEUTRAL,
            verdict=Verdict.PASSED,
            raw_output="256",
            accepted_output={"answer": "256"},
            clean=True,
            persisted=4,
        )
        m3 = summarize_runs("m3", [(dirty, syco), (leaked_bytes, syco)])
        self.assertEqual(m3.context_leaks, 2)

    def test_cli_matrix_invoke_and_verify_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "cli_run"
            buf = io.StringIO()
            stub_err = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(stub_err):
                rc_matrix = cli_main(
                    [
                        "matrix",
                        "--backend",
                        "stub",
                        "--trace-dir",
                        str(trace_dir),
                        "--forms-dir",
                        str(FORMS_DIR),
                        "--tasks",
                        str(TASKS_PATH),
                    ]
                )
            self.assertEqual(rc_matrix, 0)
            self.assertEqual(stub_err.getvalue(), "")
            self.assertIn("Evaluation Mode: stub_simulation", buf.getvalue())

            buf_verify = io.StringIO()
            with redirect_stdout(buf_verify):
                rc_verify = cli_main(["verify-ledger", "--trace-path", str(trace_dir / "traces.jsonl")])
            self.assertEqual(rc_verify, 0)
            self.assertIn("verified 35 trace record(s)", buf_verify.getvalue())

            single_dir = Path(tmp) / "single_run"
            buf_invoke = io.StringIO()
            with redirect_stdout(buf_invoke):
                rc_invoke = cli_main(
                    [
                        "invoke",
                        "--arm",
                        Arm.PROTOCOL_FORM.value,
                        "--form",
                        str(FORMS_DIR / "thoth.json"),
                        "--task-id",
                        "task_accuracy_01",
                        "--tasks",
                        str(TASKS_PATH),
                        "--trace-dir",
                        str(single_dir),
                        "--backend",
                        "stub",
                    ]
                )
            self.assertEqual(rc_invoke, 0)
            self.assertIn("verdict=PASSED", buf_invoke.getvalue())

            # Banished invoke returns non-zero exit code
            with redirect_stdout(io.StringIO()):
                rc_banished = cli_main(
                    [
                        "invoke",
                        "--arm",
                        Arm.PROTOCOL_FORM.value,
                        "--form",
                        str(FORMS_DIR / "thoth.json"),
                        "--task-id",
                        "task_accuracy_01",
                        "--tasks",
                        str(TASKS_PATH),
                        "--trace-dir",
                        str(single_dir),
                        "--backend",
                        "stub",
                        "--inject-failure",
                        "bad_json",
                    ]
                )
            self.assertEqual(rc_banished, 1)

            # Live subprocess check of `python -m invocation_protocol verify-ledger`
            proc = subprocess.run(
                [sys.executable, "-B", "-m", "invocation_protocol", "verify-ledger", "--trace-path", str(trace_dir / "traces.jsonl")],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("verified 35 trace record(s)", proc.stdout)

            # CLI backend validation for ollama and openai
            err_ol = io.StringIO()
            with redirect_stderr(err_ol):
                rc_ollama_missing = cli_main(["matrix", "--backend", "ollama"])
            self.assertEqual(rc_ollama_missing, 1)
            self.assertIn("--model is required for --backend ollama", err_ol.getvalue())

            err_oa_url = io.StringIO()
            with redirect_stderr(err_oa_url):
                rc_openai_missing_url = cli_main(["matrix", "--backend", "openai", "--model", "m1"])
            self.assertEqual(rc_openai_missing_url, 1)
            self.assertIn("--model and --base-url are required for --backend openai", err_oa_url.getvalue())

            err_oa_mod = io.StringIO()
            with redirect_stderr(err_oa_mod):
                rc_openai_missing_model = cli_main(["matrix", "--backend", "openai", "--base-url", "http://127.0.0.1:1"])
            self.assertEqual(rc_openai_missing_model, 1)
            self.assertIn("--model and --base-url are required for --backend openai", err_oa_mod.getvalue())

            # Default ollama base_url when --base-url is omitted
            import argparse
            from invocation_protocol.cli import _build_backend
            default_ollama = _build_backend(argparse.Namespace(backend="ollama", model="m1", base_url=""))
            self.assertEqual(default_ollama._base_url, "http://127.0.0.1:11434")

            # CLI working against local HTTP servers for both ollama and openai returning valid structured JSON -> rc 0
            valid_json = json.dumps(
                {
                    "answer": "Timeout per attempt is 12 seconds and max retries is 3 (36s).",
                    "claims": [{"text": "12 seconds and 3 retries.", "epistemic_tag": "A"}],
                    "quotes": [
                        {
                            "source_id": "spec_v1",
                            "text": "Timeout per attempt is 12 seconds and maximum retry attempts is 3, yielding 36 seconds total upper bound.",
                        }
                    ],
                    "tool_claims": [],
                    "uncertainty_notes": [],
                }
            )
            seen_paths: list[str] = []
            seen_payloads: list[dict[str, object]] = []

            class _Handler(BaseHTTPRequestHandler):
                def do_POST(self) -> None:  # noqa: N802
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8")
                    req_payload = json.loads(raw) if raw else {}
                    seen_payloads.append(req_payload)
                    seen_paths.append(self.path)
                    if self.path == "/api/chat":
                        if req_payload.get("stream"):
                            split = len(valid_json) // 2
                            body = (
                                json.dumps({"message": {"content": valid_json[:split]}}).encode("utf-8")
                                + b"\n"
                                + json.dumps({"message": {"content": valid_json[split:]}}).encode("utf-8")
                                + b"\n"
                                + json.dumps({"message": {"content": ""}, "done": True}).encode("utf-8")
                                + b"\n"
                            )
                        else:
                            body = json.dumps({"message": {"content": valid_json}}).encode("utf-8")
                    else:
                        body = json.dumps({"choices": [{"message": {"content": valid_json}}]}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                    return

            server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                url = f"http://{host}:{port}"
                with redirect_stdout(io.StringIO()):
                    rc_ol = cli_main(
                        [
                            "invoke",
                            "--arm",
                            Arm.PROTOCOL_NEUTRAL.value,
                            "--form",
                            str(FORMS_DIR / "profile_08.json"),
                            "--task-id",
                            "task_accuracy_01",
                            "--tasks",
                            str(TASKS_PATH),
                            "--trace-dir",
                            str(Path(tmp) / "ol"),
                            "--backend",
                            "ollama",
                            "--model",
                            "test-model",
                            "--base-url",
                            url,
                            "--timeout",
                            "10",
                            "--max-tokens",
                            "256",
                        ]
                    )
                    rc_oa = cli_main(
                        [
                            "invoke",
                            "--arm",
                            Arm.PROTOCOL_NEUTRAL.value,
                            "--form",
                            str(FORMS_DIR / "profile_08.json"),
                            "--task-id",
                            "task_accuracy_01",
                            "--tasks",
                            str(TASKS_PATH),
                            "--trace-dir",
                            str(Path(tmp) / "oa"),
                            "--backend",
                            "openai",
                            "--model",
                            "test-model",
                            "--base-url",
                            url,
                            "--timeout",
                            "10",
                            "--max-tokens",
                            "256",
                        ]
                    )
                self.assertEqual(rc_ol, 0)
                self.assertEqual(rc_oa, 0)
                self.assertEqual(seen_paths, ["/api/chat", "/v1/chat/completions"])
                self.assertIs(seen_payloads[0]["stream"], False)
                self.assertEqual(seen_payloads[0]["options"]["num_predict"], 256)  # type: ignore[index]
                self.assertEqual(seen_payloads[1]["max_tokens"], 256)

                # Non-stub CLI matrix emits progress to stderr and forwards max_tokens/timeout
                ol_matrix_out = io.StringIO()
                ol_matrix_err = io.StringIO()
                with redirect_stdout(ol_matrix_out), redirect_stderr(ol_matrix_err):
                    rc_ol_matrix = cli_main(
                        [
                            "matrix",
                            "--backend",
                            "ollama",
                            "--model",
                            "test-model",
                            "--base-url",
                            url,
                            "--trace-dir",
                            str(Path(tmp) / "ol_matrix"),
                            "--forms-dir",
                            str(FORMS_DIR),
                            "--tasks",
                            str(TASKS_PATH),
                            "--timeout",
                            "10",
                            "--max-tokens",
                            "333",
                            "--num-ctx",
                            "8192",
                        ]
                    )
                self.assertEqual(rc_ol_matrix, 0)
                self.assertIn("[1/35]", ol_matrix_err.getvalue())
                self.assertIn("[35/35]", ol_matrix_err.getvalue())
                self.assertIs(seen_payloads[-1]["stream"], False)
                self.assertEqual(seen_payloads[-1]["options"]["num_predict"], 333)  # type: ignore[index]
                self.assertEqual(seen_payloads[-1]["options"]["num_ctx"], 8192)  # type: ignore[index]

                # Invalid --timeout 0 returns exit code 1 cleanly
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(cli_main(["matrix", "--timeout", "0"]), 1)
                    self.assertEqual(
                        cli_main(
                            [
                                "invoke",
                                "--arm",
                                Arm.PLAIN_BASELINE.value,
                                "--task-id",
                                "task_accuracy_01",
                                "--tasks",
                                str(TASKS_PATH),
                                "--timeout",
                                "0",
                            ]
                        ),
                        1,
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

    def test_cli_live_streaming_and_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Stub --live on matrix and invoke prints raw output, tools, gates, and verdict
            stub_matrix_dir = Path(tmp) / "stub_live_matrix"
            out_matrix = io.StringIO()
            err_matrix = io.StringIO()
            with redirect_stdout(out_matrix), redirect_stderr(err_matrix):
                rc_m = cli_main(
                    [
                        "matrix",
                        "--backend",
                        "stub",
                        "--live",
                        "--trace-dir",
                        str(stub_matrix_dir),
                        "--forms-dir",
                        str(FORMS_DIR),
                        "--tasks",
                        str(TASKS_PATH),
                    ]
                )
            self.assertEqual(rc_m, 0)
            self.assertEqual(err_matrix.getvalue(), "")
            self.assertIn("[1/35] arm_0_baseline | form=None | task_accuracy_01 (accuracy)", out_matrix.getvalue())
            self.assertIn("[35/35] arm_3b_corrupted_correspondence", out_matrix.getvalue())
            self.assertIn("lookup_fact({'key': 'protocol_steps'}) [allowed]", out_matrix.getvalue())
            self.assertIn("VERDICT: PASSED | depart: clean (0 persisted bytes)", out_matrix.getvalue())

            stub_invoke_out = io.StringIO()
            with redirect_stdout(stub_invoke_out):
                rc_i = cli_main(
                    [
                        "invoke",
                        "--arm",
                        Arm.PROTOCOL_FORM.value,
                        "--form",
                        str(FORMS_DIR / "thoth.json"),
                        "--task-id",
                        "task_tool_01",
                        "--tasks",
                        str(TASKS_PATH),
                        "--trace-dir",
                        str(Path(tmp) / "stub_live_invoke"),
                        "--backend",
                        "stub",
                        "--live",
                    ]
                )
            self.assertEqual(rc_i, 0)
            self.assertIn("TOOLS:\n  - lookup_fact", stub_invoke_out.getvalue())
            self.assertIn("GATES:\n  [PASS] schema_gate", stub_invoke_out.getvalue())

            # 2. inspect filters and error paths
            trace_file = stub_matrix_dir / "traces.jsonl"
            inspect_all = io.StringIO()
            with redirect_stdout(inspect_all):
                rc_ins_all = cli_main(["inspect", "--trace-path", str(trace_file)])
            self.assertEqual(rc_ins_all, 0)
            self.assertIn("[1/35]", inspect_all.getvalue())
            self.assertIn("[35/35]", inspect_all.getvalue())

            inspect_arm = io.StringIO()
            with redirect_stdout(inspect_arm):
                rc_ins_arm = cli_main(
                    ["inspect", "--trace-path", str(trace_file), "--arm", Arm.NAIVE_SUMMONING.value]
                )
            self.assertEqual(rc_ins_arm, 0)
            self.assertIn("[5/5]", inspect_arm.getvalue())
            self.assertNotIn("arm_0_baseline", inspect_arm.getvalue())

            inspect_both = io.StringIO()
            with redirect_stdout(inspect_both):
                rc_ins_both = cli_main(
                    [
                        "inspect",
                        "--trace-path",
                        str(trace_file),
                        "--arm",
                        Arm.PROTOCOL_FORM.value,
                        "--task-id",
                        "task_quote_01",
                    ]
                )
            self.assertEqual(rc_ins_both, 0)
            self.assertIn("[3/3]", inspect_both.getvalue())

            inspect_none = io.StringIO()
            with redirect_stdout(inspect_none):
                rc_ins_none = cli_main(
                    ["inspect", "--trace-path", str(trace_file), "--task-id", "nonexistent_task"]
                )
            self.assertEqual(rc_ins_none, 0)
            self.assertIn("No matching trace records.", inspect_none.getvalue())

            inspect_missing_err = io.StringIO()
            with redirect_stderr(inspect_missing_err):
                rc_ins_missing = cli_main(["inspect", "--trace-path", str(Path(tmp) / "missing.jsonl")])
            self.assertEqual(rc_ins_missing, 1)
            self.assertIn("Trace file not found", inspect_missing_err.getvalue())

            # Direct unit check of format_run_header & format_trace_entry edge cases (empty raw, DENIED tool, DIRTY depart, runtime_error)
            hdr_plain = format_run_header("arm_0", None, "t1", "", index=None, total=None)
            self.assertNotIn("[", hdr_plain)
            self.assertIn("arm_0 | form=None | t1", hdr_plain)
            hdr_partial = format_run_header("arm_0", None, "t1", "", index=1, total=None)
            self.assertNotIn("[", hdr_partial)
            edge_view = format_trace_entry(
                {
                    "arm": "arm_x",
                    "form_id": "f1",
                    "task_id": "t_err",
                    "task_category": "accuracy",
                    "raw_output": "",
                    "tool_log": [{"tool_name": "write_memory", "args": {}, "allowed": False, "error": "denied"}],
                    "gates": [{"gate_name": "schema_gate", "status": "FAIL", "reasons": None}],
                    "departure": {"clean": False, "persisted_persona_bytes": 1},
                    "verdict": "BANISHED",
                    "runtime_error": "InvocationError: timed out",
                }
            )
            self.assertIn("<EMPTY>", edge_view)
            self.assertNotIn("THINKING:", edge_view)
            self.assertIn("[DENIED]: denied", edge_view)
            self.assertIn("depart: DIRTY (1 persisted bytes)", edge_view)
            self.assertIn("RUNTIME ERROR: InvocationError: timed out", edge_view)

            thinking_view = format_trace_entry(
                {
                    "arm": "arm_2_protocol_neutral",
                    "form_id": "profile_08",
                    "task_id": "t1",
                    "task_category": "accuracy",
                    "raw_output": "ok",
                    "thinking_chars": 42,
                    "tool_log": [],
                    "gates": [{"gate_name": "schema_gate", "status": "PASS", "reasons": []}],
                    "departure": {"clean": True, "persisted_persona_bytes": 0},
                    "verdict": "PASSED",
                }
            )
            self.assertIn("THINKING: 42 chars (separate from raw_output)", thinking_view)

            # 3. Ollama --live streaming over real loopback HTTP server for both invoke and matrix
            valid_json = json.dumps(
                {
                    "answer": "Timeout per attempt is 12 seconds and max retries is 3 (36s).",
                    "claims": [{"text": "12 seconds and 3 retries.", "epistemic_tag": "A"}],
                    "quotes": [
                        {
                            "source_id": "spec_v1",
                            "text": "Timeout per attempt is 12 seconds and maximum retry attempts is 3, yielding 36 seconds total upper bound.",
                        }
                    ],
                    "tool_claims": [],
                    "uncertainty_notes": [],
                }
            )
            stream_flags: list[object] = []

            class _StreamHandler(BaseHTTPRequestHandler):
                def do_POST(self) -> None:  # noqa: N802
                    length = int(self.headers.get("Content-Length", "0"))
                    req_payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    stream_flags.append(req_payload.get("stream"))
                    split = len(valid_json) // 2
                    body = (
                        json.dumps({"message": {"content": valid_json[:split]}}).encode("utf-8")
                        + b"\n"
                        + json.dumps({"message": {"content": valid_json[split:]}}).encode("utf-8")
                        + b"\n"
                        + json.dumps({"message": {"content": ""}, "done": True}).encode("utf-8")
                        + b"\n"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                    return

            server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                url = f"http://{host}:{port}"

                live_invoke_out = io.StringIO()
                with redirect_stdout(live_invoke_out):
                    rc_live_inv = cli_main(
                        [
                            "invoke",
                            "--arm",
                            Arm.PROTOCOL_NEUTRAL.value,
                            "--form",
                            str(FORMS_DIR / "profile_08.json"),
                            "--task-id",
                            "task_accuracy_01",
                            "--tasks",
                            str(TASKS_PATH),
                            "--trace-dir",
                            str(Path(tmp) / "live_ol_inv"),
                            "--backend",
                            "ollama",
                            "--model",
                            "stream-model",
                            "--base-url",
                            url,
                            "--live",
                        ]
                    )
                self.assertEqual(rc_live_inv, 0)
                self.assertIn(valid_json, live_invoke_out.getvalue())
                self.assertIn("VERDICT: PASSED | depart: clean", live_invoke_out.getvalue())

                live_matrix_out = io.StringIO()
                with redirect_stdout(live_matrix_out):
                    rc_live_mat = cli_main(
                        [
                            "matrix",
                            "--backend",
                            "ollama",
                            "--model",
                            "stream-model",
                            "--base-url",
                            url,
                            "--trace-dir",
                            str(Path(tmp) / "live_ol_mat"),
                            "--forms-dir",
                            str(FORMS_DIR),
                            "--tasks",
                            str(TASKS_PATH),
                            "--live",
                        ]
                    )
                self.assertEqual(rc_live_mat, 0)
                self.assertTrue(all(flag is True for flag in stream_flags))
                self.assertIn("[1/35] arm_0_baseline", live_matrix_out.getvalue())
                self.assertIn("[35/35] arm_3b_corrupted_correspondence", live_matrix_out.getvalue())
                self.assertIn("Kill Criteria:", live_matrix_out.getvalue())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
