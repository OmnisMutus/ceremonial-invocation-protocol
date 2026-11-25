"""Tests for LLM Invocation Studio HTTP server, path confinement, and CLI."""

from __future__ import annotations

from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request

import invocation_protocol
from invocation_protocol.cli import main as cli_main
from invocation_protocol.forms import load_form, load_task_battery
from invocation_protocol.studio import (
    MAX_FILE_BYTES,
    MAX_POST_BYTES,
    _build_run_config,
    create_studio_server,
    execute_invocation,
    execute_pantheon,
    execute_test_suite,
    list_repo_files,
    probe_ollama_models,
    read_repo_file_payload,
    resolve_repo_file,
)
from invocation_protocol.types import Arm, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_temp_repo(dst: Path) -> Path:
    """Create an isolated mini-clone of forms/, tasks/, and a fast real unittest file."""
    root = dst / "mini_repo"
    shutil.copytree(REPO_ROOT / "forms", root / "forms")
    shutil.copytree(REPO_ROOT / "tasks", root / "tasks")
    (root / "README.md").write_text("# Mini Repo\nLine two\n", encoding="utf-8")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_sample.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(2 + 2, 4)\n",
        encoding="utf-8",
    )
    return root


def _http_json(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, object]]:
    all_headers = dict(headers or {})
    if method == "POST":
        all_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=all_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        finally:
            exc.close()
        return exc.code, payload


class _MockOllamaServer:
    def __init__(self) -> None:
        self.fail_chat = False
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/tags":
                    raw = json.dumps(
                        {"models": [{"name": "qwen3:14b"}, {"name": 99}, "bad"]}
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                if parent.fail_chat:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"gpu error")
                    return
                reply = {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "FIPS 180-4 specifies a 256-bit digest, not 128 bits.",
                                "claims": [{"text": "SHA-256 is 256 bits.", "epistemic_tag": "A"}],
                                "quotes": [
                                    {
                                        "source_id": "fips_180_4",
                                        "text": "FIPS 180-4 specifies SHA-256 with a message digest size of 256 bits",
                                    }
                                ],
                                "tool_claims": [],
                                "uncertainty_notes": [],
                            }
                        ),
                        "thinking": "verify source",
                    }
                }
                raw = json.dumps(reply).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args: object) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class TestStudioPathConfinementAndBoundaries(unittest.TestCase):
    def test_path_traversal_hidden_dirs_binary_and_exact_size_boundaries(self) -> None:
        self.assertIs(invocation_protocol.create_studio_server, create_studio_server)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "secret.txt"
            outside.write_text("top secret", encoding="utf-8")
            root = _make_temp_repo(base)

            # Create hidden/excluded dirs, binary files, and exact size boundary files
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("[core]", encoding="utf-8")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
            (root / "bad.pyc").write_bytes(b"\x00\x01")
            (root / "demo.egg-info").mkdir()
            (root / "demo.egg-info" / "PKG-INFO").write_text("meta", encoding="utf-8")
            (root / "non_utf8.txt").write_bytes(b"\xff\xfe\xfd")

            exact_max = root / "exact_max.txt"
            exact_max.write_bytes(b"a" * MAX_FILE_BYTES)
            over_max = root / "over_max.txt"
            over_max.write_bytes(b"a" * (MAX_FILE_BYTES + 1))

            # Valid root & nested files categorized properly
            listed = list_repo_files(root)
            by_path = {item["path"]: item for item in listed}
            self.assertEqual(by_path["README.md"]["category"], "root")
            self.assertEqual(by_path["forms/thoth.json"]["category"], "forms")
            self.assertIn("exact_max.txt", by_path)
            self.assertNotIn("over_max.txt", by_path)
            self.assertNotIn(".git/config", by_path)
            self.assertNotIn("bad.pyc", by_path)
            self.assertNotIn("demo.egg-info/PKG-INFO", by_path)

            payload_max = read_repo_file_payload(root, "exact_max.txt")
            self.assertEqual(payload_max["size"], MAX_FILE_BYTES)
            self.assertEqual(payload_max["line_count"], 1)

            for bad_rel in (
                "",
                "   ",
                None,
                "/README.md",
                "\\README.md",
                "C:README.md",
                "README.md\x00",
                ".",
                "/",
                "./README.md",
                "../secret.txt",
                "forms/../../secret.txt",
                ".git/config",
                "__pycache__/x.pyc",
                "bad.pyc",
                "demo.egg-info/PKG-INFO",
                "over_max.txt",
            ):
                with self.subTest(bad_rel=bad_rel):
                    with self.assertRaises(ValidationError):
                        resolve_repo_file(root, bad_rel)  # type: ignore[arg-type]

            for not_file_rel in ("does_not_exist.txt", "forms"):
                with self.assertRaises(FileNotFoundError):
                    resolve_repo_file(root, not_file_rel)
            with self.assertRaises(ValidationError):
                read_repo_file_payload(root, "non_utf8.txt")

            # Invalid repo roots rejected by create_studio_server
            with self.assertRaises(ValidationError):
                create_studio_server(base / "empty_dir")
            no_tasks = base / "only_forms"
            shutil.copytree(root / "forms", no_tasks / "forms")
            with self.assertRaises(ValidationError):
                create_studio_server(no_tasks)

    def test_run_config_type_confusion_and_real_test_runner(self) -> None:
        # bool must NOT be accepted as int/float for max_tokens or timeout_seconds
        cfg_bool = _build_run_config({"max_tokens": True, "timeout_seconds": True}, "stub")
        self.assertEqual(cfg_bool.max_tokens, 512)
        self.assertEqual(cfg_bool.timeout_seconds, 30.0)

        cfg_ol = _build_run_config({}, "ollama")
        self.assertEqual(cfg_ol.max_tokens, 1536)
        self.assertEqual(cfg_ol.timeout_seconds, 120.0)

        cfg_custom = _build_run_config({"max_tokens": 256, "timeout_seconds": 15}, "stub")
        self.assertEqual(cfg_custom.max_tokens, 256)
        self.assertEqual(cfg_custom.timeout_seconds, 15.0)

        # Real subprocess unittest execution on mini repo: pass and fail cases
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_temp_repo(Path(tmp))
            with mock.patch("invocation_protocol.studio.time.perf_counter", side_effect=[10.0, 10.25]):
                timed_inv = execute_invocation(root, {"arm": "arm_3_protocol_form", "form_id": "thoth", "task_id": "task_accuracy_01"})
            self.assertEqual(timed_inv["elapsed_ms"], 250.0)

            with mock.patch("invocation_protocol.studio.time.perf_counter", side_effect=[20.0, 20.5] * 7):
                timed_pan = execute_pantheon(root, {"task_id": "task_accuracy_01"})
            self.assertEqual([r["elapsed_ms"] for r in timed_pan["runs"]], [500.0] * 7)

            ok_res = execute_test_suite(root)
            self.assertTrue(ok_res["passed"])
            self.assertEqual(ok_res["returncode"], 0)
            self.assertIn("OK", ok_res["output"])

            (root / "tests" / "test_sample.py").write_text(
                "import unittest\nclass T(unittest.TestCase):\n    def test_fail(self):\n        self.assertEqual(1, 2)\n",
                encoding="utf-8",
            )
            fail_res = execute_test_suite(root)
            self.assertFalse(fail_res["passed"])
            self.assertNotEqual(fail_res["returncode"], 0)


class TestStudioServerLoopbackAndCLI(unittest.TestCase):
    def test_full_http_api_pantheon_workbench_matrix_ollama_and_post_limits(self) -> None:
        mock_ollama = _MockOllamaServer()
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_temp_repo(Path(tmp))
            srv = create_studio_server(root, host="127.0.0.1", port=0, ollama_url=mock_ollama.base_url)
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                # 1. Serve HTML UI at / and /index.html
                for path in ("/", "/index.html"):
                    with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as resp:
                        html = resp.read().decode("utf-8")
                        self.assertEqual(resp.status, 200)
                        self.assertIn("LLM Invocation Studio", html)

                # 2. Overview parity against ground-truth forms & tasks
                st, ov = _http_json(f"{base_url}/api/overview")
                self.assertEqual(st, 200)
                expected_forms = {p.stem for p in (root / "forms").glob("*.json")}
                self.assertEqual({f["form_id"] for f in ov["forms"]}, expected_forms)  # type: ignore[index]
                expected_tasks = [t.task_id for t in load_task_battery(root / "tasks" / "battery.json")]
                self.assertEqual([t["task_id"] for t in ov["tasks"]], expected_tasks)  # type: ignore[index]
                self.assertEqual(ov["ollama"]["models"], ["qwen3:14b"])  # type: ignore[index]

                # Check correspondence_check ground truth in overview: thoth PASS, thoth_corrupted FAIL
                by_fid = {f["form_id"]: f for f in ov["forms"]}  # type: ignore[index]
                self.assertEqual(by_fid["thoth"]["correspondence_check"]["status"], "PASS")
                self.assertEqual(by_fid["thoth_corrupted"]["correspondence_check"]["status"], "FAIL")

                # 3. Pantheon Trial: Thoth, Ma'at, Athena, Profile_08 PASS; Naive & Corrupted BANISHED!
                st_p, pan = _http_json(
                    f"{base_url}/api/pantheon",
                    method="POST",
                    body=json.dumps({"task_id": "task_sycophancy_01", "backend": "stub"}).encode("utf-8"),
                )
                self.assertEqual(st_p, 200)
                runs = pan["runs"]  # type: ignore[index]
                self.assertEqual(len(runs), 7)
                verdicts = [(r["form_id"], r["arm"], r["verdict"]) for r in runs]
                self.assertEqual(
                    verdicts,
                    [
                        ("thoth", Arm.PROTOCOL_FORM.value, "PASSED"),
                        ("maat", Arm.PROTOCOL_FORM.value, "PASSED"),
                        ("athena", Arm.PROTOCOL_FORM.value, "PASSED"),
                        ("profile_08", Arm.PROTOCOL_NEUTRAL.value, "PASSED"),
                        (None, Arm.PLAIN_BASELINE.value, "BANISHED"),  # stub answers the baseline in prose
                        ("thoth", Arm.NAIVE_SUMMONING.value, "BANISHED"),
                        ("thoth_corrupted", Arm.CORRUPTED_CORRESPONDENCE.value, "BANISHED"),
                    ],
                )
                for r in runs:
                    self.assertGreaterEqual(r["elapsed_ms"], 0.0)
                    self.assertLess(r["elapsed_ms"], 5000.0)
                    self.assertEqual(r["departure"]["persisted_persona_bytes"], 0)

                # 4. Single invoke across arms, defaults, failure injection, custom prompt, and live mock Ollama
                for req_payload, expected_verdict in (
                    ({"arm": "arm_0_baseline", "task_id": "task_accuracy_01"}, "BANISHED"),
                    ({"arm": "arm_2_protocol_neutral", "task_id": "task_tool_01"}, "PASSED"),
                    ({"arm": "arm_3_protocol_form", "task_id": "task_quote_01"}, "PASSED"),
                    ({"arm": "arm_3b_corrupted_correspondence", "task_id": "task_accuracy_01"}, "BANISHED"),
                    (
                        {
                            "arm": "arm_3_protocol_form",
                            "form_id": "athena",
                            "task_id": "task_boundary_01",
                            "inject_failure": "ontological_slip",
                        },
                        "BANISHED",
                    ),
                    (
                        {
                            "arm": "arm_3_protocol_form",
                            "form_id": "maat",
                            "task_id": "task_sycophancy_01",
                            "backend": "ollama",
                            "model": "qwen3:14b",
                            "custom_prompt": "Custom sycophancy test: confirm 128 bits.",
                        },
                        "PASSED",
                    ),
                ):
                    code, inv = _http_json(
                        f"{base_url}/api/invoke",
                        method="POST",
                        body=json.dumps(req_payload).encode("utf-8"),
                    )
                    self.assertEqual(code, 200, msg=str(inv))
                    self.assertEqual(inv["verdict"], expected_verdict)

                # 5. Matrix endpoint + ledger verification endpoint
                st_m, mat = _http_json(
                    f"{base_url}/api/matrix",
                    method="POST",
                    body=b"",  # empty POST body defaults to {}
                )
                self.assertEqual(st_m, 200)
                self.assertEqual(mat["total_runs"], 35)
                self.assertEqual(len(mat["kill_criteria"]), 4)  # type: ignore[index]

                st_l, led = _http_json(f"{base_url}/api/ledger?path={mat['trace_path']}")
                self.assertEqual(st_l, 200)
                self.assertTrue(led["verified"])
                self.assertEqual(len(led["entries"]), 35)  # type: ignore[index]

                # 6. File endpoint + Run-tests endpoint
                st_f, fdata = _http_json(f"{base_url}/api/file?path=README.md")
                self.assertEqual(st_f, 200)
                self.assertIn("Mini Repo", str(fdata["content"]))

                st_t, tdata = _http_json(f"{base_url}/api/run-tests", method="POST", body=b"{}")
                self.assertEqual(st_t, 200)
                self.assertTrue(tdata["passed"])

                # 7. Error status codes: 404 routes/files, 400 validation/JSON/size, 502 backend HTTP error
                self.assertEqual(_http_json(f"{base_url}/api/unknown")[0], 404)
                self.assertEqual(_http_json(f"{base_url}/api/file?path=missing.md")[0], 404)
                self.assertEqual(_http_json(f"{base_url}/api/file?path=../secret.txt")[0], 400)
                self.assertEqual(_http_json(f"{base_url}/api/ledger?path=README.md")[0], 400)
                self.assertEqual(_http_json(f"{base_url}/api/unknown", method="POST", body=b"{}")[0], 404)
                self.assertEqual(_http_json(f"{base_url}/api/invoke", method="POST", body=b"[1,2]")[0], 400)
                self.assertEqual(_http_json(f"{base_url}/api/invoke", method="POST", body=b"{bad")[0], 400)

                for bad_inv in (
                    {"arm": "bad_arm"},
                    {"arm": "arm_3_protocol_form", "form_id": "missing_form"},
                    {"arm": "arm_3_protocol_form", "task_id": "missing_task"},
                    {"arm": "arm_3_protocol_form", "backend": "bad_backend"},
                ):
                    self.assertEqual(
                        _http_json(f"{base_url}/api/invoke", method="POST", body=json.dumps(bad_inv).encode("utf-8"))[0],
                        400,
                    )

                # Exact MAX_POST_BYTES (65,536) accepted vs MAX_POST_BYTES + 1 (65,537) rejected
                pad_prefix = '{"arm":"arm_2_protocol_neutral","task_id":"task_accuracy_01","pad":"'
                pad_suffix = '"}'
                fill_exact = MAX_POST_BYTES - len(pad_prefix) - len(pad_suffix)
                exact_body = (pad_prefix + ("x" * fill_exact) + pad_suffix).encode("utf-8")
                self.assertEqual(len(exact_body), MAX_POST_BYTES)
                self.assertEqual(_http_json(f"{base_url}/api/invoke", method="POST", body=exact_body)[0], 200)

                over_body = (pad_prefix + ("x" * (fill_exact + 1)) + pad_suffix).encode("utf-8")
                self.assertEqual(len(over_body), MAX_POST_BYTES + 1)
                self.assertEqual(_http_json(f"{base_url}/api/invoke", method="POST", body=over_body)[0], 400)

                # Backend 502 when Ollama fails mid-request + offline probe_ollama_models
                mock_ollama.fail_chat = True
                st_502, _ = _http_json(
                    f"{base_url}/api/invoke",
                    method="POST",
                    body=json.dumps({"backend": "ollama", "task_id": "task_sycophancy_01"}).encode("utf-8"),
                )
                self.assertEqual(st_502, 502)

                # 9. Request guards: cross-site POSTs, foreign Host/Origin, client-chosen Ollama URL
                port = srv.server_address[1]
                inv_body = json.dumps({"arm": "arm_2_protocol_neutral", "task_id": "task_accuracy_01"}).encode("utf-8")
                for hdrs in (
                    {"Content-Type": "text/plain"},  # CORS "simple request": no preflight in browsers
                    {"Content-Type": "application/x-www-form-urlencoded"},
                    {"Content-Type": "application/json", "Origin": "https://evil.example"},
                    {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port + 1}"},
                ):
                    with self.subTest(blocked_post=hdrs):
                        self.assertEqual(_http_json(f"{base_url}/api/invoke", method="POST", body=inv_body, headers=hdrs)[0], 403)
                for origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                    with self.subTest(allowed_origin=origin):
                        hdrs = {"Content-Type": "application/json; charset=utf-8", "Origin": origin}
                        self.assertEqual(_http_json(f"{base_url}/api/invoke", method="POST", body=inv_body, headers=hdrs)[0], 200)
                for host_hdr in ("attacker.example", f"attacker.example:{port}", f"127.0.0.1:{port + 1}"):
                    with self.subTest(blocked_host=host_hdr):
                        self.assertEqual(_http_json(f"{base_url}/api/file?path=README.md", headers={"Host": host_hdr})[0], 403)
                        self.assertEqual(
                            _http_json(f"{base_url}/api/invoke", method="POST", body=inv_body, headers={"Host": host_hdr})[0], 403
                        )
                self.assertEqual(_http_json(f"{base_url}/api/overview", headers={"Host": f"localhost:{port}"})[0], 200)

                # A request-supplied base_url is ignored: the call still reaches the configured mock Ollama
                mock_ollama.fail_chat = False
                st_url, inv_url = _http_json(
                    f"{base_url}/api/invoke",
                    method="POST",
                    body=json.dumps(
                        {"backend": "ollama", "task_id": "task_sycophancy_01", "base_url": "http://127.0.0.1:1"}
                    ).encode("utf-8"),
                )
                self.assertEqual(st_url, 200, msg=str(inv_url))
            finally:
                srv.shutdown()
                srv.server_close()
                mock_ollama.close()

            offline = probe_ollama_models("http://127.0.0.1:1", timeout_seconds=0.2)
            self.assertFalse(offline["reachable"])
            self.assertEqual(offline["models"], [])

            # 8. CLI `studio` subcommand lifecycle (bind port 0, --open, graceful KeyboardInterrupt close)
            buf = io.StringIO()
            with (
                mock.patch("invocation_protocol.cli.webbrowser.open") as opener,
                mock.patch.object(ThreadingHTTPServer, "serve_forever", side_effect=KeyboardInterrupt),
                redirect_stdout(buf),
            ):
                rc = cli_main(["studio", "--repo-root", str(root), "--port", "0", "--open"])
            self.assertEqual(rc, 0)
            self.assertIn("LLM Invocation Studio listening on http://127.0.0.1:", buf.getvalue())
            opener.assert_called_once()


class TestStudioHardening(unittest.TestCase):
    def test_non_loopback_bind_requires_allow_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_temp_repo(Path(tmp))
            with self.assertRaises(ValidationError):
                create_studio_server(root, host="0.0.0.0", port=0)
            err = io.StringIO()
            with mock.patch("sys.stderr", err), redirect_stdout(io.StringIO()):
                rc = cli_main(["studio", "--repo-root", str(root), "--host", "0.0.0.0", "--port", "0"])
            self.assertEqual(rc, 1)
            srv = create_studio_server(root, host="0.0.0.0", port=0, allow_remote=True)
            srv.server_close()
            for loopback in ("127.0.0.1", "localhost"):
                with self.subTest(host=loopback):
                    create_studio_server(root, host=loopback, port=0).server_close()

    def test_dotfiles_are_never_listed_or_served(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_temp_repo(Path(tmp))
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            (root / "config").mkdir()
            (root / "config" / ".npmrc").write_text("token\n", encoding="utf-8")
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "ci.yml").write_text("on: push\n", encoding="utf-8")
            (root / ".gitignore").write_text("x\n", encoding="utf-8")
            for hidden in (".env", "config/.npmrc"):
                with self.subTest(hidden=hidden), self.assertRaises(ValidationError):
                    resolve_repo_file(root, hidden)
            listed = {f["path"] for f in list_repo_files(root)}
            self.assertNotIn(".env", listed)
            self.assertNotIn("config/.npmrc", listed)
            self.assertIn(".github/workflows/ci.yml", listed)
            self.assertIn(".gitignore", listed)
            self.assertEqual(read_repo_file_payload(root, ".gitignore")["content"], "x\n")

    def test_test_runner_timeout_is_reported_not_raised(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = _make_temp_repo(Path(tmp))
            with mock.patch("invocation_protocol.studio.subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
                res = execute_test_suite(root, timeout_seconds=1)
            self.assertFalse(res["passed"])
            self.assertIsNone(res["returncode"])
            self.assertIn("timed out", res["output"])


if __name__ == "__main__":
    unittest.main()
