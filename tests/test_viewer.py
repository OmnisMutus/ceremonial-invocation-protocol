"""Tests for the self-contained HTML ledger viewer and the `view` CLI command."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from invocation_protocol.cli import main as cli_main
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.types import ValidationError
from invocation_protocol.viewer import (
    DATA_PLACEHOLDER,
    ledger_label,
    load_ledger_payload,
    render_viewer_html,
    write_viewer,
)


def _embedded_data(html: str) -> list:
    match = re.search(r"const DATA = (.*?);\n", html)
    assert match is not None
    return json.loads(match.group(1))


class TestViewer(unittest.TestCase):
    def test_ledger_label(self) -> None:
        self.assertEqual(ledger_label(Path("runs/live_qwen3/traces.jsonl")), "live_qwen3")
        self.assertEqual(ledger_label(Path("runs/other/custom.jsonl")), "custom")
        self.assertEqual(ledger_label(Path("traces.jsonl")), "traces")

    def test_payload_reports_verified_chain_and_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_a" / "traces.jsonl"
            ledger = LiberELedger(path)
            ledger.append_trace({"trace_id": "a", "note": "Δίκη"})
            head = ledger.append_trace({"trace_id": "b"})
            payload = load_ledger_payload(path)
            self.assertTrue(payload["verified"])
            self.assertEqual(payload["errors"], [])
            self.assertEqual(payload["head"], head)
            self.assertEqual(payload["label"], "run_a")
            self.assertEqual([e["trace_id"] for e in payload["entries"]], ["a", "b"])

    def test_payload_flags_tampered_and_malformed_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            LiberELedger(path).append_trace({"trace_id": "a", "verdict": "PASSED"})
            path.write_text(path.read_text(encoding="utf-8").replace("PASSED", "BANISHED"), encoding="utf-8")
            tampered = load_ledger_payload(path)
            self.assertFalse(tampered["verified"])
            self.assertTrue(any("entry_hash mismatch" in e for e in tampered["errors"]))
            self.assertEqual(len(tampered["entries"]), 1)

            path.write_text("{not json\n", encoding="utf-8")
            broken = load_ledger_payload(path)
            self.assertFalse(broken["verified"])
            self.assertEqual(broken["entries"], [])
            self.assertIsNone(broken["head"])

            with self.assertRaises(ValidationError):
                load_ledger_payload(Path(tmp) / "missing.jsonl")

    def test_render_escapes_markup_in_record_text(self) -> None:
        hostile = "</script><script>alert(1)</script><!-- x"
        payloads = [{"label": "x", "entries": [{"raw_output": hostile}]}]
        template = "<script>const DATA = " + DATA_PLACEHOLDER + ";\n</script>"
        html = render_viewer_html(payloads, template=template)
        self.assertEqual(html.count("</script>"), 1)
        self.assertNotIn("<!--", html)
        self.assertEqual(_embedded_data(html)[0]["entries"][0]["raw_output"], hostile)

        with self.assertRaises(ValidationError):
            render_viewer_html(payloads, template="<html>no placeholder</html>")

        packaged = render_viewer_html(payloads)
        self.assertIn("Liber E", packaged)
        self.assertNotIn(DATA_PLACEHOLDER, packaged)
        self.assertEqual(_embedded_data(packaged), payloads)

    def test_write_viewer_and_cli(self) -> None:
        with self.assertRaises(ValidationError):
            write_viewer([], "unused.html")
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "run_b" / "traces.jsonl"
            LiberELedger(trace).append_trace({"trace_id": "z"})
            out = Path(tmp) / "nested" / "viewer.html"

            buf = io.StringIO()
            with mock.patch("invocation_protocol.cli.webbrowser.open") as opener, redirect_stdout(buf):
                rc = cli_main(["view", str(trace), "--out", str(out)])
            self.assertEqual(rc, 0)
            opener.assert_not_called()
            self.assertIn("Wrote viewer for 1 ledger(s)", buf.getvalue())
            data = _embedded_data(out.read_text(encoding="utf-8"))
            self.assertEqual(data[0]["label"], "run_b")
            self.assertTrue(data[0]["verified"])

            with mock.patch("invocation_protocol.cli.webbrowser.open") as opener, redirect_stdout(io.StringIO()):
                cli_main(["view", str(trace), "--out", str(out), "--open"])
            opener.assert_called_once_with(out.resolve().as_uri())

            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                rc_missing = cli_main(["view", str(Path(tmp) / "nope.jsonl"), "--out", str(out)])
            self.assertEqual(rc_missing, 1)


if __name__ == "__main__":
    unittest.main()
