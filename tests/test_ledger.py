"""Tests for LiberELedger SHA-256 hash chaining, tamper detection, and commentary isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from invocation_protocol.ledger import GENESIS_HASH, LiberELedger
from invocation_protocol.types import ValidationError


class TestLiberELedger(unittest.TestCase):
    def test_chain_matches_independent_sha256_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"
            commentary_path = Path(tmp) / "commentary.jsonl"
            ledger = LiberELedger(trace_path, commentary_path)

            ok_empty, errs_empty = ledger.verify_chain()
            self.assertTrue(ok_empty)
            self.assertEqual(errs_empty, [])

            h1 = ledger.append_trace({"trace_id": "t1", "verdict": "PASSED", "note": "Δίκη 𓅝"})
            h2 = ledger.append_trace({"trace_id": "t2", "verdict": "BANISHED", "note": "second"})

            ok, errs = ledger.verify_chain()
            self.assertTrue(ok, msg=str(errs))
            self.assertEqual(errs, [])

            # Independent ground-truth SHA-256 computation directly from raw file bytes
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            expected_prev = GENESIS_HASH
            for expected_hash, raw_line in zip((h1, h2), lines):
                obj = json.loads(raw_line)
                self.assertEqual(obj["prev_hash"], expected_prev)
                self.assertEqual(obj["entry_hash"], expected_hash)
                body = {k: v for k, v in obj.items() if k not in {"prev_hash", "entry_hash"}}
                canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                recomputed = hashlib.sha256(f"{expected_prev}\n{canonical}".encode("utf-8")).hexdigest()
                self.assertEqual(recomputed, expected_hash)
                expected_prev = expected_hash

            # Commentary stays physically isolated and rejects missing/invalid inputs
            traces_before = trace_path.read_bytes()
            with self.assertRaises(ValidationError):
                ledger.append_commentary("missing_id", "alice", "note")
            with self.assertRaises(ValidationError):
                ledger.append_commentary("t1", "", "note")
            with self.assertRaises(ValidationError):
                ledger.append_commentary("t1", "alice", "   ")

            entry = ledger.append_commentary("t1", "researcher", "Verified manually")
            self.assertEqual(entry["trace_id"], "t1")
            self.assertEqual(trace_path.read_bytes(), traces_before)
            self.assertEqual(len(ledger.read_commentary()), 1)

            # Single-character tamper in line 1 invalidates chain
            tampered_obj = json.loads(lines[0])
            tampered_obj["verdict"] = "TAMPERED"
            tampered_text = json.dumps(tampered_obj, sort_keys=True, separators=(",", ":")) + "\n" + lines[1] + "\n"
            trace_path.write_text(tampered_text, encoding="utf-8")

            ok_after, errs_after = ledger.verify_chain()
            self.assertFalse(ok_after)
            self.assertGreater(len(errs_after), 0)

            # Corrupt JSON line detected cleanly
            trace_path.write_text("{not-json\n", encoding="utf-8")
            ok_bad_json, errs_bad_json = ledger.verify_chain()
            self.assertFalse(ok_bad_json)
            self.assertIn("invalid JSON", errs_bad_json[0])


if __name__ == "__main__":
    unittest.main()
