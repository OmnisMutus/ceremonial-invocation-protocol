"""Append-only SHA-256 hash-chained Liber E ledger with separate human commentary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from invocation_protocol.types import ValidationError, require_non_empty_str

GENESIS_HASH: str = "0" * 64


def _canonical_json(obj: Mapping[str, Any]) -> str:
    """Serialize a mapping into deterministic compact UTF-8 JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_entry_hash(prev_hash: str, body: Mapping[str, Any]) -> str:
    """Compute SHA-256 chain hash over prev_hash and canonical JSON body."""
    payload = f"{prev_hash}\n{_canonical_json(body)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LiberELedger:
    """Manages immutable raw execution traces separately from human commentary."""

    def __init__(
        self,
        trace_path: str | Path,
        commentary_path: str | Path | None = None,
    ) -> None:
        self._trace_path = Path(trace_path)
        self._commentary_path = (
            Path(commentary_path)
            if commentary_path is not None
            else self._trace_path.with_name("commentary.jsonl")
        )

    @property
    def trace_path(self) -> Path:
        return self._trace_path

    @property
    def commentary_path(self) -> Path:
        return self._commentary_path

    def append_trace(self, record: Mapping[str, Any]) -> str:
        """Append a raw execution trace linked by SHA-256 prev_hash and return entry_hash."""
        if not isinstance(record, Mapping):
            raise ValidationError("Trace record must be a mapping")
        body = {k: v for k, v in record.items() if k not in {"prev_hash", "entry_hash"}}
        require_non_empty_str(body.get("trace_id"), "trace_id")

        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.read_traces()
        prev_hash = existing[-1]["entry_hash"] if existing else GENESIS_HASH
        entry_hash = _compute_entry_hash(prev_hash, body)

        envelope = dict(body)
        envelope["prev_hash"] = prev_hash
        envelope["entry_hash"] = entry_hash

        with self._trace_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(_canonical_json(envelope) + "\n")
        return entry_hash

    def read_traces(self) -> list[dict[str, Any]]:
        """Read all raw execution trace records in append order."""
        if not self._trace_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        raw_lines = self._trace_path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(raw_lines, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Malformed JSON in {self._trace_path} line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValidationError(f"Trace line {line_no} is not a JSON object")
            entries.append(obj)
        return entries

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify SHA-256 prev_hash and entry_hash integrity across the entire trace ledger."""
        if not self._trace_path.exists():
            return True, []

        errors: list[str] = []
        expected_prev = GENESIS_HASH
        raw_lines = self._trace_path.read_text(encoding="utf-8").splitlines()

        for line_no, line in enumerate(raw_lines, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_no}: invalid JSON ({exc.msg})")
                break
            if not isinstance(obj, dict):
                errors.append(f"Line {line_no}: expected JSON object")
                break

            actual_prev = obj.get("prev_hash")
            actual_hash = obj.get("entry_hash")
            if actual_prev != expected_prev:
                errors.append(f"Line {line_no}: prev_hash mismatch")

            body = {k: v for k, v in obj.items() if k not in {"prev_hash", "entry_hash"}}
            recomputed = _compute_entry_hash(expected_prev, body)
            if actual_hash != recomputed:
                errors.append(f"Line {line_no}: entry_hash mismatch")

            if isinstance(actual_hash, str):
                expected_prev = actual_hash

        return len(errors) == 0, errors

    def head(self) -> tuple[int, str]:
        """Return (record_count, head_entry_hash); GENESIS_HASH when the ledger is empty.

        The hash chain alone cannot detect records deleted from the end of the file, and
        anyone can recompute it after an edit. Publish the head hash out of band (README,
        release notes, a signed tag) and check it with ``verify-ledger --expect-head``.
        """
        entries = self.read_traces()
        if not entries:
            return 0, GENESIS_HASH
        return len(entries), str(entries[-1].get("entry_hash", ""))

    def append_commentary(self, trace_id: str, author: str, note: str) -> dict[str, Any]:
        """Append human commentary to commentary.jsonl without touching raw traces."""
        clean_trace_id = require_non_empty_str(trace_id, "trace_id")
        clean_author = require_non_empty_str(author, "author")
        clean_note = require_non_empty_str(note, "note")

        known_ids = {entry.get("trace_id") for entry in self.read_traces()}
        if clean_trace_id not in known_ids:
            raise ValidationError(f"Cannot attach commentary: trace_id {clean_trace_id!r} not in trace ledger")

        self._commentary_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "trace_id": clean_trace_id,
            "author": clean_author,
            "note": clean_note,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._commentary_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(_canonical_json(record) + "\n")
        return record

    def read_commentary(self) -> list[dict[str, Any]]:
        """Read all human commentary entries."""
        if not self._commentary_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line_no, line in enumerate(self._commentary_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValidationError(f"Commentary line {line_no} is not a JSON object")
            entries.append(obj)
        return entries
