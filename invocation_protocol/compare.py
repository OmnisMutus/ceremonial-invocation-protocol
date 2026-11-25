"""Multi-ledger summary and comparison utilities for Liber E traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from invocation_protocol.ledger import LiberELedger
from invocation_protocol.types import Arm, ValidationError


@dataclass(frozen=True)
class LedgerArmStats:
    """Aggregated evaluation metrics for one group (or whole ledger)."""

    key: str
    total: int
    passed: int
    banished: int
    honest: int
    slips: int
    correspondence_failures: int
    leaks: int
    mean_thinking_chars: float


@dataclass(frozen=True)
class LedgerSummary:
    """Tamper-verified summary of one traces.jsonl ledger."""

    label: str
    trace_path: str
    backend: str
    total: int
    overall: LedgerArmStats
    groups: dict[str, LedgerArmStats]


def _gate_status(entry: dict[str, Any], gate_name: str) -> str:
    for gate in entry.get("gates", []):
        if gate.get("gate_name") == gate_name:
            return str(gate.get("status", ""))
    return ""


def _summarize_group(key: str, entries: Sequence[dict[str, Any]]) -> LedgerArmStats:
    """Compute pass, honesty, slip, correspondence, leak, and thinking stats."""
    if not entries:
        raise ValidationError(f"Cannot summarize empty group: {key!r}")
    total = len(entries)
    passed = sum(1 for e in entries if e.get("verdict") == "PASSED")
    banished = total - passed
    honest = sum(
        1
        for e in entries
        if _gate_status(e, "epistemic_gate") == "PASS"
        and _gate_status(e, "quote_gate") == "PASS"
        and _gate_status(e, "tool_claim_gate") == "PASS"
    )
    slips = sum(1 for e in entries if _gate_status(e, "slip_gate") != "PASS")
    corr_fail = sum(1 for e in entries if _gate_status(e, "correspondence_gate") != "PASS")
    leaks = sum(
        1
        for e in entries
        if (not bool(e.get("departure", {}).get("clean", True)))
        or int(e.get("departure", {}).get("persisted_persona_bytes", 0)) > 0
    )
    thinking_values = [int(e.get("thinking_chars", 0)) for e in entries]
    mean_thinking = round(float(mean(thinking_values)), 1)
    return LedgerArmStats(
        key=key,
        total=total,
        passed=passed,
        banished=banished,
        honest=honest,
        slips=slips,
        correspondence_failures=corr_fail,
        leaks=leaks,
        mean_thinking_chars=mean_thinking,
    )


def summarize_ledger(
    trace_path: str | Path,
    *,
    label: str | None = None,
    by_form: bool = False,
) -> LedgerSummary:
    """Verify hash-chain integrity and summarize one JSONL trace ledger."""
    path = Path(trace_path)
    if not path.is_file():
        raise ValidationError(f"Trace ledger not found: {path}")

    ledger = LiberELedger(path)
    ok, errors = ledger.verify_chain()
    if not ok:
        raise ValidationError(
            f"Ledger chain verification failed for {path}: {'; '.join(errors)}"
        )

    entries = ledger.read_traces()
    if not entries:
        raise ValidationError(f"Ledger contains no trace records: {path}")

    if label is not None:
        stripped = label.strip()
        if not stripped:
            raise ValidationError("Ledger label must not be blank")
        resolved_label = stripped
    elif path.name == "traces.jsonl" and path.parent.name:
        resolved_label = path.parent.name
    else:
        resolved_label = path.stem

    backends = sorted({str(e.get("backend", "unknown")) for e in entries})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        arm = str(entry.get("arm", ""))
        if by_form:
            form_id = entry.get("form_id") or "none"
            key = f"{arm}|{form_id}"
        else:
            key = arm
        grouped[key].append(entry)

    groups = {k: _summarize_group(k, v) for k, v in grouped.items()}
    overall = _summarize_group("ALL", entries)
    return LedgerSummary(
        label=resolved_label,
        trace_path=str(path),
        backend=", ".join(backends),
        total=len(entries),
        overall=overall,
        groups=groups,
    )


def compare_ledgers(
    trace_paths: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
    by_form: bool = False,
) -> list[LedgerSummary]:
    """Load, verify, and summarize one or more trace ledgers for comparison."""
    if not trace_paths:
        raise ValidationError("At least one trace ledger path is required")
    if labels is not None and len(labels) != len(trace_paths):
        raise ValidationError(
            f"labels length ({len(labels)}) must match trace_paths length ({len(trace_paths)})"
        )
    resolved_labels: Sequence[str | None] = (
        labels if labels is not None else [None] * len(trace_paths)
    )
    return [
        summarize_ledger(path, label=lbl, by_form=by_form)
        for path, lbl in zip(trace_paths, resolved_labels)
    ]


def _ordered_keys(summaries: Iterable[LedgerSummary]) -> list[str]:
    arm_order = {arm.value: idx for idx, arm in enumerate(Arm)}
    all_keys = {key for summary in summaries for key in summary.groups}

    def _sort_key(key: str) -> tuple[int, str, str]:
        arm_part, _, form_part = key.partition("|")
        return (arm_order.get(arm_part, len(arm_order)), arm_part, form_part)

    return sorted(all_keys, key=_sort_key)


def _render_grid(headers: list[str], rows: list[list[str]], *, markdown: bool) -> str:
    """Format columns once as either aligned terminal text or Markdown table."""
    widths = [
        max([len(headers[i]), *(len(row[i]) for row in rows)])
        for i in range(len(headers))
    ]
    prefix = "| " if markdown else ""
    joiner = " | " if markdown else "  "
    suffix = " |" if markdown else ""

    def _fmt(cells: Sequence[str]) -> str:
        return (
            prefix
            + joiner.join(cells[i].ljust(widths[i]) for i in range(len(headers)))
            + suffix
        )

    sep_cells = ["-" * w for w in widths]
    return "\n".join([_fmt(headers), _fmt(sep_cells), *(_fmt(row) for row in rows)])


def _stats_cells(stats: LedgerArmStats) -> list[str]:
    n = stats.total
    return [
        str(n),
        f"{stats.passed}/{n}",
        f"{stats.honest}/{n}",
        f"{stats.slips}/{n}",
        f"{stats.correspondence_failures}/{n}",
        f"{stats.leaks}/{n}",
        f"{stats.mean_thinking_chars:.1f}",
    ]


def format_comparison(
    summaries: Sequence[LedgerSummary],
    *,
    markdown: bool = False,
) -> str:
    """Render an overall and per-arm comparison across verified ledgers."""
    if not summaries:
        raise ValidationError("At least one LedgerSummary is required")

    overall_headers = [
        "Ledger",
        "Backend",
        "N",
        "Passed",
        "Honest",
        "Slips",
        "CorrFail",
        "Leaks",
        "MeanThink",
    ]
    overall_rows = [
        [s.label, s.backend, *_stats_cells(s.overall)] for s in summaries
    ]

    group_headers = ["Group", *overall_headers]
    group_rows: list[list[str]] = []
    for key in _ordered_keys(summaries):
        for s in summaries:
            if key in s.groups:
                group_rows.append([key, s.label, s.backend, *_stats_cells(s.groups[key])])

    overall_title = "## Ledger Totals" if markdown else "=== Ledger Totals ==="
    group_title = "## By Group" if markdown else "=== By Group ==="
    return "\n".join(
        [
            overall_title,
            _render_grid(overall_headers, overall_rows, markdown=markdown),
            "",
            group_title,
            _render_grid(group_headers, group_rows, markdown=markdown),
        ]
    )
