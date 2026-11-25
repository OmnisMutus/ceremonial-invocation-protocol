"""Self-contained offline HTML viewer for verified Liber E trace ledgers."""

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
from typing import Any, Sequence

from invocation_protocol.ledger import LiberELedger
from invocation_protocol.types import ValidationError

TEMPLATE_NAME: str = "viewer_template.html"
DATA_PLACEHOLDER: str = "/*__LEDGER_DATA__*/[]"


def ledger_label(path: Path) -> str:
    """Name a ledger after its run directory (``runs/<name>/traces.jsonl``) or its file stem."""
    if path.name == "traces.jsonl" and path.parent.name:
        return path.parent.name
    return path.stem


def load_ledger_payload(trace_path: str | Path) -> dict[str, Any]:
    """Verify one ledger's hash chain and package its records for the viewer.

    Verification runs here, in Python, with the same canonical JSON the ledger was written
    with; the page only displays the result. A ledger whose chain fails still loads, flagged
    as unverified, so a broken ledger can be inspected rather than silently skipped.
    """
    path = Path(trace_path)
    if not path.is_file():
        raise ValidationError(f"Trace ledger not found: {path}")
    ledger = LiberELedger(path)
    ok, errors = ledger.verify_chain()
    try:
        entries = ledger.read_traces()
    except ValidationError as exc:
        entries = []
        errors = [*errors, str(exc)]
        ok = False
    head = str(entries[-1].get("entry_hash", "")) if entries else None
    return {
        "label": ledger_label(path),
        "path": path.as_posix(),
        "verified": ok,
        "errors": errors,
        "head": head,
        "entries": entries,
    }


def render_viewer_html(payloads: Sequence[dict[str, Any]], template: str | None = None) -> str:
    """Embed ledger payloads into the viewer template and return a standalone HTML page."""
    if template is None:
        template = resources.files("invocation_protocol").joinpath(TEMPLATE_NAME).read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in template:
        raise ValidationError("Viewer template is missing its data placeholder")
    # Escaping every '<' keeps record text (for example '</script>' or '<!--' inside model
    # output) from terminating or altering the script element. JSON allows < anywhere.
    data = json.dumps(list(payloads), ensure_ascii=False).replace("<", "\\u003c")
    return template.replace(DATA_PLACEHOLDER, data, 1)


def write_viewer(trace_paths: Sequence[str | Path], out_path: str | Path) -> Path:
    """Verify each ledger, render the viewer, and write it to ``out_path``."""
    if not trace_paths:
        raise ValidationError("At least one trace ledger path is required")
    payloads = [load_ledger_payload(p) for p in trace_paths]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_viewer_html(payloads), encoding="utf-8")
    return out
