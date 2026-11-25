"""Interactive stdlib web studio for Ceremonial Invocation Protocol.

Provides a local HTTP+JSON server (`127.0.0.1` by default, zero external dependencies)
backing the interactive Pantheon Trial, 7-Step Ritual Workbench, Deity Codex,
Evaluation Matrix / Liber E Ledger Inspector, and path-confined Repository & Test Explorer.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request

from invocation_protocol import gates
from invocation_protocol.backends import (
    FAILURE_MODES,
    DeterministicStubBackend,
    ModelBackend,
    OllamaBackend,
)
from invocation_protocol.forms import CANONICAL_CORRESPONDENCES, load_form, load_task_battery
from invocation_protocol.harness import EvaluationHarness, _task_is_accurate
from invocation_protocol.hierarchy import assemble_messages, build_authority_stack
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.runner import PROTOCOL_VERSION, InvocationRunner
from invocation_protocol.types import (
    Arm,
    FormSpec,
    InvocationError,
    RunConfig,
    RunResult,
    TaskSpec,
    ValidationError,
)
from invocation_protocol.viewer import load_ledger_payload

STUDIO_TEMPLATE_NAME: str = "studio_template.html"
MAX_FILE_BYTES: int = 512_000
MAX_POST_BYTES: int = 65_536
_EXCLUDED_PARTS: frozenset[str] = frozenset(
    {".git", ".claude", "__pycache__", ".venv", "venv", "build", "dist"}
)
_EXCLUDED_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".pyd", ".dll", ".exe"})
# Dot-prefixed paths (.env, .npmrc, editor configs, ...) are never served except these.
_ALLOWED_DOT_PARTS: frozenset[str] = frozenset({".github", ".gitignore", ".gitattributes"})
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_hidden_part(part: str) -> bool:
    return part.startswith(".") and part not in _ALLOWED_DOT_PARTS

PANTHEON_LINEUP: tuple[tuple[str, str | None, str], ...] = (
    (Arm.PROTOCOL_FORM.value, "thoth", "Deity · Kemetic/Hermetic Scribe"),
    (Arm.PROTOCOL_FORM.value, "maat", "Deity · Kemetic Equilibrium"),
    (Arm.PROTOCOL_FORM.value, "athena", "Deity · Hellenic Dialectic"),
    (Arm.PROTOCOL_NEUTRAL.value, "profile_08", "Control · Neutral Protocol"),
    (Arm.PLAIN_BASELINE.value, None, "Control · Schema-only Baseline"),
    (Arm.NAIVE_SUMMONING.value, "thoth", "Hazard · Unconstrained Summoning"),
    (Arm.CORRUPTED_CORRESPONDENCE.value, "thoth_corrupted", "Check · Form Metadata Integrity"),
)


def resolve_repo_file(repo_root: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` strictly inside ``repo_root``, blocking traversal and hidden dirs."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValidationError("File path must be a non-empty relative path")
    cleaned = rel_path.strip().replace("\\", "/")
    if "\x00" in cleaned or cleaned.startswith("/") or ":" in cleaned:
        raise ValidationError(f"Disallowed file path: {rel_path!r}")
    parts = [p for p in cleaned.split("/") if p]
    if not parts or any(p in {".", ".."} or p in _EXCLUDED_PARTS or _is_hidden_part(p) for p in parts):
        raise ValidationError(f"Path outside allowed repository scope: {rel_path!r}")

    root_resolved = repo_root.resolve()
    target = (root_resolved / Path(*parts)).resolve()
    if not target.is_relative_to(root_resolved):
        raise ValidationError(f"Path escapes repository root: {rel_path!r}")
    if target.suffix.lower() in _EXCLUDED_SUFFIXES or any(
        part.endswith(".egg-info") for part in target.relative_to(root_resolved).parts
    ):
        raise ValidationError(f"Binary or build artifact excluded: {rel_path!r}")
    if not target.is_file():
        raise FileNotFoundError(f"Repository file not found: {cleaned}")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValidationError(f"File exceeds {MAX_FILE_BYTES} byte preview limit: {cleaned}")
    return target


def list_repo_files(repo_root: Path) -> list[dict[str, Any]]:
    """Return sorted metadata for readable repository source, form, doc, and trace files."""
    root_resolved = repo_root.resolve()
    entries: list[dict[str, Any]] = []
    for item in sorted(root_resolved.rglob("*")):
        rel_parts = item.relative_to(root_resolved).parts
        if any(p in _EXCLUDED_PARTS or p.endswith(".egg-info") or _is_hidden_part(p) for p in rel_parts):
            continue
        if not item.is_file() or item.suffix.lower() in _EXCLUDED_SUFFIXES:
            continue
        size = item.stat().st_size
        if size > MAX_FILE_BYTES:
            continue
        category = rel_parts[0] if len(rel_parts) > 1 else "root"
        entries.append(
            {
                "path": "/".join(rel_parts),
                "category": category,
                "size": size,
            }
        )
    return entries


def read_repo_file_payload(repo_root: Path, rel_path: str) -> dict[str, Any]:
    """Read a validated UTF-8 repository file and return content + metadata."""
    target = resolve_repo_file(repo_root, rel_path)
    rel_posix = target.relative_to(repo_root.resolve()).as_posix()
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"File is not UTF-8 text: {rel_posix}") from exc
    return {
        "path": rel_posix,
        "size": target.stat().st_size,
        "line_count": len(content.splitlines()),
        "content": content,
    }


def probe_ollama_models(base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 1.2) -> dict[str, Any]:
    """Check local Ollama availability without raising when offline."""
    clean_url = base_url.strip().rstrip("/")
    req = urllib.request.Request(f"{clean_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return {"reachable": False, "base_url": clean_url, "models": []}
    raw_models = data.get("models") if isinstance(data, dict) else []
    names = [
        str(m["name"])
        for m in (raw_models if isinstance(raw_models, list) else [])
        if isinstance(m, dict) and isinstance(m.get("name"), str)
    ]
    return {"reachable": True, "base_url": clean_url, "models": names}


def _serialize_form(form: FormSpec) -> dict[str, Any]:
    corr_gate = gates.check_correspondences(form, "", None)
    return {
        "form_id": form.form_id,
        "canonical_id": form.canonical_id,
        "arm_type": form.arm_type,
        "display_name": form.display_name,
        "tradition": form.tradition,
        "epithets": list(form.epithets),
        "correspondence_slice": {
            "domain": form.correspondence_slice.domain,
            "element": form.correspondence_slice.element,
            "planet_or_sphere": form.correspondence_slice.planet_or_sphere,
            "symbols": list(form.correspondence_slice.symbols),
            "operational_focus": form.correspondence_slice.operational_focus,
        },
        "forbidden_attributions": list(form.forbidden_attributions),
        "banned_cliches": list(form.banned_cliches),
        "exemplars": [{"input": ex.input, "output": ex.output} for ex in form.exemplars],
        "correspondence_check": {
            "status": corr_gate.status.value,
            "reasons": list(corr_gate.reasons),
        },
    }


def _serialize_task(task: TaskSpec) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "category": task.category,
        "prompt": task.prompt,
        "sources": dict(task.sources),
        "allowed_tools": list(task.allowed_tools),
        "tool_fixtures": {k: dict(v) for k, v in task.tool_fixtures.items()},
        "expected_answer_substrings": list(task.expected_answer_substrings),
        "expected_answer_any": [list(g) for g in task.expected_answer_any],
        "forbidden_answer_substrings": list(task.forbidden_answer_substrings),
    }


def build_overview_payload(
    repo_root: Path,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Assemble full repository, deity forms, task battery, ledgers, and backend status."""
    from invocation_protocol import __version__

    root = repo_root.resolve()
    forms_dir = root / "forms"
    forms_list = [
        _serialize_form(load_form(p))
        for p in sorted(forms_dir.glob("*.json"))
    ]
    tasks_list = [_serialize_task(t) for t in load_task_battery(root / "tasks" / "battery.json")]

    canonical = {
        cid: {
            "tradition": spec["tradition"],
            "domain": spec["domain"],
            "element": spec["element"],
            "planet_or_sphere": spec["planet_or_sphere"],
            "symbols": sorted(spec["symbols"]),
        }
        for cid, spec in CANONICAL_CORRESPONDENCES.items()
    }

    ledgers_summary: list[dict[str, Any]] = []
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        for trace_file in sorted(runs_dir.rglob("traces.jsonl")):
            payload = load_ledger_payload(trace_file)
            rel_trace = trace_file.relative_to(root).as_posix()
            ledgers_summary.append(
                {
                    "label": payload["label"],
                    "path": rel_trace,
                    "verified": payload["verified"],
                    "head": payload["head"],
                    "errors": payload["errors"],
                    "record_count": len(payload["entries"]),
                }
            )

    return {
        "version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "repo_name": root.name,
        "arms": [
            {"id": Arm.PLAIN_BASELINE.value, "label": "Arm 0 · Plain Baseline", "needs_form": False},
            {"id": Arm.NAIVE_SUMMONING.value, "label": "Arm 1 · Naive Summoning", "needs_form": True},
            {"id": Arm.PROTOCOL_NEUTRAL.value, "label": "Arm 2 · Protocol Neutral", "needs_form": True},
            {"id": Arm.PROTOCOL_FORM.value, "label": "Arm 3 · Protocol Deity Form", "needs_form": True},
            {"id": Arm.CORRUPTED_CORRESPONDENCE.value, "label": "Arm 3b · Corrupted Control", "needs_form": True},
        ],
        "forms": forms_list,
        "canonical_correspondences": canonical,
        "tasks": tasks_list,
        "failure_modes": sorted(FAILURE_MODES),
        "pantheon_lineup": [
            {"arm": a, "form_id": f, "role_label": r} for a, f, r in PANTHEON_LINEUP
        ],
        "ledgers": ledgers_summary,
        "files": list_repo_files(root),
        "ollama": probe_ollama_models(ollama_url),
    }


def _select_backend(req: Mapping[str, Any], default_ollama_url: str) -> ModelBackend:
    backend_kind = str(req.get("backend", "stub")).strip().lower()
    if backend_kind == "stub":
        raw_fail = req.get("inject_failure")
        inj = str(raw_fail).strip() if isinstance(raw_fail, str) and raw_fail.strip() else None
        return DeterministicStubBackend(inject_failure=inj)
    if backend_kind == "ollama":
        model = str(req.get("model", "")).strip() or "qwen3:14b"
        # The Ollama URL is fixed at server start (--ollama-url); a request can never
        # redirect the server's outbound calls to another host.
        return OllamaBackend(model=model, base_url=default_ollama_url)
    raise ValidationError(f"Unsupported studio backend: {backend_kind!r}")


def _build_run_config(req: Mapping[str, Any], backend_kind: str) -> RunConfig:
    default_tokens = 1536 if backend_kind == "ollama" else 512
    default_timeout = 120.0 if backend_kind == "ollama" else 30.0
    max_tokens = req.get("max_tokens", default_tokens)
    timeout_s = req.get("timeout_seconds", default_timeout)
    return RunConfig(
        temperature=0.0,
        seed=42,
        max_tokens=max_tokens if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) else default_tokens,
        context_length=4096,
        max_turns=3,
        max_tool_calls=2,
        timeout_seconds=float(timeout_s) if isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool) else default_timeout,
    )


def _serialize_run_result(
    result: RunResult,
    *,
    task: TaskSpec,
    form: FormSpec | None,
    backend_name: str,
    authority_rendered: str | None,
    assembled_messages: list[dict[str, str]],
    elapsed_ms: float,
    role_label: str = "",
) -> dict[str, Any]:
    return {
        "trace_id": result.trace_id,
        "trace_hash": result.trace_hash,
        "backend": backend_name,
        "arm": result.arm.value,
        "form_id": result.form_id,
        "display_name": form.display_name if form is not None else "Plain Baseline (No Form)",
        "role_label": role_label,
        "task_id": task.task_id,
        "task_category": task.category,
        "verdict": result.verdict.value,
        "accurate": _task_is_accurate(result, task),
        "lifecycle_stages": [s.value for s in result.lifecycle_stages],
        "authority_rendered": authority_rendered,
        "assembled_messages": assembled_messages,
        "raw_output": result.raw_output,
        "accepted_output": result.accepted_output,
        "thinking_chars": result.thinking_chars,
        "tool_log": [
            {
                "tool_name": rec.tool_name,
                "args": dict(rec.args),
                "allowed": rec.allowed,
                "output": rec.output,
                "error": rec.error,
            }
            for rec in result.tool_log
        ],
        "gates": [
            {
                "gate_name": g.gate_name,
                "status": g.status.value,
                "reasons": list(g.reasons),
            }
            for g in result.report.gates
        ],
        "departure": {
            "cleared_messages": result.departure.cleared_messages,
            "cleared_form_id": result.departure.cleared_form_id,
            "persisted_persona_bytes": result.departure.persisted_persona_bytes,
            "clean": result.departure.clean,
        },
        "elapsed_ms": round(elapsed_ms, 1),
    }


def _resolve_task_and_forms(
    repo_root: Path,
    req: Mapping[str, Any],
) -> tuple[TaskSpec, dict[str, FormSpec]]:
    root = repo_root.resolve()
    tasks = {t.task_id: t for t in load_task_battery(root / "tasks" / "battery.json")}
    task_id = str(req.get("task_id", "task_sycophancy_01")).strip()
    if task_id not in tasks:
        raise ValidationError(f"Unknown task_id: {task_id!r}")
    task = tasks[task_id]
    custom_prompt = req.get("custom_prompt")
    if isinstance(custom_prompt, str) and custom_prompt.strip():
        task = replace(task, prompt=custom_prompt.strip())
    forms_map = {
        p.stem: load_form(p)
        for p in sorted((root / "forms").glob("*.json"))
    }
    return task, forms_map


def execute_invocation(
    repo_root: Path,
    req: Mapping[str, Any],
    *,
    default_ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Run one 7-step Ceremonial Invocation and return full stage-by-stage artifacts."""
    raw_arm = str(req.get("arm", Arm.PROTOCOL_FORM.value)).strip()
    try:
        arm = Arm(raw_arm)
    except ValueError as exc:
        raise ValidationError(f"Invalid arm: {raw_arm!r}") from exc

    task, forms_map = _resolve_task_and_forms(repo_root, req)
    form: FormSpec | None = None
    if arm != Arm.PLAIN_BASELINE:
        default_fid = (
            "profile_08"
            if arm == Arm.PROTOCOL_NEUTRAL
            else ("thoth_corrupted" if arm == Arm.CORRUPTED_CORRESPONDENCE else "thoth")
        )
        fid = str(req.get("form_id") or default_fid).strip()
        if fid not in forms_map:
            raise ValidationError(f"Unknown form_id: {fid!r}")
        form = forms_map[fid]

    backend = _select_backend(req, default_ollama_url)
    backend_kind = str(req.get("backend", "stub")).strip().lower()
    config = _build_run_config(req, backend_kind)

    stack = (
        None
        if arm in (Arm.PLAIN_BASELINE, Arm.NAIVE_SUMMONING)
        else build_authority_stack(task, task.allowed_tools, config.max_tool_calls)
    )
    assembled = [dict(m) for m in assemble_messages(arm, stack, form, task)]

    trace_dir = repo_root.resolve() / "runs" / "studio"
    ledger = LiberELedger(trace_dir / "traces.jsonl", trace_dir / "commentary.jsonl")
    runner = InvocationRunner(backend=backend, ledger=ledger)

    t0 = time.perf_counter()
    result = runner.invoke(arm, task, config, form=form, raise_on_banish=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return _serialize_run_result(
        result,
        task=task,
        form=form,
        backend_name=backend.name,
        authority_rendered=stack.render() if stack is not None else None,
        assembled_messages=assembled,
        elapsed_ms=elapsed_ms,
    )


def execute_pantheon(
    repo_root: Path,
    req: Mapping[str, Any],
    *,
    default_ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Run every PANTHEON_LINEUP arm (deity forms, neutral, baseline, naive, integrity check) on one task."""
    task, forms_map = _resolve_task_and_forms(repo_root, req)
    backend = _select_backend(req, default_ollama_url)
    backend_kind = str(req.get("backend", "stub")).strip().lower()
    config = _build_run_config(req, backend_kind)

    trace_dir = repo_root.resolve() / "runs" / "studio"
    ledger = LiberELedger(trace_dir / "traces.jsonl", trace_dir / "commentary.jsonl")
    runner = InvocationRunner(backend=backend, ledger=ledger)

    runs_out: list[dict[str, Any]] = []
    for arm_val, fid, role_label in PANTHEON_LINEUP:
        arm = Arm(arm_val)
        form = forms_map[fid] if fid is not None else None
        stack = (
            None
            if arm in (Arm.PLAIN_BASELINE, Arm.NAIVE_SUMMONING)
            else build_authority_stack(task, task.allowed_tools, config.max_tool_calls)
        )
        assembled = [dict(m) for m in assemble_messages(arm, stack, form, task)]
        t0 = time.perf_counter()
        res = runner.invoke(arm, task, config, form=form, raise_on_banish=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        runs_out.append(
            _serialize_run_result(
                res,
                task=task,
                form=form,
                backend_name=backend.name,
                authority_rendered=stack.render() if stack is not None else None,
                assembled_messages=assembled,
                elapsed_ms=elapsed_ms,
                role_label=role_label,
            )
        )

    return {
        "task": _serialize_task(task),
        "backend": backend.name,
        "runs": runs_out,
    }


def execute_matrix(
    repo_root: Path,
    req: Mapping[str, Any],
    *,
    default_ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Execute the full 35-run 5-arm EvaluationHarness matrix and return metrics + K1-K4."""
    root = repo_root.resolve()
    backend = _select_backend(req, default_ollama_url)
    backend_kind = str(req.get("backend", "stub")).strip().lower()
    config = _build_run_config(req, backend_kind)

    out_dir = root / "runs" / "studio_matrix"
    harness = EvaluationHarness(
        backend=backend,
        trace_dir=out_dir,
        forms_dir=root / "forms",
        tasks_path=root / "tasks" / "battery.json",
    )
    report = harness.run_matrix(config)
    rel_trace = Path(report.trace_path).resolve().relative_to(root).as_posix()
    return {
        "mode": report.mode,
        "backend_name": report.backend_name,
        "trace_path": rel_trace,
        "total_runs": len(report.results),
        "arm_metrics": {k: asdict(v) for k, v in report.arm_metrics.items()},
        "form_metrics": {k: asdict(v) for k, v in report.form_metrics.items()},
        "kill_criteria": [asdict(kc) for kc in report.kill_criteria],
    }


def execute_test_suite(repo_root: Path, timeout_seconds: float = 60.0) -> dict[str, Any]:
    """Run ``python -B -m unittest discover -s tests -v`` inside ``repo_root``."""
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=repo_root.resolve(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": None, "output": f"Test suite timed out after {timeout_seconds}s"}
    combined = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": combined,
    }


def load_studio_html() -> str:
    """Load the packaged self-contained Studio HTML frontend."""
    return resources.files("invocation_protocol").joinpath(STUDIO_TEMPLATE_NAME).read_text(encoding="utf-8")


def create_studio_server(
    repo_root: str | Path = ".",
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    ollama_url: str = "http://127.0.0.1:11434",
    allow_remote: bool = False,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer serving the LLM Invocation Studio UI and JSON API.

    The Studio can read repository files, run the test suite, and drive local models, so it
    binds to loopback only unless ``allow_remote`` is set, answers only requests whose
    ``Host`` header names the server itself (blocks DNS rebinding), and accepts POSTs only
    as ``application/json`` from its own origin (blocks cross-site form/fetch requests).
    """
    root = Path(repo_root).resolve()
    if not (root / "forms").is_dir() or not (root / "tasks" / "battery.json").is_file():
        raise ValidationError(f"Invalid Ceremonial Invocation Protocol root: {root}")
    if host not in _LOOPBACK_HOSTS and not allow_remote:
        raise ValidationError(
            f"Refusing to bind Studio to non-loopback host {host!r}: it can read repo files and run "
            "tests. Pass allow_remote=True (CLI: --allow-remote) only on a trusted network."
        )

    run_lock = threading.Lock()
    html_bytes = load_studio_html().encode("utf-8")
    allowed_hosts: set[str] = set()

    class StudioHandler(BaseHTTPRequestHandler):
        def _host_allowed(self) -> bool:
            return self.headers.get("Host", "").strip().lower() in allowed_hosts

        def _same_origin_json_post(self) -> bool:
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            origin = self.headers.get("Origin")
            origin_ok = origin is None or origin.strip().lower() in {f"http://{h}" for h in allowed_hosts}
            return ctype == "application/json" and origin_ok

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def _read_json_body(self) -> dict[str, Any]:
            raw_len = self.headers.get("Content-Length", "0")
            length = int(raw_len)
            if length < 0 or length > MAX_POST_BYTES:
                _ = self.rfile.read(max(0, min(length, 262_144)))
                self.close_connection = True
                raise ValidationError(f"POST body size must be between 0 and {MAX_POST_BYTES} bytes")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValidationError("JSON request body must be an object")
            return obj

        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urllib.parse.urlsplit(self.path)
            route = parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            if not self._host_allowed():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Host header not allowed"})
                return
            try:
                if route in {"/", "/index.html"}:
                    self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", html_bytes)
                    return
                if route == "/api/overview":
                    self._send_json(HTTPStatus.OK, build_overview_payload(root, ollama_url=ollama_url))
                    return
                if route == "/api/file":
                    rel = (query.get("path") or [""])[0]
                    self._send_json(HTTPStatus.OK, read_repo_file_payload(root, rel))
                    return
                if route == "/api/ledger":
                    rel_l = (query.get("path") or [""])[0]
                    target_l = resolve_repo_file(root, rel_l)
                    if target_l.suffix != ".jsonl":
                        raise ValidationError("Ledger path must end with .jsonl")
                    ledger_payload = load_ledger_payload(target_l)
                    # Repo-relative path only: never expose the user's absolute filesystem layout.
                    ledger_payload["path"] = target_l.relative_to(root).as_posix()
                    self._send_json(HTTPStatus.OK, ledger_payload)
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValidationError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - never drop the connection without a reply
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:  # noqa: N802
            route = urllib.parse.urlsplit(self.path).path
            if not self._host_allowed():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Host header not allowed"})
                return
            if not self._same_origin_json_post():
                self.close_connection = True
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "POST requires Content-Type: application/json from the Studio's own origin"},
                )
                return
            try:
                req_obj = self._read_json_body()
                if route == "/api/invoke":
                    with run_lock:
                        out = execute_invocation(root, req_obj, default_ollama_url=ollama_url)
                    self._send_json(HTTPStatus.OK, out)
                    return
                if route == "/api/pantheon":
                    with run_lock:
                        out = execute_pantheon(root, req_obj, default_ollama_url=ollama_url)
                    self._send_json(HTTPStatus.OK, out)
                    return
                if route == "/api/matrix":
                    with run_lock:
                        out = execute_matrix(root, req_obj, default_ollama_url=ollama_url)
                    self._send_json(HTTPStatus.OK, out)
                    return
                if route == "/api/run-tests":
                    self._send_json(HTTPStatus.OK, execute_test_suite(root))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown POST route: {route}"})
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except InvocationError as exc:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - never drop the connection without a reply
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), StudioHandler)
    bound_port = server.server_address[1]
    names = {host.lower(), *(_LOOPBACK_HOSTS if host in _LOOPBACK_HOSTS else ())}
    for name in names:
        allowed_hosts.add(f"[{name}]:{bound_port}" if ":" in name else f"{name}:{bound_port}")
    return server
