"""Stdlib CLI entrypoint for running invocations, matrices, ledger checks, and inspection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
import webbrowser

from invocation_protocol.backends import (
    DeterministicStubBackend,
    ModelBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
)
from invocation_protocol.compare import compare_ledgers, format_comparison
from invocation_protocol.forms import load_form, load_task_battery, resolve_data_paths
from invocation_protocol.harness import EvaluationHarness, format_summary
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.runner import InvocationRunner
from invocation_protocol.types import Arm, InvocationError, RunConfig, RunResult
from invocation_protocol.studio import create_studio_server
from invocation_protocol.viewer import write_viewer


def resolve_cli_data_paths(forms_arg: str | None, tasks_arg: str | None) -> tuple[Path, Path]:
    """Use explicit --forms-dir / --tasks when given; fill the rest from the checkout or bundled data."""
    if forms_arg is not None and tasks_arg is not None:
        return Path(forms_arg), Path(tasks_arg)
    default_forms, default_tasks = resolve_data_paths()
    return (
        Path(forms_arg) if forms_arg is not None else default_forms,
        Path(tasks_arg) if tasks_arg is not None else default_tasks,
    )


def resolve_form_arg(value: str, forms_dir: Path) -> Path:
    """Accept a FormSpec file path, or a bare form id ('thoth') looked up in ``forms_dir``."""
    as_path = Path(value)
    if as_path.is_file():
        return as_path
    by_id = forms_dir / f"{value}.json"
    if by_id.is_file():
        return by_id
    known = sorted(p.stem for p in forms_dir.glob("*.json"))
    raise InvocationError(f"Form {value!r} is neither a file nor a known form id {known}")


def format_run_header(
    arm: str,
    form_id: str | None,
    task_id: str,
    category: str,
    *,
    index: int | None = None,
    total: int | None = None,
) -> str:
    """Format a concise section banner for one invocation run."""
    prefix = f"[{index}/{total}] " if index is not None and total is not None else ""
    cat_part = f" ({category})" if category else ""
    return f"{'=' * 70}\n{prefix}{arm} | form={form_id} | {task_id}{cat_part}"


def format_trace_entry(
    entry: Mapping[str, Any],
    *,
    index: int | None = None,
    total: int | None = None,
    include_header: bool = True,
    include_raw_output: bool = True,
) -> str:
    """Format one trace dictionary (or live result view) into readable terminal output."""
    lines: list[str] = []
    if include_header:
        lines.append(
            format_run_header(
                str(entry.get("arm")),
                entry.get("form_id"),
                str(entry.get("task_id")),
                str(entry.get("task_category", "")),
                index=index,
                total=total,
            )
        )
    if include_raw_output:
        lines.append("-" * 70)
        raw = str(entry.get("raw_output", ""))
        lines.append(raw if raw else "<EMPTY>")

    thinking_chars = entry.get("thinking_chars", 0)
    if isinstance(thinking_chars, int) and thinking_chars > 0:
        lines.append(f"THINKING: {thinking_chars} chars (separate from raw_output)")

    lines.append("-" * 70)
    tool_log = entry.get("tool_log") or []
    if not tool_log:
        lines.append("TOOLS: none")
    else:
        lines.append("TOOLS:")
        for rec in tool_log:
            status = "allowed" if rec.get("allowed") else "DENIED"
            detail = rec.get("output") if rec.get("allowed") else rec.get("error")
            lines.append(f"  - {rec.get('tool_name')}({rec.get('args')}) [{status}]: {detail}")

    lines.append("GATES:")
    for gate in entry.get("gates") or []:
        reasons = "; ".join(gate.get("reasons") or ())
        suffix = f": {reasons}" if reasons else ""
        lines.append(f"  [{gate.get('status')}] {gate.get('gate_name')}{suffix}")

    dep = entry.get("departure") or {}
    dep_state = "clean" if dep.get("clean") else "DIRTY"
    lines.append(
        f"VERDICT: {entry.get('verdict')} | "
        f"depart: {dep_state} ({dep.get('persisted_persona_bytes')} persisted bytes)"
    )
    if entry.get("runtime_error"):
        lines.append(f"RUNTIME ERROR: {entry.get('runtime_error')}")
    return "\n".join(lines)


def _trace_from_result(result: RunResult, category: str) -> dict[str, Any]:
    """Build a display dictionary matching the LiberELedger trace schema from a RunResult."""
    return {
        "arm": result.arm.value,
        "form_id": result.form_id,
        "task_id": result.task_id,
        "task_category": category,
        "raw_output": result.raw_output,
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
                "gate_name": gate.gate_name,
                "status": gate.status.value,
                "reasons": list(gate.reasons),
            }
            for gate in result.report.gates
        ],
        "departure": {
            "clean": result.departure.clean,
            "persisted_persona_bytes": result.departure.persisted_persona_bytes,
        },
        "verdict": result.verdict.value,
        "runtime_error": None,
    }


def _build_backend(
    args: argparse.Namespace,
    *,
    on_token: Callable[[str], None] | None = None,
) -> ModelBackend:
    if args.backend == "stub":
        return DeterministicStubBackend(inject_failure=getattr(args, "inject_failure", None))
    if args.backend == "ollama":
        if not args.model:
            raise ValueError("--model is required for --backend ollama")
        return OllamaBackend(
            model=args.model,
            base_url=args.base_url or "http://127.0.0.1:11434",
            on_token=on_token,
        )
    if args.backend == "openai":
        if not args.model or not args.base_url:
            raise ValueError("--model and --base-url are required for --backend openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAICompatibleBackend(model=args.model, base_url=args.base_url, api_key=api_key)
    raise ValueError(f"Unsupported backend: {args.backend!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main returning process exit status."""
    from invocation_protocol import __version__

    parser = argparse.ArgumentParser(prog="invocation-protocol")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_matrix = sub.add_parser("matrix", help="Run the 4-arm (+ Arm 3b) evaluation matrix")
    p_matrix.add_argument("--backend", choices=("stub", "ollama", "openai"), default="stub")
    p_matrix.add_argument("--model", default="")
    p_matrix.add_argument("--base-url", default="")
    p_matrix.add_argument("--trace-dir", default="runs/latest")
    p_matrix.add_argument(
        "--forms-dir", default=None, help="FormSpec directory (default: ./forms in a checkout, else the bundled forms)"
    )
    p_matrix.add_argument(
        "--tasks", default=None, help="Task battery JSON (default: ./tasks/battery.json in a checkout, else the bundled battery)"
    )
    p_matrix.add_argument("--seed", type=int, default=42)
    p_matrix.add_argument("--temperature", type=float, default=0.0)
    p_matrix.add_argument("--timeout", type=float, default=120.0)
    p_matrix.add_argument("--max-tokens", type=int, default=512)
    p_matrix.add_argument("--num-ctx", type=int, default=4096)
    p_matrix.add_argument("--live", action="store_true", help="Stream live output, gates, and verdict to stdout")

    p_invoke = sub.add_parser("invoke", help="Run a single 7-step invocation")
    p_invoke.add_argument("--arm", choices=[arm.value for arm in Arm], required=True)
    p_invoke.add_argument("--form", default="", help="FormSpec JSON path, or a bundled form id such as 'thoth'")
    p_invoke.add_argument("--task-id", required=True)
    p_invoke.add_argument(
        "--tasks", default=None, help="Task battery JSON (default: ./tasks/battery.json in a checkout, else the bundled battery)"
    )
    p_invoke.add_argument("--trace-dir", default="runs/single")
    p_invoke.add_argument("--backend", choices=("stub", "ollama", "openai"), default="stub")
    p_invoke.add_argument("--model", default="")
    p_invoke.add_argument("--base-url", default="")
    p_invoke.add_argument("--timeout", type=float, default=120.0)
    p_invoke.add_argument("--max-tokens", type=int, default=512)
    p_invoke.add_argument("--num-ctx", type=int, default=4096)
    p_invoke.add_argument("--inject-failure", default=None)
    p_invoke.add_argument("--raise-on-banish", action="store_true")
    p_invoke.add_argument("--live", action="store_true", help="Stream live output, gates, and verdict to stdout")

    p_verify = sub.add_parser("verify-ledger", help="Verify SHA-256 chain integrity of traces.jsonl")
    p_verify.add_argument("--trace-path", required=True)
    p_verify.add_argument(
        "--expect-head",
        default=None,
        help="Fail unless the last entry_hash equals this published value (detects truncation)",
    )

    p_inspect = sub.add_parser("inspect", help="Pretty-print raw traces from a Liber E ledger")
    p_inspect.add_argument("--trace-path", required=True)
    p_inspect.add_argument("--arm", choices=[arm.value for arm in Arm], default=None)
    p_inspect.add_argument("--task-id", default=None)

    p_compare = sub.add_parser("compare", help="Compare one or more Liber E traces.jsonl ledgers")
    p_compare.add_argument("trace_paths", nargs="+", help="Paths to traces.jsonl ledger files")
    p_compare.add_argument("--by-form", action="store_true", help="Break down metrics by arm and form_id")
    p_compare.add_argument("--markdown", action="store_true", help="Render comparison tables as Markdown")

    p_view = sub.add_parser("view", help="Write a self-contained HTML viewer for verified ledgers")
    p_view.add_argument("trace_paths", nargs="+", help="Paths to traces.jsonl ledger files")
    p_view.add_argument("--out", default="runs/viewer.html", help="Output HTML path (default: runs/viewer.html)")
    p_view.add_argument("--open", action="store_true", help="Open the viewer in the default browser")

    p_studio = sub.add_parser("studio", help="Launch interactive web studio for deities, gates, ledgers, and repo")
    p_studio.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_studio.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")
    p_studio.add_argument("--repo-root", default=".", help="Repository root path (default: current directory)")
    p_studio.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Local Ollama base URL")
    p_studio.add_argument("--open", action="store_true", help="Open Studio URL in default browser")
    p_studio.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding to a non-loopback --host (exposes file reading and test running; trusted networks only)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "verify-ledger":
            ledger = LiberELedger(trace_path=args.trace_path)
            ok, errors = ledger.verify_chain()
            if ok:
                count, head_hash = ledger.head()
                if args.expect_head is not None and head_hash != args.expect_head.strip():
                    print(
                        f"FAILED: head hash {head_hash} does not match expected {args.expect_head.strip()} "
                        f"in {args.trace_path} (records missing or replaced)",
                        file=sys.stderr,
                    )
                    return 1
                print(f"OK: verified {count} trace record(s) in {args.trace_path}")
                print(f"head: {head_hash}")
                return 0
            print(f"FAILED: {len(errors)} chain error(s) in {args.trace_path}", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1

        if args.command == "inspect":
            trace_file = Path(args.trace_path)
            if not trace_file.exists():
                print(f"ERROR: Trace file not found: {args.trace_path}", file=sys.stderr)
                return 1
            entries = LiberELedger(trace_path=trace_file).read_traces()
            filtered = [
                entry
                for entry in entries
                if (args.arm is None or entry.get("arm") == args.arm)
                and (args.task_id is None or entry.get("task_id") == args.task_id)
            ]
            if not filtered:
                print("No matching trace records.")
                return 0
            total = len(filtered)
            for idx, entry in enumerate(filtered, start=1):
                print(format_trace_entry(entry, index=idx, total=total))
            return 0

        if args.command == "view":
            out = write_viewer(args.trace_paths, args.out)
            print(f"Wrote viewer for {len(args.trace_paths)} ledger(s): {out}")
            if args.open:
                webbrowser.open(out.resolve().as_uri())
            return 0

        if args.command == "studio":
            server = create_studio_server(
                args.repo_root,
                host=args.host,
                port=args.port,
                ollama_url=args.ollama_url,
                allow_remote=args.allow_remote,
            )
            bound_host, bound_port = server.server_address[:2]
            url = f"http://{bound_host}:{bound_port}"
            print(f"LLM Invocation Studio listening on {url}", flush=True)
            if args.open:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0

        if args.command == "compare":
            summaries = compare_ledgers(args.trace_paths, by_form=args.by_form)
            print(format_comparison(summaries, markdown=args.markdown))
            return 0

        if args.command == "matrix":
            stream_tokens = bool(args.live and args.backend == "ollama")
            on_token = (lambda tok: print(tok, end="", flush=True)) if stream_tokens else None
            backend = _build_backend(args, on_token=on_token)
            config = RunConfig(
                temperature=args.temperature,
                seed=args.seed,
                max_tokens=args.max_tokens,
                context_length=args.num_ctx,
                timeout_seconds=args.timeout,
            )
            forms_dir, tasks_path = resolve_cli_data_paths(args.forms_dir, args.tasks)
            harness = EvaluationHarness(
                backend=backend,
                trace_dir=Path(args.trace_dir),
                forms_dir=forms_dir,
                tasks_path=tasks_path,
            )
            tasks_by_id = {t.task_id: t.category for t in load_task_battery(tasks_path)}

            before_cb = None
            progress_cb = None
            if args.live:
                if stream_tokens:
                    def before_cb(idx: int, total: int, arm: Arm, form: Any, task: Any) -> None:
                        fid = form.form_id if form is not None else None
                        print(
                            format_run_header(
                                arm.value, fid, task.task_id, task.category, index=idx, total=total
                            )
                            + f"\n{'-' * 70}",
                            flush=True,
                        )

                    def progress_cb(_idx: int, _total: int, res: RunResult) -> None:
                        print(
                            "\n"
                            + format_trace_entry(
                                _trace_from_result(res, tasks_by_id.get(res.task_id, "")),
                                include_header=False,
                                include_raw_output=False,
                            ),
                            flush=True,
                        )
                else:
                    def progress_cb(idx: int, total: int, res: RunResult) -> None:
                        print(
                            format_trace_entry(
                                _trace_from_result(res, tasks_by_id.get(res.task_id, "")),
                                index=idx,
                                total=total,
                                include_header=True,
                                include_raw_output=True,
                            ),
                            flush=True,
                        )
            elif args.backend != "stub":
                def progress_cb(idx: int, total: int, res: RunResult) -> None:
                    print(
                        f"[{idx}/{total}] {res.arm.value} / {res.task_id} -> {res.verdict.value}",
                        file=sys.stderr,
                        flush=True,
                    )

            report = harness.run_matrix(
                config,
                before_run_callback=before_cb,
                progress_callback=progress_cb,
            )
            if args.live:
                print("=" * 70)
            print(format_summary(report))
            return 0

        if args.command == "invoke":
            stream_tokens = bool(args.live and args.backend == "ollama")
            on_token = (lambda tok: print(tok, end="", flush=True)) if stream_tokens else None
            backend = _build_backend(args, on_token=on_token)
            forms_dir, tasks_path = resolve_cli_data_paths(None, args.tasks)
            tasks = {t.task_id: t for t in load_task_battery(tasks_path)}
            if args.task_id not in tasks:
                print(f"Unknown task_id {args.task_id!r}", file=sys.stderr)
                return 2
            task = tasks[args.task_id]
            form = load_form(resolve_form_arg(args.form, forms_dir)) if args.form else None
            trace_dir = Path(args.trace_dir)
            ledger = LiberELedger(trace_dir / "traces.jsonl", trace_dir / "commentary.jsonl")
            runner = InvocationRunner(backend=backend, ledger=ledger)
            config = RunConfig(
                max_tokens=args.max_tokens,
                context_length=args.num_ctx,
                timeout_seconds=args.timeout,
            )
            if stream_tokens:
                fid = form.form_id if form is not None else None
                print(
                    format_run_header(args.arm, fid, task.task_id, task.category) + f"\n{'-' * 70}",
                    flush=True,
                )
            result = runner.invoke(
                Arm(args.arm),
                task,
                config,
                form=form,
                raise_on_banish=args.raise_on_banish,
            )
            if args.live:
                trace_view = _trace_from_result(result, task.category)
                if stream_tokens:
                    print(
                        "\n"
                        + format_trace_entry(
                            trace_view, include_header=False, include_raw_output=False
                        ),
                        flush=True,
                    )
                else:
                    print(format_trace_entry(trace_view), flush=True)
            print(f"trace_id={result.trace_id} verdict={result.verdict.value} hash={result.trace_hash}")
            return 0 if result.verdict.value == "PASSED" else 1
    except (InvocationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 2
