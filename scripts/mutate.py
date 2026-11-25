"""Uncapped AST mutation testing runner (STANDARD.md §4).

Executes every comparison (<, <=, >, >=, ==, !=), boolean (and, or), and
arithmetic (+, -, *) mutation inside isolated temporary worker sandboxes so
zero bytes of the live working tree are ever modified on disk.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "invocation_protocol"

_CMP_MUTATIONS: dict[type[ast.cmpop], tuple[type[ast.cmpop], str]] = {
    ast.Lt: (ast.LtE, "< -> <="),
    ast.LtE: (ast.Lt, "<= -> <"),
    ast.Gt: (ast.GtE, "> -> >="),
    ast.GtE: (ast.Gt, ">= -> >"),
    ast.Eq: (ast.NotEq, "== -> !="),
    ast.NotEq: (ast.Eq, "!= -> =="),
}

_BOOL_MUTATIONS: dict[type[ast.boolop], tuple[type[ast.boolop], str]] = {
    ast.And: (ast.Or, "and -> or"),
    ast.Or: (ast.And, "or -> and"),
}

_BINOP_MUTATIONS: dict[type[ast.operator], tuple[type[ast.operator], str]] = {
    ast.Add: (ast.Sub, "+ -> -"),
    ast.Sub: (ast.Add, "- -> +"),
    ast.Mult: (ast.Add, "* -> +"),
}

_FAST_TEST_MODULES: dict[str, str] = {
    "backends": "tests.test_backends",
    "cli": "tests.test_compare",
    "compare": "tests.test_compare",
    "forms": "tests.test_forms",
    "gates": "tests.test_gates",
    "harness": "tests.test_harness",
    "hierarchy": "tests.test_hierarchy",
    "ledger": "tests.test_ledger",
    "runner": "tests.test_runner",
    "studio": "tests.test_studio",
    "tools": "tests.test_tools",
    "types": "tests.test_forms",
    "viewer": "tests.test_viewer",
}


def _collect_sites(tree: ast.Module) -> list[tuple[int, str, str]]:
    sites: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                entry = _CMP_MUTATIONS.get(type(op))
                if entry is not None:
                    sites.append((node.lineno, "cmp", entry[1]))
        elif isinstance(node, ast.BoolOp):
            entry = _BOOL_MUTATIONS.get(type(node.op))
            if entry is not None:
                sites.append((node.lineno, "bool", entry[1]))
        elif isinstance(node, ast.BinOp):
            entry = _BINOP_MUTATIONS.get(type(node.op))
            if entry is not None:
                sites.append((node.lineno, "binop", entry[1]))
    return sites


def _apply_mutation(tree: ast.Module, target_index: int) -> tuple[ast.Module, int, str]:
    mutated = copy.deepcopy(tree)
    cursor = 0
    for node in ast.walk(mutated):
        if isinstance(node, ast.Compare):
            for idx, op in enumerate(node.ops):
                entry = _CMP_MUTATIONS.get(type(op))
                if entry is not None:
                    if cursor == target_index:
                        node.ops[idx] = entry[0]()
                        return mutated, node.lineno, entry[1]
                    cursor += 1
        elif isinstance(node, ast.BoolOp):
            entry = _BOOL_MUTATIONS.get(type(node.op))
            if entry is not None:
                if cursor == target_index:
                    node.op = entry[0]()
                    return mutated, node.lineno, entry[1]
                cursor += 1
        elif isinstance(node, ast.BinOp):
            entry = _BINOP_MUTATIONS.get(type(node.op))
            if entry is not None:
                if cursor == target_index:
                    node.op = entry[0]()
                    return mutated, node.lineno, entry[1]
                cursor += 1
    raise IndexError(f"Mutation index {target_index} out of range")


# Generous per-mutant timeout: a timeout counts as a kill (it catches mutants that loop
# forever), so it must sit far above the suite's runtime under parallel load or slow
# machines would report false kills.
MUTANT_TIMEOUT_SECONDS: int = 120


def _run_unittest(cwd: Path, args: list[str], timeout_seconds: int = MUTANT_TIMEOUT_SECONDS) -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", *args],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
        return proc.returncode != 0
    except subprocess.TimeoutExpired:
        return True


def _populate_sandbox(dst: Path) -> None:
    shutil.copytree(REPO_ROOT / "invocation_protocol", dst / "invocation_protocol", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO_ROOT / "tests", dst / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO_ROOT / "forms", dst / "forms")
    shutil.copytree(REPO_ROOT / "tasks", dst / "tasks")
    for trace in sorted((REPO_ROOT / "runs").glob("live_*/traces.jsonl")):
        (dst / "runs" / trace.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace, dst / "runs" / trace.parent.name / "traces.jsonl")
    shutil.copy2(REPO_ROOT / "pyproject.toml", dst / "pyproject.toml")
    shutil.copy2(REPO_ROOT / "README.md", dst / "README.md")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uncapped sandbox-isolated AST mutator")
    parser.add_argument("--modules", nargs="*", default=None, help="Optional module stems to restrict to")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if _run_unittest(REPO_ROOT, ["discover", "-s", "tests", "-f"], timeout_seconds=35):
        print("Baseline tests FAILED before mutation; aborting.", file=sys.stderr, flush=True)
        return 2

    py_files = sorted(PKG_DIR.glob("*.py"))
    if args.modules:
        wanted = set(args.modules)
        py_files = [p for p in py_files if p.stem in wanted]

    workers = min(6, max(2, os.cpu_count() or 4))
    total_all = 0
    killed_all = 0
    survivors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="cip_mut_") as tmp_base:
        sandboxes: queue.Queue[Path] = queue.Queue()
        for w_idx in range(workers):
            sb = Path(tmp_base) / f"w_{w_idx}"
            _populate_sandbox(sb)
            sandboxes.put(sb)

        # The unmutated code must pass INSIDE a sandbox too. If a sandbox lacked a file the
        # tests need, every mutant would fail for that reason and be miscounted as killed.
        if _run_unittest(Path(tmp_base) / "w_0", ["discover", "-s", "tests", "-f"]):
            print("Baseline tests FAILED inside the mutation sandbox; aborting.", file=sys.stderr, flush=True)
            return 2

        def _eval_one(rel: str, stem: str, orig_b: bytes, mut_b: bytes, lineno: int, label: str) -> tuple[bool, str]:
            sb = sandboxes.get()
            target = sb / rel
            try:
                target.write_bytes(mut_b)
                fast = _FAST_TEST_MODULES.get(stem)
                killed = (bool(fast) and _run_unittest(sb, [fast, "-f"])) or _run_unittest(
                    sb, ["discover", "-s", "tests", "-f"]
                )
                return killed, f"{rel}:{lineno} ({label})"
            finally:
                target.write_bytes(orig_b)
                sandboxes.put(sb)

        print("Uncapped AST Mutation Results:", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for file_path in py_files:
                rel_path = file_path.relative_to(REPO_ROOT).as_posix()
                orig_bytes = file_path.read_bytes()
                tree = ast.parse(orig_bytes.decode("utf-8"), filename=rel_path)
                sites = _collect_sites(tree)
                if not sites:
                    print(f"  {rel_path}: 0 / 0 (no mutable nodes)", flush=True)
                    continue
                jobs = []
                for idx in range(len(sites)):
                    mutated_tree, lineno, label = _apply_mutation(tree, idx)
                    mutated_bytes = ast.unparse(mutated_tree).encode("utf-8")
                    jobs.append(pool.submit(_eval_one, rel_path, file_path.stem, orig_bytes, mutated_bytes, lineno, label))

                mod_killed = 0
                for fut in jobs:
                    killed, desc = fut.result()
                    if killed:
                        mod_killed += 1
                    else:
                        survivors.append(desc)
                total_all += len(sites)
                killed_all += mod_killed
                print(f"  {rel_path}: {mod_killed} / {len(sites)}", flush=True)

    print(f"\nTotal: {killed_all} / {total_all} killed, {len(survivors)} survived", flush=True)
    if survivors:
        print("\nSurviving Mutants (verbatim):", flush=True)
        for item in survivors:
            print(f"  - {item}", flush=True)
        return 1
    print("\nSurviving Mutants: NONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
