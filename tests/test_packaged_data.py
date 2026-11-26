"""Bundled forms/tasks: data resolution, CLI fallbacks, and running outside a checkout."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from invocation_protocol import forms as forms_mod
from invocation_protocol.cli import main as cli_main
from invocation_protocol.cli import resolve_cli_data_paths, resolve_form_arg
from invocation_protocol.forms import (
    load_form,
    load_task_battery,
    packaged_forms_dir,
    packaged_tasks_path,
    resolve_data_paths,
)
from invocation_protocol.types import InvocationError, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent


class _InDir:
    """Temporarily chdir into a directory (restored on exit)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Path:
        self.prev = Path.cwd()
        os.chdir(self.path)
        return self.path

    def __exit__(self, *exc: object) -> None:
        os.chdir(self.prev)


class TestDataResolution(unittest.TestCase):
    def test_checkout_empty_and_partial_directories(self) -> None:
        # Repo checkout -> its own copies
        self.assertEqual(resolve_data_paths(REPO_ROOT), (REPO_ROOT / "forms", REPO_ROOT / "tasks" / "battery.json"))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Neither present -> bundled copies (the repo's files when running from source)
            self.assertEqual(resolve_data_paths(base), (packaged_forms_dir(), packaged_tasks_path()))
            with _InDir(base):
                self.assertEqual(resolve_data_paths(), (packaged_forms_dir(), packaged_tasks_path()))
            # Only forms/ -> ambiguous
            (base / "forms").mkdir()
            with self.assertRaises(ValidationError):
                resolve_data_paths(base)
            # Only tasks/battery.json -> ambiguous
            shutil.rmtree(base / "forms")
            (base / "tasks").mkdir()
            (base / "tasks" / "battery.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValidationError):
                resolve_data_paths(base)
            # Both -> local copies win (local edits take effect)
            (base / "forms").mkdir()
            self.assertEqual(resolve_data_paths(base), (base / "forms", base / "tasks" / "battery.json"))

    def test_wheel_layout_is_preferred_and_missing_data_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_pkg = Path(tmp) / "site" / "invocation_protocol"
            (fake_pkg / "data_forms").mkdir(parents=True)
            (fake_pkg / "data_tasks").mkdir()
            # A decoy repo-style forms/ next to the package must NOT win over the wheel layout
            (fake_pkg.parent / "forms").mkdir()
            with mock.patch.object(forms_mod, "_PKG_DIR", fake_pkg):
                self.assertEqual(packaged_forms_dir(), fake_pkg / "data_forms")
                self.assertEqual(packaged_tasks_path(), fake_pkg / "data_tasks" / "battery.json")
            empty_pkg = Path(tmp) / "empty" / "invocation_protocol"
            empty_pkg.mkdir(parents=True)
            with mock.patch.object(forms_mod, "_PKG_DIR", empty_pkg):
                with self.assertRaises(ValidationError):
                    packaged_forms_dir()
                with self.assertRaises(ValidationError):
                    packaged_tasks_path()

    def test_bundled_data_is_complete_and_valid(self) -> None:
        bundled = sorted(p.stem for p in packaged_forms_dir().glob("*.json"))
        self.assertEqual(bundled, ["athena", "maat", "profile_08", "thoth", "thoth_corrupted"])
        for p in packaged_forms_dir().glob("*.json"):
            with self.subTest(form=p.stem):
                self.assertEqual(load_form(p).form_id, p.stem)
        self.assertEqual(len(load_task_battery(packaged_tasks_path())), 5)

    def test_pyproject_maps_the_data_directories_into_the_wheel(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for needle in (
            '"invocation_protocol.data_forms" = "forms"',
            '"invocation_protocol.data_tasks" = "tasks"',
            '"invocation_protocol.data_forms" = ["*.json"]',
            '"invocation_protocol.data_tasks" = ["*.json"]',
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)


class TestCliFallbacks(unittest.TestCase):
    def test_resolve_cli_data_paths_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _InDir(Path(tmp)):
            bundled = (packaged_forms_dir(), packaged_tasks_path())
            self.assertEqual(resolve_cli_data_paths(None, None), bundled)
            self.assertEqual(resolve_cli_data_paths("f", None), (Path("f"), bundled[1]))
            self.assertEqual(resolve_cli_data_paths(None, "t.json"), (bundled[0], Path("t.json")))
            # Both explicit: no resolution needed, even inside an ambiguous half-checkout
            (Path(tmp) / "forms").mkdir()
            self.assertEqual(resolve_cli_data_paths("f", "t.json"), (Path("f"), Path("t.json")))
            with self.assertRaises(ValidationError):
                resolve_cli_data_paths("f", None)

    def test_resolve_form_arg_path_id_and_unknown(self) -> None:
        fdir = packaged_forms_dir()
        self.assertEqual(resolve_form_arg(str(fdir / "maat.json"), fdir), fdir / "maat.json")
        self.assertEqual(resolve_form_arg("athena", fdir), fdir / "athena.json")
        with self.assertRaises(InvocationError):
            resolve_form_arg("not_a_form", fdir)

    def test_matrix_and_invoke_run_outside_a_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _InDir(Path(tmp)):
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                rc = cli_main(["matrix", "--backend", "stub", "--trace-dir", "m"])
            self.assertEqual(rc, 0, msg=out.getvalue())
            self.assertIn("(35 total runs)", out.getvalue())
            self.assertEqual(len((Path(tmp) / "m" / "traces.jsonl").read_text(encoding="utf-8").splitlines()), 35)

            out2 = io.StringIO()
            with redirect_stdout(out2), redirect_stderr(io.StringIO()):
                rc2 = cli_main(
                    ["invoke", "--arm", "arm_3_protocol_form", "--form", "thoth", "--task-id", "task_tool_01", "--trace-dir", "i"]
                )
            self.assertEqual(rc2, 0, msg=out2.getvalue())
            record = json.loads((Path(tmp) / "i" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual((record["form_id"], record["verdict"]), ("thoth", "PASSED"))

            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                rc3 = cli_main(["invoke", "--arm", "arm_3_protocol_form", "--form", "nope", "--task-id", "task_tool_01"])
            self.assertEqual(rc3, 1)
            self.assertIn("known form id", err.getvalue())


if __name__ == "__main__":
    unittest.main()
