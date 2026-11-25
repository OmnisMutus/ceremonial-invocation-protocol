"""Tests for declarative FormSpecs, TaskSpecs, RunConfig boundaries, and zero-dependency imports."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

from invocation_protocol.forms import (
    CANONICAL_CORRESPONDENCES,
    load_form,
    load_task_battery,
    parse_form_dict,
    parse_task_dict,
)
from invocation_protocol.types import RunConfig, ValidationError, require_int

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "invocation_protocol"
FORMS_DIR = REPO_ROOT / "forms"
TASKS_PATH = REPO_ROOT / "tasks" / "battery.json"


def _minimal_valid_form() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "form_id": "test_form",
        "canonical_id": "thoth",
        "arm_type": "deity",
        "display_name": "Thoth Δίκη 𓅝",
        "tradition": "kemetic_hermetic",
        "epithets": ["Scribe Δίκη"],
        "correspondence_slice": {
            "domain": "hod_mercury_scribe",
            "element": "air",
            "planet_or_sphere": "mercury",
            "symbols": ["ibis"],
            "operational_focus": "archival_precision",
        },
        "forbidden_attributions": ["mars"],
        "banned_cliches": ["behold mortal"],
        "exemplars": [{"input": "in", "output": "{}"}],
    }


def _minimal_valid_task() -> dict[str, object]:
    return {
        "task_id": "t1",
        "category": "accuracy",
        "prompt": "Report value.",
        "sources": {"s1": "Value is 5."},
        "allowed_tools": [],
        "tool_fixtures": {},
        "expected_answer_substrings": ["5"],
        "forbidden_answer_substrings": ["9"],
    }


class TestZeroDependencies(unittest.TestCase):
    def test_package_uses_only_stdlib_imports(self) -> None:
        stdlib = sys.stdlib_module_names
        py_files = sorted(PKG_DIR.glob("*.py"))
        self.assertGreater(len(py_files), 0)
        for py_file in py_files:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self.assertIn(top, stdlib | {"invocation_protocol"}, msg=f"{py_file}: {top}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module is not None:
                        top = node.module.split(".")[0]
                        self.assertIn(top, stdlib | {"invocation_protocol"}, msg=f"{py_file}: {top}")

    def test_version_matches_pyproject_toml_and_cli(self) -> None:
        from contextlib import redirect_stdout
        import io
        import tomllib

        import invocation_protocol
        from invocation_protocol.cli import main as cli_main

        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        expected_version = pyproject["project"]["version"]
        self.assertEqual(invocation_protocol.__version__, expected_version)

        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as ctx:
                cli_main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn(expected_version, buf.getvalue())


class TestFormLoadingAndValidation(unittest.TestCase):
    def test_all_declarative_forms_match_raw_json_ground_truth(self) -> None:
        expected_files = {"profile_08.json", "thoth.json", "maat.json", "athena.json", "thoth_corrupted.json"}
        actual_files = {p.name for p in FORMS_DIR.glob("*.json")}
        self.assertEqual(actual_files, expected_files)

        for form_path in sorted(FORMS_DIR.glob("*.json")):
            raw = json.loads(form_path.read_text(encoding="utf-8"))
            spec = load_form(form_path)
            self.assertEqual(spec.schema_version, raw["schema_version"])
            self.assertEqual(spec.form_id, raw["form_id"])
            self.assertEqual(spec.canonical_id, raw["canonical_id"])
            self.assertEqual(spec.arm_type, raw["arm_type"])
            self.assertEqual(spec.display_name, raw["display_name"])
            self.assertEqual(spec.tradition, raw["tradition"])
            self.assertEqual(list(spec.epithets), raw["epithets"])
            self.assertEqual(list(spec.correspondence_slice.symbols), raw["correspondence_slice"]["symbols"])
            self.assertEqual(spec.correspondence_slice.domain, raw["correspondence_slice"]["domain"])
            self.assertEqual(spec.correspondence_slice.element, raw["correspondence_slice"]["element"])
            self.assertEqual(
                spec.correspondence_slice.planet_or_sphere,
                raw["correspondence_slice"]["planet_or_sphere"],
            )

    def test_canonical_correspondences_immutable(self) -> None:
        with self.assertRaises(TypeError):
            CANONICAL_CORRESPONDENCES["extra"] = {}  # type: ignore[index]

    def test_unicode_form_loads_cleanly(self) -> None:
        spec = parse_form_dict(_minimal_valid_form())
        self.assertIn("𓅝", spec.display_name)
        self.assertEqual(spec.epithets, ("Scribe Δίκη",))

    def test_invalid_form_inputs_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            parse_form_dict([])
        with self.assertRaises(ValidationError):
            load_form(FORMS_DIR / "does_not_exist.json")

        bad_version = _minimal_valid_form()
        bad_version["schema_version"] = "2.0"
        with self.assertRaises(ValidationError):
            parse_form_dict(bad_version)

        bad_arm_type = _minimal_valid_form()
        bad_arm_type["arm_type"] = "oracle"
        with self.assertRaises(ValidationError):
            parse_form_dict(bad_arm_type)

        bare_str_symbols = _minimal_valid_form()
        bare_str_symbols["correspondence_slice"] = {
            **bare_str_symbols["correspondence_slice"],  # type: ignore[dict-item]
            "symbols": "ibis",
        }
        with self.assertRaises(ValidationError):
            parse_form_dict(bare_str_symbols)

        empty_exemplars = _minimal_valid_form()
        empty_exemplars["exemplars"] = []
        with self.assertRaises(ValidationError):
            parse_form_dict(empty_exemplars)

        bad_exemplar_entry = _minimal_valid_form()
        bad_exemplar_entry["exemplars"] = ["not_a_dict"]
        with self.assertRaises(ValidationError):
            parse_form_dict(bad_exemplar_entry)

        bad_slice_type = _minimal_valid_form()
        bad_slice_type["correspondence_slice"] = "invalid"
        with self.assertRaises(ValidationError):
            parse_form_dict(bad_slice_type)

        empty_canonical = _minimal_valid_form()
        empty_canonical["canonical_id"] = "   "
        with self.assertRaises(ValidationError):
            parse_form_dict(empty_canonical)

        non_str_item = _minimal_valid_form()
        non_str_item["epithets"] = [123]
        with self.assertRaises(ValidationError):
            parse_form_dict(non_str_item)

        empty_str_item = _minimal_valid_form()
        empty_str_item["epithets"] = ["  "]
        with self.assertRaises(ValidationError):
            parse_form_dict(empty_str_item)


class TestTaskBatteryAndRunConfig(unittest.TestCase):
    def test_load_task_battery_matches_json_ground_truth(self) -> None:
        raw = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
        tasks = load_task_battery(TASKS_PATH)
        self.assertEqual(len(tasks), len(raw["tasks"]))
        self.assertEqual(len(tasks), 5)
        for task_obj, raw_item in zip(tasks, raw["tasks"]):
            self.assertEqual(task_obj.task_id, raw_item["task_id"])
            self.assertEqual(task_obj.category, raw_item["category"])
            self.assertEqual(task_obj.sources, raw_item["sources"])

    def test_invalid_task_inputs_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            load_task_battery(REPO_ROOT / "tasks" / "missing.json")
        with self.assertRaises(ValidationError):
            parse_task_dict("not_an_object")

        bad_sources = _minimal_valid_task()
        bad_sources["sources"] = []
        with self.assertRaises(ValidationError):
            parse_task_dict(bad_sources)

        bad_fixtures = _minimal_valid_task()
        bad_fixtures["tool_fixtures"] = {"lookup_fact": "not_a_dict"}
        with self.assertRaises(ValidationError):
            parse_task_dict(bad_fixtures)

        with tempfile.TemporaryDirectory() as tmp:
            bad_root = Path(tmp) / "bad_root.json"
            bad_root.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_task_battery(bad_root)

            empty_tasks = Path(tmp) / "empty_tasks.json"
            empty_tasks.write_text('{"tasks": []}', encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_task_battery(empty_tasks)

            nonlist_tasks = Path(tmp) / "nonlist_tasks.json"
            nonlist_tasks.write_text('{"tasks": "bad"}', encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_task_battery(nonlist_tasks)

    def test_require_int_upper_bound_exact(self) -> None:
        self.assertEqual(require_int(3, "x", min_value=1, max_value=3), 3)
        with self.assertRaises(ValidationError):
            require_int(4, "x", min_value=1, max_value=3)

    def test_run_config_exact_boundaries_and_type_confusion(self) -> None:
        # Exact valid boundaries
        cfg_min = RunConfig(
            temperature=0.0,
            seed=0,
            max_tokens=1,
            context_length=256,
            max_turns=1,
            max_tool_calls=0,
            timeout_seconds=0.001,
            fail_closed=True,
        )
        self.assertEqual(cfg_min.temperature, 0.0)
        self.assertEqual(cfg_min.context_length, 256)
        self.assertEqual(cfg_min.max_tool_calls, 0)

        cfg_max_temp = RunConfig(temperature=2.0)
        self.assertEqual(cfg_max_temp.temperature, 2.0)

        # Past boundaries and invalid values
        invalid_kwargs: list[dict[str, object]] = [
            {"temperature": -0.01},
            {"temperature": 2.01},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"temperature": True},
            {"seed": -1},
            {"seed": True},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"context_length": 255},
            {"context_length": True},
            {"max_turns": 0},
            {"max_turns": -1},
            {"max_turns": False},
            {"max_tool_calls": -1},
            {"max_tool_calls": True},
            {"timeout_seconds": 0.0},
            {"timeout_seconds": -1.0},
            {"timeout_seconds": float("inf")},
            {"timeout_seconds": True},
            {"fail_closed": 1},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    RunConfig(**kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
