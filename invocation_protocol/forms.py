"""Declarative FormSpec and TaskSpec loaders plus canonical correspondence table."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from invocation_protocol.types import (
    CorrespondenceSlice,
    FormExemplar,
    FormSpec,
    TaskSpec,
    ValidationError,
    require_non_empty_str,
    require_str_tuple,
)

VALID_ARM_TYPES: frozenset[str] = frozenset({"neutral", "deity", "corrupted_control"})

CANONICAL_CORRESPONDENCES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "thoth": MappingProxyType(
            {
                "tradition": "kemetic_hermetic",
                "domain": "hod_mercury_scribe",
                "element": "air",
                "planet_or_sphere": "mercury",
                "symbols": frozenset({"ibis", "stylus", "papyrus", "lunar_disk"}),
            }
        ),
        "maat": MappingProxyType(
            {
                "tradition": "kemetic",
                "domain": "atu_viii_adjustment_balance",
                "element": "air",
                "planet_or_sphere": "libra",
                "symbols": frozenset({"ostrich_feather", "scales", "plumb_line"}),
            }
        ),
        "athena": MappingProxyType(
            {
                "tradition": "hellenic",
                "domain": "pallas_dialectic_strategy",
                "element": "air",
                "planet_or_sphere": "jupiter_athena",
                "symbols": frozenset({"owl", "aegis", "olive_branch"}),
            }
        ),
    }
)


def parse_form_dict(data: Any) -> FormSpec:
    """Validate structural FormSpec fields without enforcing canonical correspondences.

    Note: Correspondence verification is intentionally deferred to test_spirit()
    so Arm 3b corrupted controls can load structurally and fail deterministically
    in the correspondence gate.
    """
    if not isinstance(data, Mapping):
        raise ValidationError("FormSpec JSON root must be an object")

    schema_version = require_non_empty_str(data.get("schema_version"), "schema_version")
    if schema_version != "1.0":
        raise ValidationError(f"Unsupported FormSpec schema_version: {schema_version!r}")

    form_id = require_non_empty_str(data.get("form_id"), "form_id")
    raw_canonical = data.get("canonical_id")
    if raw_canonical is None:
        canonical_id: str | None = None
    else:
        canonical_id = require_non_empty_str(raw_canonical, "canonical_id")

    arm_type = require_non_empty_str(data.get("arm_type"), "arm_type")
    if arm_type not in VALID_ARM_TYPES:
        raise ValidationError(f"Invalid arm_type {arm_type!r}; expected one of {sorted(VALID_ARM_TYPES)}")

    display_name = require_non_empty_str(data.get("display_name"), "display_name")
    tradition = require_non_empty_str(data.get("tradition"), "tradition")
    epithets = require_str_tuple(data.get("epithets"), "epithets")

    raw_slice = data.get("correspondence_slice")
    if not isinstance(raw_slice, Mapping):
        raise ValidationError("correspondence_slice must be an object")
    correspondence_slice = CorrespondenceSlice(
        domain=require_non_empty_str(raw_slice.get("domain"), "correspondence_slice.domain"),
        element=require_non_empty_str(raw_slice.get("element"), "correspondence_slice.element"),
        planet_or_sphere=require_non_empty_str(
            raw_slice.get("planet_or_sphere"),
            "correspondence_slice.planet_or_sphere",
        ),
        symbols=require_str_tuple(raw_slice.get("symbols"), "correspondence_slice.symbols"),
        operational_focus=require_non_empty_str(
            raw_slice.get("operational_focus"),
            "correspondence_slice.operational_focus",
        ),
    )

    forbidden_attributions = require_str_tuple(data.get("forbidden_attributions"), "forbidden_attributions")
    banned_cliches = require_str_tuple(data.get("banned_cliches"), "banned_cliches")

    raw_exemplars = data.get("exemplars")
    if not isinstance(raw_exemplars, (list, tuple)) or len(raw_exemplars) == 0:
        raise ValidationError("exemplars must be a non-empty list")
    exemplars: list[FormExemplar] = []
    for idx, entry in enumerate(raw_exemplars):
        if not isinstance(entry, Mapping):
            raise ValidationError(f"exemplars[{idx}] must be an object")
        exemplars.append(
            FormExemplar(
                input=require_non_empty_str(entry.get("input"), f"exemplars[{idx}].input"),
                output=require_non_empty_str(entry.get("output"), f"exemplars[{idx}].output"),
            )
        )

    return FormSpec(
        schema_version=schema_version,
        form_id=form_id,
        canonical_id=canonical_id,
        arm_type=arm_type,
        display_name=display_name,
        tradition=tradition,
        epithets=epithets,
        correspondence_slice=correspondence_slice,
        forbidden_attributions=forbidden_attributions,
        banned_cliches=banned_cliches,
        exemplars=tuple(exemplars),
    )


def load_form(path: str | Path) -> FormSpec:
    """Load and structurally validate a FormSpec JSON file."""
    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except OSError as exc:
        raise ValidationError(f"Failed to read FormSpec {file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in FormSpec {file_path}: {exc}") from exc
    return parse_form_dict(data)


def parse_task_dict(data: Any) -> TaskSpec:
    """Validate and construct a TaskSpec from a decoded JSON object."""
    if not isinstance(data, Mapping):
        raise ValidationError("Task entry must be an object")

    task_id = require_non_empty_str(data.get("task_id"), "task_id")
    category = require_non_empty_str(data.get("category"), "category")
    prompt = require_non_empty_str(data.get("prompt"), "prompt")

    raw_sources = data.get("sources")
    if not isinstance(raw_sources, Mapping):
        raise ValidationError(f"{task_id}.sources must be an object")
    sources: dict[str, str] = {}
    for key, val in raw_sources.items():
        s_key = require_non_empty_str(key, f"{task_id}.sources key")
        s_val = require_non_empty_str(val, f"{task_id}.sources[{s_key}]")
        sources[s_key] = s_val

    allowed_tools = require_str_tuple(data.get("allowed_tools"), f"{task_id}.allowed_tools")

    raw_fixtures = data.get("tool_fixtures")
    if not isinstance(raw_fixtures, Mapping):
        raise ValidationError(f"{task_id}.tool_fixtures must be an object")
    tool_fixtures: dict[str, dict[str, str]] = {}
    for tool_name, mapping in raw_fixtures.items():
        t_name = require_non_empty_str(tool_name, f"{task_id}.tool_fixtures key")
        if not isinstance(mapping, Mapping):
            raise ValidationError(f"{task_id}.tool_fixtures[{t_name}] must be an object")
        records: dict[str, str] = {}
        for rec_k, rec_v in mapping.items():
            records[require_non_empty_str(rec_k, f"{t_name} key")] = require_non_empty_str(
                rec_v, f"{t_name}[{rec_k}]"
            )
        tool_fixtures[t_name] = records

    expected = require_str_tuple(data.get("expected_answer_substrings"), f"{task_id}.expected_answer_substrings")
    forbidden = require_str_tuple(data.get("forbidden_answer_substrings"), f"{task_id}.forbidden_answer_substrings")

    raw_any = data.get("expected_answer_any", [])
    if not isinstance(raw_any, list):
        raise ValidationError(f"{task_id}.expected_answer_any must be a list of string lists")
    expected_any: list[tuple[str, ...]] = []
    for idx, group in enumerate(raw_any):
        alternatives = require_str_tuple(group, f"{task_id}.expected_answer_any[{idx}]")
        if not alternatives:
            raise ValidationError(f"{task_id}.expected_answer_any[{idx}] must list at least one alternative")
        expected_any.append(alternatives)

    return TaskSpec(
        task_id=task_id,
        category=category,
        prompt=prompt,
        sources=sources,
        allowed_tools=allowed_tools,
        tool_fixtures=tool_fixtures,
        expected_answer_substrings=expected,
        forbidden_answer_substrings=forbidden,
        expected_answer_any=tuple(expected_any),
    )


def load_task_battery(path: str | Path) -> tuple[TaskSpec, ...]:
    """Load and validate a task battery JSON file."""
    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except OSError as exc:
        raise ValidationError(f"Failed to read task battery {file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in task battery {file_path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ValidationError("Task battery root must be an object")
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) == 0:
        raise ValidationError("Task battery must contain a non-empty 'tasks' list")
    return tuple(parse_task_dict(item) for item in raw_tasks)
