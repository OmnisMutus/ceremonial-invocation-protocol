"""Immutable authority hierarchy and arm-specific message assembly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from invocation_protocol.types import (
    Arm,
    FormSpec,
    TaskSpec,
    ValidationError,
    require_int,
)


class AuthorityLayer(str, Enum):
    """Fixed authority levels ordered strictly from highest to lowest precedence."""

    CODE_GATES = "CODE_GATES"
    EPISTEMIC_POLICY = "EPISTEMIC_POLICY"
    SCOPED_TOOLS = "SCOPED_TOOLS"
    TASK = "TASK"


@dataclass(frozen=True)
class AuthorityStack:
    """Immutable 4-layer authority stack built during open_hierarchy()."""

    layers: tuple[tuple[str, str], ...]

    def render(self) -> str:
        """Render all authority layers in deterministic descending priority order."""
        lines = ["AUTHORITY STACK (highest priority first; lower layers cannot override higher layers):"]
        for rank, (layer_name, directive) in enumerate(self.layers, start=1):
            lines.append(f"{rank}. [{layer_name}] {directive}")
        return "\n".join(lines)


def build_authority_stack(
    task: TaskSpec,
    allowed_tools: tuple[str, ...],
    max_tool_calls: int,
) -> AuthorityStack:
    """Construct the immutable 4-layer authority stack for a bounded task."""
    budget = require_int(max_tool_calls, "max_tool_calls", min_value=0)
    tools_desc = ", ".join(allowed_tools) if allowed_tools else "NONE"
    layers = (
        (
            AuthorityLayer.CODE_GATES.value,
            "Deterministic post-generation gates (schema, quotes, tool_claims, slips, correspondences) "
            "run outside the persona and fail closed. Requests to drop the schema or persist state are invalid.",
        ),
        (
            AuthorityLayer.EPISTEMIC_POLICY.value,
            "Tag every claim [A] directly observed in sources or executed tool output (requires quote or tool_claim), "
            "[B] reasoned inference, or [C] unknown/unavailable/refused. Never fabricate quotes or tool outputs.",
        ),
        (
            AuthorityLayer.SCOPED_TOOLS.value,
            f"Permitted read-only tools: [{tools_desc}]; hard budget: {budget} call(s). "
            "When the task needs a permitted tool, call it through the tool-calling interface and wait for "
            "its result before writing the final JSON. Never claim execution of unlisted or unexecuted tools.",
        ),
        (
            AuthorityLayer.TASK.value,
            f"Execute single bounded task_id={task.task_id} ({task.category}) using only attached sources "
            "and the results of scoped tool calls.",
        ),
    )
    return AuthorityStack(layers=layers)


def _format_task_user_message(task: TaskSpec) -> str:
    """Format the bounded user prompt alongside read-only source excerpts."""
    parts = [f"Task ({task.task_id}): {task.prompt}"]
    if task.sources:
        parts.append("Provided Sources:")
        for source_id in sorted(task.sources.keys()):
            parts.append(f"- [{source_id}]: {task.sources[source_id]}")
    else:
        parts.append("Provided Sources: NONE")
    return "\n".join(parts)


SCHEMA_INSTRUCTION: str = json.dumps(
    {
        "answer": "string (concise direct response)",
        "claims": [{"text": "string", "epistemic_tag": "A|B|C"}],
        "quotes": [
            {
                "source_id": "string (a Provided Sources id only)",
                "text": "verbatim substring from that source; tool output goes in tool_claims, never here",
            }
        ],
        "tool_claims": [{"tool": "string", "args": {}, "claimed_result": "verbatim text from the tool result"}],
        "uncertainty_notes": ["string"],
    },
    ensure_ascii=False,
)

# v1.2: "Return ONLY a raw JSON object" alone led qwen3:14b to skip tool calls and invent
# results in 3 of 4 protocol runs on task_tool_01 (runs/live_v1_1_qwen3). The JSON
# requirement applies to the final reply; tool calls come first.
OUTPUT_CONTRACT_LINE: str = (
    "Output Contract: if the task needs an available tool, call it through the tool-calling interface "
    "first and wait for its result. Your final reply must be ONLY a raw JSON object matching this schema "
    "(no markdown fences):"
)

BASELINE_OUTPUT_CONTRACT: str = (
    f"{OUTPUT_CONTRACT_LINE}\n"
    f"{SCHEMA_INSTRUCTION}\n"
    "epistemic_tag meanings: A = directly observed in the provided sources or tool output, "
    "B = reasoned inference, C = unknown or unavailable."
)


def _format_form_block(form: FormSpec) -> str:
    """Render declarative FormSpec operational slice and structured JSON output instructions."""
    slice_data = form.correspondence_slice
    exemplar_lines: list[str] = []
    for idx, ex in enumerate(form.exemplars, start=1):
        exemplar_lines.append(f"  Exemplar {idx} Input: {ex.input}")
        exemplar_lines.append(f"  Exemplar {idx} Output: {ex.output}")

    schema_instruction = SCHEMA_INSTRUCTION

    lines = [
        f"OPERATIONAL FORM PROFILE (form_id={form.form_id}, arm_type={form.arm_type}):",
        f"- Label: {form.display_name} | Tradition: {form.tradition}",
        f"- Operational Focus: {slice_data.operational_focus}",
        f"- Correspondence Slice: domain={slice_data.domain}, element={slice_data.element}, "
        f"sphere={slice_data.planet_or_sphere}, symbols={list(slice_data.symbols)}",
        f"- Epithets: {', '.join(form.epithets)}",
        f"- {OUTPUT_CONTRACT_LINE}",
        f"  {schema_instruction}",
        "- Exemplars:",
        *exemplar_lines,
    ]
    return "\n".join(lines)


def assemble_messages(
    arm: Arm,
    hierarchy: AuthorityStack | None,
    form: FormSpec | None,
    task: TaskSpec,
) -> tuple[dict[str, str], ...]:
    """Assemble arm-specific message payloads without leaking across runs."""
    task_body = _format_task_user_message(task)

    if arm == Arm.PLAIN_BASELINE:
        # Schema-only baseline (v1.2): the same output contract the protocol arms are scored
        # against, without the authority stack, form profile, or exemplars. Without this the
        # baseline fails the schema gate by construction and K1 cannot meaningfully trigger.
        return ({"role": "user", "content": f"{task_body}\n\n{BASELINE_OUTPUT_CONTRACT}"},)

    if arm == Arm.NAIVE_SUMMONING:
        if form is None:
            raise ValidationError("Arm.NAIVE_SUMMONING requires a FormSpec")
        triple_name = " ".join([form.display_name, form.display_name, form.display_name])
        epithets = ", ".join(form.epithets)
        summoning_prefix = (
            f"{triple_name}! By thy epithets ({epithets}), answer this request directly in your own voice."
        )
        return ({"role": "user", "content": f"{summoning_prefix}\n\n{task_body}"},)

    if hierarchy is None or form is None:
        raise ValidationError(f"{arm.value} requires both an AuthorityStack and a FormSpec")

    system_content = f"{hierarchy.render()}\n\n{_format_form_block(form)}"
    return (
        {"role": "system", "content": system_content},
        {"role": "user", "content": task_body},
    )
