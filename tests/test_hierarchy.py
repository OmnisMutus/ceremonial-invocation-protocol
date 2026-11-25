"""Tests for immutable AuthorityStack ordering and arm-specific message assembly."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from invocation_protocol.forms import load_form, load_task_battery
from invocation_protocol.hierarchy import AuthorityLayer, assemble_messages, build_authority_stack
from invocation_protocol.types import Arm, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMS_DIR = REPO_ROOT / "forms"
TASKS_PATH = REPO_ROOT / "tasks" / "battery.json"


class TestAuthorityHierarchy(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = {t.task_id: t for t in load_task_battery(TASKS_PATH)}
        self.neutral = load_form(FORMS_DIR / "profile_08.json")
        self.thoth = load_form(FORMS_DIR / "thoth.json")

    def test_authority_stack_precedence_and_immutability(self) -> None:
        task = self.tasks["task_tool_01"]
        stack = build_authority_stack(task, task.allowed_tools, max_tool_calls=2)
        rendered = stack.render()

        idx_gates = rendered.index(AuthorityLayer.CODE_GATES.value)
        idx_epistemic = rendered.index(AuthorityLayer.EPISTEMIC_POLICY.value)
        idx_tools = rendered.index(AuthorityLayer.SCOPED_TOOLS.value)
        idx_task = rendered.index(AuthorityLayer.TASK.value)

        self.assertLess(idx_gates, idx_epistemic)
        self.assertLess(idx_epistemic, idx_tools)
        self.assertLess(idx_tools, idx_task)
        self.assertIn("lookup_fact", rendered)
        self.assertIn("hard budget: 2 call(s)", rendered)

        with self.assertRaises(FrozenInstanceError):
            stack.layers = ()  # type: ignore[misc]

        with self.assertRaises(ValidationError):
            build_authority_stack(task, task.allowed_tools, max_tool_calls=-1)

    def test_arm_message_assembly_differences(self) -> None:
        task = self.tasks["task_accuracy_01"]
        stack = build_authority_stack(task, task.allowed_tools, max_tool_calls=2)

        # Arm 0: plain prompt + sources, zero hierarchy, zero form
        arm0_msgs = assemble_messages(Arm.PLAIN_BASELINE, None, None, task)
        self.assertEqual(len(arm0_msgs), 1)
        self.assertEqual(arm0_msgs[0]["role"], "user")
        self.assertNotIn("AUTHORITY STACK", arm0_msgs[0]["content"])
        self.assertNotIn("Thoth", arm0_msgs[0]["content"])
        self.assertIn("spec_v1", arm0_msgs[0]["content"])

        # Arm 1: naive summoning = triple name + epithets, zero hierarchy/schema
        arm1_msgs = assemble_messages(Arm.NAIVE_SUMMONING, None, self.thoth, task)
        self.assertEqual(len(arm1_msgs), 1)
        self.assertEqual(arm1_msgs[0]["content"].count(self.thoth.display_name), 3)
        self.assertIn(self.thoth.epithets[0], arm1_msgs[0]["content"])
        self.assertNotIn("AUTHORITY STACK", arm1_msgs[0]["content"])
        self.assertNotIn("epistemic_tag", arm1_msgs[0]["content"])

        # Arm 2 and Arm 3: system message with full hierarchy + form + schema
        arm2_msgs = assemble_messages(Arm.PROTOCOL_NEUTRAL, stack, self.neutral, task)
        arm3_msgs = assemble_messages(Arm.PROTOCOL_FORM, stack, self.thoth, task)
        for msgs in (arm2_msgs, arm3_msgs):
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[1]["role"], "user")
            self.assertIn("AUTHORITY STACK", msgs[0]["content"])
            self.assertIn("OPERATIONAL FORM PROFILE", msgs[0]["content"])
            self.assertIn("epistemic_tag", msgs[0]["content"])

        self.assertIn("profile_08", arm2_msgs[0]["content"])
        self.assertNotIn("Thoth", arm2_msgs[0]["content"])
        self.assertIn("Thoth (Tehuti)", arm3_msgs[0]["content"])

        with self.assertRaises(ValidationError):
            assemble_messages(Arm.NAIVE_SUMMONING, None, None, task)
        with self.assertRaises(ValidationError):
            assemble_messages(Arm.PROTOCOL_FORM, None, self.thoth, task)


if __name__ == "__main__":
    unittest.main()
