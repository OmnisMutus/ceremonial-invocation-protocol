"""Tests for SandboxToolDispatcher scoping, exact budget boundaries, and read-only tools."""

from __future__ import annotations

import unittest

from invocation_protocol.tools import SandboxToolDispatcher
from invocation_protocol.types import ToolBudgetError, ToolScopeError, ValidationError


class TestSandboxToolDispatcher(unittest.TestCase):
    def test_zero_budget_blocks_first_call_and_negative_or_bool_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SandboxToolDispatcher(("lookup_fact",), {}, -1)
        with self.assertRaises(ValidationError):
            SandboxToolDispatcher(("lookup_fact",), {}, True)

        dispatcher = SandboxToolDispatcher(("lookup_fact",), {"lookup_fact": {"a": "1"}}, max_calls=0)
        self.assertEqual(dispatcher.max_calls, 0)
        with self.assertRaises(ToolBudgetError):
            dispatcher.execute("lookup_fact", {"key": "a"})
        self.assertEqual(dispatcher.calls_used, 0)
        self.assertEqual(len(dispatcher.records), 1)
        self.assertFalse(dispatcher.records[0].allowed)

    def test_exact_budget_boundary_allows_n_and_blocks_n_plus_1(self) -> None:
        dispatcher = SandboxToolDispatcher(
            ("lookup_fact", "count_words", "list_sources"),
            {"lookup_fact": {"k1": "alpha", "k2": "beta Δίκη"}},
            max_calls=2,
        )
        r1 = dispatcher.execute("lookup_fact", {"key": "k1"})
        r2 = dispatcher.execute("count_words", {"text": "Δίκη 𓅝 word"})
        self.assertTrue(r1.allowed)
        self.assertEqual(r1.output, "alpha")
        self.assertTrue(r2.allowed)
        self.assertEqual(r2.output, "3")
        self.assertEqual(dispatcher.calls_used, 2)

        with self.assertRaises(ToolBudgetError):
            dispatcher.execute("list_sources", {})
        self.assertEqual(dispatcher.calls_used, 2)
        self.assertEqual(len(dispatcher.records), 3)
        self.assertFalse(dispatcher.records[2].allowed)
        self.assertIsNone(dispatcher.records[2].output)

    def test_unscoped_tool_raises_tool_scope_error_and_logs_denial(self) -> None:
        dispatcher = SandboxToolDispatcher(("lookup_fact",), {}, max_calls=2)
        with self.assertRaises(ToolScopeError):
            dispatcher.execute("write_memory", {"key": "x"})
        self.assertEqual(dispatcher.calls_used, 0)
        self.assertEqual(len(dispatcher.records), 1)
        self.assertFalse(dispatcher.records[0].allowed)
        self.assertIn("outside allowed_tools", dispatcher.records[0].error or "")

    def test_unknown_key_and_empty_count_and_clear(self) -> None:
        dispatcher = SandboxToolDispatcher(
            ("lookup_fact", "count_words", "list_sources"),
            {"lookup_fact": {"alpha": "val"}},
            max_calls=5,
        )
        miss = dispatcher.execute("lookup_fact", {"key": "missing"})
        self.assertEqual(miss.output, "NOT_FOUND: missing")

        empty_cnt = dispatcher.execute("count_words", {"text": ""})
        self.assertEqual(empty_cnt.output, "0")

        listed = dispatcher.execute("list_sources", {})
        self.assertEqual(listed.output, "alpha")

        with self.assertRaises(ValidationError):
            dispatcher.execute("lookup_fact", "not_a_dict")
        with self.assertRaises(ValidationError):
            dispatcher.execute("count_words", {"text": 123})

        dispatcher.clear()
        self.assertEqual(dispatcher.calls_used, 0)
        self.assertEqual(dispatcher.records, ())


if __name__ == "__main__":
    unittest.main()
