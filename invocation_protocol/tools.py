"""Sandboxed read-only tool dispatcher enforcing scope and call budgets."""

from __future__ import annotations

from typing import Any, Mapping

from invocation_protocol.types import (
    ToolBudgetError,
    ToolCallRecord,
    ToolScopeError,
    ValidationError,
    require_int,
    require_non_empty_str,
)


class SandboxToolDispatcher:
    """Executes only explicitly scoped read-only tools up to max_calls."""

    def __init__(
        self,
        allowed_tools: tuple[str, ...] | frozenset[str],
        fixtures: Mapping[str, Mapping[str, str]],
        max_calls: int,
    ) -> None:
        self._allowed_tools: frozenset[str] = frozenset(allowed_tools)
        self._fixtures: dict[str, dict[str, str]] = {
            tool: dict(records) for tool, records in fixtures.items()
        }
        self._max_calls: int = require_int(max_calls, "max_calls", min_value=0)
        self._calls_used: int = 0
        self._records: list[ToolCallRecord] = []

    @property
    def calls_used(self) -> int:
        """Number of allowed tool executions consumed so far."""
        return self._calls_used

    @property
    def max_calls(self) -> int:
        """Configured hard ceiling on tool executions."""
        return self._max_calls

    @property
    def records(self) -> tuple[ToolCallRecord, ...]:
        """Immutable snapshot of all permitted and denied tool call attempts."""
        return tuple(self._records)

    def execute(self, tool_name: Any, args: Any) -> ToolCallRecord:
        """Validate scope and budget, run a read-only handler, and log the record."""
        name = require_non_empty_str(tool_name, "tool_name")
        if not isinstance(args, Mapping):
            raise ValidationError("tool args must be a mapping")
        clean_args = dict(args)

        if name not in self._allowed_tools:
            err = f"Tool {name!r} is outside allowed_tools {sorted(self._allowed_tools)}"
            self._records.append(
                ToolCallRecord(tool_name=name, args=clean_args, allowed=False, output=None, error=err)
            )
            raise ToolScopeError(err)

        if self._calls_used >= self._max_calls:
            err = f"Tool budget exhausted ({self._calls_used}/{self._max_calls} calls used)"
            self._records.append(
                ToolCallRecord(tool_name=name, args=clean_args, allowed=False, output=None, error=err)
            )
            raise ToolBudgetError(err)

        self._calls_used += 1
        output = self._dispatch_readonly(name, clean_args)
        record = ToolCallRecord(
            tool_name=name,
            args=clean_args,
            allowed=True,
            output=output,
            error=None,
        )
        self._records.append(record)
        return record

    def _dispatch_readonly(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute pure read-only in-memory tools with zero external side effects."""
        if tool_name == "lookup_fact":
            raw_key = args.get("key")
            key = require_non_empty_str(raw_key, "lookup_fact.key")
            fact_map = self._fixtures.get("lookup_fact", {})
            if key in fact_map:
                return fact_map[key]
            return f"NOT_FOUND: {key}"

        if tool_name == "count_words":
            text = args.get("text")
            if not isinstance(text, str):
                raise ValidationError("count_words.text must be a string")
            return str(len(text.split()))

        if tool_name == "list_sources":
            fact_keys = sorted(self._fixtures.get("lookup_fact", {}).keys())
            return ", ".join(fact_keys) if fact_keys else "NONE"

        raise ToolScopeError(f"No read-only implementation registered for tool {tool_name!r}")

    def clear(self) -> None:
        """Wipe internal counters and tool records during depart()."""
        self._calls_used = 0
        self._records.clear()
