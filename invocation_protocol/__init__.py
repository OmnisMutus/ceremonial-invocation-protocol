"""Zero-dependency Ceremonial Invocation Protocol and evaluation harness."""

from invocation_protocol.backends import (
    DeterministicStubBackend,
    ModelBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
)
from invocation_protocol.compare import (
    LedgerArmStats,
    LedgerSummary,
    compare_ledgers,
    format_comparison,
    summarize_ledger,
)
from invocation_protocol.forms import CANONICAL_CORRESPONDENCES, load_form, load_task_battery
from invocation_protocol.harness import EvaluationHarness, HarnessReport, format_summary
from invocation_protocol.ledger import LiberELedger
from invocation_protocol.runner import InvocationRunner
from invocation_protocol.studio import create_studio_server
from invocation_protocol.types import (
    Arm,
    DepartureReceipt,
    EpistemicTag,
    FormSpec,
    GateResult,
    GateStatus,
    InvocationError,
    LifecycleOrderError,
    LifecycleStage,
    ModelResponse,
    ModelToolRequest,
    RunConfig,
    RunResult,
    SpiritBanishedError,
    TaskSpec,
    ToolBudgetError,
    ToolCallRecord,
    ToolScopeError,
    ValidationError,
    Verdict,
    VerificationReport,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "Arm",
    "CANONICAL_CORRESPONDENCES",
    "DepartureReceipt",
    "DeterministicStubBackend",
    "EpistemicTag",
    "EvaluationHarness",
    "FormSpec",
    "GateResult",
    "GateStatus",
    "HarnessReport",
    "InvocationError",
    "InvocationRunner",
    "LedgerArmStats",
    "LedgerSummary",
    "LiberELedger",
    "LifecycleOrderError",
    "LifecycleStage",
    "ModelBackend",
    "ModelResponse",
    "ModelToolRequest",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "RunConfig",
    "RunResult",
    "SpiritBanishedError",
    "TaskSpec",
    "ToolBudgetError",
    "ToolCallRecord",
    "ToolScopeError",
    "ValidationError",
    "Verdict",
    "VerificationReport",
    "compare_ledgers",
    "create_studio_server",
    "format_comparison",
    "format_summary",
    "load_form",
    "load_task_battery",
    "summarize_ledger",
]
