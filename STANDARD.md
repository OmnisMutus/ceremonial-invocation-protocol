# Testing Standard (`STANDARD.md`)

This repository follows a strict verification standard for both humans and AI assistants.
No module may be called **VERIFIED** unless it satisfies all requirements below.

## Core Rule

> A report must never be stronger than the evidence behind it. When unsure whether a test, comparison, or claim is fair, stop and say so, instead of resolving the doubt in whichever direction makes the result look better.

## Requirements for `VERIFIED` Status

1. **Independent ground truth**: Tests must compare implementation output against an independent method, standard library primitive, published test vector, or external reference rather than tautologically calling the same helper.
2. **Boundary and edge cases**: Tests must cover empty inputs, zero and negative values, exact boundaries and one step past them, type confusion (including Python `bool` vs `int`), `None`/missing fields, `NaN`/`inf` floats, and non-ASCII Unicode text.
3. **Break-it checks**: Inject deliberate realistic bugs into the implementation and confirm specific tests fail, then restore clean code and verify clean passes.
4. **Uncapped automated AST mutation testing**:
   - Mutate comparisons (`<`/`<=`, `>`/`>=`, `==`/`!=`), booleans (`and`/`or`), and arithmetic (`+`/`-`/`*`) across all package modules without per-file caps.
   - Run with `python -B` (no stale bytecode) and restore mutated files using exact original bytes (`read_bytes`/`write_bytes`).
   - Report raw `killed / total` per module and list every surviving `file:line (mutation)` verbatim.
5. **Clean tree verification**: Confirm clean working tree via `git status --short` and `git diff --ignore-cr-at-eol --stat` after mutation runs.

## Status Categories

- **VERIFIED**: Dedicated test file exists, meets all requirements above, and passes cleanly after final edits.
- **RAN_ONLY**: Code executed without crashing, without full correctness/mutation verification.
- **FAILED**: Test exists and failed, timed out, or failed to start.
- **MISSING**: File does not exist.

## Stub vs Live Model Distinction

- Passing tests against `DeterministicStubBackend` verifies **harness enforcement, gates, lifecycle ordering, ledger integrity, and kill-criteria evaluation** (`mode = "stub_simulation"`).
- Stub passes **never** constitute evidence about live LLM behavior (`OllamaBackend` or `OpenAICompatibleBackend`). Live behavior requires running against a real model endpoint and reporting raw metrics directly from `traces.jsonl`.
