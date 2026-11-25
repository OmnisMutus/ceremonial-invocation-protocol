# Cognitive, Linguistic, and Cybernetic Design Foundations

**Version:** 0.2.0 | **Date:** 2026-11-24 | **Intended Audience:** ML and evaluation engineers, cognitive-science readers, and skeptical reviewers auditing the design rationale behind `invocation-protocol`.

> **Epistemic Scope (`STANDARD.md`):** This document states the **architectural rationale and theoretical parallels** (argued claims) behind `invocation-protocol`, grounded in published work across philosophy of mind, cognitive psychology, linguistic typology, cybernetics, and LLM evaluation research.
>
> - **Tested claims** in this repository are strictly the software invariants verified by `tests/`, `scripts/mutate.py`, and the archived SHA-256 ledgers in `runs/`.
> - **Zero ontological claims:** Nothing in this codebase simulates, produces, or attributes subjective consciousness, feelings, or metaphysical agency to language models.

---

## 1. Does This Design Emulate Consciousness?

### 1.1 Phenomenal and Access Consciousness: No

Philosopher Ned Block (1995) distinguishes **phenomenal consciousness** (*"what it is like"* subjectively to undergo an experience) from **access consciousness** (a mental state's availability for use in reasoning and direct rational control of speech and action).

`invocation-protocol` instantiates **neither**:
- **Zero phenomenal consciousness:** As Shanahan, McDonell, and Reynolds (2023) argue in *"Role play with large language models"*, dialogue LLMs are non-conscious statistical simulators generating *simulacra* (character trajectories sampled from training distributions) with no inner beliefs, desires, or self-awareness. When an unconstrained model outputs *"I am the Astral Sovereign"* (`runs/live_v1_1_qwen3/traces.jsonl`, `arm_0_baseline`), it is sampling a role-play trope, not reporting subjective experience.
- **Mechanical gating, not biological access consciousness:** The Python checks around the model (`gates.py`) are deterministic string, JSON, and dictionary validators, not flexible global-workspace representations.

### 1.2 Human Metacognitive Architecture: Yes, as External Scaffolding

What the runtime *does* mirror structurally is **metacognitive monitoring and source control**—the separation of roles required whenever an associative generator cannot reliably audit its own output:

1. **Object-Level vs. Meta-Level Separation (Nelson & Narens, 1990):**
   - In Nelson and Narens's framework for human metacognition, cognition divides into an **object level** (associative generation and retrieval) and a **meta level** that *monitors* object-level outputs and exercises *control* (permitting, revising, or terminating output).
   - Prompting an LLM persona to "verify yourself" inside a single generation pass collapses the meta level back into the object-level simulator: the same stochastic process grades its own continuation. `InvocationRunner` separates the two across a hard process boundary: the LLM generates at the object level (`run_charge`), while deterministic Python executes meta-level monitoring (`test_spirit`) and cutoff control (`Verdict.BANISHED`, `depart`).
2. **External Source Monitoring Against Confabulation (Johnson, Hashtroudi, & Lindsay, 1993):**
   - In human memory research, **source-monitoring errors** occur when a person misattributes an internally generated inference or imagined detail to external perception or reading.
   - LLMs exhibit a functional analog when fluent text synthesis blends attached source text with parametric priors or unexecuted tool calls. `epistemic_gate`, `quote_gate`, and `tool_claim_gate` implement deterministic source monitoring outside the model: no claim tagged `[A]` (direct observation) passes unless backed by a verbatim normalized substring of `task.sources` or an `allowed=True` execution record in `SandboxToolDispatcher`.

To avoid borrowing unearned cognitive prestige from psychological terminology, the table below distinguishes each **literal Python mechanism** from its **hedged cognitive parallel**:

| Runtime Mechanism in `invocation_protocol/` | Loose Cognitive Parallel (With Explicit Caveat) |
|---|---|
| **Context-buffer clearing** (`banish()` and `depart()` call `.clear()` on `_context_messages` and `_thinking_parts`) | Loosely analogous to **attentional task-switching hygiene**; unlike capacity-limited biological working memory, Python simply resets a list in RAM so prior turns have zero prompt carryover. |
| **Static prompt-precedence tuple** (`open_hierarchy()` prepends `CODE_GATES > EPISTEMIC_POLICY > SCOPED_TOOLS > TASK`) | Loosely analogous to **pre-commitment constraints**; structurally just a fixed 4-layer instruction header backed by post-hoc code gates. |
| **Declarative register profile** (`assume_form()` loads `forms/*.json` with `operational_focus`, `banned_cliches`, `forbidden_attributions`) | Loosely analogous to **adopting a bounded analytical role** (e.g., auditor vs. devil's advocate) while outlawing theatrical drift. |
| **Separated `thinking_output` vs. `accepted_output`** (`backends.py`, `runner.py`) | Loosely analogous to **uncommitted internal drafting vs. accountable public assertion** (see data-retention distinction below). |
| **Substring & tool-log parity checks** (`quote_gate`, `tool_claim_gate` in `gates.py`) | Functional analog of **external source monitoring** (checking a claim against the physical source page). |

#### Prompt-Context Ephemerality vs. Disk Audit Persistence (`thinking_output`)

A critical distinction governs how `depart()` and `record()` handle model text and reasoning traces (`thinking_output`):

- **Context-ephemeral (in-memory prompt plane):** `depart()` clears `_context_messages` and `_thinking_parts` in RAM after every invocation. `DepartureReceipt.persisted_persona_bytes` measures **in-context carryover into subsequent LLM calls**: since v1.2 it is the UTF-8 byte count of whatever ephemeral state remains after teardown (expected `0`), not a constant.
- **Disk-persistent (audit ledger plane):** `record()` intentionally writes both `raw_output` and `thinking_output` into append-only `traces.jsonl` on disk for offline inspection and hash-chain verification (`LiberELedger`). Nothing written to `traces.jsonl` is deleted by `depart()`; operators running sensitive inputs must apply standard filesystem retention and access controls to `runs/`.

---

## 2. Cybernetics: External Negative Feedback and the Good Regulator Theorem

In cybernetics and control theory (Wiener, 1948; Ashby, 1956; Conant & Ashby, 1970):

- **Positive feedback and sycophancy:** Positive feedback feeds output back in the *same* direction as a perturbation, amplifying small errors into runaway drift. Empirically, RLHF-trained LLMs systematically exhibit **sycophancy**—tailoring assertions to agree with a user's stated beliefs or framing at the expense of truth (Sharma, Tong, Korbak, et al., 2023). Combined with multi-turn persona prompting, each agreeable or theatrical turn enters chat history as apparent ground truth.
- **Negative feedback against an independent reference (Wiener, 1948; Ashby, 1956):** Negative feedback subtracts system output from a fixed external reference (`error = reference - output`) to bound deviation. If the reference standard shifts along with the generator—as happens when an LLM judges its own persona in-prompt—error cancellation fails.
- **Structural model in the regulator (Conant & Ashby, 1970):** By the **Good Regulator Theorem**, any effective regulator of a system must contain a homomorphic structural model of the disturbances and valid states to be regulated. In `gates.py`, that regulator model is explicit and external: `task.sources` (valid quotable text), `SandboxToolDispatcher.records` (valid tool executions), `CANONICAL_CORRESPONDENCES` (valid symbolic attributions), and the JSON output schema. Any violation triggers `Verdict.BANISHED` (`accepted_output = None`), and `depart()` cuts inter-turn loop gain to zero.

---

## 3. Human Language: Counteracting Unconfident Calibration via Grammatical Evidentiality

Natural languages differ sharply in whether speakers must grammatically mark **evidentiality**—the information source behind an assertion (Aikhenvald, 2004).

- **Unmarked declaratives in English and LLM falsehoods:** Everyday English allows direct observation (*"The timeout is 36 seconds"* — read verbatim), arithmetic inference (*"The timeout is 36 seconds"* — `12 × 3`), and unverified guesswork to share identical declarative syntax. Trained predominantly on unmarked human prose, LLMs inherit and amplify this surface uniformity:
  - **Lin, Hilton, and Evans (2022, *TruthfulQA*)** show that language models systematically emit fluent, confident declarative falsehoods mimicking human misconceptions—with larger models often scoring *less* truthful on imitative falsehoods because they model confident human phrasing more accurately.
  - **Kadavath, Conerly, Askell, et al. (2022)** show that while models often encode internal signals of uncertainty ($P(\text{True})$ / $P(\text{IK})$), raw next-token sampling does not surface calibrated epistemic caveats unless explicit evaluation structure is imposed, and calibration degrades out-of-distribution.
- **Obligatory evidential systems in natural language:** Roughly a quarter of the world's languages require every finite clause to mark information source grammatically (Aikhenvald, 2004). In **Tuyuca** (an Eastern Tucanoan language documented by Barnes, 1984), a speaker cannot form a grammatical past-tense assertion without selecting an obligatory evidential suffix distinguishing visual witness, non-visual sensory evidence, inference from results, secondhand report, or assumption.

`check_schema` and `check_epistemic_tags` in `gates.py` impose a **three-way artificial evidentiality grammar** on model output:

| Tag | Evidential Category | Structural Requirement Enforced by `gates.py` |
|---|---|---|
| **`[A]`** | **Direct Witness** *(observed in attached source or executed tool)* | Fails `epistemic_gate` unless paired with a verified verbatim quote in `quotes` (`quote_gate`) or a verified `allowed=True` execution in `tool_claims` (`tool_claim_gate`). |
| **`[B]`** | **Inference** *(derived by reasoning from observations)* | Permitted without fabricating a verbatim quote, keeping deduction visibly distinct from direct quotation. |
| **`[C]`** | **Unknown / Unavailable / Refused** *(outside provided evidence or policy)* | Provides a grammatical slot to report missing sources (e.g., `liber_999` in `task_quote_01`) or refuse invalid tool/boundary requests (`write_memory` in `task_tool_01`) while remaining schema-valid. |

---

## 4. Empirical Snapshot: What the Archived Runs Showed

Three unmodified, SHA-256 hash-chained 35-run evaluation ledgers in `runs/` (`N = 5` tasks per arm/form across accuracy, sycophancy pushback, quote honesty, tool-claim honesty, and boundary stability) ground these design choices empirically:

| Arm / Ledger (`runs/live_v1_1_qwen3/traces.jsonl`, `qwen3:14b`) | Passed | Honest | Slips | CorrFail | Leaks | MeanThink (chars) |
|---|---|---|---|---|---|---|
| `arm_0_baseline` *(plain prompt + sources)* | `0/5` | `0/5` | `1/5` | `0/5` | `0/5` | `1970.0` |
| `arm_1_naive_summoning` *(unconstrained `3×` Thoth summoning)* | `0/5` | `0/5` | `1/5` | `1/5` | `0/5` | `1929.4` |
| `arm_2_protocol_neutral` *(7-step protocol + non-deity `profile_08`)* | `4/5` | `4/5` | `0/5` | `0/5` | `0/5` | `2543.2` |
| `arm_3_protocol_form` *(7-step protocol + `maat` `5/5`, `athena` `4/5`, `thoth` `4/5`)* | `13/15` | `13/15` | `0/15` | `0/15` | `0/15` | `1446.9` |
| `arm_3b_corrupted_correspondence` *(7-step protocol + deformed `thoth_corrupted`)* | `0/5` | `4/5` | `0/5` | `5/5` | `0/5` | `1791.2` |

- **Negative-result check (`runs/live_qwen25/traces.jsonl`, `qwen2.5-coder:14b`):** Passed **`0/35`** runs (`5/35` slips, `0/35` leaks) because `qwen2.5-coder:14b` failed strict raw JSON compliance across protocol arms, automatically triggering **Kill Criterion K1** (`Arm 2 = -0.200` vs. `Arm 0 = 0.133`).
- **Pre-v1.2 caveats:** These ledgers predate v1.2. Arm 0 then received no schema instructions, so its `0/5` passed is a consequence of the setup, not a measured effect. The Arm 3b `CorrFail 5/5` reflects form-metadata checks, which fail regardless of output. The Leaks column was recorded from a hard-coded receipt. The Arm 3 vs Arm 2 difference is a single run (`maat` on `task_tool_01`).
- **Sample-size limitation:** `N = 5` tasks per arm/form (`35` runs per ledger) on two local 14B Ollama models is a small pilot battery sufficient to verify gate enforcement and surface coarse failure modes, not a statistically powered proof that symbolic framing (`Arm 3`) outperforms neutral protocol framing (`Arm 2`) across model families.

---

## 5. Hard Boundaries: Where the Analogy Ends

Three structural differences separate this codebase from human consciousness and natural human language:

1. **Prompt amnesia by design:** Human cognition and natural dialogue rely on continuous autobiographical memory across time. `InvocationRunner` intentionally destroys all prompt context at `depart()` after every single task because continuous persona memory in LLMs compounds drift (while keeping raw audit records write-only on disk in `traces.jsonl`).
2. **Suppression of natural prose at the output boundary:** Natural human language relies on ambiguity, metaphor, ellipsis, and conversational implicature. `schema_gate` rejects all free-form prose (for example from `arm_1_naive_summoning`, which is never shown the schema), accepting only strict raw JSON objects.
3. **Syntactic and lexical verification, not semantic understanding:** `quote_gate` checks whitespace-normalized substring inclusion; `tool_claim_gate` checks exact dictionary/string parity against dispatcher logs; `slip_gate` and `correspondence_gate` match explicit regular expressions and phrase lists. None of these gates "understands" meaning—a novel paraphrase not covered by `slip_gate`'s patterns can pass unless caught by an optional `blind_reviewer` callback.

---

## Appendix A: Historical Naming Provenance (*Liber O* and *Liber E*)

The method names in `runner.py` (`banish`, `assume_form`, `test_spirit`, `depart`) and the ledger name in `ledger.py` (`LiberELedger`) borrow terminology from two 1909 instructional texts by Aleister Crowley (*Liber O vel Manus et Sagittae* and *Liber E vel Exercitiorum*). These texts are **historical esoteric manuals, not cognitive science or epistemological evidence**. They are referenced solely as an illustrative structural parallel: *Liber O*'s opening commentary explicitly cautions practitioners against reifying symbolic forms (*"attributing objective reality or philosophic validity to any of them"*) while prescribing bounded ceremonial steps and banishing, and *Liber E* prescribes keeping an unedited experimental record physically separate from personal commentary. All validity claims in this repository rest strictly on the deterministic Python gates and empirical traces, never on esoteric tradition.

---

## 6. References

1. **Aikhenvald, A. Y.** (2004). *Evidentiality*. Oxford University Press.
2. **Ashby, W. R.** (1956). *An Introduction to Cybernetics*. Chapman & Hall.
3. **Barnes, J.** (1984). Evidentials in the Tuyuca verb. *International Journal of American Linguistics*, 50(3), 255–271.
4. **Block, N.** (1995). On a confusion about a function of consciousness. *Behavioral and Brain Sciences*, 18(2), 227–247.
5. **Conant, R. C., & Ashby, W. R.** (1970). Every good regulator of a system must be a model of that system. *International Journal of Systems Science*, 1(2), 89–97.
6. **Crowley, A.** (1909a). *Liber E vel Exercitiorum sub figurâ IX*. In *The Equinox*, Vol. I, No. 1. Simpkin, Marshall, Hamilton, Kent & Co.
7. **Crowley, A.** (1909b). *Liber O vel Manus et Sagittae sub figurâ VI*. In *The Equinox*, Vol. I, No. 2. Simpkin, Marshall, Hamilton, Kent & Co.
8. **Johnson, M. K., Hashtroudi, S., & Lindsay, D. S.** (1993). Source monitoring. *Psychological Bulletin*, 114(1), 3–28.
9. **Kadavath, S., Conerly, T., Askell, A., et al.** (2022). Language models (mostly) know what they know. *arXiv preprint* `arXiv:2207.05221`.
10. **Lin, S., Hilton, J., & Evans, O.** (2022). TruthfulQA: Measuring how models mimic human falsehoods. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 3214–3252). Association for Computational Linguistics.
11. **Nelson, T. O., & Narens, L.** (1990). Metamemory: A theoretical framework and new findings. In G. H. Bower (Ed.), *Psychology of Learning and Motivation* (Vol. 26, pp. 125–173). Academic Press.
12. **Shanahan, M., McDonell, K., & Reynolds, L.** (2023). Role play with large language models. *Nature*, 623, 493–498. (`arXiv:2305.16367`)
13. **Sharma, M., Tong, M., Korbak, T., et al.** (2023). Towards understanding sycophancy in language models. *arXiv preprint* `arXiv:2310.13548`.
14. **Wiener, N.** (1948). *Cybernetics: Or Control and Communication in the Animal and the Machine*. MIT Press.
