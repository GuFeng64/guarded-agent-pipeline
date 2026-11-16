# Guarded Agent Pipeline

**A decision layer with guarded LLM agents, deterministic guardrails, and evidence-backed guarantees.**

Models predict. Someone still has to decide **whether to trust a number** — report it, re-measure it, or escalate to a reference method. This repository is that decision layer: a modular pipeline where an LLM *proposes* actions, deterministic rules *gate* them, and every guarantee ships with a test that can refute it.

> **Scope.** This repo contains the decision layer only. The upstream regression models are out of scope; the pipeline consumes predictions `(ŷ, σ)` and never inspects the model. A synthetic data generator makes the whole thing runnable without any private dataset.

---

## Quick start

```bash
pip install -r requirements.txt

python code/mems_data.py      # inspect the synthetic dataset
python demo/run_demo.py       # end-to-end decision layer
python agent_system/run_tests.py   # test suites (see "Testing" below)
```

No API key required. Without `LLM_API_KEY`, every LLM component degrades to a deterministic rule path by design — that fallback is itself tested.

---

## Architecture

```
                    upstream model (out of scope)
                              │  ŷ, σ
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  A4  Assess      conformal intervals ▸ RPD grading        │
   │  A5  Route       budget-constrained re-measurement plan   │
   │  A5b Guarded     LLM proposes ▸ 8 guard rules ▸ or fallback│
   │  A5c Deliberate  propose ⇄ critique ⇄ revise (≤3 rounds)  │
   │  A6  Report      structured text, numbers provenance-checked│
   │  V   Validate    the only component that sees ground truth │
   │  A7  Feedback    flag high-value samples for re-labelling  │
   └──────────────────────────────────────────────────────────┘
```

Two agents have **different capabilities**, which is what makes the interaction real rather than cosmetic:

| Agent | Can do | Cannot do |
|---|---|---|
| **A5b decision agent** | choose actions; query structured fields | **see the spectrum** |
| **A1 perception agent** | inspect the spectrum (spikes/saturation/NaN) | make a decision |

So A5b must *request* `request_spectral_recheck` from A1. That is a genuine cross-agent call, not a data lookup.

---

## Guard rules

Every rule has a test that **must fail** when the rule is violated — a guard that can never fire proves nothing.

| Rule | Rejects |
|---|---|
| G1 | action outside `{report, retest, reference}` |
| G1b | missing target |
| G2 | reporting a target that is not reportable |
| G3 | `report` while a whole model family is degraded |
| G4 | reason code outside the allowed vocabulary |
| G5 | a number in the proposal that no tool ever returned |
| G6 | `report` when `report` has the highest expected loss |
| **G7** | *(batch level)* proposals whose re-measure count exceeds the budget |

**G7 exists because local validity is not global feasibility.** Ten individually-valid proposals can jointly break a budget. Verified: 10 samples all proposing `retest` with a 20% budget → exactly **2** survive, **8** fall back to the rule path.

---

## What the demo prints

Real output from `python demo/run_demo.py` on synthetic data:

```
Fat    R²=+0.8668  RPD= 2.74     ┐
Prot   R²=+0.8711  RPD= 2.78     ├ reportable
TS     R²=+0.8603  RPD= 2.68     ┘
Lact   R²=+0.0546  RPD= 1.03     ┐ not reportable
Urea   R²=-0.0120  RPD= 0.99     ┘

conformal coverage   89.7%  (nominal 90%)
guard on first sample ok=True
validation pass rate 297/297 = 100.0%

family missing           degradation decision
[]                       ok                 Δgap +0.0000
['b9']                   accept_degraded    Δgap +0.0006
['b8']                   retry              Δgap +0.0022
['b7','b8','b9']         abort              Δgap +0.0258
```

The synthetic generator deliberately reproduces the *structure* of the real problem — the reportable set `[Fat, Prot, TS]` emerges from the RPD grades rather than being hard-coded.

---

## Measured results (original dataset, not included)

These numbers come from the original study dataset and are **not reproducible in this repo** (the dataset is not redistributable). They are listed so the claims are auditable.

| Claim | Measurement |
|---|---|
| Reliability signal ranks errors | Spearman ρ = **0.165–0.196** (p < 0.01) |
| Signal beats the industrial heuristic | AURC **+19.8% … +26.5%** vs threshold-proximity, across **5 threshold sets**, **100 random draws** (P = 0.99–1.00), **5 budget points** |
| Signal combinations do not help | **9** combined signals tested — **all worse** than σ alone (−3% … −32%) |
| Conformal calibration | **90.3%** coverage at nominal 90%, **95.1%** at 95% |
| Process isolation | 3 workers, `module_arrays=0`, negative control detects a leaked array (2 vs 0) |
| Fault behaviour | worker killed → 3.01 s; worker hung → 2.01 s; unauthorized field → `PermissionError`; component still usable afterwards |
| LLM faithfulness | CFS **21.8%** (n=124, 95% CI 15.4–29.8%); concept accuracy **46.7%** (n=75, CI 35.8–57.8%) |
| LLM adds decision value | Δ = **−0.002** — not significant; value is explanation and safety, not accuracy |

---

## Design decisions worth arguing about

### The LLM is not allowed to decide alone

Measured concept-level decision accuracy is **46.7%** (95% CI 35.8–57.8%, three-class chance 33%). That is not good enough to decide, and it is *also* not evidence that reasoning is useless — the ablation flip rate (CFS) is 21.8%, so the concepts do drive roughly one decision in five. Hence: **propose → gate**, never **decide → explain**.

### Numbers in reports are provenance-checked, not just formatted

`verify_numbers_strict` extracts every number from the rendered text and requires each one to trace back to a structured field, using **only the formats the template actually emits**. An earlier version allowed all rounding variants of each value, which let `12.3456` whitelist the string `12.3` — a fabricated number could slip through. Both directions are tested: missing numbers *and* invented numbers.

### The pipeline reports what it cannot do

`SYSTEM_README.md` carries an explicit maturity table. Isolation is payload-isolation, **not filesystem isolation** — a process can still read a file it has permission to read, and a negative control proved it. Decision-layer triage only has content when the cost ratio `c_test / E|err|` lies in **[0.20, 2.00]**; outside that range the three-way decision collapses to a single policy and "optimising" it optimises a degenerate problem.

---

## Testing

```bash
python agent_system/run_tests.py
```

```
TOTAL  72 checks pass  |  18 suites skipped (require the original dataset)
```

The suite is split by what it can honestly prove:

| Kind | Runs without data? | Examples |
|---|---|---|
| **Logic** | ✅ yes | guards (8 rules, one violation each), deliberation loop, report templating, number provenance, confidence intervals, analytic metric values (1.60 / 3.20) |
| **Data-anchored** | ⏭ skipped | suites asserting values measured on the original dataset (e.g. gap = 0.0324, coverage = 90.3%) |

Skipping is deliberate: those suites pin the numbers that are supposed to be *reproducible on the real dataset*. Letting them run on synthetic data would either fail spuriously or — worse — be relaxed until they pass, which is how a test stops meaning anything. Each skipped suite prints why it was skipped.

To run them, place the experiment artifacts under `experiments/` (see `DATA_MARKERS` in `run_tests.py`).

**Testing philosophy: every guarantee gets a test that can refute it.**

| Guarantee | How it is refuted |
|---|---|
| "The guard rejects bad proposals" | 7 hand-built violations, one per rule |
| "Isolation keeps data out" | a deliberately leaking worker must be detected |
| "The metric direction is right" | oracle ranking must score best; analytic values 1.60 / 3.20 must be hit exactly |
| "Reports are not tampered" | flip a digit in the text — validation must fail |
| "Degradation is correct" | drop a branch and compare against the measured degradation table |

---

## Project structure

```
code/            synthetic data generator, metrics
demo/            end-to-end decision-layer demo
agent_system/    the pipeline
  agents/        A1 acquisition · A2 branch · A3 aggregate · A4 assess
                 A5 route · A5b guarded · A5c deliberation
                 A6 report/LLM · A7 feedback
  context.py     per-agent access whitelist
  orchestrator.py  DAG execution, timeouts
  degradation.py   three-tier degradation thresholds
  validator.py     the only component that sees ground truth
  process_isolation.py  workers in separate OS processes
  run_tests.py     logic + data-anchored suites (skip-aware)
```

---

## Limitations

- The upstream regression models are **not** included; the demo substitutes a ridge baseline.
- Isolation is payload-level. Filesystem-level isolation would need containers or permission boundaries.
- The triage problem degenerates outside `c_test / E|err| ∈ [0.20, 2.00]`.
- LLM components were evaluated on 25 samples (75 judgments); confidence intervals are reported alongside every proportion rather than the point estimate alone.
- No user study on report readability.

## License

MIT — see `LICENSE`.
