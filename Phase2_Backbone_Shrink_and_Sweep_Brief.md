# Phase 2 Brief — Shrunk Backbone + Parameter Sweep (EuroSAT)

**Hand this to Claude Code in the existing QCNN_Proj repo.** It extends the current
codebase — do not start a new repo or rewrite the training loop.

---

## Why this phase exists

The Phase 1 result (quantum vs. control, both at 54 params) showed no accuracy or
efficiency advantage for the quantum layer. But the shared CNN backbone was 241,824
parameters — the 54-param slot was 0.02% of the model. At that ratio, no comparison
of 12 vs 24 vs 54 vs 108 slot-parameters can move total-model storage, memory, or
latency by a measurable amount. The slot's contribution is invisible against a
backbone two orders of magnitude larger.

This phase shrinks the backbone to a few thousand parameters, so the slot becomes a
real fraction of the model, then sweeps the slot's parameter count and measures four
axes — accuracy, storage, memory, and latency — not accuracy alone.

**Decision rule, fixed in advance:** if quantum matches or beats parameter-matched
control on accuracy AND wins on any one of storage / memory / latency, that is a
positive signal worth generalizing to a second dataset. If not, that is the answer,
and it gets reported as such — not chased across datasets until something looks better.

---

## Non-negotiable rules (same as Phase 1, restated because they matter)

1. All arms train under **identical conditions** — same backbone, epochs, batch size,
   optimizer, LR, split, seeds. Only the middle-block slot differs between arms.
2. Every experiment runs on **≥3 seeds minimum** (5 preferred if compute allows).
   Report mean ± std. Never report a single run.
3. **Parameter-matching assertion is mandatory** for every quantum/control pair,
   exactly like the existing `[classical_control] target_params=... actual_params=...
   rel_error=...` check. It must fail loudly on mismatch, not warn.
4. **Report accuracy, storage, memory, and latency as four separate metrics.** Never
   let a win on one imply a win on another.
5. Log every run to the same `results.csv` schema already in use, with two new
   columns: `slot_target_params` and `backbone_variant`.

---

## Step 1 — Shrink the shared backbone

**Target: ~3,000–5,000 total backbone parameters**, down from 241,824.

Current backbone (`qrs/models/backbone.py`):
```
block1: Conv2d(4→32, 3x3) + BN + ReLU + MaxPool     → 1,248 params
block2: Conv2d(32→64, 3x3) + BN + ReLU + MaxPool    → 18,624 params
block3: Conv2d(64→128, 3x3) + BN + ReLU + MaxPool   → 74,112 params
block4: Conv2d(128→128, 3x3) + BN + ReLU + MaxPool  → 147,840 params
GlobalAvgPool                                        → 0 params
```

Proposed shrunk backbone — narrow the channel counts, drop to 3 blocks:
```
block1: Conv2d(4→8, 3x3) + BN + ReLU + MaxPool      → ~320 params
block2: Conv2d(8→16, 3x3) + BN + ReLU + MaxPool     → ~1,168 params
block3: Conv2d(16→24, 3x3) + BN + ReLU + MaxPool    → ~3,480 params
GlobalAvgPool → 24-d feature vector                  → 0 params
```
Total: **~4,968 params** — roughly 49x smaller than the original, and now the same
order of magnitude as the slot sizes being tested.

Implementation instructions for Claude Code:

> In `qrs/models/backbone.py`, add a `backbone_variant: str` config field with values
> `"large"` (existing 32/64/128/128 backbone, keep unchanged and default for any
> config that doesn't set this field, so Phase 1 configs and results remain
> reproducible) and `"small"` (new 8/16/24 backbone, 3 blocks, as specified above).
> Route `build_model()` to construct the correct one based on this field. Add a unit
> test asserting the small backbone's parameter count is between 4,500 and 5,500.
>
> The output feature width changes from 128 to 24 for the small backbone — update
> the middle-block input dimension accordingly (the `Linear(128→N)` projection
> becomes `Linear(24→N)` for every arm, where N is the qubit/slot count). Make this
> dimension read from the backbone's actual output width, not hardcoded, so it
> can't silently mismatch.

---

## Step 2 — Re-baseline at the new backbone size

Before sweeping, re-run the existing 54-param point (quantum + control) with the
**small** backbone, 3 seeds each. This is not optional — Phase 1's 54-param numbers
were measured against the old backbone and are not comparable to anything measured
against the new one.

```python
p = make("small54_quantum_s012", arm="quantum", backbone_variant="small",
        n_qubits=6, n_layers=3, epochs=30, seeds=[0,1,2])
```
```python
p = make("small54_control_s012", arm="control", backbone_variant="small",
        n_qubits=6, n_layers=3, epochs=30, seeds=[0,1,2])
```

**Checkpoint:** compare these F1 numbers to Phase 1's (quantum 0.9273, control
0.9300). Expect both to drop somewhat — a 4,968-param backbone extracts weaker
features than a 241,824-param one. If either arm collapses to near-random, the small
backbone is undersized for EuroSAT's 64x64x4 input and needs a wider middle block
(e.g., 8/16/32 instead of 8/16/24) before proceeding — do not sweep on top of a
broken baseline.

---

## Step 3 — The parameter sweep

Four budgets, quantum config on the left, matched by the existing assertion:

| Target params | Qubits | Layers | Formula (qubits x layers x 3) |
|---|---|---|---|
| ~12 | 2 | 2 | 12 |
| ~24 | 4 | 2 | 24 |
| 54  | 6 | 3 | 54  (already run in Step 2) |
| ~108 | 6 | 6 | 108 |

For each budget, run quantum and control, 3 seeds each, small backbone:

```python
for qubits, layers, name in [(2,2,"p12"), (4,2,"p24"), (6,6,"p108")]:
    p = make(f"sweep_{name}_quantum", arm="quantum", backbone_variant="small",
            n_qubits=qubits, n_layers=layers, epochs=30, seeds=[0,1,2])
    p = make(f"sweep_{name}_control", arm="control", backbone_variant="small",
            n_qubits=qubits, n_layers=layers, epochs=30, seeds=[0,1,2])
```

**Before committing all 18 new runs (3 budgets x 2 arms x 3 seeds):** smoke-test the
2-qubit point at 1 seed first. Very small circuits (2 qubits, 2 layers) can hit the
same trainability failure that broke the original semester-5 circuit. Confirm it
trains above random (>0.5 F1 on 10 classes) before spending the remaining seeds on
it.

> Ask Claude Code: add `n_qubits` and `n_layers` as first-class fields the config
> factory (`make()`) can override per call, if they are not already. Confirm the
> quantum layer module rebuilds its circuit correctly for each combination rather
> than caching a stale circuit from a previous config.

---

## Step 4 — Measure all four axes, not just accuracy

For every (budget, arm) pair, in addition to the training run, run the existing
efficiency profiler:

```python
!python profile_models.py --arms quantum control \
    --backbone-variant small --n-qubits <Q> --n-layers <L>
```

> Ask Claude Code: extend `profile_models.py` to accept `--backbone-variant`,
> `--n-qubits`, and `--n-layers` flags so it can profile each sweep point, not just
> the default architecture. It should already report params, memory (MB),
> inference latency/FPS, and the split classical-GFLOPs / quantum-gate-ops /
> quantum-sim-GFLOPs columns from Phase 1 — reuse that logic unchanged.

This gives you, per budget: F1 (mean ± std), storage (MB), peak inference memory
(MB), inference latency (ms/image, FPS), and the GFLOPs breakdown — for both quantum
and control, so every axis is a matched pair, never quantum alone.

---

## Step 5 — Build the sweep figure and apply the decision rule

Plot parameter count (x, log scale) against F1 (y), one line for quantum, one for
control, shaded std-dev bands — same style as the brief's requested figure. Build
three more small tables (not full line plots needed) for storage, memory, and
latency at each budget, quantum vs. control side by side.

Then apply the decision rule from the top of this document, explicitly, in writing:

- Does quantum match or beat control's F1 at any budget (paired comparison, same
  seeds)?
- Does quantum win on storage, memory, or latency at any budget, at equal or better
  accuracy?

**If yes to both at any single budget:** that budget is the candidate configuration
to carry into the hyperspectral generalization study (Pavia University first, per
the brief's own sequencing advice).

**If no:** write up the sweep as a complete negative result on EuroSAT before
touching a new dataset. A second dataset does not rescue an effect that isn't there
on the first one — it only adds compute cost, per your own reasoning for deferring
it.

---

## Time and compute estimate

| Step | Runs | Rough time |
|---|---|---|
| 2 (re-baseline) | 6 (2 arms x 3 seeds) | ~3-8h depending on shrunk-backbone speed |
| 3 smoke test | 1 | ~15-30 min |
| 3 full sweep | 18 (3 budgets x 2 arms x 3 seeds) | ~15-25h, dominated by the 108-param quantum point |
| 4 profiling | 8 (4 budgets x 2 arms) | ~20 min total, no training |

**Total: roughly 20-30h** — similar order to Phase 1's full sweep. Given this is a
new phase rather than an Aug 31 deliverable, treat it as September work and pace it
across sessions rather than compressing it.

---

## First message to Claude Code

> Read this brief. Start with Step 1 only: add the `backbone_variant` config field
> and the small (8/16/24, 3-block) backbone alongside the existing large one,
> defaulting to large for backward compatibility. Add the parameter-count unit test.
> Update the middle-block projection to read the backbone's actual output width
> instead of hardcoding 128. Show me the diff and the test output before moving to
> Step 2.

Work one step at a time, exactly as with Phase 1. Confirm each step's checkpoint
before proceeding to the next.
