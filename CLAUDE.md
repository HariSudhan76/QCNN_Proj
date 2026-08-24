# Project Brief — Hybrid Quantum–Classical Remote Sensing Classifier

**Hand this whole file to Claude Code at the start of the session.**
Save Part 1 as `CLAUDE.md` in the repo root so it persists across sessions.

---

# PART 1 — `CLAUDE.md` (persistent project rules)

## What this project is

A parameter-efficient hybrid quantum–classical neural network for satellite land-cover
classification, targeting IEEE TGRS. The scientific claim is **not** "quantum beats classical."
The claim is:

> Under a controlled, parameter-matched, multi-seed protocol, does replacing a classical
> compression step with a quantum layer — using physically-grounded HSI-based encoding
> rather than a learned bottleneck — reduce trainable parameters at acceptable accuracy cost?

A rigorous negative result is a valid and publishable outcome. Do not tune the quantum arm
harder than the classical arm to manufacture a win.

## Non-negotiable methodology rules

These exist because the previous version of this project produced an invalid comparison.
Violating any of them invalidates the results.

1. **All arms train under identical conditions.** Same input resolution, same dataset size,
   same epochs, same batch size, same optimiser, same LR schedule, same early-stopping rule.
   The *only* thing that varies between arms is the architecture under test.
2. **Every experiment runs on ≥5 seeds.** Report mean ± std. Never report a single run.
3. **A parameter-matched classical control is mandatory** for every quantum arm. Replace the
   quantum layer with a classical dense layer sized to the same trainable parameter count.
4. **Parameter count and wall-clock time are separate metrics.** Never conflate them. Log both.
   Quantum layers have few parameters but are *expensive to simulate* — say so.
5. **Log everything to a single results CSV.** One row per (arm, seed, dataset, epoch-final).
   Columns: `arm, dataset, seed, n_trainable_params, n_quantum_params, f1_weighted,
   accuracy, precision, recall, train_wallclock_s, inference_wallclock_s, epochs_run, git_sha`.
6. **Fix and record all seeds**: Python `random`, NumPy, PyTorch, and the quantum device seed.

## Tech stack — use exactly this

| Component | Choice | Notes |
|---|---|---|
| Quantum framework | PennyLane ≥ 0.43 | |
| Simulator device | `lightning.qubit` | NOT `default.qubit` — it is far slower |
| ML interface | **PyTorch** | PennyLane's TensorFlow interface is deprecated and removed in 0.44. Do not use TF. |
| Differentiation | `adjoint` on simulator | Note in code comments that hardware would need `parameter-shift` |
| Qubit count | **6–8 qubits**, default 8 | Simulation cost is exponential in qubits. Never default to 16. |
| Data | EuroSAT RGB (Phase 1) | BigEarthNet deferred to September |
| Config | YAML + dataclass | No hardcoded hyperparameters anywhere |

## Hard constraints

- **Never** use `default.qubit` in benchmarks.
- **Never** hardcode a hyperparameter that differs between arms.
- **Never** write TensorFlow/Keras code.
- **Never** silently change a training setting for one arm only.
- If a quantum experiment would take > 30 min on CPU for a smoke test, reduce qubits or
  subset size — do not just let it run.

## Repo conventions

- Python 3.10+, type hints on public functions, `ruff` clean.
- All randomness flows through a single `set_all_seeds(seed)` helper.
- Every experiment is reproducible from a YAML config alone: `python -m qrs.run --config X.yaml`.
- No notebook logic. Notebooks only call into the package.

---

# PART 2 — Build specification

## Environment note: Claude Code local, experiments on Colab

Claude Code works on a local repo; the heavy runs happen on Colab. Use this workflow:

1. Claude Code builds a proper installable package locally (`pip install -e .`).
2. Push to GitHub.
3. Colab notebook does: clone repo → `pip install -e .` → run configs → write results to
   Google Drive.
4. Notebooks stay thin — they must contain no logic worth version-controlling.

Create `notebooks/colab_runner.ipynb` that does exactly this and nothing more.

## Target repo structure

```
qrs/                          # package: "quantum remote sensing"
  __init__.py
  config.py                   # dataclasses + YAML loader
  seeds.py                    # set_all_seeds()
  data/
    __init__.py
    eurosat.py                # download, split, cache
    preprocessing.py          # RGB→HSI, edge channel, 4-ch tensor
    loaders.py                # DataLoader construction (identical across arms)
  models/
    __init__.py
    backbone.py               # shared CNN trunk
    attention.py              # HSI channel attention gate
    quantum_layer.py          # PennyLane PQC as torch.nn.Module
    classical_control.py      # parameter-matched dense replacement
    heads.py                  # classifier heads
    build.py                  # assembles an arm from config
  train/
    __init__.py
    loop.py                   # single training loop used by ALL arms
    metrics.py                # F1/precision/recall/confusion
    callbacks.py              # early stopping, checkpointing
  analysis/
    __init__.py
    results.py                # CSV logging + aggregation across seeds
    explain.py                # Grad-CAM + quantum sensitivity
    circuit_metrics.py        # expressibility, entangling capability
  run.py                      # CLI entrypoint
configs/
  base.yaml
  arm_classical.yaml
  arm_quantum.yaml
  arm_control.yaml            # parameter-matched
  arm_quantum_attn.yaml
tests/
notebooks/
  colab_runner.ipynb
results/
  results.csv
```

## Architecture spec

### Preprocessing (`data/preprocessing.py`)

Input: RGB tile (64×64×3), values in [0,1].

1. **RGB → HSI conversion.** Implement explicitly (do not use a library shortcut — the
   thesis documents these formulas):
   - `I = (R+G+B)/3`
   - `S = 1 - 3*min(R,G,B)/(R+G+B)` (guard divide-by-zero)
   - `θ = arccos( 0.5*((R-G)+(R-B)) / sqrt((R-G)² + (R-B)(G-B)) )` (guard)
   - `H = θ if B<=G else 2π-θ`, normalised to [0,1]
2. **Edge channel.** Adjacency-based: for each pixel, Euclidean distance in (H,S,I) space to
   its 4-neighbours, averaged. Vectorise with array shifts — no Python loops.
3. **Stack** → 4-channel tensor (H, S, I, Edge).

Cache the preprocessed tensors to disk. This is deterministic and must not be recomputed
every run.

### HSI Channel Attention Gate (`models/attention.py`)

Squeeze-and-excitation over the 4 channels, applied **before** the backbone:

```
x: (B, 4, H, W)
s = global_avg_pool(x)          -> (B, 4)
w = sigmoid(Linear(2->4)(ReLU(Linear(4->2)(s))))   -> (B, 4)
out = x * w.view(B,4,1,1)
```

Expose `w` via a `.last_weights` attribute — the per-class distribution of these weights is a
reportable result and feeds the explainability chapter. Make it toggleable by config.

### Quantum layer (`models/quantum_layer.py`)

A `torch.nn.Module` wrapping a PennyLane QNode.

- Device: `qml.device("lightning.qubit", wires=n_qubits)`
- Interface: `torch`, `diff_method="adjoint"`
- **Encoding:** RY angle encoding, one feature per qubit. Input must be scaled to [0, π].
- **Variational block:** use `qml.StronglyEntanglingLayers` with `n_layers` from config.
  **Do NOT build an RZ-only trainable layer** — RZ does not change the Z-measurement
  outcome on its own qubit, which makes it a near-dead parameterisation. This was a real
  defect in the previous version.
- **Entanglement toggle:** config flag `entangle: true|false` that removes the entangling
  gates (for the ablation). When false, use only single-qubit rotation layers.
- **Readout:** `[qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]`
- Expose `n_quantum_params` property.

Also implement an optional **data re-uploading** mode (config flag): repeat the
encoding block between variational layers. This is the highest-value expressivity upgrade
for a small qubit budget.

### Parameter-matched classical control (`models/classical_control.py`)

Given a quantum layer's trainable parameter count `P`, construct a classical MLP with the
same input width, same output width, and trainable parameter count as close to `P` as
possible. Assert at construction time that the counts match within 5%, and log the exact
counts. This assertion must fail loudly, not warn.

### Model assembly (`models/build.py`)

```
build_model(config) -> nn.Module
```
Arms to support:
- `classical` — backbone + classical head only
- `quantum` — backbone + compression + quantum layer + head
- `control` — backbone + compression + parameter-matched dense + head
- `quantum_attn` — as `quantum`, with attention gate enabled
- (September) `fused` — both branches + fusion layer

The backbone, compression width, head, and all training settings must be **identical**
across arms. Only the swapped block differs.

## Training loop (`train/loop.py`)

**One loop, used by every arm.** No per-arm branches in training logic.

Defaults (in `configs/base.yaml`, identical for all arms):
```yaml
epochs: 30
batch_size: 32
optimizer: adam
lr: 1.0e-3
early_stopping:
  monitor: val_loss
  patience: 5
split: [0.70, 0.15, 0.15]
image_size: 64
n_qubits: 8
n_layers: 3
seeds: [0, 1, 2, 3, 4]
```

Log per run: final weighted F1, accuracy, precision, recall, confusion matrix, trainable
parameter count, quantum parameter count, train wall-clock, inference wall-clock.

## Explainability (`analysis/explain.py`)

1. **Grad-CAM** on the classical backbone — standard implementation.
2. **Quantum input sensitivity** — for each encoded angle θᵢ, compute
   `S_i = |f(θᵢ+ε) - f(θᵢ-ε)| / (2ε)`. This reuses parameter-shift machinery.
   Produce per-class mean attribution over the 4 HSI channels.
3. **Do NOT implement SHAP or LIME on the quantum circuit.** Finite-shot randomness makes
   them unreliable (Gil-Fuster et al., 2024). Add a code comment saying so.

## Circuit metrics (`analysis/circuit_metrics.py`)

Pure simulation, no training required — cheap and high-value for the paper:
- **Expressibility**: KL divergence between the circuit's fidelity distribution (sampled over
  random parameter pairs) and the Haar distribution.
- **Entangling capability**: Meyer–Wallach measure averaged over random parameter settings.
- **Gradient variance vs qubit count**: sweep n_qubits ∈ {2,4,6,8,10,12}, plot Var[∂C] on a
  log axis. This is the barren-plateau diagnostic.

---

# PART 3 — Seven-day build order (Aug 24 → Aug 31)

Deliverable on Aug 31 is **working code with intended results on EuroSAT**, not a finished
paper. Build in this order and stop adding scope.

| Day | Goal | Done when |
|---|---|---|
| 1 | Repo scaffold, config system, seeds, EuroSAT download + split, preprocessing with cache | `pytest` green; cached 4-ch tensors on disk; one sample tile visualised |
| 2 | Backbone + classical arm + training loop + results CSV | Classical arm trains end-to-end, writes a row to results.csv |
| 3 | Quantum layer (lightning.qubit, StronglyEntanglingLayers, torch interface) | Quantum arm completes 1 epoch on a small subset without error |
| 4 | Parameter-matched control + full quantum run under matched conditions | 3 arms (classical / quantum / control) each produce results rows |
| 5 | HSI channel attention gate + ablation config | `quantum_attn` arm runs; attention weights logged |
| 6 | Multi-seed sweep (5 seeds × 4 arms), aggregation script, results table | Aggregated mean±std table generated from results.csv |
| 7 | Minimal explainability (Grad-CAM + quantum sensitivity) + buffer | One figure of each; buffer for overruns |

**Explicitly deferred to September:** classical–quantum fusion layer, BigEarthNet,
quantum Shapley, expressibility/entangling sweeps, significance testing.

## Acceptance criteria for Aug 31

- [ ] `python -m qrs.run --config configs/arm_quantum.yaml` runs end-to-end on Colab
- [ ] 4 arms × 5 seeds complete and are logged in `results/results.csv`
- [ ] Parameter-count assertion between quantum and control arms passes
- [ ] Aggregated table shows mean ± std weighted F1 per arm
- [ ] Quantum arm F1 is meaningfully above random (≫ 0.10 on 10 classes) — if it is not,
      **stop and debug**; do not proceed to add features
- [ ] Attention weights are logged per class
- [ ] One Grad-CAM figure and one quantum sensitivity figure exist

## Debug ladder if the quantum arm collapses

Run in this order — do not skip ahead:
1. Train on a **2-class** EuroSAT subset. A working model must solve this easily. If it
   cannot, the bug is in the pipeline, not the physics.
2. Check gradients w.r.t. quantum parameters are non-zero (`.grad` after backward).
3. Verify encoding scaling — inputs must reach the circuit in [0, π], not [0,1] or unbounded.
4. Verify the torch↔PennyLane boundary passes gradients (small unit test on a 2-qubit toy).
5. Widen the compression bottleneck — it may be starving the circuit.
6. Only then consider barren plateaus.

---

# PART 4 — First message to Claude Code

Paste this after the file above:

> Read `CLAUDE.md` and the build specification. Start with Day 1 only: scaffold the repo
> structure, implement `config.py`, `seeds.py`, `data/eurosat.py`, and
> `data/preprocessing.py` with disk caching, plus tests for the HSI conversion (verify
> known RGB→HSI values) and the edge channel. Do not implement models yet. Show me the
> file tree and the test output when done.

Work one day-block at a time. Review before moving on.
