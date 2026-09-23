"""Generate results/results_annotated.csv from results/results.csv.

results.csv is the canonical append-only log written by qrs.analysis.results
and must keep its exact schema (see the comment in that module: widening
RESULT_COLUMNS would not rewrite the existing header, so later rows would
carry more values than the header names). This script derives a wider,
human-readable view instead, and is safe to re-run after new experiments.

Adds per row: the qubit/layer configuration, the input representation, a
design label, and the per-group (design x arm) seed count, mean and std, plus
a plain-language statement of what that group's result shows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = Path("results/results.csv")
OUT = Path("results/results_annotated.csv")

# n_trainable_params uniquely identifies a design across this study, because
# it encodes compression width + slot size + head width together.
# (qubits, layers, slot_params, input_repr, experiment)
DESIGN = {
    246: (4, 8, 96, "HSI+Edge 2x3 (24-d)", "No-backbone smoke: 4 qubits x 8 layers"),
    250: (None, None, 0, "HSI+Edge 2x3 (24-d)", "No-backbone linear probe (no slot)"),
    274: (6, 3, 54, "HSI+Edge 2x3 (24-d)", "No-backbone ablation, 54-param slot"),
    328: (6, 6, 108, "HSI+Edge 2x3 (24-d)", "No-backbone smoke: 6 qubits x 6 layers, 10 epochs"),
    514: (6, 3, 54, "HSI+Edge 4x4 (64-d)", "No-backbone, finer pooling grid"),
    556: (2, 2, 12, "rich2 (256-d)", "Rich features, minimum 12-param slot"),
    650: (None, None, 0, "HSI+Edge 4x4 (64-d)", "No-backbone linear probe (no slot)"),
    1666: (6, 3, 54, "rich2 (256-d)", "Rich features, 54-param slot"),
    2170: (8, 1, 24, "rich2 (256-d)", "Rich features, wide and shallow (8q x 1L)"),
    2194: (8, 2, 48, "rich2 (256-d)", "Rich features, wide and deeper (8q x 2L)"),
    2710: (10, 1, 30, "rich2 (256-d)", "Rich features, 10 qubits - EXCEEDS 8-qubit cap"),
    5130: (None, None, 0, "Frozen ResNet-18 (512-d)", "Pretrained linear probe (no slot)"),
    243018: (8, 3, 72, "Trained CNN, large (128-d)", "HEADLINE: real CNN backbone + 72-param slot"),
    243114: (None, None, 0, "Trained CNN, large (128-d)", "Real CNN backbone, no slot (upper bound)"),
}

# What each (design, arm) group actually shows. Keyed (n_trainable_params, arm).
STATES = {
    (246, "quantum"): "Single-seed smoke test only. Deeper circuit (96 params) did not beat the 54-param slot.",
    (246, "control"): "Single-seed smoke test only. Matched control slightly ahead of quantum.",
    (250, "classical"): "Floor reference: 24 pooled numbers and a linear head reach 0.43. Any slot must beat this to justify itself.",
    (274, "quantum"): "Quantum 0.538 vs control 0.403. Apparent quantum win, but the control collapsed on 1 of 5 seeds (dying ReLU).",
    (274, "control"): "Collapsed on seed 1 (F1 0.022). Healthy seeds average ~0.50; the reported mean is dragged down by the failure.",
    (274, "control_scaled"): "Giving the classical slot the quantum arm's sigmoid-x-pi input squashing did NOT help - it scored slightly worse. Rules out input scaling as the source of quantum's lead.",
    (328, "quantum"): "Single-seed, 10 epochs only. Doubling depth to 108 params made accuracy worse, not better.",
    (328, "control"): "Single-seed, 10 epochs only. Control ahead of quantum at this budget and schedule.",
    (514, "quantum"): "Quantum 0.550 vs control 0.503 on a richer 64-d input. Same ~0.05 gap as the 24-d grid.",
    (514, "control"): "No collapses here. Finer pooling grid barely helped control (0.503 vs 0.498 at 24-d) - the bottleneck is feature type, not grid size.",
    (556, "quantum"): "Quantum 0.460 vs control 0.148. Largest apparent gap in the study - but the control collapsed on 2 of 3 seeds.",
    (556, "control"): "Collapsed on 2 of 3 seeds. At 12 params the matched MLP is forced to a near-rank-1 hidden layer and frequently dies. NOT retested with Tanh.",
    (650, "classical"): "Floor reference on the 64-d input. Linear probe reaches 0.485 - within 0.02 of the 54-param control, showing the slot adds little here.",
    (1666, "quantum"): "Quantum 0.689 vs control 0.685: a statistical tie. The richer input closed the gap that existed at 24-d and 64-d.",
    (1666, "control"): "No collapses - the 54-param budget affords ~4 hidden units, enough to avoid the rank-1 failure. Ties quantum.",
    (2170, "quantum"): "Quantum 0.625 with only 24 slot params. Looks far ahead of control, but control was broken here.",
    (2170, "control"): "Collapsed on 2 of 3 seeds. At 24 params with 8-wide I/O the search is forced to a SINGLE hidden neuron (8->1->8). NOT retested with Tanh.",
    (2194, "quantum"): "Quantum 0.691 - best rich-feature result, tying the 54-param configuration from a different direction.",
    (2194, "control"): "ReLU control: collapsed on 1 of 3 seeds, mean 0.446 with huge variance. Superseded by control_tanh below.",
    (2194, "control_tanh"): "THE KEY CONTROL. Same 48 params, same shape, only Tanh instead of ReLU: zero collapses, 0.676 vs quantum's 0.691. A 0.245 gap closed to 0.015 - inside noise. The apparent quantum advantage was a dying-ReLU artifact.",
    (2710, "quantum"): "10 qubits exceeds the project's 8-qubit cap; reported separately. 0.625 - no better than 8 qubits, confirming diminishing returns.",
    (5130, "classical"): "Frozen pretrained ResNet-18 features reach 0.916 with a bare linear head - but this bypasses the compression and slot entirely, so it is NOT a quantum comparison. Shows the accuracy ceiling was the features.",
    (243018, "quantum"): "HEADLINE RESULT. Quantum 0.9277 vs control 0.9420 over 5 seeds: significantly WORSE (p=0.0017), with complete seed separation, at 4.8x the training cost and 8.3x inference.",
    (243018, "control"): "HEADLINE CONTROL. 0.9420 over 5 seeds, no collapses, and matches the unbottlenecked classical arm - so the bottleneck architecture itself costs nothing. The accuracy loss is specific to the quantum substitution.",
    (243114, "classical"): "Upper bound: full CNN, no bottleneck, 0.9412. The control matches it, so quantum's 1.4-point deficit is a real cost, not a bottleneck artifact.",
}

COLLAPSE_F1 = 0.05


def main() -> None:
    df = pd.read_csv(SRC)

    design = df.n_trainable_params.map(DESIGN)
    unknown = df[design.isna()].n_trainable_params.unique()
    if len(unknown):
        raise SystemExit(f"Unmapped designs in {SRC}: {sorted(unknown)} - add them to DESIGN.")

    # Int64 (nullable) so "8" prints as 8, not 8.0, while arms with no slot stay blank.
    df["n_qubits"] = pd.array([d[0] for d in design], dtype="Int64")
    df["n_layers"] = pd.array([d[1] for d in design], dtype="Int64")
    df["slot_params"] = [d[2] for d in design]
    df["input_repr"] = [d[3] for d in design]
    df["experiment"] = [d[4] for d in design]
    df["collapsed"] = (df.f1_weighted < COLLAPSE_F1).map({True: "yes", False: "no"})

    grp = df.groupby(["n_trainable_params", "arm"]).f1_weighted
    df["group_n_seeds"] = grp.transform("count")
    df["group_mean_f1"] = grp.transform("mean").round(4)
    df["group_std_f1"] = grp.transform("std").round(4)

    df["states"] = [
        STATES.get((p, a), "") for p, a in zip(df.n_trainable_params, df.arm, strict=True)
    ]
    missing = {(p, a) for p, a in zip(df.n_trainable_params, df.arm, strict=True)} - set(STATES)
    if missing:
        print(f"[warn] no interpretation for: {sorted(missing)}")

    df = df.sort_values(["n_trainable_params", "arm", "seed"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} - {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
