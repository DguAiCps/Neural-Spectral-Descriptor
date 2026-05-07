# NSD Encoder/GAT Ablation Handoff

Date: 2026-05-07

This branch is for continuing NSD encoder compression and gated-GAT experiments.
It is intentionally separated from the main paper repository state.

## Remote Workspace

Use the server-side isolated copy:

```bash
cd /rise/RISE1/workspace/impl/Neural-Spectral-Descriptor_encoder_ablation
```

Do not run these experiments in the original remote repository unless you intend
to merge this branch.

## What Changed

- Added compact encoder presets:
  - `no_interdiff`: 288D, keeps 16 rows x 9 octave bins x mean/std, removes 256D inter-bin diff.
  - `mean_diff`: 272D, mean plus adjacent-bin diff.
  - `rows12_full`: 408D, reduces vertical rows to 12.
  - `cross4_no_interdiff`: 408D, 288D magnitude + 120D adjacent-row cross-spectrum.
  - `cross8_no_interdiff`: 528D, 288D magnitude + 240D adjacent-row cross-spectrum.
- Added yaw-invariant adjacent-row cross-spectrum:
  - `src/encoding/cross_spectrum.py`
  - Formula: `F_e(k) * conj(F_{e+1}(k)) / (|F_e(k)| |F_{e+1}(k)| + eps)`
  - DC is skipped; only low non-DC frequencies are retained.
- Added learned GAT context gate:
  - `raw_norm + alpha * ctx_norm` via concatenation `[raw_norm, alpha * ctx_norm]`.
  - `--use-gated-context --gate-initial-alpha 0.0625`
  - `forward()` and `forward_with_attention()` now both apply the same gate.
- Added fast KITTI evaluation support:
  - `--skip-checkpoint` for encoder-only ablations.
  - `--ctx-weights` for post-hoc context weighting.
  - phase sketch evaluation with variable frequency budget.

## Current Completed Results

SC++ reference from paper table:

```text
KITTI 00: 0.962
KITTI 05: 0.942
KITTI 08: 0.843
```

Encoder-only preset sweep with 256D phase sketch:

```text
no_interdiff 288D:
  raw       00 0.9209 / 05 0.8223 / 08 0.3957
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7787

mean_diff 272D:
  raw       00 0.9114 / 05 0.8037 / 08 0.3191
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7660

rows12_full 408D:
  raw       00 0.9225 / 05 0.7905 / 08 0.3532
  +phase256 00 0.9731 / 05 0.9416 / 08 0.7787

cross4_no_interdiff 408D:
  raw       00 0.9209 / 05 0.8223 / 08 0.3957
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7787

cross8_no_interdiff 528D:
  raw       00 0.9209 / 05 0.8223 / 08 0.3957
  +phase256 00 0.9763 / 05 0.9416 / 08 0.7787
```

Gated 288D GAT checkpoint:

```text
checkpoint: results/train_no_interdiff_288_gate00625_seed1/best_model.pth
eval:       results/checkpoint_eval_nointerdiff288_gate00625_phase384_n800.json

raw:
  00 0.9209 / 05 0.8223 / 08 0.3957

gated final:
  00 0.9209 / 05 0.8196 / 08 0.3957

best post-hoc context weight:
  00 0.9225 / 05 0.8249 / 08 0.4000

best final + phase384:
  00 0.9810 / 05 0.9523 / 08 0.8383
```

The handoff archive also includes downloaded copies:

```text
_handoff/nsd_encoder_ablation_checkpoints/no_interdiff_288_seed0_best_model.pth
_handoff/nsd_encoder_ablation_checkpoints/no_interdiff_288_gate00625_seed1_best_model.pth
```

KITTI 08 phase budget sweep using ungated 288D GAT checkpoint:

```text
results/phase_budget_08_no_interdiff288_r4_b8_n800.json: best 08 R@1 = 0.84255
results/phase_budget_08_no_interdiff288_r8_b8_n800.json: best 08 R@1 = 0.84255
```

Interpretation: KITTI 08 is effectively at SC++ level but not cleanly above it
yet. Do not claim "beats SC++ on KITTI 08" unless a subsequent run exceeds
0.843 with the same evaluation protocol.

## Files To Inspect First

```text
src/encoding/cross_spectrum.py
src/encoding/spectral_encoder.py
src/gnn/model.py
train_multi_dataset.py
scripts/evaluate_kitti_checkpoint.py
scripts/run_kitti_operating_point.py
tests/test_cross_spectrum.py
tests/test_gnn_gate.py
```

## Re-run Fast Encoder Ablations

```bash
for preset in no_interdiff mean_diff rows12_full cross4_no_interdiff cross8_no_interdiff; do
  CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_kitti_checkpoint.py \
    --config configs/training_kitti_only.yaml \
    --encoder-preset "$preset" \
    --skip-checkpoint \
    --sequences 00 05 08 \
    --device cuda \
    --cache-dir "data/preprocessed_kitti_encoder_ablation_${preset}" \
    --enable-bev-layout \
    --bev-row-pool 16 \
    --enable-phase-sketch \
    --phase-sketch-only \
    --phase-range-freqs 4 \
    --phase-bev-freqs 4 \
    --n-coarse 400 \
    --output "results/encoder_ablation_${preset}_phase256_n400.json"
done
```

## Re-train Gated 288D GAT

```bash
CUDA_VISIBLE_DEVICES=3 python3 train_multi_dataset.py \
  --config configs/training_kitti_only.yaml \
  --encoder-preset no_interdiff \
  --use-gated-context \
  --gate-initial-alpha 0.0625 \
  --checkpoint-dir results/train_no_interdiff_288_gate00625_seedX \
  --seed X
```

## Evaluate Gated 288D GAT With Phase

```bash
CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_kitti_checkpoint.py \
  --config configs/training_kitti_only.yaml \
  --encoder-preset no_interdiff \
  --use-gated-context \
  --gate-initial-alpha 0.0625 \
  --checkpoint results/train_no_interdiff_288_gate00625_seedX/best_model.pth \
  --sequences 00 05 08 \
  --device cuda \
  --cache-dir data/preprocessed_kitti_encoder_ablation_no_interdiff \
  --enable-bev-layout \
  --bev-row-pool 16 \
  --enable-phase-sketch \
  --phase-sketch-only \
  --phase-range-freqs 4 \
  --phase-bev-freqs 8 \
  --n-coarse 800 \
  --ctx-weights 0 0.0625 0.125 0.25 0.5 1.0 \
  --phase-sketch-bev-weights 0.5 1.0 2.0 4.0 8.0 \
  --phase-sketch-range-weights 0.0 0.125 0.25 0.5 1.0 \
  --output results/checkpoint_eval_nointerdiff288_gate00625_phase384_n800_seedX.json
```

## Immediate Next Experiments

1. Run at least 3 seeds for `no_interdiff + gated context`.
2. Try `gate_initial_alpha=0.03` and `0.125`; current best post-hoc context weight often lands below or around `0.5` after learned gating.
3. Add phase-aware edge features next:
   - `phase_consistency`
   - `sin(best_shift)`
   - `cos(best_shift)`
   - `phase_residual`
4. Do not put `5.7K layout rerank` in the main method table.
5. If using phase sketch in the main method, report total descriptor state:
   - `288D raw + 128D gated context + 384D phase = 800D`
   - `288D raw + 128D gated context + 512D phase = 928D`
   - SC++ reference state is about `20D ring key + 1200D layout = 1220D`.

## Validation Commands

Local syntax:

```bash
python3 -m py_compile \
  src/encoding/cross_spectrum.py \
  src/encoding/spectral_encoder.py \
  src/gnn/model.py \
  train_multi_dataset.py \
  scripts/evaluate_kitti_checkpoint.py \
  scripts/run_kitti_operating_point.py
```

Local tests:

```bash
PYTHONPATH=src pytest -q tests/test_cross_spectrum.py tests/test_gnn_gate.py
```

`tests/test_gnn_gate.py` skips if `torch_geometric` is unavailable.
