# NSD 800D Release Runbook

Date: 2026-05-10

This file is the student-facing entry point for continuing the NeurIPS 2026
800D NSD experiments. Older configs and archived result files describe the
historical 544D/672D baseline; use this runbook and `EXPERIMENT_HANDOFF.md`
for the current paper state.

## 1. Canonical split

Do not collapse these rows into one result.

| Row | Retrieval key | Phase sketch | Reranker | Scope |
| --- | --- | --- | --- | --- |
| Four-sensor headline | 288D `no_interdiff` + 128D fixed-alpha GAT = 416D | cylindrical+BEV 384D (`range 16x4x2 + BEV 16x8x2`) | closed-form cyclic-shift cosine | KITTI/NCLT/HeLiPR/MulRan |
| KITTI learned-residual sub-row | same 416D key | BEV-only max-height 384D (`16x12x2`) | zero-init residual MLP | KITTI 00/05/08 only |
| Ablation only | same or sensor-aware key | physics3 384D (`48x4x2`) | closed-form or residual | appendix only |

The stored state is always 800D for the reported rows:

```text
288D magnitude key + 128D GAT context + 384D phase-alignment sketch = 800D
```

## 2. Critical flags

The YAML defaults still contain the older 544D encoder policy. The 800D paper
rows require this override in every training/evaluation command:

```bash
--encoder-preset no_interdiff
```

The paper rows also require fixed-alpha GAT:

```bash
--use-gated-context --gate-initial-alpha 0.0625
```

Do not enable these for paper-main numbers:

```text
gnn.sensor_gate.enabled=true
gnn.dual_stream.enabled=true
bev.height_encoding=physics3
scripts/run_retrain_combine_eval.sh
```

Those are appendix ablations only.

## 3. Local sanity check

Run from repo root:

```bash
python3 -m py_compile \
  train_multi_dataset.py \
  scripts/evaluate_kitti_checkpoint.py \
  scripts/evaluate_nclt_checkpoint.py \
  scripts/evaluate_nclt_learned_reranker.py \
  scripts/train_kitti_learned_reranker.py \
  src/encoding/bev_image.py \
  src/encoding/spectral_encoder.py \
  src/gnn/model.py \
  src/gnn/learned_reranker.py \
  src/keyframe/graph_manager.py

PYTHONPATH=src pytest -q \
  tests/test_cross_spectrum.py \
  tests/test_gnn_gate.py \
  tests/test_phase_alignment.py \
  tests/test_phase_coherence.py
```

Some GNN tests skip on machines without `torch_geometric`.

## 4. Reproduce KITTI closed-form max-BEV control

This is the BEV-only control used in `tab:reranker_ablation`, not the
four-sensor headline row.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_kitti_checkpoint.py \
  --config configs/training_kitti_only.yaml \
  --encoder-preset no_interdiff \
  --use-gated-context --gate-initial-alpha 0.0625 \
  --checkpoint results/train_no_interdiff_288_gate00625_seed1/best_model.pth \
  --sequences 00 05 08 \
  --device cuda \
  --cache-dir data/preprocessed_kitti_encoder_ablation_no_interdiff \
  --enable-bev-layout --bev-row-pool 16 --bev-height-encoding max \
  --enable-phase-sketch --phase-sketch-only \
  --phase-range-freqs 0 --phase-bev-freqs 12 \
  --phase-sketch-range-weights 0.0 \
  --phase-sketch-bev-weights 0.5 1.0 2.0 4.0 8.0 \
  --phase-rerank-mode sketch_fft \
  --n-coarse 800 \
  --output results/kitti_checkpoint_eval_nointerdiff288_gate00625_bevonly384_sketch_fft_n800.json
```

Expected KITTI control:

```text
00 0.9794 / 05 0.9523 / 08 0.8468
```

## 5. Train/evaluate KITTI learned residual

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_kitti_learned_reranker.py \
  --config configs/training_kitti_only.yaml \
  --encoder-preset no_interdiff \
  --use-gated-context --gate-initial-alpha 0.0625 \
  --checkpoint results/train_no_interdiff_288_gate00625_seed1/best_model.pth \
  --train-sequences 01 02 06 07 \
  --val-sequences 00 05 08 \
  --cache-dir data/preprocessed_kitti_encoder_ablation_no_interdiff \
  --bev-cache-dir data/preprocessed_kitti_bev_layout \
  --bev-height-encoding max --bev-row-pool 16 --bev-freqs 12 \
  --max-candidates 800 \
  --epochs 20 --seed 1 \
  --output results/kitti_learned_reranker_bev384_residual.json \
  --checkpoint-out results/kitti_learned_reranker_bev384_residual.pth
```

Reported two-seed mean:

```text
00 0.9691 / 05 0.9589 / 08 0.8617
```

This row is KITTI-only. Do not claim cross-sensor learned-residual transfer.

## 6. Four-sensor headline

Use `configs/training_multi_dataset.yaml` plus the same fixed-alpha and
`no_interdiff` overrides. The headline phase sketch is cylindrical+BEV:

```text
range phase: 16 rows x 4 freqs x 2 = 128D
BEV phase:   16 rows x 8 freqs x 2 = 256D
```

Headline Table 1 row:

```text
KITTI 00/05/08:        0.986 / 0.963 / 0.877
NCLT 12-01/13-01:      0.487 / 0.221
HeLiPR Town01:         0.414
MulRan DCC/KAIST/Riv:  0.751 / 0.998 / 0.863
```

## 7. Appendix-only ablations

Use `scripts/run_retrain_combine_eval.sh` only for the sensor-aware GAT +
physics3 appendix chain. It is intentionally not a paper-main reproduction
script.

Current interpretation:

```text
sensor-aware GAT: stabilizes alpha but no retrieval gain
physics3 KITTI:   00 0.9794 / 05 0.9655 / 08 0.7787
physics3 NCLT:    macro 0.4619 vs max-BEV 0.3218 with KITTI+NCLT checkpoint
```

Physics3 is a sparse-sensor direction, not a replacement for the KITTI
reverse-loop main row.
