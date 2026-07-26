# NSD 800D Release Runbook

Date: 2026-05-10

This file is the student-facing entry point for continuing the NeurIPS 2026
800D NSD experiments. Older configs and archived result files describe the
historical 544D/672D baseline; use this runbook and `EXPERIMENT_HANDOFF.md`
for the current paper state.

## 1. Canonical configuration

There is exactly one deployed NSD architecture; it applies identically to KITTI, NCLT, HeLiPR, and MulRan.

| Retrieval key | Phase sketch | Reranker | Scope |
| --- | --- | --- | --- |
| 288D `no_interdiff` + 128D fixed-alpha GAT = 416D | cylindrical+BEV 384D (`range 16x4x2 + BEV 16x8x2`) | closed-form cyclic-shift cosine (Eq. 6) | KITTI/NCLT/HeLiPR/MulRan |
| same or sensor-aware key | physics3 384D (`48x4x2`) | closed-form | appendix ablation only |

The stored state is 800D for the reported row:

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
  src/encoding/bev_image.py \
  src/encoding/spectral_encoder.py \
  src/gnn/model.py \
  src/keyframe/graph_manager.py

PYTHONPATH=src pytest -q \
  tests/test_cross_spectrum.py \
  tests/test_gnn_gate.py \
  tests/test_phase_alignment.py \
  tests/test_phase_coherence.py
```

Some GNN tests skip on machines without `torch_geometric`.

## 4. Four-sensor NSD row

Use `configs/training_multi_dataset.yaml` plus the same fixed-alpha and
`no_interdiff` overrides. The phase sketch is cylindrical+BEV:

```text
range phase: 16 rows x 4 freqs x 2 = 128D
BEV phase:   16 rows x 8 freqs x 2 = 256D
```

Table 1 row:

```text
KITTI 00/05/08:        0.986 / 0.963 / 0.877
NCLT 12-01/13-01:      0.487 / 0.221
HeLiPR Town01:         0.414
MulRan DCC/KAIST/Riv:  0.751 / 0.998 / 0.863
```

## 5. Appendix-only ablations

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
