# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

NeurIPS 2026 release code for the Neural Spectral Descriptor (NSD), a LiDAR place-recognition descriptor. The frozen reported state is **800D**: `288D magnitude key + 128D gated-GAT context + 384D phase-alignment sketch`. The repository is code-only and ships no datasets or checkpoints.

Authoritative docs (read in this order before touching anything):
1. [README.md](README.md) — entry summary and quick start.
2. [RELEASE_800D.md](RELEASE_800D.md) — student-facing runbook with the canonical commands.
3. [DATA.md](DATA.md) — dataset roots, sensor metadata, NCLT eval protocol.
4. [artifacts/MANIFEST.md](artifacts/MANIFEST.md) — checkpoint restore paths.
5. [EXPERIMENT_HANDOFF.md](EXPERIMENT_HANDOFF.md) — full experiment history and interpretation.
6. [configs/README.md](configs/README.md) — which YAML belongs to paper-main vs. appendix.

## Paper-Main vs. Appendix — DO NOT CONFLATE

Two reported 800D rows share the same retrieval key but differ in phase sketch / reranker:

| Row | Phase sketch | Reranker | Scope |
| --- | --- | --- | --- |
| Four-sensor headline (Table 1) | cylindrical+BEV 384D (`range 16×4×2 + BEV 16×8×2`) | closed-form cyclic-shift cosine | KITTI/NCLT/HeLiPR/MulRan |
| KITTI learned-residual sub-row | BEV-only max-height 384D (`16×12×2`) | zero-init residual MLP | KITTI 00/05/08 only |

The following are **appendix ablations only** — never use them for paper-main numbers:
- `gnn.sensor_gate.enabled=true`
- `gnn.dual_stream.enabled=true`
- `bev.height_encoding=physics3`
- [scripts/run_retrain_combine_eval.sh](scripts/run_retrain_combine_eval.sh) (annotated `ABLATION-ONLY` in its header)

NCLT zero-shot transfer of the learned residual is a paper-acknowledged limitation. Do not claim cross-sensor learned-residual transfer.

## Mandatory CLI Overrides

The YAML defaults still describe the older 544D encoder policy. Every paper-main training/eval command needs:

```bash
--encoder-preset no_interdiff                  # 288D magnitude key
--use-gated-context --gate-initial-alpha 0.0625  # fixed-alpha GAT
```

Without these overrides you get the historical 544D/672D baseline, which is in the repo for ablation context but is **not** the reported paper-main path.

## Common Commands

Install:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

Release smoke check (file-presence + py_compile, plus optional pytest):
```bash
bash scripts/verify_release_smoke.sh
RUN_TESTS=1 bash scripts/verify_release_smoke.sh   # requires torch_geometric
```

Local syntax + minimal test sanity:
```bash
python3 -m py_compile \
  train_multi_dataset.py \
  scripts/evaluate_kitti_checkpoint.py \
  scripts/evaluate_nclt_checkpoint.py \
  scripts/evaluate_nclt_learned_reranker.py \
  scripts/train_kitti_learned_reranker.py \
  src/encoding/bev_image.py src/encoding/spectral_encoder.py \
  src/gnn/model.py src/gnn/learned_reranker.py \
  src/keyframe/graph_manager.py

PYTHONPATH=src pytest -q \
  tests/test_cross_spectrum.py \
  tests/test_gnn_gate.py \
  tests/test_phase_alignment.py \
  tests/test_phase_coherence.py
```

Run a single test:
```bash
PYTHONPATH=src pytest -q tests/test_phase_alignment.py::test_name -v
```

Paper-main reproduction (shell wrappers around the Python entry points):
```bash
bash scripts/run_paper_kitti_closed_form.sh       # KITTI BEV-only closed-form control
bash scripts/run_paper_kitti_residual.sh          # KITTI learned residual reranker
bash scripts/run_paper_nclt_physics3_control.sh   # NCLT max-BEV vs physics3 (control pair)
```

Restore handoff checkpoint into the expected path:
```bash
bash scripts/restore_release_artifacts.sh
```

Build a code-only release tarball (excludes `data/`, `results/`, `_handoff/`, checkpoints):
```bash
bash scripts/make_release_bundle.sh
```

## Architecture (Big Picture)

The pipeline ([src/pipeline.py](src/pipeline.py) wraps the offline-training / online-inference flow; the production training entry is the top-level [train_multi_dataset.py](train_multi_dataset.py)):

```
LiDAR scan ──► keyframe selector ──► spectral encoder ──► temporal graph ──► GNN ──► retrieval key
                                                                                      │
                                                            phase-alignment sketch ──┴──► two-stage retrieval ──► loop closures (g2o)
```

Module roles (all live under [src/](src/), imported as top-level packages because `PYTHONPATH=src` / `sys.path.insert(0, 'src')`):

- [src/encoding/](src/encoding/) — point cloud → descriptor. Key files: `spectral_encoder.py` (range-image FFT magnitude → 288D `no_interdiff` key), `bev_image.py` (BEV layout + height encodings: `max`, `iris`, `physics3`), `phase_alignment.py` / `phase_coherence.py` / `cross_spectrum.py` (384D phase sketch components), `spectral_policy.py` (encoder preset selection logic mirrored in `train_multi_dataset.apply_encoder_preset`).
- [src/gnn/](src/gnn/) — graph reasoning over keyframes. `model.py` defines the DiffAttnConv-based encoder; `phase_diff_conv.py` is the attention layer; gating is controlled by `--use-gated-context --gate-initial-alpha` (adds the 128D context to the 288D key). `learned_reranker.py` is the zero-init residual MLP for the KITTI sub-row. `trainer.py` / `triplet_miner.py` drive training.
- [src/keyframe/](src/keyframe/) — `selector.py` picks keyframes by pose/overlap/time; `graph_manager.py` builds the temporal graph (`temporal_neighbors` edges, capped at `max_active_nodes`).
- [src/retrieval/](src/retrieval/) — `two_stage_retrieval.py` does coarse (FAISS over the 416D key) + fine (geometric verification). `wasserstein.py` and `geometric_verification.py` support fine-stage scoring.
- [src/utils/cyclic_shift_distance.py](src/utils/cyclic_shift_distance.py) — the closed-form cyclic-shift cosine reranker used by the four-sensor row.

The "800D stored state" is conceptual: 288D magnitude + 128D GAT context (416D retrieval key written to the database) **plus** 384D phase sketch computed on-the-fly during rerank. The retrieval key alone is 416D; phase contributes only at the cyclic-shift / residual MLP stage.

## Config Layout

YAMLs under [configs/](configs/) are grouped in [configs/README.md](configs/README.md). For paper-main work you almost always start from one of two configs:

- [configs/training_multi_dataset.yaml](configs/training_multi_dataset.yaml) — four-sensor headline (KITTI+NCLT+HeLiPR+MulRan).
- [configs/training_kitti_only.yaml](configs/training_kitti_only.yaml) — KITTI learned-residual sub-row.

Every other config in the directory is either an appendix ablation (sensor_gate / dual_stream / physics3 / phase-edge variants) or a historical negative result. Don't pick one by filename — match it against the table in `configs/README.md`.

## Repo State Caveats

- **`src/data/` is not in this checkout.** [src/pipeline.py](src/pipeline.py) and [train_multi_dataset.py](train_multi_dataset.py) import from `data.kitti_loader`, `data.nclt_loader`, `data.helipr_loader`, `data.mulran_loader`, `data.multi_dataset_loader`, and `data.pose_utils`. `scripts/verify_release_smoke.sh` also requires them. If the smoke check fails on missing `src/data/*.py`, the data-loader package needs to be restored from the upstream branch — don't try to rewrite or stub the imports.
- The release archive is code-only; `data/`, `results/`, `_handoff/`, and `*.pth` are intentionally not present and not part of the public bundle.

## When Editing Configs or Scripts

`scripts/verify_release_smoke.sh` greps for stale wording (e.g. `0.9496`, `older README`, `current main upgrade`, `GAT learns phase`, `NSD full`, `full NSD`, `zero-shot performance`) in README/RELEASE_800D/DATA/EXPERIMENT_HANDOFF/configs/MANIFEST. Avoid reintroducing those phrases when editing docs, or the smoke check will fail.
