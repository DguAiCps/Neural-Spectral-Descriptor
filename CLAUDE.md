# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

NeurIPS 2026 release code for the Neural Spectral Descriptor (NSD), a LiDAR place-recognition descriptor. The frozen reported state is **800D**: `288D magnitude key + 128D gated-GAT context + 384D phase-alignment sketch`. The repository is code-only and ships no datasets or checkpoints.

Authoritative docs (read in this order before touching anything):
1. [PAPER_MAIN_B3.md](PAPER_MAIN_B3.md) — **the frozen AAAI-27 paper-main spec** (pipeline, constants, 3-seed numbers, protocol disclosures, repro commands, negative-results ledger). If you are writing or editing the paper, start here.
2. [README.md](README.md) — entry summary and quick start.
3. [RELEASE_800D.md](RELEASE_800D.md) — student-facing runbook with the canonical commands (describes the pre-freeze base configuration).
4. [DATA.md](DATA.md) — dataset roots, sensor metadata, NCLT eval protocol.
5. [artifacts/MANIFEST.md](artifacts/MANIFEST.md) — checkpoint/artifact restore paths.
6. [EXPERIMENT_HANDOFF.md](EXPERIMENT_HANDOFF.md) — full experiment history and interpretation.
7. [configs/README.md](configs/README.md) — which YAML belongs to paper-main vs. appendix.

## Paper-Main Configuration — uniform across all datasets

**Frozen 2026-07-09 (branch `feat/remetrize-twopass`)**: paper main = the three frozen 800D checkpoints + fixed key re-metrization (A) + classifier-selected similarity edges (B3) + raw-scale phase rerank. 3-seed full-pipeline result: R̄q 0.796 / R̄s 0.767 / σ_cross 0.201 / R_min 0.467. Zero retraining; storage stays 800D. Full spec in [PAPER_MAIN_B3.md](PAPER_MAIN_B3.md).

There is exactly one deployed NSD architecture; it applies identically to KITTI, NCLT, HeLiPR, and MulRan.

| Key metric | Edge selection | Phase sketch | Reranker | Scope |
| --- | --- | --- | --- | --- |
| 416D `[normalize(r⊙d); g·ĉ]`, r = `artifacts/key_remetrize_r.npy` | two-pass: top-30 candidates → `artifacts/edge_classifier.pt` → top-10 edges | cylindrical+BEV 384D (`range 16×4×2 + BEV 16×8×2`) | closed-form cyclic-shift cosine (Eq. 6), `fusion_norm="raw"` | KITTI/NCLT/HeLiPR/MulRan (all four sensors) |

Paper-main eval entry points: [scripts/run_b3_rerank.py](scripts/run_b3_rerank.py) (main numbers), [scripts/run_remetrize_twopass_eval.py](scripts/run_remetrize_twopass_eval.py) + [scripts/summarize_remetrize_twopass.py](scripts/summarize_remetrize_twopass.py) (ablation ladder), [scripts/build_and_train_edge_classifier.py](scripts/build_and_train_edge_classifier.py) (classifier rebuild).

The following are **appendix ablations / negative results only** — never use them as the paper-main architecture:
- `gnn.sensor_gate.enabled=true`
- `gnn.dual_stream.enabled=true`
- `bev.height_encoding=physics3`
- [scripts/run_retrain_combine_eval.sh](scripts/run_retrain_combine_eval.sh) (annotated `ABLATION-ONLY` in its header)
- per-query minmax sketch fusion (the pre-freeze rerank default; harms NCLT/HeLiPR)
- `gnn.two_pass` train-time refinement configs (`configs/training_multi_dataset_remetrize_twopass*.yaml`) and the `checkpoints/remetrize_twopass_s*` retrains — ablation evidence only, do not stack with B3
- `gnn.key_remetrize.learnable=true` and the EdgeConfidenceGate/edge-aux path (diverges; see PAPER_MAIN_B3.md §4)

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
  src/encoding/bev_image.py src/encoding/spectral_encoder.py \
  src/gnn/model.py \
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
- [src/gnn/](src/gnn/) — graph reasoning over keyframes. `model.py` defines the DiffAttnConv-based encoder; `phase_diff_conv.py` is the attention layer; gating is controlled by `--use-gated-context --gate-initial-alpha` (adds the 128D context to the 288D key). `trainer.py` / `triplet_miner.py` drive training.
- [src/keyframe/](src/keyframe/) — `selector.py` picks keyframes by pose/overlap/time; `graph_manager.py` builds the temporal graph (`temporal_neighbors` edges, capped at `max_active_nodes`).
- [src/retrieval/](src/retrieval/) — `two_stage_retrieval.py` does coarse (FAISS over the 416D key) + fine (geometric verification). `wasserstein.py` and `geometric_verification.py` support fine-stage scoring.
- [src/utils/cyclic_shift_distance.py](src/utils/cyclic_shift_distance.py) — the closed-form cyclic-shift cosine reranker used by the four-sensor row.

The "800D stored state" is conceptual: 288D magnitude + 128D GAT context (416D retrieval key written to the database) **plus** 384D phase sketch (stored and used during rerank). The retrieval key alone is 416D; phase contributes only at the cyclic-shift reranker stage.

## Config Layout

YAMLs under [configs/](configs/) are grouped in [configs/README.md](configs/README.md). For paper-main work you start from a single config:

- [configs/training_multi_dataset.yaml](configs/training_multi_dataset.yaml) — the four-sensor architecture (KITTI+NCLT+HeLiPR+MulRan), which is the only reported deployed configuration.

Every other config in the directory is either an appendix ablation (sensor_gate / dual_stream / physics3 / phase-edge variants) or a historical negative result. Don't pick one by filename — match it against the table in `configs/README.md`.

## Repo State Caveats

- **`src/data/` is not in this checkout.** [src/pipeline.py](src/pipeline.py) and [train_multi_dataset.py](train_multi_dataset.py) import from `data.kitti_loader`, `data.nclt_loader`, `data.helipr_loader`, `data.mulran_loader`, `data.multi_dataset_loader`, and `data.pose_utils`. `scripts/verify_release_smoke.sh` also requires them. If the smoke check fails on missing `src/data/*.py`, the data-loader package needs to be restored from the upstream branch — don't try to rewrite or stub the imports.
- The release archive is code-only; `data/`, `results/`, `_handoff/`, and `*.pth` are intentionally not present and not part of the public bundle.

## When Editing Configs or Scripts

`scripts/verify_release_smoke.sh` greps for stale wording (e.g. `0.9496`, `older README`, `current main upgrade`, `GAT learns phase`, `NSD full`, `full NSD`, `zero-shot performance`) in README/RELEASE_800D/DATA/EXPERIMENT_HANDOFF/configs/MANIFEST. Avoid reintroducing those phrases when editing docs, or the smoke check will fail.
