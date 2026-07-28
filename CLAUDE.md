# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Release code for the Neural Spectral Descriptor (NSD), a LiDAR place-recognition descriptor. The current paper (AAAI-27, `docs/paper/body_v2.0/`) reports **two operating points that share one architecture**: **NSD** (800-float stored state) and **NSD-H** (1,024-float), differing only in the auxiliary projection of the retrieval-time complex spectrum. Both build a 416D retrieval key (`288D magnitude key + 128D gated context`) and store a complex spectrum used only at retrieval time. The repository is code-only and ships no datasets or checkpoints.

Authoritative docs (read in this order before touching anything):
1. [PAPER_MAIN_B3.md](PAPER_MAIN_B3.md) — **the frozen AAAI-27 paper-main spec** (pipeline, constants, 3-seed numbers, protocol disclosures, repro commands, negative-results ledger). If you are writing or editing the paper, start here.
2. [README.md](README.md) — entry summary and quick start.
3. [RELEASE_800D.md](RELEASE_800D.md) — student-facing runbook with the canonical commands (describes the pre-freeze base configuration).
4. [DATA.md](DATA.md) — dataset roots, sensor metadata, NCLT eval protocol.
5. [artifacts/MANIFEST.md](artifacts/MANIFEST.md) — checkpoint/artifact restore paths.
6. [EXPERIMENT_HANDOFF.md](EXPERIMENT_HANDOFF.md) — full experiment history and interpretation.
7. [configs/README.md](configs/README.md) — which YAML belongs to paper-main vs. appendix.

## Paper-Main Configuration — uniform across all datasets

**Current paper-main (branch `feat/pbev-stored-spectrum`; supersedes the 2026-07-09 `feat/remetrize-twopass` freeze).** Paper main is now **two operating points** on the same three frozen refiner checkpoints + fixed key re-metrization (A) + two-pass classifier-selected edges (B3), extended with a retrieval-time reuse of the stored complex spectrum (a second magnitude index + cyclic alignment). Zero retraining.

- **NSD** — 800-float state, auxiliary BEV spectrum. 3-seed: R̄q 0.888 / R̄s 0.896 / σ_cross 0.048 / R_min 0.813.
- **NSD-H** — 1,024-float state, auxiliary height-polar spectrum. 3-seed: R̄q 0.887 / R̄s 0.914 / σ_cross 0.082 / R_min 0.770.

Numbers verified 2026-07-28 against committed `results/height_sota/{pbev,hspec}_branch_fusion*.json`. The authoritative spec is the paper `docs/paper/body_v2.0/aaai2027_body.tex`. **[PAPER_MAIN_B3.md](PAPER_MAIN_B3.md) and [RELEASE_800D.md](RELEASE_800D.md) predate this and describe only the earlier single-800D freeze (R̄q 0.785/0.796, no NSD-H)** — their key/edge-stage pipeline detail is still valid, but do not trust their headline numbers.

One deployed architecture, two storage/accuracy operating points; it applies identically to KITTI, NCLT, HeLiPR, and MulRan.

| Stage | Spec |
| --- | --- |
| retrieval key | 416D `[normalize(ω⊙d); g·ĉ]`, ω = `artifacts/key_remetrize_r.npy` |
| edge selection | two-pass: top-30 candidates → `artifacts/edge_classifier.pt` → top-10 edges |
| stored complex spectrum | shared cylindrical `16×4×2`=128; NSD adds BEV `16×8×2`=256 (→ 800), NSD-H adds height-polar `20×12×2`=480 (→ 1,024) |
| retrieval-time matching | two top-100 cosine indexes (key + variant magnitude) → union → per-query min–max 4-channel fusion + cyclic alignment; tuples NSD `(0.5,1,0,0.5)`, NSD-H `(1,1,1,2)` |

Paper-main eval entry points: [scripts/_pbev_branch_fusion.py](scripts/_pbev_branch_fusion.py) (NSD), [scripts/_hspec_branch_fusion.py](scripts/_hspec_branch_fusion.py) (NSD-H; deployed variant is `hspec12`, not the 784-float `hspec6` control), [scripts/_final_fix8_agg.py](scripts/_final_fix8_agg.py) (fusion-tuple selection + 3-seed aggregation). Upstream B3 stages: [scripts/run_b3_rerank.py](scripts/run_b3_rerank.py), [scripts/build_and_train_edge_classifier.py](scripts/build_and_train_edge_classifier.py) (classifier rebuild).

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

The stored state is conceptual: 288D magnitude + 128D gated context (416D retrieval key written to the database) **plus** the complex spectrum (stored and used only at retrieval time). NSD stores a 384D spectrum (cyl 128 + BEV 256) → 800 floats; NSD-H stores 608D (cyl 128 + height-polar 480) → 1,024 floats. The retrieval key alone is 416D; the spectrum contributes at retrieval time as a second magnitude index and via cyclic alignment.

## Config Layout

YAMLs under [configs/](configs/) are grouped in [configs/README.md](configs/README.md). For paper-main work you start from a single config:

- [configs/training_multi_dataset.yaml](configs/training_multi_dataset.yaml) — the four-sensor architecture (KITTI+NCLT+HeLiPR+MulRan), which is the only reported deployed configuration.

Every other config in the directory is either an appendix ablation (sensor_gate / dual_stream / physics3 / phase-edge variants) or a historical negative result. Don't pick one by filename — match it against the table in `configs/README.md`.

## Repo State Caveats

- **`src/data/` is not in this checkout.** [src/pipeline.py](src/pipeline.py) and [train_multi_dataset.py](train_multi_dataset.py) import from `data.kitti_loader`, `data.nclt_loader`, `data.helipr_loader`, `data.mulran_loader`, `data.multi_dataset_loader`, and `data.pose_utils`. `scripts/verify_release_smoke.sh` also requires them. If the smoke check fails on missing `src/data/*.py`, the data-loader package needs to be restored from the upstream branch — don't try to rewrite or stub the imports.
- The release archive is code-only; `data/`, `results/`, `_handoff/`, and `*.pth` are intentionally not present and not part of the public bundle.

## When Editing Configs or Scripts

`scripts/verify_release_smoke.sh` greps for stale wording (e.g. `0.9496`, `older README`, `current main upgrade`, `GAT learns phase`, `NSD full`, `full NSD`, `zero-shot performance`) in README/RELEASE_800D/DATA/EXPERIMENT_HANDOFF/configs/MANIFEST. Avoid reintroducing those phrases when editing docs, or the smoke check will fail.
