# Neural Spectral Descriptor 800D Release

This branch contains the NSD 800D paper code plus appendix ablations. The
current source of truth is:

1. `RELEASE_800D.md` — commands for students continuing the experiments.
2. `EXPERIMENT_HANDOFF.md` — full experiment history and interpretation.
3. `body_v1.3/neurips_2026.tex` — paper text and reported numbers.
4. `RELEASE_CHECKLIST.md` — final checks before sharing the code.

The old 544D/672D encoder-bandwidth baseline is still present in configs and
results for ablation context, but it is not the current paper-main path.

## Current 800D State

```text
288D no_interdiff magnitude key
+ 128D fixed-alpha gated DiffAttnConv context
+ 384D phase-alignment sketch
= 800D stored state
```

Reported rows:

| Row | Phase sketch | Reranker | Scope |
| --- | --- | --- | --- |
| Four-sensor headline | cylindrical+BEV 384D | closed-form cyclic-shift cosine | 9 validation sequences |
| KITTI learned-residual | BEV-only max-height 384D | zero-init residual MLP | KITTI 00/05/08 |
| Appendix ablation | physics3 384D | closed-form or residual | not paper-main |

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

Run a release smoke check:

```bash
bash scripts/verify_release_smoke.sh
```

Restore the default paper checkpoint path if the `_handoff` archive is present:

```bash
bash scripts/restore_release_artifacts.sh
```

Read the runbook before launching GPU jobs:

```bash
less RELEASE_800D.md
```

Create a clean source bundle:

```bash
bash scripts/make_release_bundle.sh
```

## Paper-Main Commands

KITTI closed-form max-BEV control:

```bash
bash scripts/run_paper_kitti_closed_form.sh
```

KITTI learned residual:

```bash
bash scripts/run_paper_kitti_residual.sh
```

NCLT held-out max-BEV / physics3 control:

```bash
bash scripts/run_paper_nclt_physics3_control.sh
```

The four-sensor headline row uses `configs/training_multi_dataset.yaml` with
`--encoder-preset no_interdiff`, fixed-alpha GAT, and the cylindrical+BEV
closed-form phase sketch. KITTI and NCLT have standalone release runners; the
remaining HeLiPR/MulRan validation path is retained through the multi-dataset
training/validation code and the paper result artifacts.

## Appendix-Only Runner

Do not use this as a paper-main reproduction script:

```bash
bash scripts/run_retrain_combine_eval.sh
```

It reproduces the sensor-aware GAT + physics3 appendix chain.

## Data

Dataset paths, splits, and sensor metadata are documented in `DATA.md`.

## Artifacts

Checkpoint restore instructions are documented in `artifacts/MANIFEST.md`.

## License

GNU General Public License v3.0
