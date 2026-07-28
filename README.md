# Neural Spectral Descriptor 800D Release

This branch contains the NSD paper code plus appendix ablations. The
current source of truth is:

1. `CLAUDE.md` — current paper-main spec (NSD + NSD-H operating points).
2. `docs/paper/body_v2.0/aaai2027_body.tex` — paper text and reported numbers.
3. `EXPERIMENT_HANDOFF.md` — full experiment history and interpretation.
4. `RELEASE_800D.md` — base 800D runbook for students continuing the experiments.
5. `RELEASE_CHECKLIST.md` — final checks before sharing the code.

The old 544D/672D encoder-bandwidth baseline is still present in configs and
results for ablation context, but it is not the current paper-main path.

## Current State (two operating points)

The current paper (AAAI-27) reports two operating points that share one architecture
and differ only in the auxiliary projection of the retrieval-time complex spectrum:

```text
416D retrieval key = 288D no_interdiff magnitude key (re-metrized)
                   + 128D fixed-alpha gated DiffAttnConv context
stored per keyframe = 416D key + complex spectrum
  NSD   : + cylindrical 128 + BEV 256           =   800 floats
  NSD-H : + cylindrical 128 + height-polar 480   = 1,024 floats
```

Reported numbers (3-seed, 9 held-out sequences across KITTI/NCLT/HeLiPR/MulRan):

| Operating point | Stored | R̄q | R̄s | σ_cross | R_min |
| --- | --- | --- | --- | --- | --- |
| NSD | 800 | 0.888 | 0.896 | 0.048 | 0.813 |
| NSD-H | 1,024 | 0.887 | 0.914 | 0.082 | 0.770 |

The complex spectrum is used only at retrieval time: its magnitude is a second cosine index, and
its coefficients are cyclically aligned over the candidate union. `physics3` height encoding is an
appendix ablation, not paper-main.

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

NCLT held-out max-BEV / physics3 control:

```bash
bash scripts/run_paper_nclt_physics3_control.sh
```

The four-sensor NSD row uses `configs/training_multi_dataset.yaml` with
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
