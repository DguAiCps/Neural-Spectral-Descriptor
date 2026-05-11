# Artifact Manifest

Large checkpoints and remote result snapshots are not committed to the release
branch. The handoff archive contains the small set needed to resume the paper
experiments.

## Included Handoff Checkpoints

```text
_handoff/nsd_encoder_ablation_checkpoints/no_interdiff_288_seed0_best_model.pth
_handoff/nsd_encoder_ablation_checkpoints/no_interdiff_288_gate00625_seed1_best_model.pth
```

## Restore Expected Paths

Run from repo root:

```bash
bash scripts/restore_release_artifacts.sh
```

The release scripts assume this restored path unless `CHECKPOINT` is set.

## Remote Result Snapshots

```text
_handoff/nsd_encoder_ablation_remote_results/
```

These files are historical diagnostics and should not be used to overwrite the
frozen paper tables without checking `EXPERIMENT_HANDOFF.md`.

## Public Bundle Policy

`scripts/make_release_bundle.sh` excludes `_handoff`, `results`, `data`,
checkpoints, and model weights. Share those separately only when the recipient
has the right dataset/license context.
