# Artifact Manifest

Large checkpoints and remote result snapshots are not committed to the release
branch. The handoff archive contains the small set needed to resume the paper
experiments.

## Committed Paper-Main Artifacts (in this directory)

These two small files ARE tracked in git and are required by the frozen
paper-main pipeline (see `PAPER_MAIN_B3.md`):

```text
artifacts/key_remetrize_r.npy    # 288-float diagonal re-metrization vector r
                                 # (closed-form within-pair whitening, fit on
                                 # train revisit pairs; loaded via
                                 # gnn.key_remetrize.init_path)
artifacts/edge_classifier.pt     # 14-feature edge MLP (14-64-64-1, ~5.2k params)
                                 # + feature mu/sd; selects similarity edges in
                                 # the two-pass eval (scripts/run_b3_*.py)
```

Rebuild commands if either is lost:
`scripts/fit_key_remetrize.py` (r vector) and
`scripts/build_and_train_edge_classifier.py` (classifier; writes
`results/_edge_cls/classifier.pt`, copy to `artifacts/edge_classifier.pt`).

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
