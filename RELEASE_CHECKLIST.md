# NSD 800D Release Checklist

Use this checklist before handing the repository to a student or creating a
public archive.

## Required Checks

```bash
bash scripts/verify_release_smoke.sh
```

Optional test run, if `torch_geometric` is installed:

```bash
RUN_TESTS=1 bash scripts/verify_release_smoke.sh
```

## Required Reading Order

1. `README.md`
2. `RELEASE_800D.md`
3. `DATA.md`
4. `artifacts/MANIFEST.md`
5. `EXPERIMENT_HANDOFF.md`

## Paper-Main Policy

- Main 800D state is `288D + 128D + 384D`.
- Use `--encoder-preset no_interdiff` for all paper-main rows.
- Use fixed-alpha GAT: `--use-gated-context --gate-initial-alpha 0.0625`.
- Do not use `sensor_gate`, `dual_stream`, `physics3`, or
  `run_retrain_combine_eval.sh` for paper-main numbers.

## Bundle

Create a clean source archive without data, logs, results, checkpoints, or
handoff snapshots:

```bash
bash scripts/make_release_bundle.sh
```

The release archive is code-only by design. Do not include `data/`, `results/`,
checkpoints, logs, or `_handoff` snapshots in the public bundle.
