# Config Groups

Most YAML files are historical experiment configs. Use this grouping to avoid
mixing paper-main and appendix ablation settings.

## Paper-Main Retrieval Key

Use these with command-line overrides:

```bash
--encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625
```

For a portable training run, also pass `--data-root "$NSD_DATA_ROOT"` (or set
`NSD_DATA_ROOT`; it defaults to `data`). The paper-main config intentionally
retains its historical server paths for auditability; the training entry point
overrides them before any loader is constructed.

| Config | Use |
| --- | --- |
| `training_multi_dataset.yaml` | Four-sensor NSD 800D (applied identically to KITTI/NCLT/HeLiPR/MulRan) |

The raw YAML defaults still describe the older 544D encoder. The override
`--encoder-preset no_interdiff` is mandatory for the reported 288D magnitude
key.

## Place-Disjoint Retrain Protocol (2026-08-03 revision work)

Same architecture and overrides as paper-main; only the split and cache dir
change. Test places are held out from training and all selection (see the
config headers for the split rationale).

| Config | Use |
| --- | --- |
| `training_place_disjoint.yaml` | primary: three-way place-disjoint split, NCLT in training |
| `training_place_disjoint_no_nclt.yaml` | ablation: NCLT excluded from train/val, test-only |

## Paper Appendix / Diagnostics

| Config | Use |
| --- | --- |
| `training_multi_dataset_sensor_gat_absdiff.yaml` | sensor-aware GAT + physics3 ablation |
| `training_kitti_nclt_sensor_gat_absdiff.yaml` | server-available subset for the same ablation |
| `training_server_available_sensor_gat_absdiff.yaml` | reduced server subset |
| `training_kitti_nclt_compact_fast.yaml` | KITTI+NCLT held-out controls |

## Historical Phase/GAT Experiments

These are retained for negative ablations and should not be used as paper-main
configs:

```text
training_kitti_learned_phase_fast.yaml
training_kitti_learned_phase_cross_fast.yaml
training_kitti_phase_alignment_aux_gat_fast.yaml
training_kitti_phase_alignment_gat_fast.yaml
training_kitti_phase_bispectrum_dualstream_fast.yaml
training_kitti_phase_edge_gat_fast.yaml
training_kitti_phase_edge_value_gat_fast.yaml
training_kitti_phase_power_dualstream_fast.yaml
training_kitti_nclt_phase_identity_fast.yaml
```
