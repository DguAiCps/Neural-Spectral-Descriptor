# Config Groups

Most YAML files are historical experiment configs. Use this grouping to avoid
mixing paper-main and appendix ablation settings.

## Paper-Main Retrieval Key

Use these with command-line overrides:

```bash
--encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625
```

| Config | Use |
| --- | --- |
| `training_multi_dataset.yaml` | Four-sensor closed-form 800D headline |
| `training_kitti_only.yaml` | KITTI learned-residual 800D sub-row |

The raw YAML defaults still describe the older 544D encoder. The override
`--encoder-preset no_interdiff` is mandatory for the reported 288D magnitude
key.

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

