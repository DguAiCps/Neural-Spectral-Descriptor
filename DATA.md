# Dataset Layout

This repository does not redistribute KITTI, NCLT, HeLiPR, or MulRan. The
release archive is code-only; point the configs or runner environment variables
at local dataset copies.

## Portable Dataset Root

Set `NSD_DATA_ROOT` to a directory containing the four dataset folders. Paper
scripts default to `./data` when it is unset; training also accepts an explicit
`--data-root /path/to/datasets`, which overrides the historical roots embedded
in old YAML files.

```bash
export NSD_DATA_ROOT=/absolute/path/to/NSD_datasets
```

| Dataset | Relative path below data root | Sensor key | Beam count | Validation split |
| --- | --- | --- | ---: | --- |
| KITTI | `kitti/dataset` | `kitti` | 64 | `00`, `05`, `08` |
| NCLT | `nclt` | `nclt` | 32 | `2012-01-08`, `2013-01-10` |
| HeLiPR | `helipr` | `helipr` | 16 | `Town01` |
| MulRan | `mulran` | `mulran` | 64 | `DCC03`, `KAIST03`, `Riverside03` |

## Sensor Elevation Ranges

These are fixed in the configs and used for calibrated range-image pooling:

```yaml
kitti:  [-24.8, 2.0]
nclt:   [-30.67, 10.67]
helipr: [-15.0, 15.0]
mulran: [-16.6, 16.6]
```

## NCLT Evaluation Protocol

The NCLT checkpoint evaluator uses:

```text
scan_stride = 5
skip_frames = 6
distance_threshold = 5 m
sensor_key = nclt
elevation_range = [-30.67, 10.67]
```

Do not compare NCLT numbers from a KITTI-only zero-shot residual run with the
KITTI+NCLT held-out physics3 control. They answer different questions.
