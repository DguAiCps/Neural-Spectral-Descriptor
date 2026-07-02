# Dataset Layout

This repository does not redistribute KITTI, NCLT, HeLiPR, or MulRan. The
release archive is code-only; point the configs or runner environment variables
at local dataset copies.

## Default Server Paths

| Dataset | Default root | Sensor key | Beam count | Validation split |
| --- | --- | --- | ---: | --- |
| KITTI | `/rise/RISE1/workspace/data/kitti/dataset` | `kitti` | 64 | `00`, `05`, `08` |
| NCLT | `/rise/RISE1/workspace/data/nclt` | `nclt` | 32 | `2012-01-08`, `2013-01-10` |
| HeLiPR | `/workspace/data/helipr` | `helipr` | 16 | `Town01` |
| MulRan | `/workspace/data/mulran` | `mulran` | 64 | `DCC03`, `KAIST03`, `Riverside03` |

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
