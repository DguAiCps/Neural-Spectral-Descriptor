# NSD — reference implementation (supplementary material)

This folder contains the code of the Neural Spectral Descriptor (NSD) for online LiDAR place recognition, the trained
weights of the three reported runs, and the fixed constants of the evaluated configuration. It is provided for review;
the datasets are public and are not included.

The code covers the full path of the paper, from raw scans to the reported metrics:

| Step | Paper | Code |
|---|---|---|
| Scans, reference poses, LiDAR odometry, keyframes | Sec. 3, Sec. 4.1 | `nsd/data.py`, `nsd/odometry.py`, `nsd/keyframes.py` |
| Spectral encoding: invariant spectral key `d`, complex DFT features `z` | Sec. 3.1 | `nsd/encoding.py` |
| Temporal spectrum refining with two DiffAttnConv layers | Sec. 3.2 | `nsd/model.py` |
| Place indexing, alignment, temporal evidence, acceptance score | Sec. 3.1, 3.3 | `nsd/index.py`, `nsd/matching.py` |
| Training objective and schedule | Sec. 3.4, 4.1 | `nsd/train.py` |
| Evaluation protocol and metrics | Sec. 4.1 | `nsd/metrics.py` |

Command-line entry points are in `tools/`. Fixed constants are in `assets/`:

| File | Content |
|---|---|
| `weights/s4.pth`, `s5.pth`, `s6.pth` | trained DiffAttnConv weights of the three reported runs (training seeds 4, 5, 6) |
| `remetric_weights.npy` | fixed per-coordinate weights of the spectral key index (288) |
| `whitening_matrix.npy` | decorrelation matrices of the acceptance score (8 x 16 x 16) |
| `matching.json` | fusion weights of place selection, decorrelated-bank weight, score calibration |
| `loss_calibration.json` | fixed affine map of the training loss |
| `splits.json`, `test_cities.json` | training (8), validation (9) and test (46) sequences; test sequences by city |

## Requirements

Python >= 3.10, NumPy, PyTorch (CUDA recommended), `kiss-icp==1.3.0` for LiDAR odometry, and `rosbags` only for
extracting MCD. The results below were produced with PyTorch 2.10 on a single NVIDIA GPU.

```bash
pip install numpy torch kiss-icp==1.3.0 rosbags
```

## Data

Download KITTI odometry, KITTI-360, HeLiPR, MulRan, NCLT and MCD from their official sources. The expected layout of
each dataset folder is listed at the top of `nsd/data.py`. MCD is distributed as ROS bags; extract each sequence first:

```bash
python tools/extract_mcd.py --bag MCD/kth_day_06/kth_day_06_os1_64.bag \
    --gt MCD/kth_day_06/gt/kth_day_06/pose_inW.csv --out MCD_extracted/kth_day_06
```

## Evaluation of the released weights

1. Encode every test sequence (`assets/splits.json`). This runs KISS-ICP on all scans, selects keyframes on the
   odometry poses and encodes each keyframe. The output name must be `<dataset>_<sequence>.npz`.

   ```bash
   python tools/encode_sequence.py --dataset kitti --root KITTI --sequence 00 --out encoded/kitti_00.npz
   python tools/encode_sequence.py --dataset mcd --root MCD_extracted --sequence kth_day_06 --out encoded/mcd_kth_day_06.npz
   ```

2. Refine the spectra with each released model.

   ```bash
   python tools/refine.py --inputs encoded --weights assets/weights/s4.pth --out refined_s4
   ```

3. Run indexing, matching and acceptance for every query, then compute the metrics. `match` writes one ledger per
   sequence (query, returned place, score) and the 6-city macro average; `summarize` averages the three runs.

   ```bash
   python tools/evaluate.py match --inputs refined_s4 --out run_s4
   python tools/evaluate.py summarize run_s4 run_s5 run_s6
   ```

Expected result (Table 1, NSD row; 6-city macro average, mean +- SD over the three runs):

| AP | max F1 | R@100P | revisit R@1 |
|---|---|---|---|
| 89.88 +- 0.20 | 90.39 +- 0.18 | 72.17 +- 0.56 | 91.74 +- 0.19 |

## Cost measurement

`tools/measure_cost.py` measures the cost columns of Table 1 for NSD. Keyframes arrive one at a time; each arrival
encodes the scan, updates the map state, and decides the query 15 keyframes earlier by exact search over the map.
It needs MulRan Sejong01 encoded as above (about 20 minutes, mostly LiDAR odometry).

```bash
python tools/encode_sequence.py --dataset mulran --root MulRan --sequence Sejong01 --out encoded/mulran_Sejong01.npz
python tools/measure_cost.py --dataset mulran --root MulRan --sequence Sejong01 \
    --encoded encoded/mulran_Sejong01.npz --weights assets/weights/s4.pth --n-map 10000
```

The tool times 96 arrivals, repeated three times, with 10,000 candidates in the map at the first query, on one CPU
thread. It then checks the returned places and scores against the full-sequence computation of `tools/refine.py`
and `tools/evaluate.py`. Note that the measured times depend on the hardware and on the library versions.

## Training

The model is trained from random initialisation on the eight training sequences (`assets/splits.json`). Training
sequences are encoded on the dataset reference poses rather than on odometry (`--reference-poses`); for KITTI this
also stores the LiDAR-to-camera calibration, which the model uses for the heading of the camera poses.

```bash
for s in 01 02 05 06; do
  python tools/encode_sequence.py --reference-poses --dataset kitti --root KITTI --sequence $s --out train/kitti_$s.npz
done
for s in Bridge01 Bridge02 Bridge03 Bridge04; do
  python tools/encode_sequence.py --reference-poses --dataset helipr --root HeLiPR --sequence $s --out train/helipr_$s.npz
done
python tools/train.py --inputs train --seed 4 --out weights_s4.pth
```

`tools/train.py` first fixes one candidate per training query (the place-selection winner on the unrefined
spectra) and caches these pairs next to the inputs; this takes a few minutes per long sequence. The 512 updates then
take well under a minute on a single GPU. The reported runs use seeds 4, 5 and 6.

## Reproducibility notes

* The method runs online: the decision for query `q = t - 15` is made when keyframe `t` arrives, using only keyframes
  up to `t`. `tools/refine.py` computes the refined spectra of a whole sequence at once; each refined spectrum depends
  only on its +-10 keyframe window, so this equals the online computation.
* Candidate search is exact (brute force); no approximate index is used.
* Spectral features are computed in float32 NumPy on the CPU.
* All tools use deterministic PyTorch settings (`nsd.deterministic`). Starting from the same encoded features,
  `refine.py` and `evaluate.py` with the released weights reproduce the reported per-query ledgers of all 46 test
  sequences bit for bit on the GPU type used for the paper.
* Training is bitwise reproducible on the same GPU type. Seed 4 retrained from raw scans on a different GPU type gave
  AP 89.93, max F1 90.34, R@100P 72.18 and revisit R@1 91.82, inside the range of the three released runs.
