# Neural Spectral Codec for LiDAR Loop Closing

LiDAR loop closing system using spectral histograms and Graph Neural Networks.

## Overview

Neural Spectral Codec is a graph-enhanced loop closing system for LiDAR SLAM that achieves rotation-invariant place recognition through FFT-based spectral histograms and trajectory-aware GNN enhancement.

**Key Features:**
- **Rotation invariant** via FFT magnitude spectrum (phase discarded)
- **Sensor-agnostic** BEV polar grid projection (no elevation calibration needed)
- **Trajectory-aware GNN** with DiffAttnConv (feature differences) and dual-edge graph
- **Cross-sensor generalization** trained on HeLiPR(VLP-16), validated on KITTI(HDL-64E)/NCLT(HDL-32E)

## Architecture

### Pipeline
```
LiDAR Point Cloud (N, 3)
       ↓
[1] BEV Polar Grid Projection (79 × 360)
    ├── 1m radial × 1° angular resolution
    ├── Max z-height per cell
    └── Sensor-agnostic (no elevation calibration)
       ↓
[2] Row-wise 1D FFT + Magnitude (79 × 181)
       ↓
[3] Exponential Frequency Binning (79 × 4)
    ├── Statistics: [mean, std] → 632D
    └── Inter-bin diff → 474D
       ↓
[4] Spectral Descriptor (1106D)
       ↓
[5] Keyframe Selection → Temporal Graph
       ↓
[6] GNN Enhancement (DiffAttnConv × 2)
    └── cat(raw_1106, context_256) = 1362D
       ↓
[7] Loop Closure Retrieval (FAISS + GICP)
```

### GNN Architecture (DiffAttnConv)

Operates on **feature differences** (h_j - h_i) to capture trajectory change rates:

```
Input(1106) → Proj(128) → DiffAttnConv×2(128, 4 heads) → Proj(256) → Output
                                    ↑
                          EdgeEncoder(d_edge=32)
                          ├── Temporal: sinusoidal rotation encoding
                          └── Similarity: descriptor distance encoding
```

- **Output:** cat(L2-norm raw descriptor, L2-norm GNN context) = 1362D
- **Dual edges:** temporal (k=10 neighbors) + similarity (Bayesian/threshold)
- **Edge attributes:** 4D [dist_norm, rot_norm, cos_sim, l2_dist_norm]

### Training

- **Loss:** InfoNCE with temperature τ=0.07
- **Mining:** Online hard negative mining per epoch, per-sequence
- **Positives:** < 5m apart + ≥30 frame gap
- **Negatives:** ≥ 10m apart
- **Optimization:** AdamW, mixed precision (AMP), gradient accumulation (4 steps)

## Installation

```bash
pip install -r requirements.txt
# Or with dev dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.8+
- PyTorch 2.1.0+
- PyTorch Geometric
- Open3D
- FAISS
- NumPy, SciPy, PyYAML

## Training

```bash
# Train on HeLiPR + KITTI + NCLT (default config)
python train_multi_dataset.py

# Custom config
python train_multi_dataset.py --config configs/training.yaml --checkpoint-dir checkpoints/

# Specific GPU
CUDA_VISIBLE_DEVICES=1 python train_multi_dataset.py
```

### Configuration

Edit `configs/training_multi_dataset.yaml`:

```yaml
encoding:
  projection_type: 'bev'    # 'bev' or 'range_image'
  n_bins: 4                 # Frequency bins per row
  max_range: 80.0           # BEV range (meters)
  min_range: 1.0
  bin_statistics: ['mean', 'std']
  inter_bin_statistics: ['diff']

gnn:
  input_dim: 1106           # 79×4×2 + 79×3×2
  hidden_dim: 128
  context_dim: 256
  n_layers: 2
  n_heads: 4
```

### Datasets

| Dataset | Sensor | Train | Validation |
|---------|--------|-------|------------|
| HeLiPR | VLP-16 | Town02-03, Roundabout, Bridge, KAIST, DCC, Riverside | Town01 |
| KITTI | HDL-64E | 02, 06, 07 | 00, 05, 08 |
| NCLT | HDL-32E | 2012-05/08/11 | 2012-01-08, 2013-01-10 |

## Project Structure

```
Neural-Spectral-Codec/
├── train_multi_dataset.py      # Main training script
├── configs/
│   ├── training_multi_dataset.yaml  # Primary config
│   ├── default.yaml
│   └── ...
├── src/
│   ├── data/                   # Dataset loaders
│   │   ├── kitti_loader.py
│   │   ├── nclt_loader.py
│   │   └── helipr_loader.py
│   ├── encoding/               # Spectral encoding
│   │   ├── spectral_encoder.py # FFT + histogram
│   │   ├── bev_image.py        # BEV polar grid projection
│   │   └── range_image.py      # Range image projection (legacy)
│   ├── keyframe/               # Keyframe management
│   │   ├── selector.py
│   │   └── graph_manager.py
│   ├── gnn/                    # GNN model
│   │   ├── model.py            # DiffAttnConv + EdgeEncoder
│   │   ├── trainer.py
│   │   └── triplet_miner.py
│   └── retrieval/              # Loop closing
│       └── two_stage_retrieval.py
├── scripts/                    # Analysis tools
│   └── analyze_perceptual_aliasing.py
├── baselines/                  # Baseline comparisons
├── docs/                       # Design documentation
└── logs/                       # Training logs
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| projection_type | bev | BEV polar grid (1m × 1°) |
| n_bins | 4 | Frequency bins per row |
| input_dim | 1106 | Spectral descriptor dimension |
| hidden_dim | 128 | GNN internal dimension |
| context_dim | 256 | GNN context output |
| output_dim | 1362 | cat(raw_1106, ctx_256) |
| n_layers | 2 | DiffAttnConv layers |
| n_heads | 4 | Attention heads (4×32D) |
| d_edge | 32 | Edge embedding dimension |
| temperature | 0.07 | InfoNCE temperature |

## Documentation

Detailed design documents in `docs/20260128/`:
- `overall_approach.md` - System overview
- `spectral_encoding_detail.md` - FFT + histogram encoding
- `gnn_detail.md` - GNN architecture details
- `keyframe_detail.md` - Keyframe selection strategy
- `training_detail.md` - Training methodology

## License

GNU General Public License v3.0

## Authors

- Kimun Park (Dongguk University)
- Moon Gi Seok (Dongguk University)
