# Neural Spectral Descriptor (NSD) for LiDAR Place Recognition

Cross-sensor LiDAR loop closure via closed-form spectral encoding + trajectory-aware GNN refinement.

> **Paper-canonical config** (NeurIPS 2026 submission): [`configs/training_multi_dataset.yaml`](configs/training_multi_dataset.yaml) — range-image projection, 16 elevation × 360 azimuth, 9 octave bins → **544D** raw + **128D** GNN context = **672D** final, 4 sensors / 24 train sequences / 173,100 keyframes. All paper numbers (Table 2: $\sigma_{\text{cross}}=0.183$ across 3 seeds with 95% CI [0.165, 0.201]) come from this config. The paper's Method section and this YAML are the source of truth.

## Overview

Neural Spectral Descriptor (NSD) instantiates an *invariance-by-compression, discriminability-by-refinement* design principle:

- **Encoder** (closed-form): per-row DFT magnitudes on cylindrical projections give exact discrete-shift rotation invariance; learnable soft Gaussian binning summarizes the magnitude spectrum into a compact 544D descriptor.
- **Refiner** (learned): DiffAttnConv on a dual-edge trajectory graph attends to *feature differences* between keyframes, recovering discriminability lost during compression.
- **Diagnostic toolkit**: AliasRate (compression-induced collision) + yaw-conditioned R@1 (forward/reverse channel decomposition).

**Cross-sensor scope**: 4 LiDAR sensors (64/32/16/64 beams), trained jointly on KITTI + NCLT + HeLiPR + MulRan.

## Architecture

### Pipeline

```
LiDAR Point Cloud (N, 3)
       ↓
[1] Range-image projection (E_0 × 360 azimuth)
       ↓
[2] Adaptive elevation pooling → (16 × 360)
    └── Per-sensor elevation ranges normalized to 16 rows
       ↓
[3] Per-row 1D FFT magnitude (16 × 181)        ← closed-form, non-learned
       ↓
[4] Learnable soft Gaussian binning (9 bins)   ← learnable bin centers/widths
    └── Statistics: [mean, std] intra + [diff] inter-bin
       ↓
[5] 544D spectral descriptor d
       ↓
[6] Keyframe selection → Dual-edge trajectory graph
    ├── Temporal edges (k=10 nearest in time)
    └── Bayesian similarity edges (posterior ≥ 0.085 AND cosine ≥ 0.993)
       ↓
[7] DiffAttnConv ×2 (128 hidden, 4 heads, d_edge=32)
    └── Operates on h_j − h_i (feature differences)
       ↓
[8] Final descriptor f = cat(d̂_544, ĉ_128) = 672D
       ↓
[9] FAISS cosine top-K retrieval
```

### Encoder details

- **Projection**: cylindrical range image (16 elevation × 360 azimuth), per-sensor elevation ranges (KITTI HDL-64E: [-24.8°, 2.0°], NCLT HDL-32E: [-30.67°, 10.67°], HeLiPR VLP-16: [-15°, 15°], MulRan OS1-64: [-16.6°, 16.6°]) all pooled to 16 rows.
- **Spectral compression**: per-row real-FFT magnitude (16 × 181 components).
- **Soft binning**: 9 learnable Gaussian bins per elevation; centers/widths initialized from octave edges, jointly trained with the refiner at 0.1× base lr.
- **Descriptor**: 16 × 9 × 2 (mean+std intra-bin) + 16 × 8 × 2 (mean+std diffs across adjacent bins) = **288 + 256 = 544D**.

### Refiner (DiffAttnConv)

Operates on **feature differences** $h_j - h_i$, not raw features:

```
α_ij = softmax( q_i · k(h_j − h_i) / √d_h + b(e_ij) )
m_i  = Σ_j α_ij · v(h_j − h_i)
```

- **Architecture**: input projection 544 → 128, 2 layers × 4 heads (32D/head), residual + BatchNorm + dropout 0.1, output projection to 128D context.
- **Edge encoder** (type-aware): temporal → sinusoidal rotation encoding; similarity → cosine + L2-distance + Bayesian posterior. Both project to d_edge=32.
- **Final descriptor**: $f = [\hat{d}; \hat{c}] \in \mathbb{R}^{672}$ (L2-normalized concat preserves encoder's invariance even if refiner fails).

### Bayesian similarity edges

Replaces fixed cosine threshold with density-adaptive posterior:

1. Fisher z-transform: $z_{ij} = \tanh^{-1}(s_{ij})$ where $s_{ij} = \langle\hat{d}_i, \hat{d}_j\rangle$
2. Fit Gaussians: $p(z|\text{same}) = \mathcal{N}(\hat{\mu}_+, \hat{\sigma}_+^2)$, $p(z|\text{diff}) = \mathcal{N}(\hat{\mu}_-, \hat{\sigma}_-^2)$
3. Density-adaptive prior: $\pi_i = \pi_0 / (1 + \rho_i / \rho_{\text{ref}})^{\beta}$ with $\pi_0 = 0.01$, $\beta = 10$, $\rho_i$ = mean cosine similarity to k=50 nearest descriptors
4. **Dual gate**: edge included only when both (a) posterior $P(\text{same}|z_{ij}) \geq 0.085$ AND (b) cosine floor $s_{ij} \geq 0.993$ (sanity guard against unbounded posterior optimism in low-density regions)
5. Capped at 10 edges per node, excluding temporal neighbors.

### Training

- **Loss**: InfoNCE with temperature $\tau = 0.1$
- **Mining**: online hard negative mining every 5 epochs, on raw $d$ (not refined $f$ — mining on $f$ creates feedback loop, costs −11.9%p R@1)
- **Positives**: < 5m apart + ≥ 30 frame gap
- **Negatives**: ≥ 10m apart + same temporal gap
- **Optimizer**: Adam (lr = 5×10⁻⁴, cosine annealing, weight decay 10⁻⁵), FP16 mixed precision
- **Soft-binning policy**: frozen for 3-epoch warmup, then trained at 0.1× base lr
- **Hardware**: single RTX 5080 (17 GB), ~1.6–1.8 hours per seed × 100 epochs (best epoch 80–95)

## Installation

```bash
pip install -r requirements.txt
# Or with dev dependencies (pytest)
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
# Train on KITTI + NCLT + HeLiPR + MulRan (paper-canonical config)
python train_multi_dataset.py

# Custom config / checkpoint dir
python train_multi_dataset.py --config configs/training_multi_dataset.yaml --checkpoint-dir checkpoints/

# Specific GPU + seed (for multi-seed bootstrap)
CUDA_VISIBLE_DEVICES=1 python train_multi_dataset.py --seed 42

# Per-query result dump (for yaw-conditioned analysis)
python train_multi_dataset.py --dump-per-query-dir results/per_query/
```

### Datasets

| Dataset | Sensor | Train sequences | Validation |
|---------|--------|-----------------|------------|
| **HeLiPR** | VLP-16 (16 beams) | Town02, Roundabout01, Bridge01, KAIST04, DCC04, Riverside04 | Town01 |
| **KITTI** | HDL-64E (64 beams) | 01, 02, 06, 07 | 00, 05, 08 |
| **NCLT** | HDL-32E (32 beams) | 2012-05-11, 2012-08-04, 2012-11-04, 2012-11-16, 2013-02-23 | 2012-01-08, 2013-01-10 |
| **MulRan** | OS1-64 (64 beams) | DCC01, DCC02, KAIST01, KAIST02, Riverside01, Riverside02, Sejong01–03 | DCC03, KAIST03, Riverside03 |

**Total**: 24 train sequences / 9 validation sequences / 173,100 keyframes. Cross-sensor evaluation produces 12,894 revisit queries (R@1 evaluated on all queries returning within 5m of a prior keyframe with ≥ 30 frame gap).

## Project Structure

```
Neural-Spectral-Codec/
├── train_multi_dataset.py              # Main training script
├── configs/
│   ├── training_multi_dataset.yaml     # Paper-canonical config
│   └── ...
├── src/
│   ├── data/                           # Dataset loaders
│   │   ├── kitti_loader.py
│   │   ├── nclt_loader.py
│   │   ├── helipr_loader.py
│   │   ├── mulran_loader.py
│   │   └── multi_dataset_loader.py
│   ├── encoding/                       # Spectral encoding
│   │   ├── spectral_encoder.py         # FFT + soft binning
│   │   ├── spectral_policy.py          # Learnable soft binning
│   │   ├── range_image.py              # Cylindrical projection (paper config)
│   │   └── bev_image.py                # BEV projection (alternative mode)
│   ├── keyframe/                       # Keyframe + graph management
│   │   ├── selector.py
│   │   └── graph_manager.py
│   ├── gnn/                            # GNN refiner
│   │   ├── model.py                    # DiffAttnConv + EdgeEncoder
│   │   ├── trainer.py
│   │   └── triplet_miner.py
│   └── retrieval/                      # Loop closing
│       └── two_stage_retrieval.py
├── baselines/                          # Baseline comparisons
│   ├── scan_context.py / m2dp.py / fresco.py / lidar_iris.py
│   ├── bevplace.py + _bevplace_official/
│   ├── evaluate_baselines.py
│   └── evaluate_rotation_invariance.py
├── scripts/                            # Analysis tools
│   ├── aggregate_seeds.py              # 3-seed bootstrap CI
│   ├── analyze_kitti08_failure.py      # KITTI 08 failure decomposition
│   ├── compute_aliasrate.py            # AliasRate metric
│   ├── compute_yaw_recall.py           # Yaw-conditioned R@1
│   ├── finetune_bevplace.py            # BEVPlace++ multi-sensor fine-tune
│   └── ...
├── docs/paper/                         # NeurIPS 2026 paper sources
└── logs/                               # Training logs
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| projection_type | range_image | Cylindrical (16 elev × 360 az) |
| n_elevation | 16 | Pooled elevation rows |
| n_azimuth | 360 | Azimuth columns (1°) |
| n_bins | 9 (octave init) | Learnable soft Gaussian bins per elevation |
| bin_statistics | [mean, std] | Intra-bin per-row statistics |
| inter_bin_statistics | [diff] | Adjacent-bin diffs (mean + std) |
| input_dim | 544 | 16×9×2 intra + 16×8×2 inter-diff |
| hidden_dim | 128 | GNN hidden |
| context_dim | 128 | GNN context output |
| output_dim | 672 | cat($\hat{d}_{544}$, $\hat{c}_{128}$) |
| n_layers | 2 | DiffAttnConv layers |
| n_heads | 4 | Attention heads (4×32D) |
| d_edge | 32 | Edge embedding dim |
| temporal_neighbors | 10 | k-nearest temporal (5 past + 5 future) |
| similarity_max_k | 10 | Max similarity edges per node |
| edge_method | bayesian | Posterior threshold + cosine floor |
| confidence_level | 0.085 | Bayesian posterior threshold (F1-optimal) |
| similarity_threshold | 0.993 | Cosine floor (sanity guard) |
| temperature | 0.1 | InfoNCE temperature |
| n_epochs | 100 | Training epochs (best 80–95) |
| patience | 10 | Early stopping epochs |

## Bundled checkpoints

The repository ships two checkpoints used in the paper so that evaluation and Appendix M analyses can be run without re-training:

| Path | Size | Description |
|------|------|-------------|
| [`results/ctx128_cosine_bayesian/best_model.pth`](results/ctx128_cosine_bayesian/best_model.pth) | 2.3 MB | Paper-canonical NSD checkpoint (epoch 92, best val R@1 = 0.7244, seed 42 of the 3-seed bootstrap in Appendix L). 194,344 parameters total. |
| [`baselines/weights/bevplace_finetune.pth`](baselines/weights/bevplace_finetune.pth) | 5.3 MB | BEVPlace++ multi-sensor fine-tuned checkpoint (paper Appendix M; 5 epochs / 50 min on RTX 5080 from the official KITTI checkpoint). |

The official BEVPlace++ KITTI checkpoint (`bevplace_kitti.pth.tar`, ~16 MB) is **not bundled** (third-party release). To run the BEVPlace++ released-as-is comparison or rerun the multi-sensor fine-tune from scratch, download it from the official repo:

```bash
# https://github.com/zjuluolun/BEVPlace2 — see authors' README for the latest download link
mkdir -p baselines/weights
curl -L -o baselines/weights/bevplace_kitti.pth.tar <URL_FROM_BEVPLACE2_REPO>
# or set NSD_BEVPLACE_WEIGHTS environment variable to point at any local copy
```

Datasets (KITTI / NCLT / HeLiPR / MulRan) are publicly available; download instructions are in each dataset's official site (no scraping). Pre-computed FFT/descriptor caches under `data/preprocessed/` are auto-generated on first training run (~15 GB total).

## Inference / paper-Table-2 evaluation (no training)

```bash
# Validate using the bundled NSD checkpoint
python train_multi_dataset.py \
  --config configs/training_multi_dataset.yaml \
  --checkpoint-dir results/ctx128_cosine_bayesian/ \
  --validate-only

# Run all baselines (Table 2: SC++, M2DP, FreSCo, LiDAR-Iris, BEVPlace++ ms-ft)
python baselines/evaluate_baselines.py --config configs/training_multi_dataset.yaml

# Rotation invariance (Table 3)
python baselines/evaluate_rotation_invariance.py --cache-key 056e0a02
```

## Reproducing Paper Results

```bash
# 3-seed training (paper Table 2 + Appendix L bootstrap CI)
for SEED in 42 123 456; do
    python train_multi_dataset.py --seed $SEED --checkpoint-dir results/seed_$SEED/
done

# Aggregate seed results into bootstrap CI
python scripts/aggregate_seeds.py --seeds 42 123 456 --runs-dir results/ --output results/seed_aggregate.json

# AliasRate (Table 4)
python scripts/compute_aliasrate.py --cache-key 056e0a02 --output results/aliasrate.json

# Yaw-conditioned R@1 (Table 7, Appendix K)
python scripts/compute_yaw_recall.py --output results/yaw_recall.json

# KITTI 08 failure decomposition (Appendix K)
python scripts/analyze_kitti08_failure.py

# BEVPlace++ multi-sensor fine-tuning (Appendix M; ~50 min on RTX 5080)
python scripts/finetune_bevplace.py --config configs/training_multi_dataset.yaml --epochs 5
```

## License

GNU General Public License v3.0

## Authors

- Kimun Park (Dongguk University)
- Moon Gi Seok (Dongguk University)
