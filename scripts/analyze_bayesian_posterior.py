"""Bayesian Posterior TP vs FP Analysis on actual training data.

Diagnostic: do Bayesian posteriors (cosine + l2) actually separate same-place from
different-place pairs at the per-edge level? Production setup is FAISS top-K
candidate edges; we replicate that exactly and compute posterior + GT pose label
per candidate, then plot histograms / ROC / PR and report AUC + KL.

Decision criteria:
- AUC > 0.7  → posterior is meaningfully discriminative
- 0.55-0.7  → weak signal, may help with confidence_level tuning
- < 0.55    → posterior is essentially random; Bayesian edge selection won't work
"""
import sys
import os
import json
sys.path.insert(0, 'src')
import numpy as np
import yaml
import faiss
from pathlib import Path

from utils.similarity_stats import SimilarityDistribution
from utils.standardization_stats import StandardizationStats


def find_prefix(dtype: str, seq: str) -> str:
    for f in os.listdir('data/preprocessed'):
        if f.endswith(f'_{dtype}_{seq}.npz'):
            try:
                d = np.load(f'data/preprocessed/{f}', mmap_mode='r')
                if d['descriptors'].shape[1] == 544:
                    return f.replace('cache_', '').split('_', 1)[0]
            except Exception:
                pass
    return None


def load_train_data():
    cfg = yaml.safe_load(open('configs/training_multi_dataset.yaml'))
    train_cfg = cfg['data']['datasets']['train']
    prefixes = {}
    for ds_cfg in train_cfg:
        dtype = ds_cfg['type']
        for seq in ds_cfg['sequences']:
            p = find_prefix(dtype, seq)
            if p:
                prefixes[dtype] = p
                break
    print(f'544D cache prefixes: {prefixes}')

    all_descs, all_poses, all_seq, seq_labels = [], [], [], []
    cur = 0
    for ds_cfg in train_cfg:
        dtype = ds_cfg['type']
        pfx = prefixes.get(dtype)
        if pfx is None:
            continue
        for seq in ds_cfg['sequences']:
            fname = f'data/preprocessed/cache_{pfx}_{dtype}_{seq}.npz'
            if not os.path.exists(fname):
                print(f'  MISSING: {fname}')
                continue
            d = np.load(fname)
            n = len(d['poses'])
            all_descs.append(d['descriptors'])
            all_poses.append(d['poses'])
            all_seq.append(np.full(n, cur, dtype=np.int64))
            seq_labels.append(f'{dtype}_{seq}')
            cur += 1
    descs = np.concatenate(all_descs, axis=0).astype(np.float32)
    poses = np.concatenate(all_poses, axis=0)
    seq_ids = np.concatenate(all_seq, axis=0)
    print(f'Loaded {len(descs)} nodes from {cur} sequences')
    return descs, poses, seq_ids, seq_labels


def build_faiss_candidates(
    descs: np.ndarray,
    seq_ids: np.ndarray,
    k: int = 10,
    temporal_window: int = 5,
):
    """Per-sequence FAISS top-K candidate edges (production-identical).

    Returns:
        src_idx, dst_idx (global), cos_obs (1D), l2_obs (z-scored 1D)
    """
    # Normalize for cosine similarity
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    descs_normed = (descs / norms).astype(np.float32)

    # Standardize for L2 distance
    std_stats = StandardizationStats().fit(descs)
    descs_std = std_stats.transform(descs)

    src_list, dst_list, cos_list, l2_list = [], [], [], []
    unique_seqs = np.unique(seq_ids)
    for sid in unique_seqs:
        mask = seq_ids == sid
        seq_global_idx = np.where(mask)[0]
        n_seq = len(seq_global_idx)
        if n_seq < k + temporal_window + 2:
            continue

        seq_descs_normed = descs_normed[seq_global_idx]
        seq_descs_std = descs_std[seq_global_idx]

        # FAISS top-K on cosine (we'll compute l2 obs separately)
        index_cos = faiss.IndexFlatIP(seq_descs_normed.shape[1])
        index_cos.add(seq_descs_normed)
        # Over-fetch to allow exclusion of temporal neighbors and self
        fetch_k = min(k + 2 * temporal_window + 1, n_seq)
        sim_scores, sim_idx = index_cos.search(seq_descs_normed, fetch_k)

        # Per-source filtering
        for local_i in range(n_seq):
            count = 0
            for j_pos in range(fetch_k):
                local_j = int(sim_idx[local_i, j_pos])
                if local_j == local_i:
                    continue
                if abs(local_j - local_i) <= temporal_window:
                    continue
                global_i = int(seq_global_idx[local_i])
                global_j = int(seq_global_idx[local_j])
                cos_obs = float(sim_scores[local_i, j_pos])
                l2_obs = float(np.linalg.norm(seq_descs_std[local_i] - seq_descs_std[local_j]))
                src_list.append(global_i)
                dst_list.append(global_j)
                cos_list.append(cos_obs)
                l2_list.append(l2_obs)
                count += 1
                if count >= k:
                    break

    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(dst_list, dtype=np.int64),
        np.asarray(cos_list, dtype=np.float32),
        np.asarray(l2_list, dtype=np.float32),
        descs_std,
        std_stats,
    )


def compute_metrics(posteriors_pos, posteriors_neg):
    """AUC, AP, KL divergence, optimal F1 threshold."""
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

    n_pos = len(posteriors_pos)
    n_neg = len(posteriors_neg)
    scores = np.concatenate([posteriors_pos, posteriors_neg])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))

    # KL divergence between two histogram distributions (50 bins)
    hist_pos, edges = np.histogram(posteriors_pos, bins=50, range=(0, 1), density=True)
    hist_neg, _ = np.histogram(posteriors_neg, bins=50, range=(0, 1), density=True)
    eps = 1e-10
    p = hist_pos + eps
    q = hist_neg + eps
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = float(np.sum(p * np.log(p / q)))

    # Optimal F1 threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    best_idx = int(np.argmax(f1[:-1]))
    optimal_f1 = float(f1[best_idx])
    optimal_threshold = float(thresholds[best_idx])

    return {
        'auc': auc,
        'ap': ap,
        'kl': kl_pq,
        'mean_post_pos': float(posteriors_pos.mean()),
        'mean_post_neg': float(posteriors_neg.mean()),
        'median_post_pos': float(np.median(posteriors_pos)),
        'median_post_neg': float(np.median(posteriors_neg)),
        'optimal_f1_threshold': optimal_threshold,
        'optimal_f1_value': optimal_f1,
        'n_pos': int(n_pos),
        'n_neg': int(n_neg),
        'roc_data': (scores, labels),
        'pr_data': (precision, recall, thresholds),
    }


def main():
    out_dir = Path('results/posterior_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('Bayesian Posterior TP vs FP Diagnostic')
    print('=' * 78)
    print('\n[1/5] Loading train data...')
    descs, poses, seq_ids, seq_labels = load_train_data()
    positions = poses[:, :3, 3]

    print('\n[2/5] Building FAISS top-K=10 candidate edges per sequence...')
    src, dst, cos_obs, l2_obs, descs_std, std_stats = build_faiss_candidates(
        descs, seq_ids, k=10, temporal_window=5,
    )
    print(f'  Total candidate edges: {len(src):,}')

    print('\n[3/5] Fitting SimilarityDistribution (cosine + l2, 1M samples)...')
    sim_l2 = SimilarityDistribution(metric='l2').fit(
        descs_std, poses, pos_dist=5.0, neg_dist=10.0,
        min_temporal_gap=30, n_samples=1_000_000,
    )
    print(f'  L2: same N({sim_l2.mu_same:.4f}, {sim_l2.sigma_same:.4f}) | '
          f'diff N({sim_l2.mu_diff:.4f}, {sim_l2.sigma_diff:.4f})')

    sim_cos = SimilarityDistribution(metric='cosine').fit(
        descs, poses, pos_dist=5.0, neg_dist=10.0,
        min_temporal_gap=30, n_samples=1_000_000,
    )
    print(f'  Cos: same N({sim_cos.mu_same:.4f}, {sim_cos.sigma_same:.4f}) | '
          f'diff N({sim_cos.mu_diff:.4f}, {sim_cos.sigma_diff:.4f})')

    if not sim_l2.fitted or not sim_cos.fitted:
        print('ERROR: Bayesian fitting failed (too few same-place pairs)')
        return

    print('\n[4/5] Computing per-edge posteriors + GT labels...')
    posteriors_l2 = sim_l2.posterior(l2_obs, prior=0.01)
    posteriors_cos = sim_cos.posterior(cos_obs, prior=0.01)

    pose_dists = np.linalg.norm(positions[src] - positions[dst], axis=1)
    pos_mask = pose_dists < 5.0
    neg_mask = pose_dists > 10.0
    print(f'  TP (pose<5m):   {int(pos_mask.sum()):,}  ({100*pos_mask.mean():.2f}%)')
    print(f'  FP (pose>10m):  {int(neg_mask.sum()):,}  ({100*neg_mask.mean():.2f}%)')
    print(f'  Gray (5-10m):   {int((~pos_mask & ~neg_mask).sum()):,}  '
          f'({100*((~pos_mask & ~neg_mask).mean()):.2f}%) [excluded]')

    metrics_l2 = compute_metrics(posteriors_l2[pos_mask], posteriors_l2[neg_mask])
    metrics_cos = compute_metrics(posteriors_cos[pos_mask], posteriors_cos[neg_mask])

    # Save metrics (drop curve data for json)
    save_metrics = {
        'n_candidates': int(len(src)),
        'n_pos': int(pos_mask.sum()),
        'n_neg': int(neg_mask.sum()),
        'fit': {
            'l2': {'mu_same': float(sim_l2.mu_same), 'sigma_same': float(sim_l2.sigma_same),
                   'mu_diff': float(sim_l2.mu_diff), 'sigma_diff': float(sim_l2.sigma_diff)},
            'cosine': {'mu_same': float(sim_cos.mu_same), 'sigma_same': float(sim_cos.sigma_same),
                       'mu_diff': float(sim_cos.mu_diff), 'sigma_diff': float(sim_cos.sigma_diff)},
        },
        'l2': {k: v for k, v in metrics_l2.items() if k not in ('roc_data', 'pr_data')},
        'cosine': {k: v for k, v in metrics_cos.items() if k not in ('roc_data', 'pr_data')},
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(save_metrics, f, indent=2)

    print('\n[5/5] Plotting...')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    # Histogram (TP vs FP per metric)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, metric) in zip(axes, [('L2', metrics_l2), ('Cosine', metrics_cos)]):
        post_pos = posteriors_l2[pos_mask] if name == 'L2' else posteriors_cos[pos_mask]
        post_neg = posteriors_l2[neg_mask] if name == 'L2' else posteriors_cos[neg_mask]
        ax.hist(post_neg, bins=50, alpha=0.5, label=f'FP (diff-place, n={len(post_neg):,})',
                color='C0', density=True)
        ax.hist(post_pos, bins=50, alpha=0.6, label=f'TP (same-place, n={len(post_pos):,})',
                color='C3', density=True)
        ax.set_xlabel('Posterior P(same | obs)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} | AUC={metric["auc"]:.3f}, AP={metric["ap"]:.3f}, KL={metric["kl"]:.2f}')
        ax.legend()
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(out_dir / 'histogram.png', dpi=120)
    plt.close()

    # ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, metric in [('L2', metrics_l2), ('Cosine', metrics_cos)]:
        scores, labels = metric['roc_data']
        fpr, tpr, _ = roc_curve(labels, scores)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={metric["auc"]:.3f})')
        precision, recall, _ = metric['pr_data']
        axes[1].plot(recall, precision, label=f'{name} (AP={metric["ap"]:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC')
    axes[0].legend()
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_pr_curves.png', dpi=120)
    plt.close()

    # stdout summary
    print()
    print('=' * 78)
    print(f"FAISS top-K=10 candidates ({len(src):,} edges from {len(np.unique(seq_ids))} train sequences):")
    print(f"  TP (pose<5m):  {int(pos_mask.sum()):,}    FP (pose>10m): {int(neg_mask.sum()):,}")
    print()
    print(f"  {'Metric':8s} | {'AUC':6s} | {'AP':6s} | {'KL':6s} | {'mean(P|TP)':12s} {'mean(P|FP)':12s} | {'F1@τ':16s}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*12}-{'-'*12}-+-{'-'*16}")
    for name, m in [('l2', metrics_l2), ('cosine', metrics_cos)]:
        print(f"  {name:8s} | {m['auc']:.4f} | {m['ap']:.4f} | {m['kl']:.4f} | "
              f"{m['mean_post_pos']:.4f}      {m['mean_post_neg']:.4f}      | "
              f"{m['optimal_f1_value']:.3f} @ τ={m['optimal_f1_threshold']:.4f}")
    print('=' * 78)
    print()
    print('Decision criteria:')
    print('  AUC > 0.7    → posterior meaningfully discriminative')
    print('  AUC 0.55-0.7 → weak signal, try confidence_level tuning')
    print('  AUC < 0.55   → essentially random, Bayesian edge selection unworkable')
    print()
    print(f'Outputs:')
    print(f'  {out_dir}/histogram.png')
    print(f'  {out_dir}/roc_pr_curves.png')
    print(f'  {out_dir}/metrics.json')


if __name__ == '__main__':
    main()
