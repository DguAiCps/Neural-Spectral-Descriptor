"""Audit pose-GT similarity edges on actual training data."""
import sys, os
sys.path.insert(0, 'src')
import numpy as np
import yaml
from collections import defaultdict

from keyframe.graph_manager import build_similarity_edges_from_poses


def find_prefix(dtype, seq):
    for f in os.listdir('data/preprocessed'):
        if f.endswith(f'_{dtype}_{seq}.npz'):
            try:
                d = np.load(f'data/preprocessed/{f}', mmap_mode='r')
                if d['descriptors'].shape[1] == 544:
                    return f.replace('cache_', '').split('_', 1)[0]
            except Exception:
                pass
    return None


def main():
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
    print('544D cache prefixes:', prefixes)

    # Load train sequences
    all_poses, all_seq_ids, all_descs, seq_lens, seq_labels = [], [], [], [], []
    cur = 0
    for ds_cfg in train_cfg:
        dtype = ds_cfg['type']
        pfx = prefixes.get(dtype)
        if pfx is None:
            continue
        for seq in ds_cfg['sequences']:
            fname = f'data/preprocessed/cache_{pfx}_{dtype}_{seq}.npz'
            if not os.path.exists(fname):
                print(f'MISSING: {fname}')
                continue
            d = np.load(fname)
            n = len(d['poses'])
            all_poses.append(d['poses'])
            all_seq_ids.append(np.full(n, cur, dtype=np.int64))
            all_descs.append(d['descriptors'])
            seq_lens.append(n)
            seq_labels.append(f'{dtype}_{seq}')
            cur += 1

    train_poses = np.concatenate(all_poses, axis=0)
    train_sequence_ids = np.concatenate(all_seq_ids, axis=0)
    train_descs = np.concatenate(all_descs, axis=0)
    N = len(train_poses)
    print(f'Loaded: {N} nodes, {cur} sequences')

    edges = build_similarity_edges_from_poses(
        train_poses, train_descs, defaultdict(set),
        sequence_ids=train_sequence_ids,
        pos_dist=5.0, min_temporal_gap=30, similarity_max_k=10,
    )
    print(f'Built {len(edges)} edges (training log showed 595,959)')

    positions = train_poses[:, :3, 3]

    # Degree distribution
    out_deg = np.zeros(N, dtype=np.int64)
    for s, _, _, _, _ in edges:
        out_deg[s] += 1
    print(f'\n[Degree]')
    print(f'  Nodes with 0 edges:  {int((out_deg == 0).sum()):,} ({100*(out_deg==0).mean():.1f}%)')
    print(f'  Nodes with 1-3:      {int(((out_deg >= 1) & (out_deg <= 3)).sum()):,}')
    print(f'  Nodes with 4-9:      {int(((out_deg >= 4) & (out_deg <= 9)).sum()):,}')
    print(f'  Nodes at max_k=10:   {int((out_deg == 10).sum()):,} ({100*(out_deg==10).mean():.1f}%)')
    print(f'  Mean={out_deg.mean():.2f}, Max={int(out_deg.max())}')

    # Per-sequence
    print(f'\n[Per-sequence]')
    seq_e = np.zeros(cur, dtype=np.int64)
    for s, _, _, _, _ in edges:
        seq_e[train_sequence_ids[s]] += 1
    for i, (lbl, nf, ne) in enumerate(zip(seq_labels, seq_lens, seq_e)):
        print(f'  seq{i:2d} {lbl:30s} frames={nf:6d} edges={ne:7d} ({ne/nf:.2f} e/f)')

    # Constraint audit
    np.random.seed(0)
    samp = np.random.choice(len(edges), min(5000, len(edges)), replace=False)
    pv, gv, cv = 0, 0, 0
    dists = []
    for i in samp:
        s, d, _, _, _ = edges[i]
        pd = float(np.linalg.norm(positions[s] - positions[d]))
        dists.append(pd)
        if pd >= 5.0:
            pv += 1
        if abs(int(s) - int(d)) < 30:
            gv += 1
        if train_sequence_ids[s] != train_sequence_ids[d]:
            cv += 1
    dists = np.array(dists)
    print(f'\n[Constraint audit (5000 edges)]')
    print(f'  pose_dist >= 5m:   {pv}  (should be 0)')
    print(f'  frame_gap < 30:    {gv}  (should be 0)')
    print(f'  cross_sequence:    {cv}  (should be 0)')
    print(f'  dist stats: mean={dists.mean():.2f}m, median={np.median(dists):.2f}m, p99={np.percentile(dists, 99):.2f}m')


if __name__ == '__main__':
    main()
