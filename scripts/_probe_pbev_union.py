#!/usr/bin/env python3
"""8-seq candidate-union eval: 288D invariant key + |p_BEV| 192D magnitude key.
Per query: top-K candidates from each channel, union, rerank with
score = w_R * s_R + w_H * s_H. Reports per-seq R@1, union candidate recall,
and query-weighted aggregate R-bar-q on the exact paper weights."""
import sys, numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
from data.kitti_loader import KITTILoader
from data.nclt_loader import NCLTLoader
from data.mulran_loader_old import MulRanLoader
from encoding.bev_image import BEVProjector
from encoding.phase_features import _pool_rows, _phase_sketch

K = 100
WEIGHTS = [(0.25, 1.0), (0.5, 1.0), (1.0, 1.0)]


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def bin288(m):
    O = [0, 1, 2, 4, 8, 16, 32, 64, 128, 181]
    mus, sds = [], []
    for b in range(len(O) - 1):
        seg = m[..., O[b]:O[b + 1]]
        mu = seg.mean(-1)
        mus.append(mu)
        sds.append(np.sqrt(((seg - mu[..., None]) ** 2).mean(-1) + 1e-8))
    return l2(np.concatenate([np.stack(mus, -1).reshape(len(m), -1),
                              np.stack(sds, -1).reshape(len(m), -1)], -1))


BP = BEVProjector(n_sectors=60, max_range=80.0, min_range=1.0, z_min=-3.0,
                  height_encoding='max', n_height_layers=8, z_max=5.0)


def pbev_stored_keys(pts):
    out = []
    for p in pts:
        bev, _ = BP.project(p, keep_intensity=False)
        sk = _phase_sketch(_pool_rows(bev, 16), 12)
        f = np.concatenate([sk.real.reshape(-1), sk.imag.reshape(-1)])
        f = np.sign(f) * np.log1p(np.abs(f))
        out.append(np.hypot(f[:192], f[192:]))
    return l2(np.stack(out))


def union_eval(kr, kh, pos):
    n = len(kr)
    res = {w: 0 for w in WEIGHTS}
    r_only = h_only = cand_hit = e = 0
    for j in range(30, n):
        g = np.linalg.norm(pos[:j - 30 + 1] - pos[j], axis=1)
        if g.min() >= 5:
            continue
        e += 1
        sr = kr[j] @ kr.T
        sh = kh[j] @ kh.T
        sr[max(0, j - 30):j + 31] = -9
        sh[max(0, j - 30):j + 31] = -9
        r_only += np.linalg.norm(pos[int(np.argmax(sr))] - pos[j]) < 5
        h_only += np.linalg.norm(pos[int(np.argmax(sh))] - pos[j]) < 5
        cand = np.union1d(np.argpartition(-sr, K)[:K], np.argpartition(-sh, K)[:K])
        gd = np.linalg.norm(pos[cand] - pos[j], axis=1)
        cand_hit += (gd < 5).any()
        for w in WEIGHTS:
            sc = w[0] * sr[cand] + w[1] * sh[cand]
            res[w] += gd[int(np.argmax(sc))] < 5
    return e, r_only, h_only, cand_hit, res


JOBS = [('K00', 'kitti', '00', 'data/preprocessed/cache_de201724_kitti_val_00.npz', 632),
        ('K05', 'kitti', '05', 'data/preprocessed/cache_de201724_kitti_val_05.npz', 377),
        ('K08', 'kitti', '08', 'data/preprocessed/cache_de201724_kitti_val_08.npz', 235),
        ('N12', 'nclt', '2012-01-08', 'data/preprocessed/cache_3a73ece2_nclt_val_2012-01-08.npz', 1881),
        ('N13', 'nclt', '2013-01-10', 'data/preprocessed/cache_3a73ece2_nclt_val_2013-01-10.npz', 156),
        ('DCC', 'mulran', 'DCC03', 'data/preprocessed/cache_dabab5b5_mulran_val_DCC03.npz', 2375),
        ('KAIST', 'mulran', 'KAIST03', 'data/preprocessed/cache_dabab5b5_mulran_val_KAIST03.npz', 2989),
        ('Riv', 'mulran', 'Riverside03', 'data/preprocessed/cache_dabab5b5_mulran_val_Riverside03.npz', 2636)]
ROOT = '/rise/RISE1/workspace/data'
agg = {w: [0.0, 0.0] for w in WEIGHTS}   # [weighted sum, weight sum]
agg_r, agg_h, agg_c = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
for tag, dt, seq, cf, qw in JOBS:
    d = np.load(cf)
    sids = d['scan_ids']
    pos = d['poses'][:, :3, 3]
    kd = bin288(d['fft_magnitudes'].astype(np.float64))
    if dt == 'kitti':
        L = KITTILoader(f'{ROOT}/kitti/dataset', seq, lazy_load=True)
    elif dt == 'nclt':
        L = NCLTLoader(f'{ROOT}/nclt', seq, lazy_load=True)
    else:
        L = MulRanLoader(f'{ROOT}/mulran', seq, lazy_load=True)
    kh = pbev_stored_keys([L[int(s)]['points'] for s in sids])
    e, r_only, h_only, cand_hit, res = union_eval(kd, kh, pos)
    line = f'{tag}: q={e} R={r_only/e:.3f} H={h_only/e:.3f} candrec@{K}={cand_hit/e:.3f}'
    for w in WEIGHTS:
        line += f' u{w[0]:g}/{w[1]:g}={res[w]/e:.3f}'
        agg[w][0] += qw * res[w] / e
        agg[w][1] += qw
    agg_r[0] += qw * r_only / e; agg_r[1] += qw
    agg_h[0] += qw * h_only / e; agg_h[1] += qw
    agg_c[0] += qw * cand_hit / e; agg_c[1] += qw
    print(line, flush=True)
print(f'AGG(8seq, paper weights): R288={agg_r[0]/agg_r[1]:.4f} H192={agg_h[0]/agg_h[1]:.4f} candrec={agg_c[0]/agg_c[1]:.4f}', flush=True)
for w in WEIGHTS:
    print(f'  union w={w[0]:g}/{w[1]:g}: Rbar_q={agg[w][0]/agg[w][1]:.4f}', flush=True)
print('UNION_PROBE_DONE')
