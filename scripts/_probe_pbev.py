#!/usr/bin/env python3
"""Probe: reuse deployed BEV phase-sketch magnitude |p_BEV| (16x12=192D) as a
height-magnitude retrieval key. Evaluates raw-mag (pre-log), stored-mag
(from log-compressed re/im, i.e. derivable from the stored 384D p), and
weighted concat fusion with the 288D invariant key."""
import sys, numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
from data.kitti_loader import KITTILoader
from data.nclt_loader import NCLTLoader
from data.mulran_loader_old import MulRanLoader
from encoding.bev_image import BEVProjector
from encoding.phase_features import _pool_rows, _phase_sketch


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


def r1(keys, pos):
    n = len(keys)
    c = e = 0
    for j in range(30, n):
        g = np.linalg.norm(pos[:j - 30 + 1] - pos[j], axis=1)
        if g.min() >= 5:
            continue
        e += 1
        s = keys[j] @ keys.T
        s[max(0, j - 30):j + 31] = -9
        c += np.linalg.norm(pos[int(np.argmax(s))] - pos[j]) < 5
    return c / max(e, 1)


BP = BEVProjector(n_sectors=60, max_range=80.0, min_range=1.0, z_min=-3.0,
                  height_encoding='max', n_height_layers=8, z_max=5.0)


def pbev_keys(pts):
    raw, stored = [], []
    for p in pts:
        bev, _ = BP.project(p, keep_intensity=False)
        lay = _pool_rows(bev, 16)
        sk = _phase_sketch(lay, 12)                      # (16,12) complex
        raw.append(np.abs(sk).reshape(-1))
        f = np.concatenate([sk.real.reshape(-1), sk.imag.reshape(-1)])
        f = np.sign(f) * np.log1p(np.abs(f))             # stored 384D p
        fr, fi = f[:192], f[192:]
        stored.append(np.hypot(fr, fi))                  # derivable from stored p
    return l2(np.stack(raw)), l2(np.stack(stored))


JOBS = [('K00', 'kitti', '00', 'data/preprocessed/cache_de201724_kitti_val_00.npz'),
        ('K05', 'kitti', '05', 'data/preprocessed/cache_de201724_kitti_val_05.npz'),
        ('N13', 'nclt', '2013-01-10', 'data/preprocessed/cache_3a73ece2_nclt_val_2013-01-10.npz'),
        ('KAIST', 'mulran', 'KAIST03', 'data/preprocessed/cache_dabab5b5_mulran_val_KAIST03.npz'),
        ('Riv', 'mulran', 'Riverside03', 'data/preprocessed/cache_dabab5b5_mulran_val_Riverside03.npz')]
ROOT = '/rise/RISE1/workspace/data'
for tag, dt, seq, cf in JOBS:
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
    pts = [L[int(s)]['points'] for s in sids]
    kraw, kst = pbev_keys(pts)
    print(f'{tag}: 288D={r1(kd, pos):.3f}  pbev_raw192={r1(kraw, pos):.3f}  pbev_stored192={r1(kst, pos):.3f}', flush=True)
    for a in [0.3, 0.5, 0.7]:
        k = l2(np.concatenate([np.sqrt(a) * kd, np.sqrt(1 - a) * kst], 1))
        print(f'  fuse(stored) a={a}: {r1(k, pos):.3f}', flush=True)
print('PBEV_PROBE_DONE')
