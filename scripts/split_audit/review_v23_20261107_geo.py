#!/usr/bin/env python3
"""Geographic overlap audit of the CURRENT split v2 (train 8 / val 9 / test 46).

Roster authority: results/pdv2_diffattn_contrast_test_20261104/protocol.json.
Coordinates are recomputed from the dataset files (no cached trajectories are trusted;
results/split_audit/_traj_cache.pkl of the pd2026 audit is only compared against).

Frames (2D horizontal, metres; z is ignored, which can only shorten distances):
  karlsruhe_utm32  KITTI odometry: raw-drive OXTS lat/lon (kitti_raw_oxts, devkit frame ranges);
                   KITTI-360: data_poses/<drive>/oxts lat/lon at the velodyne frames in poses.txt
                   coverage. WGS84 -> UTM 32N.
  korea_utm52      MulRan global_pose.csv (UTM 52N, checked against gps.csv);
                   HeLiPR Town / Roundabout: site-local GT fitted to UTM 52N with the 01 sequence's
                   inspva.csv (02/03 share the 01 frame, checked); HeLiPR KAIST/DCC/Riverside:
                   registered onto the MulRan Daejeon union (FFT translation search + rigid ICP).
  helipr_bridge    Bridge01-04 GT (site-local ENU). No INS on disk; placed in UTM 52N only through
                   two public bridge coordinates -> worst-case lower bounds, not exact distances.
  nclt_campus, mcd_kth, mcd_tuhh   shared local frames without lat/lon (test-only datasets).
Per-session origin-reset cache poses are never compared across sessions; cache keyframes are
mapped to the global trajectory by time / frame id and the mapping is checked by a rigid fit.

Metrics for a directed pair A -> B (B = densified polyline, <= 0.5 m spacing, no interpolation
over gaps > 20 m; exact nearest neighbour by cKDTree):
  frac_pose_X  fraction of A's GT pose samples (~10 Hz, time weighted) within X m of B
  frac_len_X   fraction of A's route (5 m arc-length samples) within X m of B
  frac_kf_X    fraction of A's cache keyframes (the rows read by training / the ledgers) within X m
  min_dist_m   minimum over A pose samples
READ-ONLY w.r.t. repo inputs and /mnt/d. Single thread. Output: results/pdv2_review_closure_20261107/geography/.
"""
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geoutil import latlon_to_utm  # noqa: E402
from geo_audit import (thin, rigid2d, ang, rot, convergence_deg, haversine_km, scan_stems,  # noqa: E402
                       fft_translation_search, icp2d, ODOM)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/pdv2_review_closure_20261107/geography'
DS = Path('/mnt/d/NSD_datasets')
KOXTS = DS / 'kitti_raw_oxts'
PROTOCOL = REPO / 'results/pdv2_diffattn_contrast_test_20261104/protocol.json'
THR = (25, 50, 80, 100, 200)
RULE_M, RULE_FRAC = 50.0, 0.01
DENSE_STEP, MAX_GAP = 0.5, 20.0
ROLE_PAIRS = (('train', 'val'), ('train', 'test'), ('val', 'test'))
# city-level metadata for frames without lat/lon (centre, uncertainty km, source)
CITY_META = {
    'nclt_campus': ('Ann Arbor, MI, USA (UMich North Campus)', (42.2932, -83.7097), 5.0,
                    'NCLT dataset paper / site: University of Michigan North Campus; GT csv has no lat/lon'),
    'mcd_kth': ('Stockholm, Sweden (KTH campus)', (59.3498, 18.0707), 5.0,
                'MCD dataset paper: KTH Royal Institute of Technology campus; pose_inW has no lat/lon'),
    'mcd_tuhh': ('Hamburg, Germany (TUHH campus)', (53.4604, 9.9690), 5.0,
                 'MCD dataset paper: Hamburg University of Technology campus; pose_inW has no lat/lon'),
}
BRIDGE_WIKI = {'Hangang': (37.5159, 126.9578, 'https://en.wikipedia.org/wiki/Hangang_Bridge'),
               'Dongjak': (37.5112, 126.9822, 'https://en.wikipedia.org/wiki/Dongjak_Bridge')}
BRIDGE_SEG_T = {'Hangang': (0.0, 80.0), 'Dongjak': (585.0, 646.0)}
T0 = time.time()
VARIANTS = {}


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


class Hasher:
    """sha256 of one file, or of an ordered list of files (content concatenated)."""

    @staticmethod
    def file(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        return h.hexdigest()

    @staticmethod
    def files(ps):
        h = hashlib.sha256()
        for p in ps:
            with open(p, 'rb') as f:
                h.update(f.read())
        return h.hexdigest()


def read_csv_floats(path, ncol):
    """Fast numeric csv parse (C parser) with an exact element-count check; returns (N, ncol), sha256."""
    raw = open(path, 'rb').read()
    nlines = raw.count(b'\n') + (0 if raw.endswith(b'\n') else 1)
    txt = raw.replace(b'-nan', b'nan').replace(b',', b' ').replace(b'\n', b' ')
    a = np.fromstring(txt.decode('ascii'), dtype=np.float64, sep=' ')
    assert a.size == nlines * ncol, (path, a.size, nlines, ncol)
    return a.reshape(-1, ncol), hashlib.sha256(raw).hexdigest()


def densify(xy, step=DENSE_STEP, max_gap=MAX_GAP):
    """Linear interpolation between consecutive samples (spacing <= step), not across gaps > max_gap."""
    d = np.hypot(*np.diff(xy, axis=0).T)
    out = [xy[:1]]
    for i in np.nonzero(d > step)[0]:
        if d[i] > max_gap:
            continue
        n = int(np.ceil(d[i] / step))
        f = (np.arange(1, n) / n)[:, None]
        out.append(xy[i] + f * (xy[i + 1] - xy[i]))
    out.append(xy[1:])
    return np.vstack(out)


def rigid_check(local, glob):
    """2D rigid fit local -> global; residual stats (used to validate keyframe mapping)."""
    R, t = rigid2d(local, glob)
    r = np.linalg.norm(local @ R.T + t - glob, axis=1)
    return {'n': int(len(r)), 'rotation_deg': round(ang(R), 3), 'resid_median_m': round(float(np.median(r)), 3),
            'resid_p95_m': round(float(np.percentile(r, 95)), 3), 'resid_max_m': round(float(r.max()), 3)}


def nearest_time(t_src, t_q):
    i = np.searchsorted(t_src, t_q).clip(1, len(t_src) - 1)
    j = np.where(np.abs(t_src[i - 1] - t_q) <= np.abs(t_src[i] - t_q), i - 1, i)
    return j, np.abs(t_src[j] - t_q)


def utm_to_latlon(E, N, zone):
    """Newton inverse of latlon_to_utm (sub-mm), vectorised."""
    E, N = np.atleast_1d(E).astype(float), np.atleast_1d(N).astype(float)
    lon0 = (zone - 1) * 6 - 180 + 3
    lat, lon = np.full_like(E, 45.0 if zone == 32 else 37.0), np.full_like(E, float(lon0))
    for _ in range(10):
        e0, n0 = latlon_to_utm(lat, lon, zone)
        h = 1e-6
        e1, n1 = latlon_to_utm(lat + h, lon, zone)
        e2, n2 = latlon_to_utm(lat, lon + h, zone)
        J = np.stack([np.stack([(e1 - e0) / h, (e2 - e0) / h], -1), np.stack([(n1 - n0) / h, (n2 - n0) / h], -1)], 1)
        d = np.linalg.solve(J, np.stack([E - e0, N - n0], -1)[..., None])[..., 0]
        lat, lon = lat + d[:, 0], lon + d[:, 1]
    return lat, lon


# ---------------------------------------------------------------- cache keyframes
def cache_of(roster, key):
    return REPO / roster[key]['cache']


def cache_xy_time(path):
    z = np.load(path)
    P = z['poses_gt'] if 'poses_gt' in z.files else z['poses']
    return dict(t=z['timestamps'].astype(np.float64), sid=z['scan_ids'].astype(np.int64), P=P,
                gt_field='poses_gt' if 'poses_gt' in z.files else 'poses (GT keyframes of the train cache)')


# ---------------------------------------------------------------- loaders (keep times for keyframe mapping)
def load_kitti(seq, cache):
    drive, a, b = ODOM[seq]
    files = sorted((KOXTS / f'{drive}_sync/oxts/data').glob('*.txt'))
    assert len(files) > b, (seq, len(files))
    used = files[a:b + 1]
    ll = np.array([[float(v) for v in open(f).read().split()[:2]] for f in used])
    E, N = latlon_to_utm(ll[:, 0], ll[:, 1], 32)
    xy = np.c_[E, N]
    P = np.loadtxt(DS / f'kitti/dataset/poses/{seq}.txt')
    assert len(P) == len(xy)
    pxz = np.c_[P[:, 11], -P[:, 3]]
    ev = {'odom_poses_vs_oxts_rigid': rigid_check(pxz, xy)}
    c = cache_xy_time(cache)
    kf = xy[c['sid']]
    kloc = np.c_[c['P'][:, 2, 3], -c['P'][:, 0, 3]]
    ev['keyframe_map'] = dict(rule='scan_id = odometry frame index -> OXTS frame first+scan_id',
                              cache_gt_vs_global_rigid=rigid_check(kloc, kf),
                              sid_range=[int(c['sid'].min()), int(c['sid'].max())])
    ref = json.load(open(REPO / 'results/kitti_odometry_geo_reference.json'))['sequences']
    if seq in ref:
        rp = np.array(ref[seq]['polyline_latlon'])
        mine = ll[::ref[seq]['subsample']]
        ev['committed_geo_reference_max_dev_m'] = (round(float(np.abs(rp - mine).max() * 111e3), 4)
                                                  if len(rp) == len(mine) else f'len {len(rp)} vs {len(mine)}')
    else:
        ev['committed_geo_reference_max_dev_m'] = 'absent from results/kitti_odometry_geo_reference.json'
    return dict(frame='karlsruhe_utm32', georef='oxts_gnss_ins', xy=xy, kf=kf, latlon_c=ll.mean(0).tolist(),
                latlon_basis='measured (OXTS mean)', evidence=ev,
                sources={f'kitti_raw_oxts/{drive}_sync/oxts/data[{a}:{b}] ({len(used)} files)': Hasher.files(used),
                         f'kitti/dataset/poses/{seq}.txt': Hasher.file(DS / f'kitti/dataset/poses/{seq}.txt')},
                transform=f'WGS84 lat/lon (OXTS) -> UTM 32N; odometry frames {a}-{b} of raw drive {drive}')


def load_kitti360(drive, cache):
    seqdir = DS / f'kitti360/2013_05_28_drive_{drive}_sync'
    base = DS / f'kitti360/data_poses/2013_05_28_drive_{drive}_sync'
    pf = np.loadtxt(seqdir / 'poses.txt', usecols=0).astype(int)
    vel = scan_stems(seqdir / 'velodyne_points/data')
    files = sorted((base / 'oxts/data').glob('*.txt'))
    fr = vel[(vel >= pf[0]) & (vel <= pf[-1])]
    assert fr.max() < len(files), (drive, fr.max(), len(files))
    used = [files[i] for i in fr]
    ll = np.array([[float(v) for v in open(f).read().split()[:2]] for f in used])
    E, N = latlon_to_utm(ll[:, 0], ll[:, 1], 32)
    xy = np.c_[E, N]
    c = cache_xy_time(cache)
    kframe = np.rint(c['t'] * 10).astype(np.int64)            # loader: timestamps = frame * 0.1
    idx = np.searchsorted(fr, kframe)
    assert np.all(fr[idx.clip(0, len(fr) - 1)] == kframe), 'K360 keyframe frame not in velodyne coverage'
    kf = xy[idx]
    ev = {'keyframe_map': dict(rule='frame = round(timestamp*10) (kitti360_loader) -> oxts/data/<frame>.txt',
                               cache_gt_vs_global_rigid=rigid_check(c['P'][:, :2, 3], kf),
                               scan_id_is_index_in_coverage=bool(np.all(idx == c['sid'])))}
    return dict(frame='karlsruhe_utm32', georef='oxts_gnss_ins', xy=xy, kf=kf, latlon_c=ll.mean(0).tolist(),
                latlon_basis='measured (OXTS mean)', evidence=ev,
                sources={f'kitti360/data_poses/2013_05_28_drive_{drive}_sync/oxts/data at {len(fr)} velodyne frames':
                         Hasher.files(used),
                         f'kitti360/2013_05_28_drive_{drive}_sync/poses.txt': Hasher.file(seqdir / 'poses.txt')},
                transform='WGS84 lat/lon (OXTS) -> UTM 32N; velodyne frames inside poses.txt coverage')


def load_nclt(date, cache):
    f = DS / f'nclt/{date}/groundtruth_{date}.csv'
    a, h = read_csv_floats(f, 7)
    a = a[np.isfinite(a[:, :3]).all(1)]
    a = a[np.argsort(a[:, 0])]
    st = scan_stems(DS / f'nclt/{date}/velodyne_sync')
    span = a[(a[:, 0] >= st[0]) & (a[:, 0] <= st[-1])]
    xy = span[::10, 1:3]
    c = cache_xy_time(cache)
    ut = st[c['sid']].astype(np.float64)
    dt_ts = np.abs((ut - st[0]) / 1e6 - c['t'])
    j, dt = nearest_time(a[:, 0], ut)
    kf = a[j, 1:3]
    ev = {'keyframe_map': dict(rule='scan_id -> sorted velodyne_sync stem (utime) -> nearest GT row',
                               cache_timestamp_vs_stem_max_s=round(float(dt_ts.max()), 6),
                               gt_time_gap_max_ms=round(float(dt.max() / 1e3), 3),
                               note='the loader matches the same nearest GT row, so a GT gap affects cache and audit equally',
                               cache_gt_vs_global_rigid=rigid_check(c['P'][:, :2, 3], kf))}
    return dict(frame='nclt_campus', georef='shared_local_frame', xy=xy, kf=kf, latlon_c=None,
                latlon_basis='metadata', evidence=ev, sources={f'nclt/{date}/groundtruth_{date}.csv': h},
                transform='NCLT GT (shared campus frame, x/y), ~100 Hz -> every 10th row inside velodyne_sync span')


def load_mulran(seq, cache):
    f = DS / f'mulran/{seq}/global_pose.csv'
    p, h = read_csv_floats(f, 13)
    p = p[np.argsort(p[:, 0])][:, [0, 4, 8]]
    st = scan_stems(DS / f'mulran/{seq}/Ouster')
    p = p[(p[:, 0] >= st[0] - 1e8) & (p[:, 0] <= st[-1] + 1e8)]
    graw = np.loadtxt(DS / f'mulran/{seq}/gps.csv', delimiter=',', usecols=(0, 1, 2))
    E, N = latlon_to_utm(graw[:, 1], graw[:, 2], 52)
    gi, gdt = nearest_time(p[:, 0], graw[:, 0])
    m = gdt < 5e7
    gd = np.hypot(E[m] - p[gi[m], 1], N[m] - p[gi[m], 2])
    dec = max(1, int(round(0.1 / (np.median(np.diff(p[:, 0])) / 1e9))))
    xy = p[::dec, 1:3]
    c = cache_xy_time(cache)
    j, dt = nearest_time(p[:, 0], np.rint(c['t'] * 1e9))
    kf = p[j, 1:3]
    ev = {'gps_csv_vs_global_pose_m': {'median': round(float(np.median(gd)), 2),
                                       'p90': round(float(np.percentile(gd, 90)), 2), 'n': int(m.sum())},
          'keyframe_map': dict(rule='timestamp (s) -> nearest global_pose row (ns)',
                               time_gap_max_ms=round(float(dt.max() / 1e6), 3),
                               cache_gt_vs_global_rigid=rigid_check(c['P'][:, :2, 3], kf))}
    return dict(frame='korea_utm52', georef='gnss_global_pose', xy=xy, kf=kf,
                latlon_c=graw[:, 1:3].mean(0).tolist(), latlon_basis='measured (gps.csv mean)', evidence=ev,
                sources={f'mulran/{seq}/global_pose.csv': h,
                         f'mulran/{seq}/gps.csv': Hasher.file(DS / f'mulran/{seq}/gps.csv')},
                transform='global_pose.csv translation (UTM 52N), decimated to ~10 Hz inside Ouster scan span')


def load_helipr_local(seq, cache):
    f = DS / f'helipr/{seq}/LiDAR_GT/Velodyne_gt.txt'
    a = np.loadtxt(f, usecols=(0, 1, 2))
    c = cache_xy_time(cache)
    j, dt = nearest_time(a[:, 0], np.rint(c['t'] * 1e9))
    kf = a[j, 1:3]
    ev = {'keyframe_map': dict(rule='timestamp (s) -> nearest Velodyne_gt row (ns)',
                               time_gap_max_ms=round(float(dt.max() / 1e6), 3),
                               cache_gt_vs_local_gt_rigid=rigid_check(c['P'][:, :2, 3], kf))}
    return dict(t=a[:, 0], xy_local=a[:, 1:3], kf_local=kf, evidence=ev,
                sources={f'helipr/{seq}/LiDAR_GT/Velodyne_gt.txt': Hasher.file(f)})


def load_mcd(seq, cache):
    f = DS / f'mcd_extracted/{seq}/scan_poses.csv'
    a = np.loadtxt(f, delimiter=',', skiprows=1, usecols=(0, 1, 2))
    c = cache_xy_time(cache)
    j, dt = nearest_time(a[:, 0], np.rint(c['t'] * 1e9))
    kf = a[j, 1:3]
    ev = {'keyframe_map': dict(rule='timestamp (s) -> nearest scan_poses row (ns)',
                               time_gap_max_ms=round(float(dt.max() / 1e6), 3),
                               cache_gt_vs_global_rigid=rigid_check(c['P'][:, :2, 3], kf))}
    return dict(frame='mcd_' + seq.split('_')[0], georef='shared_local_frame', xy=a[:, 1:3], kf=kf,
                latlon_c=None, latlon_basis='metadata', evidence=ev,
                sources={f'mcd_extracted/{seq}/scan_poses.csv': Hasher.file(f)},
                transform='MCD prior-map frame (gt/pose_inW via scan_poses.csv), one row per scan')


def ins_fit(seq, gt):
    f = DS / f'helipr/{seq}/Inertial_data/inspva.csv'
    ins = []
    with open(f) as fh:
        for line in fh:
            p = line.split(',')
            ins.append((float(p[0]), float(p[1]), float(p[2])))
    ins = np.array(ins)
    E, N = latlon_to_utm(ins[:, 1], ins[:, 2], 52)
    j, dt = nearest_time(ins[:, 0], gt['t'])
    m = dt < 3e7
    P, Q = gt['xy_local'][m], np.c_[E[j[m]], N[j[m]]]
    R, t = rigid2d(P, Q)
    res = np.linalg.norm(P @ R.T + t - Q, axis=1)
    lat, lon = ins[:, 1].mean(), ins[:, 2].mean()
    return R, t, {'method': f'2D rigid fit of local GT to UTM 52N of helipr/{seq}/Inertial_data/inspva.csv',
                  'n_matched': int(m.sum()), 'rotation_deg': round(ang(R), 3),
                  'utm_grid_convergence_deg': round(convergence_deg(lat, lon, 52), 3),
                  'resid_m': {'median': round(float(np.median(res)), 3), 'p95': round(float(np.percentile(res, 95)), 3),
                              'max': round(float(res.max()), 3)},
                  'ins_latlon_mean': [float(lat), float(lon)]}, Hasher.file(f)


def shared_frame_check(first, other):
    """Does `other` share `first`'s site-local frame? (as-is NN overlap and ICP correction)."""
    B = thin(first, 1.0)
    tree = cKDTree(B)
    X = thin(other, 2.0)
    d0, _ = tree.query(X)
    R, t = np.eye(2), np.zeros(2)
    for _ in range(50):
        d, j = tree.query(X @ R.T + t)
        m = d < 40.0
        R, t = rigid2d(X[m], B[j[m]])
    return {'nn_as_is_median_m': round(float(np.median(d0)), 2), 'frac_as_is_lt_50m': round(float((d0 < 50).mean()), 4),
            'icp_rotation_deg': round(ang(R), 3), 'icp_translation_m': [round(float(v), 2) for v in t]}


def bridge_georef(hl):
    """Bounded placement of Bridge01-04 (no INS): rotation = grid convergence (+-1 deg), translation =
    feasible set where both public bridge points lie within w m of the traversed crossings."""
    t1, xy1 = hl['helipr/Bridge01']['t'], hl['helipr/Bridge01']['xy_local']
    ts = (t1 - t1[0]) / 1e9
    seg = {b: xy1[(ts >= a) & (ts <= c)] for b, (a, c) in BRIDGE_SEG_T.items()}
    cp = {b: np.array(latlon_to_utm(la, lo, 52)).ravel() for b, (la, lo, _) in BRIDGE_WIKI.items()}
    gamma = convergence_deg(37.513, 126.97, 52)
    variants = {}
    for rd in (0.0, -1.0, 1.0):
        for w in (60.0, 150.0):
            R = rot(gamma + rd)
            SH, SD = seg['Hangang'] @ R.T, seg['Dongjak'] @ R.T
            cand = cp['Hangang'] - SH
            lo, hi = cand.min(0) - w, cand.max(0) + w
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 10.0), np.arange(lo[1], hi[1], 10.0))
            T = np.c_[gx.ravel(), gy.ravel()]
            dH, _ = cKDTree(cp['Hangang'] - SH).query(T)
            dD, _ = cKDTree(cp['Dongjak'] - SD).query(T)
            T = T[(dH <= w) & (dD <= w)]
            assert len(T), ('empty feasible set', rd, w)
            c = T.mean(0)
            variants[f'rot{gamma + rd:+.3f}_w{int(w)}'] = dict(R=R, c=c, rho=float(np.hypot(*(T - c).T).max()) + 10.0,
                                                               n=int(len(T)))
    nom = variants[f'rot{gamma:+.3f}_w60']
    info = {'method': 'no INS on disk (HeLiPR download is request-form gated); rotation = UTM grid convergence '
                      '(GT axes are true-north ENU, as fitted for Town/Roundabout); translation = centre of the set '
                      'of translations putting both public bridge coordinates within w m of the traversed crossings',
            'location_source': 'HeLiPR official sequence page: Bridge01-04 "Location : Hangang & Dongjak Bridge" '
                               '(page copy results/split_audit/bridge/helipr_seq.html)',
            'control_points': {b: {'latlon': [la, lo], 'source': s} for b, (la, lo, s) in BRIDGE_WIKI.items()},
            'grid_convergence_deg': round(gamma, 3),
            'nominal_translation_utm': [round(float(v), 1) for v in nom['c']],
            'variants': {k: {'n_translations_10m_grid': v['n'], 'radius_m': round(v['rho'], 1)}
                         for k, v in variants.items()}}
    return nom, variants, info


def load_all(roster, keys):
    tr, ev = {}, {}
    hl = {}
    for k in keys + ['helipr/Town01', 'helipr/Roundabout01']:
        if k in tr or k in hl:
            continue
        ds, s = k.split('/')
        cache = cache_of(roster, k)
        log('load', k)
        if ds == 'kitti':
            tr[k] = load_kitti(s, cache)
        elif ds == 'kitti360':
            tr[k] = load_kitti360(s, cache)
        elif ds == 'nclt':
            tr[k] = load_nclt(s, cache)
        elif ds == 'mulran':
            tr[k] = load_mulran(s, cache)
        elif ds == 'mcd':
            tr[k] = load_mcd(s, cache)
        elif ds == 'helipr':
            hl[k] = load_helipr_local(s, cache)
    # HeLiPR georeference per site
    site = lambda k: ''.join(c for c in k.split('/')[1] if not c.isdigit())
    sites = {}
    for k in hl:
        sites.setdefault(site(k), []).append(k)
    hev = {}
    for st in ('Town', 'Roundabout'):
        first = f'helipr/{st}01'
        R, t, info, h = ins_fit(f'{st}01', hl[first])
        info['shared_frame_checks'] = {k: shared_frame_check(hl[first]['xy_local'], hl[k]['xy_local'])
                                       for k in sites[st] if k != first}
        hev[st] = info
        for k in sites[st]:
            tr[k] = dict(frame='korea_utm52', georef='ins_fit', xy=hl[k]['xy_local'] @ R.T + t,
                         kf=hl[k]['kf_local'] @ R.T + t, latlon_c=info['ins_latlon_mean'],
                         latlon_basis=f'measured (helipr/{st}01 INS mean)', evidence=hl[k]['evidence'],
                         sources={**hl[k]['sources'], f'helipr/{st}01/Inertial_data/inspva.csv': h},
                         transform=f'site-local GT -> UTM 52N by rigid fit to helipr/{st}01 INS (02/03 share the 01 frame)')
    M = np.vstack([thin(tr[k]['xy'], 2.0) for k in tr if k.startswith('mulran/') and 'Sejong' not in k])
    gamma = convergence_deg(36.37, 127.36, 52)
    mulran_keys = [k for k in tr if k.startswith('mulran/') and 'Sejong' not in k]
    for st in ('KAIST', 'DCC', 'Riverside'):
        H = np.vstack([thin(hl[k]['xy_local'], 5.0) for k in sites[st]])
        t0, best, second = fft_translation_search(H @ rot(gamma).T, M)
        R, t, d = icp2d(H, M, rot(gamma), t0)
        land = {}
        for k in mulran_keys:
            dd, _ = cKDTree(thin(tr[k]['xy'], 2.0)).query(H @ R.T + t)
            land[k] = round(float((dd < 5).mean()), 4)
        lat, lon = utm_to_latlon(*(H @ R.T + t).mean(0), 52)
        hev[st] = {'method': 'FFT translation search (rotation fixed to grid convergence) onto the MulRan Daejeon '
                             'union (DCC/KAIST/Riverside 01-03, global_pose UTM 52N), then rigid ICP (gate 5 m)',
                   'icp_rotation_deg': round(ang(R), 3), 'grid_convergence_deg': round(gamma, 3),
                   'fft_best_frac_within5m': round(best, 4), 'fft_second_peak_frac_within5m': round(second, 4),
                   'icp_frac_within_2m': round(float((d < 2).mean()), 4),
                   'icp_frac_within_5m': round(float((d < 5).mean()), 4),
                   'icp_median_resid_m': round(float(np.median(d)), 3), 'per_mulran_seq_frac_within5m': land}
        for k in sites[st]:
            tr[k] = dict(frame='korea_utm52', georef='registered_to_mulran', xy=hl[k]['xy_local'] @ R.T + t,
                         kf=hl[k]['kf_local'] @ R.T + t, latlon_c=[float(lat[0]), float(lon[0])],
                         latlon_basis='derived (registration onto MulRan)', evidence=hl[k]['evidence'],
                         sources=dict(hl[k]['sources']),
                         transform='site-local GT -> UTM 52N by registration onto MulRan Daejeon global_pose')
    nom, variants, binfo = bridge_georef(hl)
    binfo['shared_frame_checks'] = {k: shared_frame_check(hl['helipr/Bridge01']['xy_local'], hl[k]['xy_local'])
                                    for k in sites['Bridge'] if k != 'helipr/Bridge01'}
    hev['Bridge'] = binfo
    for k in sites['Bridge']:
        xyn = hl[k]['xy_local'] @ nom['R'].T + nom['c']
        lat, lon = utm_to_latlon(*xyn.mean(0), 52)
        tr[k] = dict(frame='helipr_bridge', georef='bounded_public_landmarks', xy=xyn,
                     kf=hl[k]['kf_local'] @ nom['R'].T + nom['c'], xy_local=hl[k]['xy_local'],
                     latlon_c=[float(lat[0]), float(lon[0])], latlon_basis='bounded (public bridge coordinates)',
                     evidence=hl[k]['evidence'], sources=dict(hl[k]['sources']),
                     transform='site-local GT; nominal UTM 52N placement from two public bridge coordinates')
    ev['helipr_georef'] = hev
    return tr, ev, variants


# ---------------------------------------------------------------- metrics
def prep(v):
    v['dense'] = densify(v['xy'])
    v['tree'] = cKDTree(v['dense'])
    v['xy5'] = thin(v['xy'], 5.0)
    v['len_km'] = round(float(np.hypot(*np.diff(v['xy5'], axis=0).T).sum() / 1000.0), 3)
    c = v['xy'].mean(0)
    v['radius_km'] = float(np.hypot(*(v['xy'] - c).T).max() / 1000.0)


def directed(A, B):
    dp, _ = B['tree'].query(A['xy'])
    dl, _ = B['tree'].query(A['xy5'])
    dk, _ = B['tree'].query(A['kf'])
    r = {'n_pose': int(len(dp)), 'n_len5m': int(len(dl)), 'n_kf': int(len(dk)),
         'min_dist_m': round(float(min(dp.min(), dk.min())), 3)}
    for th in THR:
        r[f'frac_pose_{th}m'] = float((dp < th).mean())
        r[f'frac_len_{th}m'] = float((dl < th).mean())
        r[f'frac_kf_{th}m'] = float((dk < th).mean())
    r['count_pose_50m'] = int((dp < 50).sum())
    r['count_kf_50m'] = int((dk < 50).sum())
    r['max_frac_50m'] = max(r['frac_pose_50m'], r['frac_len_50m'], r['frac_kf_50m'])
    return r


BOUND_STEP = 2.0
_BOUND_MEMO = {}


def bridge_bound(kor, brg, tr):
    """Worst-case lower bound of the min distance between a Korean route and a Bridge route over all
    feasible Bridge placements. Both routes are arc-thinned to BOUND_STEP m (every omitted sample lies
    within BOUND_STEP m of a kept one), so 2*BOUND_STEP is subtracted to keep the value a lower bound."""
    if (kor, brg) not in _BOUND_MEMO:
        A = tr[kor]
        At = cKDTree(thin(np.vstack([A['dense'], A['kf']]), BOUND_STEP))
        U = thin(tr[brg]['xy_local'], BOUND_STEP)
        lb = np.inf
        for v in VARIANTS.values():
            lb = min(lb, float(At.query(U @ v['R'].T + v['c'])[0].min()) - v['rho'])
        _BOUND_MEMO[(kor, brg)] = max(0.0, lb - 2 * BOUND_STEP)
    return _BOUND_MEMO[(kor, brg)]


def geo_lower_bound_km(A, B, ka, kb):
    """City-level separation: centroid great-circle distance minus route radii and metadata uncertainty."""
    def centre(v, k):
        if v['latlon_c'] is not None:
            unc = 1.0 if v['georef'] == 'bounded_public_landmarks' else 0.1
            return tuple(v['latlon_c']), unc, v['latlon_basis']
        name, ll, unc, src = CITY_META[v['frame']]
        return ll, unc, f'metadata: {src}'
    (la, ua, ba), (lb_, ub, bb) = centre(A, ka), centre(B, kb)
    d = haversine_km(la, lb_)
    return d, d - A['radius_km'] - B['radius_km'] - ua - ub, ba, bb


def city_name(v, roster_row):
    return f"{roster_row['city']}, {roster_row['country']}"


def brute_force_check(A, B, n_max=None, chunk=128):
    """Chunked exact NN (no N^2 matrix) vs cKDTree on A pose samples."""
    X = A['xy'] if n_max is None or len(A['xy']) <= n_max else \
        A['xy'][np.random.default_rng(0).choice(len(A['xy']), n_max, replace=False)]
    D = B['dense']
    best = np.empty(len(X))
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        best[i:i + chunk] = np.sqrt(((x[:, None, :] - D[None, :, :]) ** 2).sum(-1).min(1))
    kd, _ = B['tree'].query(X)
    return {'n_points': int(len(X)), 'n_ref': int(len(D)), 'max_abs_diff_m': float(np.abs(best - kd).max()),
            'frac_50m_brute': float((best < 50).mean()), 'frac_50m_kdtree': float((kd < 50).mean())}


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roster_doc = json.load(open(OUT / 'roster.json'))
    assert roster_doc['checks']['all_pass'], 'roster checks failed; run review_v23_20261107_roster.py'
    roster = {r['key']: r for r in roster_doc['roster']}
    prot = json.load(open(PROTOCOL))
    role = {k: r for r in ('train', 'val', 'test') for k in prot[r]}
    keys = list(role)
    assert len(keys) == 63 and set(keys) == set(roster)
    tr, ev, variants = load_all(roster, keys)
    VARIANTS.update(variants)
    for k in list(tr):
        if k not in role:
            tr.pop(k)
    for v in tr.values():
        prep(v)
    log('trajectories ready', len(tr))
    bridge_keys = [k for k in keys if tr[k]['frame'] == 'helipr_bridge']

    def pair_class(a, b):
        fa, fb = tr[a]['frame'], tr[b]['frame']
        if fa == fb:
            g = {tr[a]['georef'], tr[b]['georef']}
            if 'registered_to_mulran' in g:
                return 'measured_registered'
            if g <= {'shared_local_frame'}:
                return 'measured_shared_local'
            return 'measured_global'
        if {fa, fb} == {'helipr_bridge', 'korea_utm52'}:
            return 'bounded'
        return 'geographic_exclusion'

    rows = []
    for a in keys:
        for b in keys:
            if a == b or role[a] == role[b]:
                continue
            cls = pair_class(a, b)
            r = dict(a=a, a_split=role[a], b=b, b_split=role[b], role_pair='-'.join(
                sorted((role[a], role[b]), key=['train', 'val', 'test'].index)),
                comparison=cls, frame_a=tr[a]['frame'], frame_b=tr[b]['frame'],
                georef_a=tr[a]['georef'], georef_b=tr[b]['georef'],
                city_a=city_name(tr[a], roster[a]), city_b=city_name(tr[b], roster[b]))
            if cls.startswith('measured'):
                r.update(directed(tr[a], tr[b]))
            elif cls == 'bounded':
                kor, brg = (a, b) if tr[a]['frame'] == 'korea_utm52' else (b, a)
                lb = bridge_bound(kor, brg, tr)
                nomr = directed(tr[a], tr[b])
                r.update({f'nominal_{k}': v for k, v in nomr.items() if k.startswith(('frac_pose', 'min_dist', 'frac_kf'))})
                r['min_dist_lower_bound_m'] = round(lb, 1)
                r['max_frac_50m_upper_bound'] = 0.0 if lb >= RULE_M else None
                r['basis'] = 'Bridge placed by public bridge coordinates; worst case over rotation +-1 deg and feasible translations'
            else:
                d, lbk, ba, bb = geo_lower_bound_km(tr[a], tr[b], a, b)
                r['centroid_distance_km'] = round(d, 1)
                r['min_dist_lower_bound_km'] = round(lbk, 1)
                r['max_frac_50m_upper_bound'] = 0.0 if lbk * 1000 >= RULE_M else None
                r['basis'] = f'A: {ba}; B: {bb}'
            rows.append(r)
    log('directed cross-split pairs', len(rows))

    # per-sequence: fraction within 50 m of the union of each other split (same metric frame only)
    per_seq = []
    for a in keys:
        rec = dict(key=a, split=role[a], frame=tr[a]['frame'], georef=tr[a]['georef'], len_km=tr[a]['len_km'],
                   n_pose=int(len(tr[a]['xy'])), n_kf=int(len(tr[a]['kf'])))
        for other in ('train', 'val', 'test'):
            if other == role[a]:
                continue
            comp = [b for b in keys if role[b] == other and tr[b]['frame'] == tr[a]['frame']]
            if comp:
                U = {'tree': cKDTree(np.vstack([tr[b]['dense'] for b in comp]))}
                m = directed(tr[a], U)
                rec[f'vs_{other}_union'] = dict(n_seq=len(comp), frac_pose_50m=m['frac_pose_50m'],
                                                frac_len_50m=m['frac_len_50m'], frac_kf_50m=m['frac_kf_50m'],
                                                min_dist_m=m['min_dist_m'], comparison='same frame' +
                                                (' (nominal Bridge placement)' if tr[a]['frame'] == 'helipr_bridge' else ''))
            others = [r for r in rows if r['a'] == a and r['b_split'] == other]
            rec[f'vs_{other}_classes'] = {c: sum(r['comparison'] == c for r in others)
                                          for c in sorted({r['comparison'] for r in others})}
            mm = [r for r in others if r['comparison'].startswith('measured')]
            rec[f'vs_{other}_max_pair'] = (max(mm, key=lambda r: r['max_frac_50m'])['b'] if mm else None)
            rec[f'vs_{other}_max_pair_frac50'] = (max(r['max_frac_50m'] for r in mm) if mm else None)
            bd = [r for r in others if r['comparison'] == 'bounded']
            rec[f'vs_{other}_bounded_min_lb_m'] = min((r['min_dist_lower_bound_m'] for r in bd), default=None)
            ge = [r for r in others if r['comparison'] == 'geographic_exclusion']
            rec[f'vs_{other}_geo_min_lb_km'] = min((r['min_dist_lower_bound_km'] for r in ge), default=None)
        per_seq.append(rec)
    # for Bridge (train) sequences the union fraction against the Korean val/test is not measurable;
    # the bound is what counts. Mark explicitly.

    # role-pair summary
    summary = {}
    for ra, rb in ROLE_PAIRS:
        sel = [r for r in rows if {r['a_split'], r['b_split']} == {ra, rb}]
        und = {}
        for r in sel:
            und.setdefault(tuple(sorted((r['a'], r['b']))), r['comparison'])
        cnt = {}
        for c in und.values():
            cnt[c] = cnt.get(c, 0) + 1
        meas = [r for r in sel if r['comparison'].startswith('measured')]
        top = max(meas, key=lambda r: r['max_frac_50m']) if meas else None
        near = min(meas, key=lambda r: r['min_dist_m']) if meas else None
        bd = [r for r in sel if r['comparison'] == 'bounded']
        ge = [r for r in sel if r['comparison'] == 'geographic_exclusion']
        # union rule per sequence for this role pair
        uni = []
        for s in per_seq:
            if s['split'] in (ra, rb):
                o = rb if s['split'] == ra else ra
                u = s.get(f'vs_{o}_union')
                if u:
                    uni.append((max(u['frac_pose_50m'], u['frac_len_50m'], u['frac_kf_50m']), s['key'], o))
        umax = max(uni) if uni else None
        summary[f'{ra}-{rb}'] = dict(
            unordered_pairs=len(und), directed_pairs=len(sel), classes=cnt,
            unverified_pairs=cnt.get('unverified', 0),
            measured_max_frac_50m=(top['max_frac_50m'] if top else None),
            measured_max_frac_50m_pair=(f"{top['a']} -> {top['b']}" if top else None),
            measured_max_detail=({k: top[k] for k in ('frac_pose_50m', 'frac_len_50m', 'frac_kf_50m',
                                                      'count_pose_50m', 'n_pose', 'min_dist_m')} if top else None),
            measured_min_dist_m=(near['min_dist_m'] if near else None),
            measured_min_dist_pair=(f"{near['a']} -> {near['b']}" if near else None),
            union_max_frac_50m=(umax[0] if umax else None),
            union_max_seq=(f'{umax[1]} vs {umax[2]} union' if umax else None),
            bounded_min_lower_bound_m=min((r['min_dist_lower_bound_m'] for r in bd), default=None),
            bounded_min_pair=(min(bd, key=lambda r: r['min_dist_lower_bound_m'])['a'] + ' / ' +
                              min(bd, key=lambda r: r['min_dist_lower_bound_m'])['b'] if bd else None),
            geographic_min_lower_bound_km=min((r['min_dist_lower_bound_km'] for r in ge), default=None),
            rule_pass_measured=(top['max_frac_50m'] <= RULE_FRAC if top else True),
            rule_pass_union=(umax[0] <= RULE_FRAC if umax else True),
            rule_pass_bounded=all(r['max_frac_50m_upper_bound'] == 0.0 for r in bd),
            rule_pass_geographic=all(r['max_frac_50m_upper_bound'] == 0.0 for r in ge))
        s = summary[f'{ra}-{rb}']
        s['rule_pass'] = bool(s['rule_pass_measured'] and s['rule_pass_union'] and s['rule_pass_bounded']
                              and s['rule_pass_geographic'] and s['unverified_pairs'] == 0)
    log('summary', json.dumps({k: (v['measured_max_frac_50m'], v['rule_pass']) for k, v in summary.items()}))

    # same-split overlap groups (principle: overlapping sequences in one split)
    same = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if tr[a]['frame'] == tr[b]['frame'] and role[a] == role[b]:
                ab, ba = directed(tr[a], tr[b]), directed(tr[b], tr[a])
                same.append(dict(a=a, b=b, split=role[a], frame=tr[a]['frame'],
                                 frac_pose_50m_ab=ab['frac_pose_50m'], frac_pose_50m_ba=ba['frac_pose_50m'],
                                 min_dist_m=min(ab['min_dist_m'], ba['min_dist_m'])))
    # connected components of the >1% overlap graph over ALL comparable pairs (any split)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in same:
        if max(r['frac_pose_50m_ab'], r['frac_pose_50m_ba']) > RULE_FRAC:
            parent[find(r['a'])] = find(r['b'])
    for r in rows:
        if r['comparison'].startswith('measured') and r['max_frac_50m'] > RULE_FRAC:
            parent[find(r['a'])] = find(r['b'])
    comps = {}
    for k in keys:
        comps.setdefault(find(k), []).append(k)
    groups = [dict(members=v, splits=sorted({role[k] for k in v})) for v in comps.values() if len(v) > 1]
    groups_ok = all(len(g['splits']) == 1 for g in groups)

    # verification: brute force vs cKDTree; reverse direction; reuse consistency with the pd2026 cache
    meas = sorted([r for r in rows if r['comparison'].startswith('measured')], key=lambda r: r['min_dist_m'])
    bf = {}
    for r in meas[:6]:
        bf[f"{r['a']} -> {r['b']} (full)"] = brute_force_check(tr[r['a']], tr[r['b']])
    for r in meas[6::max(1, len(meas) // 12)]:
        bf[f"{r['a']} -> {r['b']} (sample)"] = brute_force_check(tr[r['a']], tr[r['b']], n_max=1500)
    bf_ok = all(v['max_abs_diff_m'] < 1e-6 and v['frac_50m_brute'] == v['frac_50m_kdtree'] for v in bf.values())
    reverse = {}
    for r in meas:
        rr = next(x for x in rows if x['a'] == r['b'] and x['b'] == r['a'])
        reverse[f"{r['a']}|{r['b']}"] = (r['min_dist_m'], rr['min_dist_m'])
    # min distance symmetric up to sampling (pose vs dense polyline)
    rev_gap = max(abs(a - b) for a, b in reverse.values()) if reverse else 0.0
    old = {}
    pk = REPO / 'results/split_audit/_traj_cache.pkl'
    if pk.exists():
        otr, _ = pickle.load(open(pk, 'rb'))
        for k in keys:
            mine = tr[k]['xy_local'] if tr[k]['frame'] == 'helipr_bridge' else tr[k]['xy']   # old cache: Bridge local
            if k in otr and len(otr[k]['xy']) == len(mine) and otr[k]['frame'].replace(
                    'helipr_bridge_local', 'helipr_bridge') == tr[k]['frame']:
                d = float(np.abs(otr[k]['xy'] - mine).max())
                old[k] = round(d, 6)
            elif k in otr:
                old[k] = f"shape/frame differs: old {len(otr[k]['xy'])} {otr[k]['frame']} vs new {len(tr[k]['xy'])} {tr[k]['frame']}"
            else:
                old[k] = 'absent from pd2026 cache'

    # ---------------- write
    cols = ['a', 'a_split', 'b', 'b_split', 'role_pair', 'comparison', 'frame_a', 'frame_b', 'georef_a', 'georef_b',
            'city_a', 'city_b', 'n_pose', 'n_len5m', 'n_kf', 'min_dist_m'] + \
        [f'frac_{m}_{t}m' for t in THR for m in ('pose', 'len', 'kf')] + \
        ['count_pose_50m', 'count_kf_50m', 'max_frac_50m', 'nominal_min_dist_m', 'nominal_frac_pose_50m',
         'nominal_frac_kf_50m', 'min_dist_lower_bound_m', 'centroid_distance_km', 'min_dist_lower_bound_km',
         'max_frac_50m_upper_bound', 'basis']
    with open(OUT / 'geo_pairs_cross_split.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6f}' if isinstance(v, float) and k.startswith(('frac', 'max_frac', 'nominal_frac'))
                            else v) for k, v in r.items()})
    with open(OUT / 'geo_same_split_pairs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(same[0]))
        w.writeheader()
        w.writerows(same)
    seq_cols = ['key', 'split', 'frame', 'georef', 'len_km', 'n_pose', 'n_kf']
    with open(OUT / 'geo_per_sequence.csv', 'w', newline='') as f:
        flat = []
        for s in per_seq:
            d = {k: s[k] for k in seq_cols}
            for o in ('train', 'val', 'test'):
                if o == s['split']:
                    continue
                u = s.get(f'vs_{o}_union')
                d[f'vs_{o}_union_frac_pose_50m'] = u['frac_pose_50m'] if u else ''
                d[f'vs_{o}_union_frac_kf_50m'] = u['frac_kf_50m'] if u else ''
                d[f'vs_{o}_union_min_dist_m'] = u['min_dist_m'] if u else ''
                d[f'vs_{o}_classes'] = json.dumps(s[f'vs_{o}_classes'])
                d[f'vs_{o}_bounded_min_lb_m'] = s[f'vs_{o}_bounded_min_lb_m'] if s[f'vs_{o}_bounded_min_lb_m'] is not None else ''
                d[f'vs_{o}_geo_min_lb_km'] = s[f'vs_{o}_geo_min_lb_km'] if s[f'vs_{o}_geo_min_lb_km'] is not None else ''
            flat.append(d)
        allc = []
        for d in flat:
            for k in d:
                if k not in allc:
                    allc.append(k)
        w = csv.DictWriter(f, fieldnames=allc)
        w.writeheader()
        w.writerows(flat)
    frames = {k: dict(split=role[k], frame=tr[k]['frame'], georef=tr[k]['georef'], transform=tr[k]['transform'],
                      latlon_c=tr[k]['latlon_c'], latlon_basis=tr[k]['latlon_basis'], len_km=tr[k]['len_km'],
                      radius_km=round(tr[k]['radius_km'], 3), n_pose=int(len(tr[k]['xy'])),
                      n_dense=int(len(tr[k]['dense'])), n_kf=int(len(tr[k]['kf'])),
                      sources=tr[k]['sources'], evidence=tr[k]['evidence']) for k in keys}
    kres = []
    for k in keys:
        e = frames[k]['evidence']['keyframe_map']
        rc = e.get('cache_gt_vs_global_rigid') or e.get('cache_gt_vs_local_gt_rigid')
        kres.append((rc['resid_max_m'], k))
    out = dict(
        _meta=dict(date='2026-11-07', script='scripts/split_audit/review_v23_20261107_geo.py',
                   script_sha256=Hasher.file(Path(__file__)), roster=str((OUT / 'roster.json').relative_to(REPO)),
                   roster_sha256=Hasher.file(OUT / 'roster.json'),
                   authority=str(PROTOCOL.relative_to(REPO)), authority_sha256=Hasher.file(PROTOCOL),
                   rule=f'per sequence: fraction of trajectory within {RULE_M:.0f} m of another split <= {RULE_FRAC:.0%}; '
                        'overlapping sequences in one split',
                   distance='2D horizontal Euclidean in the common metric frame (z ignored)',
                   reference_polyline=f'B samples linearly densified to <= {DENSE_STEP} m, no interpolation across gaps > {MAX_GAP} m',
                   samples='frac_pose: A GT pose samples (~10 Hz); frac_len: 5 m arc-length samples; frac_kf: cache keyframes',
                   thresholds_m=THR, runtime_s=round(time.time() - T0, 1)),
        summary=summary, same_split_overlap_groups=groups, same_split_groups_ok=groups_ok,
        verification=dict(brute_force_vs_kdtree=bf, brute_force_ok=bf_ok,
                          reverse_direction_max_min_dist_gap_m=round(rev_gap, 3),
                          keyframe_mapping_max_rigid_resid_m=max(kres), keyframe_mapping_resid_by_seq={k: v for v, k in kres},
                          pd2026_traj_cache_max_abs_xy_diff_m=old),
        evidence=ev, frames=frames, per_sequence=per_seq)
    (OUT / 'geo_audit.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    log('wrote', OUT / 'geo_audit.json')


if __name__ == '__main__':
    main()
