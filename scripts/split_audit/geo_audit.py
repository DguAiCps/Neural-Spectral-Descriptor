#!/usr/bin/env python3
"""Geographic place-disjointness audit of the pd2026 train/val/test split.

READ-ONLY w.r.t. the repo and /mnt/d/NSD_datasets. Everything is recomputed
from coordinates:
  * KITTI odometry: raw-drive OXTS lat/lon fetched independently from the
    official KITTI S3 zips (fetch_kitti_oxts.py -> kitti_raw_oxts/), devkit
    frame ranges, validated against dataset/poses/XX.txt by a 2D rigid fit and
    against results/kitti_odometry_geo_reference.json.
  * KITTI-360: on-disk data_poses/<drive>/oxts lat/lon, restricted to the
    velodyne frames inside poses.txt coverage (what the loader uses).
  -> both converted to UTM 32N (Karlsruhe).
  * MulRan: global_pose.csv (verified UTM 52N against gps.csv).
  * HeLiPR: LiDAR_GT/Velodyne_gt.txt = per-SITE local ENU frame (origin at the
    first pose of the site's first sequence). Town/Roundabout are georeferenced
    to UTM 52N with their inspva.csv (INS lat/lon). Daejeon sites (no INS on
    disk) are registered onto the MulRan Daejeon union (UTM 52N) by FFT
    translation search + free rigid ICP. Bridge: no georeference possible.
  * NCLT: one shared campus frame across sessions (verified by overlap).
  * MCD: local prior-map frames per campus (KTH / TUHH), no lat/lon.
Pairs are compared only inside a common metric frame; otherwise city-level.
Output: results/split_audit/geo_overlap.json.
"""
import json, os, sys, math
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy import ndimage, signal

sys.dont_write_bytecode = True           # never write __pycache__ into the repo
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geoutil import latlon_to_utm        # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results' / 'split_audit'
DS = Path('/mnt/d/NSD_datasets')
KOXTS = DS / 'kitti_raw_oxts'   # fetched by fetch_kitti_oxts.py
THR = (25, 50, 200)

# KITTI odometry devkit readme: seq -> (raw drive, first frame, last frame)
ODOM = {'00': ('2011_10_03_drive_0027', 0, 4540), '01': ('2011_10_03_drive_0042', 0, 1100),
        '02': ('2011_10_03_drive_0034', 0, 4660), '05': ('2011_09_30_drive_0018', 0, 2760),
        '06': ('2011_09_30_drive_0020', 0, 1100), '07': ('2011_09_30_drive_0027', 0, 1100),
        '08': ('2011_09_30_drive_0028', 1100, 5170), '09': ('2011_09_30_drive_0033', 0, 1590)}
K360 = ['0000', '0004', '0006', '0002', '0009']   # 0002/0009 = info only (published protocol)

# city-level metadata (used only for cross-frame pairs); lat/lon from data where available
CITY_META = {
    'nclt_campus': ('Ann Arbor, MI (UMich North Campus)', 42.2932, -83.7097, 'NCLT docs/paper (no GPS in GT csv)'),
    'mcd_kth': ('Stockholm (KTH campus)', 59.3498, 18.0707, 'MCD paper metadata (no GPS in pose_inW)'),
    'mcd_tuhh': ('Hamburg (TUHH campus)', 53.4604, 9.9690, 'MCD paper metadata (no GPS in pose_inW)'),
    'helipr_bridge_local': ('Bridge site, reportedly Seoul metro', 37.55, 126.95,
                            'HeLiPR paper: 3 distinct areas; no INS on disk -> location unverified'),
}


# ---------------------------------------------------------------- split
def load_split():
    cfg = yaml.safe_load(open(REPO / 'configs/training_place_disjoint.yaml'))
    roles = {}
    for role in ('train', 'val', 'test'):
        for e in cfg['data']['datasets'][role]:
            for s in e['sequences']:
                roles.setdefault(f"{e['type']}/{s}", set()).add(role)
    man = json.load(open(REPO / 'scripts/_pd_test_cache_manifest.json'))
    test42 = [f"{e['sensor']}/{e['sequence']}" for e in man['sequences']]
    sys.path.insert(0, str(REPO / 'scripts'))
    import _pdb_strict_rev3 as S
    strict = [f"{a}/{b}" for a, b in S.strict_roster()]
    groups = {f"{a}/{b}": S.group_of(a, b) for a, b in S.strict_roster()}
    excluded = sorted(f"{a}/{b}" for a, b in S.STRICT_EXCLUDE)
    for k in test42:
        roles.setdefault(k, set()).add('test')
    return {k: sorted(v) for k, v in roles.items()}, test42, strict, groups, excluded


# ---------------------------------------------------------------- helpers
def thin(xy, step):
    """Greedy arc-length thinning: keep a point once it is >= step from the last kept."""
    keep = [0]
    last = xy[0]
    for i in range(1, len(xy)):
        if (xy[i, 0] - last[0]) ** 2 + (xy[i, 1] - last[1]) ** 2 >= step * step:
            keep.append(i)
            last = xy[i]
    return xy[keep]


def rigid2d(P, Q, allow_reflection=False):
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, d if not allow_reflection else 1])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def ang(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def convergence_deg(lat, lon, zone):
    lon0 = (zone - 1) * 6 - 180 + 3
    return float(np.degrees(np.arctan(np.tan(np.radians(lon - lon0)) * np.sin(np.radians(lat)))))


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(h)))


def scan_stems(d):
    return np.array(sorted(int(e.name.split('.')[0]) for e in os.scandir(d) if e.name.endswith('.bin')),
                    dtype=np.int64)


def read_oxts(files):
    out = []
    for f in files:
        v = open(f).read().split()
        out.append((float(v[0]), float(v[1])))
    return np.array(out)


# ---------------------------------------------------------------- loaders
def load_kitti(evidence):
    trajs = {}
    ref = json.load(open(REPO / 'results/kitti_odometry_geo_reference.json'))['sequences']
    val = {}
    for seq, (drive, a, b) in ODOM.items():
        files = sorted((KOXTS / f'{drive}_sync/oxts/data').glob('*.txt'))
        assert len(files) > b, (seq, len(files))
        ll = read_oxts(files[a:b + 1])
        E, N = latlon_to_utm(ll[:, 0], ll[:, 1], 32)
        xy = np.c_[E, N]
        # validation 1: odometry GT poses (camera x-z plane) vs GPS, rigid 2D
        P = np.loadtxt(DS / f'kitti/dataset/poses/{seq}.txt')
        assert len(P) == b - a + 1, (seq, len(P), b - a + 1)
        pxz = np.c_[P[:, 11], -P[:, 3]]           # (forward z, left -x): right-handed top view
        R, t = rigid2d(pxz, xy)
        res = np.linalg.norm(pxz @ R.T + t - xy, axis=1)
        v = {'raw_drive': drive, 'frames': [a, b], 'n_frames': b - a + 1,
             'odom_pose_vs_gps_rigid_fit_resid_m': {'median': float(np.median(res)),
                                                    'p95': float(np.percentile(res, 95)),
                                                    'max': float(res.max())}}
        # validation 2: committed geo reference (every 5th frame)
        if seq in ref:
            rp = np.array(ref[seq]['polyline_latlon'])
            mine = ll[::ref[seq]['subsample']]
            if len(rp) == len(mine):
                dd = np.abs(rp - mine).max() * 111e3
                v['committed_geo_reference_max_dev_m'] = float(dd)
            else:
                v['committed_geo_reference_len_mismatch'] = [len(rp), len(mine)]
        else:
            v['committed_geo_reference'] = 'ABSENT (seq not in results/kitti_odometry_geo_reference.json)'
        val[seq] = v
        trajs[f'kitti/{seq}'] = dict(frame='karlsruhe_utm32', xy=xy, latlon_c=ll.mean(0).tolist(),
                                     src=f'KITTI raw OXTS {drive} frames {a}-{b} (S3 range fetch)')
    evidence['kitti_odometry_validation'] = val
    for d in K360:
        base = DS / f'kitti360/data_poses/2013_05_28_drive_{d}_sync'
        pf = np.loadtxt(DS / f'kitti360/2013_05_28_drive_{d}_sync/poses.txt', usecols=0).astype(int)
        vel = scan_stems(DS / f'kitti360/2013_05_28_drive_{d}_sync/velodyne_points/data') \
            if (DS / f'kitti360/2013_05_28_drive_{d}_sync/velodyne_points/data').exists() else None
        files = sorted((base / 'oxts/data').glob('*.txt'))
        if vel is not None:
            fr = vel[(vel >= pf[0]) & (vel <= pf[-1])]
            fr = fr[fr < len(files)]
            src_note = 'velodyne frames within poses.txt coverage'
        else:
            fr = np.arange(pf[0], min(pf[-1], len(files) - 1) + 1)
            src_note = 'poses.txt coverage (no velodyne on disk)'
        ll = read_oxts([files[i] for i in fr])
        E, N = latlon_to_utm(ll[:, 0], ll[:, 1], 32)
        trajs[f'kitti360/{d}'] = dict(frame='karlsruhe_utm32', xy=np.c_[E, N], latlon_c=ll.mean(0).tolist(),
                                      src=f'KITTI-360 data_poses oxts, {src_note}, {len(fr)} frames')
    return trajs


def load_nclt():
    trajs = {}
    for s in ['2012-01-08', '2012-02-04', '2012-03-17', '2012-05-11', '2012-08-04',
              '2012-11-04', '2012-11-16', '2013-01-10', '2013-02-23']:
        a = np.loadtxt(DS / f'nclt/{s}/groundtruth_{s}.csv', delimiter=',', usecols=(0, 1, 2))
        a = a[np.isfinite(a).all(1)]
        st = scan_stems(DS / f'nclt/{s}/velodyne_sync')
        a = a[(a[:, 0] >= st[0]) & (a[:, 0] <= st[-1])]
        a = a[::10]                                   # ~100 Hz -> ~10 Hz
        trajs[f'nclt/{s}'] = dict(frame='nclt_campus', xy=a[:, 1:3],
                                  src=f'nclt/{s}/groundtruth_{s}.csv within velodyne_sync time span')
    return trajs


def load_mulran(evidence):
    trajs = {}
    ev = {}
    for s in ['DCC01', 'DCC02', 'DCC03', 'KAIST01', 'KAIST02', 'KAIST03',
              'Riverside01', 'Riverside02', 'Riverside03', 'Sejong01', 'Sejong02', 'Sejong03']:
        p = np.loadtxt(DS / f'mulran/{s}/global_pose.csv', delimiter=',', usecols=(0, 4, 8))
        p = p[np.argsort(p[:, 0])]
        st = scan_stems(DS / f'mulran/{s}/Ouster')
        p = p[(p[:, 0] >= st[0] - 1e8) & (p[:, 0] <= st[-1] + 1e8)]
        g = np.loadtxt(DS / f'mulran/{s}/gps.csv', delimiter=',', usecols=(0, 1, 2))
        E, N = latlon_to_utm(g[:, 1], g[:, 2], 52)
        idx = np.searchsorted(p[:, 0], g[:, 0]).clip(0, len(p) - 1)
        m = np.abs(p[idx, 0] - g[:, 0]) < 5e7
        d = np.hypot(E[m] - p[idx[m], 1], N[m] - p[idx[m], 2])
        ev[s] = {'gps_vs_global_pose_utm52_diff_m': {'median': float(np.median(d)),
                                                     'p90': float(np.percentile(d, 90)), 'n': int(m.sum())}}
        dec = max(1, int(round(0.1 / np.median(np.diff(p[:, 0]) / 1e9))))
        trajs[f'mulran/{s}'] = dict(frame='korea_utm52', xy=p[::dec, 1:3], latlon_c=g[:, 1:3].mean(0).tolist(),
                                    src=f'mulran/{s}/global_pose.csv (UTM52N) within Ouster scan span')
    evidence['mulran_frame_check'] = ev
    return trajs


def load_helipr_local():
    trajs = {}
    for s in sorted(os.listdir(DS / 'helipr')):
        f = DS / f'helipr/{s}/LiDAR_GT/Velodyne_gt.txt'
        if s == 'Sample' or not f.exists():
            continue
        a = np.loadtxt(f, usecols=(0, 1, 2))
        trajs[f'helipr/{s}'] = dict(xy_local=a[:, 1:3], t=a[:, 0], src=f'helipr/{s}/LiDAR_GT/Velodyne_gt.txt')
    return trajs


def helipr_ins_fit(seq, gt):
    ins = []
    with open(DS / f'helipr/{seq}/Inertial_data/inspva.csv') as f:
        for line in f:
            p = line.split(',')
            ins.append((float(p[0]), float(p[1]), float(p[2])))
    ins = np.array(ins)
    E, N = latlon_to_utm(ins[:, 1], ins[:, 2], 52)
    idx = np.searchsorted(ins[:, 0], gt['t']).clip(0, len(ins) - 1)
    m = np.abs(ins[idx, 0] - gt['t']) < 3e7
    P = gt['xy_local'][m]
    Q = np.c_[E[idx[m]], N[idx[m]]]
    R, t = rigid2d(P, Q)
    res = np.linalg.norm(P @ R.T + t - Q, axis=1)
    lat, lon = ins[:, 1].mean(), ins[:, 2].mean()
    return R, t, {'n_matched': int(m.sum()), 'fit_rotation_deg': ang(R),
                  'utm_grid_convergence_deg': convergence_deg(lat, lon, 52),
                  'resid_m': {'median': float(np.median(res)), 'p95': float(np.percentile(res, 95)),
                              'max': float(res.max())},
                  'scale_ratio': float(np.sqrt(((Q - Q.mean(0)) ** 2).sum() / ((P - P.mean(0)) ** 2).sum())),
                  'ins_latlon_mean': [float(lat), float(lon)]}


def fft_translation_search(H, M, res=2.0, dil=5.0, excl=100.0):
    """Best translation t so that H + t lies within `dil` m of M. Returns t, frac, 2nd-peak frac."""
    pad = 50.0
    m0 = M.min(0) - pad
    shape = np.ceil((M.max(0) + pad - m0) / res).astype(int) + 1
    occ = np.zeros(shape, bool)
    ij = np.floor((M - m0) / res).astype(int)
    occ[ij[:, 0], ij[:, 1]] = True
    mask = (ndimage.distance_transform_edt(~occ) * res <= dil).astype(np.float32)
    h0 = H.min(0)
    hs = np.ceil((H.max(0) - h0) / res).astype(int) + 1
    Hr = np.zeros(hs, np.float32)
    hij = np.floor((H - h0) / res).astype(int)
    np.add.at(Hr, (hij[:, 0], hij[:, 1]), 1.0)
    corr = signal.fftconvolve(mask, Hr[::-1, ::-1], mode='full')
    n = np.unravel_index(np.argmax(corr), corr.shape)
    best = corr[n] / len(H)
    s = np.array(n) - (np.array(hs) - 1)
    t = m0 + s * res - h0
    # second peak outside `excl` m of the best
    c2 = corr.copy()
    r = int(excl / res)
    c2[max(0, n[0] - r):n[0] + r + 1, max(0, n[1] - r):n[1] + r + 1] = -np.inf
    second = c2.max() / len(H)
    return t, float(best), float(second)


def icp2d(H, M, R, t, iters=60, gate=5.0):
    tree = cKDTree(M)
    for _ in range(iters):
        X = H @ R.T + t
        d, j = tree.query(X)
        m = d < gate
        R, t = rigid2d(H[m], M[j[m]])
    X = H @ R.T + t
    d, _ = tree.query(X)
    return R, t, d


def georef_helipr(hl, mulran, evidence):
    """Assign a frame to every HeLiPR sequence; returns dict of final trajectories."""
    out = {}
    ev = {}
    site_of = lambda s: ''.join(c for c in s.split('/')[1] if not c.isdigit())
    sites = {}
    for k in hl:
        sites.setdefault(site_of(k), []).append(k)
    # Town / Roundabout: INS
    for site, first in (('Town', 'Town01'), ('Roundabout', 'Roundabout01')):
        R, t, info = helipr_ins_fit(first, hl[f'helipr/{first}'])
        info['method'] = f'2D rigid fit GT(local) -> UTM52N using helipr/{first}/Inertial_data/inspva.csv'
        ev[site] = info
        for k in sites[site]:
            out[k] = dict(frame='korea_utm52', xy=hl[k]['xy_local'] @ R.T + t,
                          latlon_c=info['ins_latlon_mean'], src=hl[k]['src'] + f' + INS georef ({first})')
    # Daejeon sites: register onto the MulRan Daejeon union (all 9 seqs, not only same-name site)
    M = np.vstack([thin(mulran[k]['xy'], 2.0) for k in mulran if 'Sejong' not in k])
    gamma = convergence_deg(36.37, 127.36, 52)
    for site in ('KAIST', 'DCC', 'Riverside'):
        H = np.vstack([thin(hl[k]['xy_local'], 5.0) for k in sites[site]])
        R0 = rot(gamma)
        t0, best, second = fft_translation_search(H @ R0.T, M)
        R, t, d = icp2d(H, M, R0, t0)
        # which MulRan site does it land on?
        land = {}
        for k in mulran:
            if 'Sejong' in k:
                continue
            dd, _ = cKDTree(thin(mulran[k]['xy'], 2.0)).query(H @ R.T + t)
            land[k] = float((dd < 5).mean())
        ev[site] = {'method': 'FFT translation search (rot fixed to UTM convergence) onto MulRan Daejeon '
                              'union (DCC/KAIST/Riverside 01-03), then free rigid ICP (gate 5 m)',
                    'init_rotation_deg': gamma, 'icp_rotation_deg': ang(R),
                    'fft_best_frac_within5m': best, 'fft_second_peak_frac_within5m(>100m away)': second,
                    'icp_frac_within_2m': float((d < 2).mean()), 'icp_frac_within_5m': float((d < 5).mean()),
                    'icp_median_resid_m': float(np.median(d)),
                    'per_mulran_seq_frac_within5m': land,
                    'translation_utm': t.tolist()}
        for k in sites[site]:
            out[k] = dict(frame='korea_utm52', xy=hl[k]['xy_local'] @ R.T + t, latlon_c=[36.37, 127.36],
                          src=hl[k]['src'] + f' + registered to MulRan UTM ({site})')
    for k in sites['Bridge']:
        out[k] = dict(frame='helipr_bridge_local', xy=hl[k]['xy_local'], latlon_c=None,
                      src=hl[k]['src'] + ' (local ENU, NO georeference)')
    evidence['helipr_georef'] = ev
    return out


def load_mcd():
    trajs = {}
    for s in sorted(os.listdir(DS / 'mcd_extracted')):
        a = np.loadtxt(DS / f'mcd_extracted/{s}/scan_poses.csv', delimiter=',', skiprows=1, usecols=(1, 2))
        trajs[f'mcd/{s}'] = dict(frame='mcd_' + s.split('_')[0], xy=a,
                                 src=f'mcd_extracted/{s}/scan_poses.csv (prior-map frame)')
    return trajs


# ---------------------------------------------------------------- metrics
def prep(tr):
    for k, v in tr.items():
        v['xy1'] = thin(v['xy'], 1.0)
        v['xy5'] = thin(v['xy'], 5.0)
        v['tree'] = cKDTree(v['xy1'])
        v['len_km'] = float(len(v['xy1']) / 1000.0)


def pair(A, B):
    d1, _ = B['tree'].query(A['xy1'])
    d5, _ = B['tree'].query(A['xy5'])
    ds, _ = B['tree'].query(A['xy'])          # scan/time-weighted (~10 Hz)
    r = {'min_dist_m': round(float(d1.min()), 2)}
    for th in THR:
        r[f'frac_len_{th}m'] = round(float((d5 < th).mean()), 5)
        r[f'frac_scan_{th}m'] = round(float((ds < th).mean()), 5)
    return r


def upper_bound_unknown_offset(A_xy, B_xy, rot_deg, dils=(5.0, 25.0, 50.0)):
    """Max over ALL translations of the fraction of A within d of B (B rotated by rot_deg).
    An upper bound on the true overlap when only B's absolute offset is unknown."""
    out = {}
    A5 = thin(A_xy, 5.0)
    B2 = thin(B_xy, 2.0) @ rot(rot_deg).T
    for d in dils:
        _, best, second = fft_translation_search(A5, B2, res=2.0 if d <= 5 else 4.0, dil=d)
        out[f'max_frac_within_{int(d)}m_over_all_offsets'] = round(best, 4)
    return out


def main():
    roles, test42, strict, groups, excluded = load_split()
    import pickle
    cache = OUT / '_traj_cache.pkl'
    if cache.exists() and '--reload' not in sys.argv:
        tr, evidence = pickle.load(open(cache, 'rb'))
    else:
        evidence = {}
        tr = {}
        tr.update(load_kitti(evidence))
        tr.update(load_nclt())
        mulran = load_mulran(evidence)
        tr.update(mulran)
        prep(mulran)
        tr.update(georef_helipr(load_helipr_local(), mulran, evidence))
        tr.update(load_mcd())
        prep(tr)
        pickle.dump((tr, evidence), open(cache, 'wb'))
    for k, v in tr.items():
        v['roles'] = roles.get(k, ['info_only'])
    # sanity: every roster sequence has a trajectory
    missing = [k for k in set(roles) | set(test42) if k not in tr]
    assert not missing, missing

    keys = sorted(tr)
    pairs = {}
    for a in keys:
        for b in keys:
            if a != b and tr[a]['frame'] == tr[b]['frame']:
                pairs[f'{a}|{b}'] = pair(tr[a], tr[b])

    # cross-frame city-level distances
    def city(k):
        f = tr[k]['frame']
        if f in CITY_META:
            n, la, lo, how = CITY_META[f]
            return n, (la, lo), how
        return f, tuple(tr[k]['latlon_c']), 'GPS/INS/UTM coordinates'

    # Bridge: upper bounds against every Korean test/val sequence group (unknown offset)
    bridge = np.vstack([tr[k]['xy'] for k in keys if tr[k]['frame'] == 'helipr_bridge_local'
                        and 'train' in tr[k]['roles']])
    ub = {}
    gam_seoul = convergence_deg(37.55, 126.95, 52)
    for grp, members in {
        'helipr_town (Town01-03, test)': [k for k in keys if k.startswith('helipr/Town')],
        'helipr/Roundabout03 (val)': ['helipr/Roundabout03'],
        'mulran_daejeon (test)': [k for k in keys if k.startswith('mulran/') and 'Sejong' not in k],
        'helipr_daejeon (test)': [k for k in keys if k.startswith('helipr/') and
                                  any(s in k for s in ('KAIST', 'DCC', 'Riverside'))],
    }.items():
        A = np.vstack([tr[k]['xy'] for k in members])
        ub[grp] = upper_bound_unknown_offset(A, bridge, gam_seoul)
    evidence['bridge_train_unknown_offset_upper_bounds'] = {
        'note': 'Bridge01-03 GT is local ENU with unknown absolute origin. Value = max over every '
                'possible 2D translation of the fraction of the test/val route within d of the '
                'Bridge train route (rotation fixed to ENU->UTM convergence). It is only an upper '
                'bound; a large value does NOT indicate overlap (road-grid coincidences).',
        'bounds': ub}

    # reporting tables
    TRAINVAL = [k for k in keys if ('train' in tr[k]['roles'] or 'val' in tr[k]['roles'])]
    def summarize(a):
        rows = []
        for b in TRAINVAL:
            if b == a:
                continue
            if tr[a]['frame'] == tr[b]['frame']:
                rows.append(dict(b=b, b_roles=tr[b]['roles'], comparable='same_frame', **pairs[f'{a}|{b}']))
            elif ('helipr_bridge_local' in (tr[a]['frame'], tr[b]['frame'])
                  and 'korea_utm52' in (tr[a]['frame'], tr[b]['frame'])):
                rows.append(dict(b=b, b_roles=tr[b]['roles'], comparable='NO (Bridge has no georeference)'))
            else:
                (na, la, ha), (nb, lb, hb) = city(a), city(b)
                rows.append(dict(b=b, b_roles=tr[b]['roles'], comparable='city-level',
                                 city_a=na, city_b=nb, approx_km=round(haversine_km(la, lb), 1),
                                 basis=f'{ha} / {hb}'))
        same = [r for r in rows if r['comparable'] == 'same_frame']
        worst = min(same, key=lambda r: r['min_dist_m']) if same else None
        worst50 = max(same, key=lambda r: r['frac_scan_50m']) if same else None
        return rows, worst, worst50

    per_seq = {}
    for a in sorted(set(test42) | {k for k in keys if 'val' in tr[k]['roles']} |
                    {'kitti360/0002', 'kitti360/0009'}):
        rows, worst, worst50 = summarize(a)
        per_seq[a] = dict(roles=tr[a]['roles'], frame=tr[a]['frame'], group=groups.get(a),
                          strict=a in strict, excluded=a in excluded, src=tr[a]['src'],
                          route_len_km=tr[a]['len_km'],
                          nearest_trainval=(dict(b=worst['b'], min_dist_m=worst['min_dist_m'])
                                            if worst else None),
                          max_overlap_trainval={m: max(((r[m], r['b']) for r in rows
                                                         if r['comparable'] == 'same_frame'),
                                                        default=None)
                                                for m in ('frac_scan_25m', 'frac_scan_50m', 'frac_scan_200m',
                                                          'frac_len_25m', 'frac_len_50m', 'frac_len_200m')},
                          vs_trainval=rows)

    # ---- extra analyses -------------------------------------------------------
    extra = {}
    # (a) null baselines for the Bridge upper bound: same statistic vs sites whose true offset IS known
    U = lambda ks: np.vstack([tr[k]['xy'] for k in ks])
    town = U(['helipr/Town01', 'helipr/Town02', 'helipr/Town03'])
    dj = U([k for k in keys if k.startswith('mulran/') and 'Sejong' not in k])
    rb = U(['helipr/Roundabout01', 'helipr/Roundabout02'])
    sj = U(['mulran/Sejong01', 'mulran/Sejong02'])
    extra['bridge_bound_null_baselines'] = {
        'note': 'max-over-all-offsets statistic for pairs that are KNOWN to be disjoint (true offset known); '
                'a Bridge bound below these values is consistent with an unrelated site',
        'Town vs Roundabout01-02 (true min 29.9 km)': upper_bound_unknown_offset(town, rb, 0.0),
        'Town vs Sejong01-02 (true min 118.7 km)': upper_bound_unknown_offset(town, sj, 0.0),
        'MulRan Daejeon vs Roundabout01-02': upper_bound_unknown_offset(dj, rb, 0.0),
        'Roundabout03 vs Town': upper_bound_unknown_offset(tr['helipr/Roundabout03']['xy'], town, 0.0)}
    # (b) fine-threshold detail for the closest strict pairs
    det = {}
    for a, b in [('kitti/05', 'kitti/06'), ('kitti/08', 'kitti/07'), ('kitti/08', 'kitti/00'),
                 ('kitti360/0000', 'kitti/01'), ('kitti360/0006', 'kitti/09'), ('kitti/00', 'kitti/07'),
                 ('kitti360/0004', 'kitti/09'), ('kitti/09', 'kitti360/0004'), ('kitti360/0004', 'kitti360/0006')]:
        d, _ = tr[b]['tree'].query(tr[a]['xy'])
        d5, _ = tr[b]['tree'].query(tr[a]['xy5'])
        det[f'{a}|{b}'] = {'min_dist_m': round(float(d.min()), 2), 'n_scans': int(len(d)),
                           **{f'scan_frac_<{t}m': round(float((d < t).mean()), 5) for t in (25, 50, 80, 100, 150, 200)},
                           **{f'len_frac_<{t}m': round(float((d5 < t).mean()), 5) for t in (25, 50, 80, 100, 150, 200)}}
    extra['near_pair_detail'] = det
    # (c) reproduce audit-script style (every 5th odometry frame, <50 m)
    rep = {}
    for a, b in [('kitti/00', 'kitti/07'), ('kitti/05', 'kitti/06')]:
        d, _ = tr[b]['tree'].query(tr[a]['xy'][::5])
        rep[f'{a}|{b}'] = {'frac_<50m_every5th': round(float((d < 50).mean()), 5),
                           'count': f'{int((d < 50).sum())}/{len(d)}', 'min_dist_m': round(float(d.min()), 2)}
    extra['audit_style_every5th_frame'] = rep
    # (d) val vs test
    vt = {}
    VAL = [k for k in keys if 'val' in tr[k]['roles']]
    for v in VAL:
        for t in test42:
            if tr[v]['frame'] == tr[t]['frame']:
                vt[f'{t}|{v}'] = dict(test_on_val=pairs[f'{t}|{v}'], val_on_test=pairs[f'{v}|{t}'],
                                      test_strict=t in strict)
    extra['val_vs_test_same_frame'] = vt
    # (e) Daejeon: HeLiPR 04-06 vs MulRan 01-03 (both directions), test-test
    H = [f'helipr/{s}0{i}' for s in ('DCC', 'KAIST', 'Riverside') for i in (4, 5, 6)]
    M = [f'mulran/{s}0{i}' for s in ('DCC', 'KAIST', 'Riverside') for i in (1, 2, 3)]
    extra['daejeon_helipr_vs_mulran'] = {
        'note': 'HeLiPR Daejeon absolute position obtained by registration onto MulRan (see evidence.helipr_georef)',
        'helipr_on_mulran': {a: {b: pairs[f'{a}|{b}'] for b in M} for a in H},
        'mulran_on_helipr': {a: {b: pairs[f'{a}|{b}'] for b in H} for a in M},
        'mulran_on_helipr_union_frac_scan_50m': {
            a: round(float((cKDTree(np.vstack([tr[b]['xy1'] for b in H])).query(tr[a]['xy'])[0] < 50).mean()), 4)
            for a in M}}
    # (f) MCD same-campus info
    extra['mcd_intra_campus_frac_len_25m'] = {
        a: {b: pairs[f'{a}|{b}']['frac_len_25m'] for b in keys if b != a and tr[b]['frame'] == tr[a]['frame']}
        for a in keys if a.startswith('mcd/')}
    # (g) strict-37 nearest train/val distribution
    dist = []
    for a in strict:
        v = per_seq[a]
        if v['nearest_trainval']:
            dist.append((v['nearest_trainval']['min_dist_m'], a, v['nearest_trainval']['b'],
                         v['max_overlap_trainval']['frac_scan_50m'][0], v['max_overlap_trainval']['frac_scan_200m'][0]))
        else:
            cl = [r for r in v['vs_trainval'] if r['comparable'] == 'city-level']
            dist.append((min(r['approx_km'] for r in cl) * 1000.0, a, 'city-level (' + min(cl, key=lambda r: r['approx_km'])['b'] + ')',
                         0.0, 0.0))
    dist.sort()
    extra['strict37_nearest_trainval_sorted'] = [
        dict(seq=a, nearest=b, min_dist_m=round(m, 1), max_frac_scan_50m=f50, max_frac_scan_200m=f200,
             flag=bool(m < 50 or f50 > 0.01),
             note=('Bridge01-03 (train) not comparable' if tr[a]['frame'] == 'korea_utm52' else ''))
        for m, a, b, f50, f200 in dist]

    out = dict(_meta=dict(date='2026-09-29', script='geo_audit.py',
                          thresholds_m=THR,
                          metrics={'min_dist_m': 'min over A (1 m-thinned) of NN distance to B (1 m-thinned)',
                                   'frac_len_Xm': 'fraction of A route length (5 m arc-length samples) within X m of B',
                                   'frac_scan_Xm': 'fraction of A scans/poses (~10 Hz, time-weighted) within X m of B'}),
               split=dict(roles=roles, test42=test42, strict37=strict, strict_groups=groups,
                          excluded=excluded),
               frames={k: dict(frame=tr[k]['frame'], src=tr[k]['src'], roles=tr[k]['roles'],
                               len_km=tr[k]['len_km']) for k in keys},
               evidence=evidence, extra=extra, per_sequence=per_seq, pairs_same_frame=pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'geo_overlap.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', OUT / 'geo_overlap.json', len(pairs), 'same-frame pairs')


if __name__ == '__main__':
    main()
