#!/usr/bin/env python3
"""HeLiPR Bridge01-03 (train) absolute placement WITHOUT INS, and distance to test/val groups.

The official INS file (Bridge01/Inertial_data/inspva.csv) is gated behind the HeLiPR
download-request Google Form, so it was NOT downloaded. Instead:
  * Location text: official HeLiPR "Sequence" page, Bridge01-04 "Location : Hangang & Dongjak Bridge".
  * Control points: Wikipedia coordinates of Hangang Bridge (37.5159 N, 126.9578 E) and
    Dongjak Bridge (37.5112 N, 126.9822 E).
  * The two bridge crossings are identified in the Bridge01 GT (local ENU; axes = true-north
    ENU, as verified for Town/Roundabout in geo_audit.py), rotated to UTM 52N by the grid
    convergence, and every translation that puts each control point within `w` m of its
    crossing is enumerated (the "feasible set"). Distances are reported at the feasible-set
    centre (nominal) and as the worst case over the whole feasible set, rotation +-1 deg.
READ-ONLY w.r.t. the repo and /mnt/d. Inputs: results/split_audit/_traj_cache.pkl (from geo_audit.py) and
/mnt/d/NSD_datasets/helipr/Bridge0{1,2,3,4}/LiDAR_GT/Velodyne_gt.txt.
Output: bridge_overlap.json (this directory).
"""
import json, sys, pickle
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results' / 'split_audit'

from geoutil import latlon_to_utm                                   # noqa: E402
from geo_audit import thin, rigid2d, ang, rot, convergence_deg     # noqa: E402

DS = Path('/mnt/d/NSD_datasets/helipr')
THR = (25, 50, 200)
WIKI = {'Hangang': (37.5159, 126.9578, 'https://en.wikipedia.org/wiki/Hangang_Bridge (37°30′57″N 126°57′28″E)'),
        'Dongjak': (37.5112, 126.9822, 'https://en.wikipedia.org/wiki/Dongjak_Bridge (37°30′40″N 126°58′56″E)')}
# crossing segments in Bridge01 (seconds from start), identified from the route profile:
#   Hangang: straight SSW (heading ~205 deg) run from the start, z ~3.4 m, 0-80 s
#   Dongjak: straight NNE (heading ~12 deg) elevated run, z ~10.5 m, 585-646 s
SEG_T = {'Hangang': (0.0, 80.0), 'Dongjak': (585.0, 646.0)}
GROUPS = {
    'helipr_town (test)': ['helipr/Town01', 'helipr/Town02', 'helipr/Town03'],
    'helipr_daejeon (test)': [f'helipr/{s}0{i}' for s in ('DCC', 'KAIST', 'Riverside') for i in (4, 5, 6)],
    'mulran_daejeon (test)': [f'mulran/{s}0{i}' for s in ('DCC', 'KAIST', 'Riverside') for i in (1, 2, 3)],
    'helipr_roundabout (train 01/02, val 03)': ['helipr/Roundabout01', 'helipr/Roundabout02', 'helipr/Roundabout03'],
    'mulran_sejong (train 01/02, val 03)': ['mulran/Sejong01', 'mulran/Sejong02', 'mulran/Sejong03'],
}


ROLE = {**{k: 'test' for k in GROUPS['helipr_town (test)'] + GROUPS['helipr_daejeon (test)'] + GROUPS['mulran_daejeon (test)']},
        'helipr/Roundabout01': 'train', 'helipr/Roundabout02': 'train', 'helipr/Roundabout03': 'val',
        'mulran/Sejong01': 'train', 'mulran/Sejong02': 'train', 'mulran/Sejong03': 'val'}


def load_gt(s):
    a = np.loadtxt(DS / f'{s}/LiDAR_GT/Velodyne_gt.txt', usecols=(0, 1, 2, 3))
    return a[:, 0], a[:, 1:3], a[:, 3]


def utm_to_latlon(E, N, zone=52):
    """Newton inverse of latlon_to_utm (sub-mm)."""
    E, N = np.atleast_1d(E).astype(float), np.atleast_1d(N).astype(float)
    lat = np.full_like(E, 37.5); lon = np.full_like(E, 127.0)
    for _ in range(8):
        e0, n0 = latlon_to_utm(lat, lon, zone)
        h = 1e-6
        e1, n1 = latlon_to_utm(lat + h, lon, zone); e2, n2 = latlon_to_utm(lat, lon + h, zone)
        J = np.stack([np.stack([(e1 - e0) / h, (e2 - e0) / h], -1), np.stack([(n1 - n0) / h, (n2 - n0) / h], -1)], 1)
        d = np.linalg.solve(J, np.stack([E - e0, N - n0], -1)[..., None])[..., 0]
        lat, lon = lat + d[:, 0], lon + d[:, 1]
    return lat, lon


def seg_dist(p, S):
    """distance from point p to polyline S (dense points)."""
    return float(np.min(np.hypot(S[:, 0] - p[0], S[:, 1] - p[1])))


def frame_check(gt):
    """Bridge02/03/04 share Bridge01's local frame? (start offsets + ICP onto Bridge01)."""
    out = {}
    B1 = thin(gt['Bridge01'][1], 1.0)
    tree = cKDTree(B1)
    for s in ('Bridge02', 'Bridge03', 'Bridge04'):
        X = thin(gt[s][1], 2.0)
        d0, _ = tree.query(X)
        R, t = np.eye(2), np.zeros(2)
        for _ in range(50):
            d, j = tree.query(X @ R.T + t)
            m = d < 40.0
            R, t = rigid2d(X[m], B1[j[m]])
        d1, _ = tree.query(X @ R.T + t)
        out[s] = {'start_xy_local_m': gt[s][1][0].round(1).tolist(),
                  'nn_to_Bridge01_as_is_m': {'median': round(float(np.median(d0)), 2),
                                             'frac_lt_25m': round(float((d0 < 25).mean()), 3),
                                             'frac_lt_50m': round(float((d0 < 50).mean()), 3)},
                  'icp_onto_Bridge01': {'rotation_deg': round(ang(R), 3), 'translation_m': t.round(2).tolist(),
                                        'median_resid_m': round(float(np.median(d1)), 2)}}
    out['note'] = ('Official Sequence page: "All the trajectories have been transformed to align with the 01 '
                   'sequence, where the starting location of the 01 sequence serves as the origin point." '
                   'Bridge01/04 are driven in reverse, 02/03 in normal direction (official page), so 02/03 '
                   'run on the opposite carriageway (median NN ~17-19 m) while 04 is ~0.6 m from 01.')
    return out


def metrics(A, Btree):
    d1, _ = Btree.query(A['xy1']); d5, _ = Btree.query(A['xy5']); ds, _ = Btree.query(A['xy'])
    r = {'min_dist_m': round(float(d1.min()), 1)}
    for th in THR:
        r[f'frac_len_{th}m'] = round(float((d5 < th).mean()), 5)
        r[f'frac_scan_{th}m'] = round(float((ds < th).mean()), 5)
    return r


def main():
    tr, _ = pickle.load(open(OUT / '_traj_cache.pkl', 'rb'))
    gt = {s: load_gt(s) for s in ('Bridge01', 'Bridge02', 'Bridge03', 'Bridge04')}
    res = {'_meta': {'date': '2026-09-29', 'script': 'bridge/bridge_bound.py', 'thresholds_m': THR,
                     'metrics': 'same definitions as geo_audit.py; A = test/val sequence, B = Bridge01-03 union'},
           'download': {
               'status': 'NOT DOWNLOADED (blocked by request form)',
               'official_page': 'https://sites.google.com/view/heliprdataset/download',
               'gate': 'Google Form "Request Download for HeLiPR Dataset" https://forms.gle/HwRUJ6aXh4hNELeU8 '
                       '(-> docs.google.com/forms/d/e/1FAIpQLSeVhyDyi6pbMWy7Shjs_Z-oGD7DeMpLYaHy9-1EImy6pHgMzQ/viewform); '
                       'fields: Email, Name, Affiliation, sequence checkboxes (Bridge01/02/03 available); '
                       'download link is e-mailed after submission. No public Drive file ID on the site.',
               'files_written_to_dataset': [],
               'location_source': 'https://sites.google.com/view/heliprdataset/sequence : Bridge01-04 '
                                  '"Location : Hangang & Dongjak Bridge"'}}

    # ---- sanity vs official page
    t1, xy1, z1 = gt['Bridge01']
    step = np.hypot(*np.diff(xy1, axis=0).T)
    res['bridge01_sanity'] = {'duration_s': round(float((t1[-1] - t1[0]) / 1e9), 1),
                              'path_len_m': round(float(step.sum()), 0),
                              'official_page': 'Bridge01: 2144 sec, Path Length 23056m',
                              'z_range_m': [round(float(z1.min()), 1), round(float(z1.max()), 1)]}
    res['frame_check'] = frame_check(gt)

    # ---- crossing segments (local)
    ts = (t1 - t1[0]) / 1e9
    seg = {}
    for b, (a, c) in SEG_T.items():
        P = xy1[(ts >= a) & (ts <= c)]
        dP = P[-1] - P[0]
        seg[b] = P
        res.setdefault('crossings_local', {})[b] = {
            'time_window_s': [a, c], 'start_xy': P[0].round(1).tolist(), 'end_xy': P[-1].round(1).tolist(),
            'length_m': round(float(np.hypot(*dP)), 0),
            'heading_deg_local_ENU': round(float(np.degrees(np.arctan2(dP[0], dP[1])) % 360), 1),
            'mean_z_m': round(float(z1[(ts >= a) & (ts <= c)].mean()), 1)}

    # ---- control points (UTM 52N)
    cp = {b: np.array(latlon_to_utm(la, lo, 52)).ravel() for b, (la, lo, _) in WIKI.items()}
    gamma = convergence_deg(37.513, 126.97, 52)
    res['control_points'] = {b: {'latlon': [la, lo], 'source': src, 'utm52': cp[b].round(1).tolist()}
                             for b, (la, lo, src) in WIKI.items()}
    res['control_points']['utm_grid_convergence_deg'] = round(gamma, 3)
    res['control_points']['vector_Hangang_to_Dongjak_utm_m'] = (cp['Dongjak'] - cp['Hangang']).round(0).tolist()

    # ---- feasible translations
    def feasible(rdeg, w, grid=10.0):
        R = rot(rdeg)
        SH, SD = seg['Hangang'] @ R.T, seg['Dongjak'] @ R.T
        # t must put cp['Hangang'] within w of SH+t  <=>  t in (cp - SH) (+) disk(w)
        cand = cp['Hangang'] - SH
        lo, hi = cand.min(0) - w, cand.max(0) + w
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], grid), np.arange(lo[1], hi[1], grid))
        T = np.c_[gx.ravel(), gy.ravel()]
        dH, _ = cKDTree(cp['Hangang'] - SH).query(T)
        dD, _ = cKDTree(cp['Dongjak'] - SD).query(T)
        ok = (dH <= w) & (dD <= w)
        return R, T[ok], dH[ok], dD[ok]

    # centroid (midpoint) registration, info only
    R0 = rot(gamma)
    mid = {b: seg[b].mean(0) @ R0.T for b in seg}
    tc = np.mean([cp[b] - mid[b] for b in seg], axis=0)
    res['registration_midpoint_assumption'] = {
        'note': 'assumes each Wikipedia point is the midpoint of the traversed straight crossing (info only)',
        'translation_utm': tc.round(1).tolist(),
        'residual_m': {b: round(float(np.hypot(*(mid[b] + tc - cp[b]))), 1) for b in seg}}

    variants = {}
    for rd in (0.0, -1.0, 1.0):
        for w in (60.0, 150.0):
            R, T, dH, dD = feasible(gamma + rd, w)
            variants[(rd, w)] = (R, T)
            res.setdefault('feasible_sets', {})[f'rot{gamma + rd:+.2f}_w{int(w)}'] = {
                'n_translations_10m_grid': int(len(T)),
                'extent_E_m': round(float(np.ptp(T[:, 0])), 0) if len(T) else None,
                'extent_N_m': round(float(np.ptp(T[:, 1])), 0) if len(T) else None,
                'min_ctrl_resid_m': {'Hangang': round(float(dH.min()), 1) if len(T) else None,
                                     'Dongjak': round(float(dD.min()), 1) if len(T) else None}}
    Rn, Tn = variants[(0.0, 60.0)]
    assert len(Tn), 'no translation satisfies both control points -> location text/geometry inconsistent'
    tn = Tn.mean(0)
    # nominal residuals: distance from each control point to its crossing at the nominal translation
    res['registration_nominal'] = {
        'method': 'rotation = UTM grid convergence (GT axes are ENU); translation = centre of the feasible '
                  'set (both Wikipedia bridge points within 60 m of the traversed crossings)',
        'rotation_deg': round(gamma, 3), 'translation_utm': tn.round(1).tolist(),
        'control_point_resid_m (point-to-crossing)': {
            b: round(seg_dist(cp[b], seg[b] @ Rn.T + tn), 1) for b in seg},
        'along_bridge_uncertainty_m (feasible-set extent, w=60)': [round(float(np.ptp(Tn[:, 0])), 0),
                                                                   round(float(np.ptp(Tn[:, 1])), 0)],
        'caveat': 'NOT an INS registration. Absolute error is bounded by the feasible set (hundreds of m), '
                  'not the ~1 m of the Town/Roundabout INS fits.'}

    # ---- Bridge absolute location (nominal)
    Bn = {s: gt[s][1] @ Rn.T + tn for s in ('Bridge01', 'Bridge02', 'Bridge03')}
    U = np.vstack(list(Bn.values()))
    la, lo = utm_to_latlon(U[:, 0], U[:, 1])
    lac, loc = utm_to_latlon(*U.mean(0))
    la0, lo0 = utm_to_latlon(*Bn['Bridge01'][0])
    res['bridge_location_nominal'] = {
        'area': 'Seoul, Han River between Hangang Bridge and Dongjak Bridge (Yongsan-gu north bank / Dongjak-gu south bank)',
        'centroid_latlon': [round(float(lac[0]), 5), round(float(loc[0]), 5)],
        'bbox_lat': [round(float(la.min()), 5), round(float(la.max()), 5)],
        'bbox_lon': [round(float(lo.min()), 5), round(float(lo.max()), 5)],
        'bridge01_start_latlon': [round(float(la0[0]), 5), round(float(lo0[0]), 5)]}

    # ---- distances: nominal + worst case over all feasible variants
    bridge_local = {s: thin(gt[s][1], 1.0) for s in ('Bridge01', 'Bridge02', 'Bridge03')}
    Bunion_local = np.vstack(list(bridge_local.values()))

    per_seq, per_group = {}, {}
    for g, members in GROUPS.items():
        gmin_nom, gmin_worst = np.inf, np.inf
        for k in members:
            A = tr[k]
            assert A['frame'] == 'korea_utm52', k
            Bt = cKDTree(Bunion_local @ Rn.T + tn)
            nom = metrics(A, Bt)
            nom['min_dist_by_bridge_seq_m'] = {s: round(float(cKDTree(bridge_local[s] @ Rn.T + tn).query(A['xy1'])[0].min()), 1)
                                               for s in bridge_local}
            # worst case over every feasible (rotation, w, translation): triangle inequality
            #   min_dist(B R + t, A) >= min_dist(B R + c, A) - |t - c|,  c = feasible-set centre
            At = A['tree']
            worst = np.inf
            for (rd, w), (R, T) in variants.items():
                c = T.mean(0)
                rho = float(np.hypot(*(T - c).T).max()) + 10.0     # + grid step
                worst = min(worst, float(At.query(Bunion_local @ R.T + c)[0].min()) - rho)
            worst = max(0.0, worst - 1.0)          # 1 m thinning of both routes
            worst_fr = {f'frac_scan_{th}m': (0.0 if worst >= th else None) for th in THR}
            per_seq[k] = {'group': g, 'role': ROLE[k], 'nominal': nom,
                          'worst_case_over_feasible_set': {'min_dist_m_lower_bound': round(worst, 1), **worst_fr}}
            gmin_nom = min(gmin_nom, nom['min_dist_m']); gmin_worst = min(gmin_worst, worst)
        per_group[g] = {'nominal_min_dist_m': round(gmin_nom, 1), 'worst_case_min_dist_m': round(gmin_worst, 1),
                        'max_frac_scan_200m_nominal': max(per_seq[k]['nominal']['frac_scan_200m'] for k in members),
                        'verdict': 'separated' if gmin_worst > 200 else 'CHECK'}
    res['per_group'] = per_group
    res['per_sequence'] = per_seq
    (OUT / 'bridge').mkdir(parents=True, exist_ok=True)
    (OUT / 'bridge' / 'bridge_overlap.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: res[k] for k in ('bridge01_sanity', 'crossings_local', 'control_points',
                                          'registration_midpoint_assumption', 'feasible_sets',
                                          'registration_nominal', 'bridge_location_nominal', 'per_group')},
                     indent=1, ensure_ascii=False, default=float))
    print(json.dumps(res['frame_check'], indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
