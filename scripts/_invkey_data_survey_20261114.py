#!/usr/bin/env python3
"""invkey 진단 연구용 데이터 조사 (2026-11-14).

실행 (호스트 파이썬, numpy·scipy만 쓴다. torch·GPU 불필요, 리포 루트에서):
    python3 scripts/_invkey_data_survey_20261114.py
출력:
    results/invkey_study_20261114/data_survey.json   시퀀스별 표, 장소 겹침 묶음, 후보 분할 합계
    표준 출력                                        보고서(data_survey.md)에 옮겨 쓸 마크다운 표

무엇을 재나 (63시퀀스 전부, 키프레임 캐시의 기준(GT) 위치만 쓴다):
  - 학습 8시퀀스는 data/preprocessed_v2/ 캐시의 `poses`(GT), 검증·시험 55시퀀스는 `poses_gt`
    (`poses`는 KISS-ICP 주행 거리계라 쓰지 않는다). 캐시 경로는 지리 감사 명단
    results/pdv2_review_closure_20261107/geography/roster.csv의 `cache` 열을 따르고 sha256도 대조한다.
  - 양성 쌍: 키프레임 간격 q - j >= 30, 3차원 GT 거리 <= 5 m. 재방문 질의: 그런 j가 하나라도 있는 q.
    float64 위치로 KD-tree(query_pairs)를 쓴 뒤 직접 차로 거리를 다시 재서 거른다.
  - 방향: 자세의 회전은 쓰지 않는다(데이터셋마다 자세 좌표계가 다르고 MCD GT 회전은 항등).
    궤적 자체의 진행 방향 = 키프레임 i의 앞뒤 HALF_WIN개 위치 차 P[i+3] - P[i-3]의 3차원 단위 벡터.
    그 창의 이동이 STILL_M(1 m) 미만이면 거의 정지로 보고 방향 미정(unknown)으로 둔다.
    두 진행 방향의 각도 < 30도 같은 방향, > 150도 반대 방향, 그 사이 교차.
  - 질의 단위 방향·거리·시간 간격은 각 재방문 질의의 가장 가까운 과거 짝(j <= q-30) 기준이다.
  - 검산: 정해 둔 4개 목표(SANITY)와 맞지 않거나, 무차별 대입 검산과 다르거나, 캐시 sha256이 명단과
    다르면 멈춘다. 진행 방향 분류는 자세 회전의 상대 각도와 대조해 일치율만 기록한다(분류에는 쓰지 않는다).
  - 장소 겹침 묶음: 감사 geo_audit.json의 same_split_overlap_groups(50 m 안 1% 초과 겹침)에,
    같은 분할 쌍 표에서 50 m 안에 점이 있는 1% 이하 근접 쌍을 더 묶는다(엄격 규칙).
  - 후보 분할(CANDIDATES)의 학습·held-out 합계와 사이트 몫 상한(SITE_SHARE_CAP) 적용 기대값.
  - 같은 사이트 다른 세션의 캐시 위치를 그대로 겹쳐 본 최근접 거리(세션 간 양성은 세지 않는다).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
GEO_DIR = REPO / 'results/pdv2_review_closure_20261107/geography'
ROSTER = GEO_DIR / 'roster.csv'
GEO_AUDIT = GEO_DIR / 'geo_audit.json'
SAME_PAIRS = GEO_DIR / 'geo_same_split_pairs.csv'
CROSS_PAIRS = GEO_DIR / 'geo_pairs_cross_split.csv'
BRIDGE_RF = REPO / 'results/pdv2_review_closure_20261107/geo_audit/bridge_rotation_free_check.json'
PROTOCOL = REPO / 'results/pdv2_diffattn_contrast_test_20261104/protocol.json'
OUT = REPO / 'results/invkey_study_20261114'

GAP = 30            # j <= q - GAP
RADIUS = 5.0        # m
HALF_WIN = 3        # 진행 방향 창: P[i+3] - P[i-3]
STILL_M = 1.0       # 창 이동이 이보다 짧으면 방향 미정
SAME_DEG, REV_DEG = 30.0, 150.0
DIRS = ('same', 'reverse', 'crossing', 'unknown')

SANITY = {  # key: (revisit queries, positive pairs or None)
    'helipr/Bridge01': (8121, 52537),
    'kitti360/0004': (1846, None),
    'helipr/Roundabout01': (1990, None),
    'mulran/Sejong03': (791, None),
}
BRUTE_CHECK = ('helipr/Bridge01', 'kitti360/0004', 'helipr/Roundabout01', 'mulran/Sejong03', 'kitti/02', 'mcd/kth_day_06')

SENSOR = {  # docs/데이터와 분할.md "데이터셋과 센서"
    'kitti': 'HDL-64E', 'kitti360': 'HDL-64E', 'helipr': 'VLP-16',
    'mulran': 'OS1-64 gen1', 'nclt': 'HDL-32E', 'mcd': 'OS1-64 gen2',
}

# ---------------------------------------------------------------- 후보 분할
# 묶음 이름은 build_units()가 만든다. 후보마다 held-out 묶음을 적고, 나머지는 모두 학습이다.
REF_UNITS = ['U_kitti09_k360_0004_0006', 'U_roundabout', 'U_sejong']   # 기준 수치가 있는 3시퀀스의 묶음
CANDIDATES = {
    'A': REF_UNITS + ['U_mcd_kth', 'U_mcd_tuhh'],
    'B': REF_UNITS + ['U_nclt'],
    'C': REF_UNITS + ['U_daejeon_kaist_riverside', 'U_daejeon_dcc', 'U_mcd_kth', 'U_mcd_tuhh'],
}
SITE_SHARE_CAP = 0.25   # 학습 재방문 질의에서 한 사이트 몫의 상한(가정)
# 같은 사이트 다른 세션의 캐시 GT 위치를 그대로 비교해 본다(세션 간 양성은 세지 않는다. 미해결 항목 근거).
XSESSION_PAIRS = [
    ('helipr/Bridge01', 'helipr/Bridge02'), ('helipr/Bridge01', 'helipr/Bridge03'), ('helipr/Bridge01', 'helipr/Bridge04'),
    ('helipr/Town01', 'helipr/Town02'), ('helipr/Town01', 'helipr/Town03'),
    ('helipr/Roundabout01', 'helipr/Roundabout02'), ('helipr/Roundabout01', 'helipr/Roundabout03'),
    ('helipr/KAIST04', 'helipr/KAIST05'), ('helipr/KAIST04', 'helipr/KAIST06'),
    ('helipr/DCC04', 'helipr/DCC05'), ('helipr/DCC04', 'helipr/DCC06'),
    ('helipr/Riverside04', 'helipr/Riverside05'), ('helipr/Riverside04', 'helipr/Riverside06'),
    ('helipr/Riverside05', 'helipr/Riverside06'),
    ('mulran/KAIST01', 'mulran/KAIST02'), ('mulran/Sejong01', 'mulran/Sejong02'), ('helipr/KAIST04', 'mulran/KAIST01'),
    ('kitti360/0004', 'kitti360/0006'), ('nclt/2012-01-08', 'nclt/2012-02-04'), ('nclt/2012-01-08', 'nclt/2012-05-11'),
    ('mcd/kth_day_06', 'mcd/kth_night_01'), ('mcd/tuhh_day_02', 'mcd/tuhh_night_07'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def load_roster():
    proto = json.load(open(PROTOCOL))
    role = {k: s for s in ('train', 'val', 'test') for k in proto[s]}
    rows = list(csv.DictReader(open(ROSTER)))
    assert len(rows) == 63 and len(role) == 63, (len(rows), len(role))
    for r in rows:
        if role[r['key']] != r['split']:
            sys.exit(f"STOP: roster split {r['split']} != protocol {role[r['key']]} for {r['key']}")
    return rows


def headings(P):
    n = len(P)
    i = np.arange(n)
    v = P[np.clip(i + HALF_WIN, 0, n - 1)] - P[np.clip(i - HALF_WIN, 0, n - 1)]
    nv = np.linalg.norm(v, axis=1)
    moving = nv >= STILL_M
    u = np.zeros_like(v)
    u[moving] = v[moving] / nv[moving, None]
    return u, moving


def classify(u, moving, a, b):
    ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', u[a], u[b]), -1.0, 1.0)))
    cat = np.full(len(a), 2, dtype=np.int8)            # crossing
    cat[ang < SAME_DEG] = 0
    cat[ang > REV_DEG] = 1
    cat[~(moving[a] & moving[b])] = 3                   # unknown
    return cat


def positive_pairs(P):
    tree = cKDTree(P)
    pr = tree.query_pairs(RADIUS + 1e-6, output_type='ndarray')
    if len(pr) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    j, q = pr.min(1), pr.max(1)
    keep = (q - j) >= GAP
    j, q = j[keep], q[keep]
    d = np.linalg.norm(P[q] - P[j], axis=1)            # 직접 차, float64
    keep = d <= RADIUS
    return q[keep], j[keep], d[keep]


def brute_counts(P):
    nq = npair = 0
    for q in range(GAP, len(P)):
        d = np.linalg.norm(P[: q - GAP + 1] - P[q], axis=1)
        k = int((d <= RADIUS).sum())
        npair += k
        nq += k > 0
    return nq, npair


def pct(x, p):
    return None if len(x) == 0 else round(float(np.percentile(x, p)), 3)


def survey_one(r):
    key = r['key']
    ds, seq = key.split('/')
    path = REPO / r['cache']
    z = np.load(path)
    if r['split'] == 'train':
        if 'poses_gt' in z.files:
            sys.exit(f'STOP: unexpected poses_gt in train cache {path}')
        field = 'poses'
    else:
        if 'poses_gt' not in z.files:
            sys.exit(f'STOP: poses_gt missing in {path}')
        field = 'poses_gt'
    T = np.asarray(z[field], dtype=np.float64)
    P, R = T[:, :3, 3], T[:, :3, :3]
    t = np.asarray(z['timestamps'], dtype=np.float64)
    n = len(P)
    if n != int(r['cache_keyframes']):
        sys.exit(f"STOP: {key} keyframes {n} != roster {r['cache_keyframes']}")
    u, moving = headings(P)
    q, j, d = positive_pairs(P)
    cat = classify(u, moving, q, j)
    # 질의별 가장 가까운 과거 짝
    order = np.lexsort((d, q))
    qs, first = np.unique(q[order], return_index=True)
    nn = order[first]
    nn_d, nn_cat, nn_dt = d[nn], cat[nn], t[q[nn]] - t[j[nn]]
    any_rev = np.unique(q[cat == 1])
    rec = dict(
        key=key, dataset=ds, sequence=seq, sensor=SENSOR[ds], site=r['site'], city=r['city'],
        paper_split=r['split'], cache=r['cache'], pose_field=field,
        cache_sha256_matches_roster=None,
        n_keyframes=n, path_km=round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) / 1000, 3),
        duration_s=round(float(t[-1] - t[0]), 1), stationary_keyframes=int((~moving).sum()),
        revisit_queries=int(len(qs)), positive_pairs=int(len(q)),
        pairs_by_dir={k: int((cat == i).sum()) for i, k in enumerate(DIRS)},
        queries_by_dir_nearest={k: int((nn_cat == i).sum()) for i, k in enumerate(DIRS)},
        queries_with_any_reverse_pair=int(len(any_rev)),
        nearest_dist_m=dict(median=pct(nn_d, 50), p90=pct(nn_d, 90)),
        nearest_dist_median_by_dir={k: pct(nn_d[nn_cat == i], 50) for i, k in enumerate(DIRS)},
        time_gap_s_median=pct(nn_dt, 50),
        time_gap_s_median_by_dir={k: pct(nn_dt[nn_cat == i], 50) for i, k in enumerate(DIRS)},
    )
    known = sum(rec['pairs_by_dir'][k] for k in DIRS[:3])
    rec['pair_share_of_known'] = {k: (round(rec['pairs_by_dir'][k] / known, 4) if known else None) for k in DIRS[:3]}
    # 검산 3: 진행 방향 분류를 자세 회전의 상대 각도(같은 임계값)와 대조한다. 분류에는 쓰지 않는다.
    if np.allclose(R, np.eye(3)[None], atol=1e-9):
        rec['rotation_crosscheck'] = 'GT rotation is identity; not checkable'
    elif known:
        k = cat < 3
        ang = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', R[q[k]], R[j[k]]) - 1) / 2, -1.0, 1.0)))
        rc = np.full(int(k.sum()), 2, dtype=np.int8)
        rc[ang < SAME_DEG] = 0
        rc[ang > REV_DEG] = 1
        rec['rotation_crosscheck'] = dict(
            agreement=round(float((rc == cat[k]).mean()), 4),
            travel_reverse_but_rotation_same=int(((cat[k] == 1) & (rc == 0)).sum()),
            travel_same_but_rotation_reverse=int(((cat[k] == 0) & (rc == 1)).sum()))
    else:
        rec['rotation_crosscheck'] = None
    return rec, P


def build_units(rows):
    """장소 겹침 묶음: 감사의 >1% 겹침 묶음 + 50 m 안에 점이 하나라도 있는 근접 쌍(엄격 규칙)."""
    keys = [r['key'] for r in rows]
    geo = json.load(open(GEO_AUDIT))
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for g in geo['same_split_overlap_groups']:
        for m in g['members'][1:]:
            parent[find(m)] = find(g['members'][0])
    near = []
    for p in csv.DictReader(open(SAME_PAIRS)):
        md = float(p['min_dist_m'])
        fmax = max(float(p['frac_pose_50m_ab']), float(p['frac_pose_50m_ba']))
        if md < 50.0 and fmax <= 0.01:
            near.append(dict(a=p['a'], b=p['b'], min_dist_m=md, frac_pose_50m_max=fmax))
            parent[find(p['a'])] = find(p['b'])
    cross_min = None
    cross_lt50 = []
    for p in csv.DictReader(open(CROSS_PAIRS)):
        if p['comparison'].startswith('measured'):
            md = float(p['min_dist_m'])
            cross_min = md if cross_min is None else min(cross_min, md)
            if md < 50.0 or float(p['max_frac_50m'] or 0) > 0:
                cross_lt50.append((p['a'], p['b'], md))
    comps = {}
    for k in keys:
        comps.setdefault(find(k), []).append(k)
    names = {}
    for members in comps.values():
        s = set(members)
        if s >= {'kitti360/0004'}:
            nm = 'U_kitti09_k360_0004_0006'
        elif s >= {'helipr/Bridge01'}:
            nm = 'U_bridge'
        elif s >= {'helipr/Roundabout01'}:
            nm = 'U_roundabout'
        elif s >= {'mulran/Sejong01'}:
            nm = 'U_sejong'
        elif s >= {'kitti/00'}:
            nm = 'U_kitti00_07'
        elif s >= {'helipr/Town01'}:
            nm = 'U_town'
        elif s >= {'mulran/KAIST01'}:
            nm = 'U_daejeon_kaist_riverside'
        elif s >= {'mulran/DCC01'}:
            nm = 'U_daejeon_dcc'
        elif s >= {'nclt/2012-01-08'}:
            nm = 'U_nclt'
        elif s >= {'mcd/kth_day_06'}:
            nm = 'U_mcd_kth'
        elif s >= {'mcd/tuhh_day_02'}:
            nm = 'U_mcd_tuhh'
        elif s >= {'kitti/05'}:
            nm = 'U_kitti05_06'
        else:
            assert len(members) == 1, members
            nm = 'U_' + members[0].replace('/', '_')
        assert nm not in names, nm
        names[nm] = sorted(members)
    return names, dict(near_pairs_lt50m_le1pct=near, cross_split_measured_min_dist_m=cross_min,
                       cross_split_measured_pairs_with_any_point_lt50m=cross_lt50)


def site_of(rec):
    return f"{rec['dataset']}:{rec['site']}" if rec['dataset'] != 'mulran' and rec['dataset'] != 'helipr' \
        else f"{rec['city']}:{rec['site']}"


def capped_weights(recs):
    """사이트별 균등 상한: 각 사이트의 재방문 질의 몫이 SITE_SHARE_CAP 이하가 되도록 사이트 예산 B를 정하고,
    사이트 예산을 그 사이트 시퀀스에 같게 나눈 값을 시퀀스 상한으로 쓴다(시퀀스 안에서는 질의를 고르게 뽑는다).
    반환: 시퀀스별 표본 비율 w(0..1], 시퀀스 상한(질의 수)."""
    by_site = {}
    for r in recs:
        by_site.setdefault(site_of(r), []).append(r)

    def alloc(B):
        w, cap = {}, {}
        for s, rs in by_site.items():
            per = B / len(rs)
            for r in rs:
                Q = r['revisit_queries']
                cap[r['key']] = per
                w[r['key']] = 1.0 if Q <= per else (per / Q if Q else 0.0)
        return w, cap

    def shares(w):
        tot = sum(r['revisit_queries'] * w[r['key']] for r in recs)
        return {s: sum(r['revisit_queries'] * w[r['key']] for r in rs) / tot for s, rs in by_site.items()}, tot
    total = sum(r['revisit_queries'] for r in recs)
    lo, hi = 1.0, float(total)
    w, _ = alloc(hi)
    if max(shares(w)[0].values()) <= SITE_SHARE_CAP:
        return w, None
    for _ in range(100):           # 최대 사이트 몫이 상한 이하가 되는 가장 큰 B를 찾는다
        mid = (lo + hi) / 2
        w, _ = alloc(mid)
        if max(shares(w)[0].values()) <= SITE_SHARE_CAP:
            lo = mid
        else:
            hi = mid
    w, cap = alloc(lo)
    return w, lo


def totals(recs, w=None):
    """센서별·방향별 재방문 질의(가장 가까운 짝 방향)와 양성 쌍 합계. w가 있으면 기대 표본 수."""
    out = {}
    for r in recs:
        f = 1.0 if w is None else w[r['key']]
        s = out.setdefault(r['sensor'], dict(sequences=0, revisit_queries=0.0, positive_pairs=0.0,
                                             queries_by_dir_nearest={k: 0.0 for k in DIRS},
                                             pairs_by_dir={k: 0.0 for k in DIRS}))
        s['sequences'] += 1
        s['revisit_queries'] += f * r['revisit_queries']
        s['positive_pairs'] += f * r['positive_pairs']
        for k in DIRS:
            s['queries_by_dir_nearest'][k] += f * r['queries_by_dir_nearest'][k]
            s['pairs_by_dir'][k] += f * r['pairs_by_dir'][k]
    allv = dict(sequences=0, revisit_queries=0.0, positive_pairs=0.0,
                queries_by_dir_nearest={k: 0.0 for k in DIRS}, pairs_by_dir={k: 0.0 for k in DIRS})
    for s in out.values():
        allv['sequences'] += s['sequences']
        allv['revisit_queries'] += s['revisit_queries']
        allv['positive_pairs'] += s['positive_pairs']
        for k in DIRS:
            allv['queries_by_dir_nearest'][k] += s['queries_by_dir_nearest'][k]
            allv['pairs_by_dir'][k] += s['pairs_by_dir'][k]
    out['ALL'] = allv

    def rnd(x):
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        return int(round(x)) if isinstance(x, float) else x
    return rnd(out)


def shares(recs, group, field, w=None):
    tot = {}
    for r in recs:
        f = 1.0 if w is None else w[r['key']]
        tot[group(r)] = tot.get(group(r), 0.0) + f * r[field]
    s = sum(tot.values())
    return {k: round(v / s, 4) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}


def share_block(recs, w=None):
    return dict(site_by_queries=shares(recs, site_of, 'revisit_queries', w),
                site_by_pairs=shares(recs, site_of, 'positive_pairs', w),
                city_by_queries=shares(recs, lambda r: r['city'], 'revisit_queries', w))


def evaluate_candidates(recs_by_key, units):
    res = {}
    unit_of = {k: u for u, ms in units.items() for k in ms}
    for name, held_units in CANDIDATES.items():
        for u in held_units:
            assert u in units, u
        held = [k for u in held_units for k in units[u]]
        train = [k for k in recs_by_key if k not in held]
        # 장소 분리: 학습과 held-out이 한 묶음을 나눠 갖지 않는지
        for u, ms in units.items():
            assert len({k in held for k in ms}) == 1, (name, u)
        tr = [recs_by_key[k] for k in train]
        he = [recs_by_key[k] for k in held]
        tr_sensors = {r['sensor'] for r in tr}
        he_sensors = {r['sensor'] for r in he}
        pc = place_check(train, held, {k: r['city'] for k, r in recs_by_key.items()})
        if pc['measured_pairs_lt50m'] or pc['unresolved_same_city_pairs']:
            sys.exit(f'STOP: candidate {name} is not place-disjoint: {pc}')
        w, B = capped_weights(tr)
        n_site = {}
        for r in tr:
            n_site[site_of(r)] = n_site.get(site_of(r), 0) + 1
        res[name] = dict(
            heldout_units=held_units, train_units=sorted({unit_of[k] for k in train}),
            train=train, heldout=held,
            train_sensors=sorted(tr_sensors),
            heldout_unseen_sensors=sorted(he_sensors - tr_sensors),
            heldout_seen_sensors=sorted(he_sensors & tr_sensors),
            reference_sequences_heldout={k: k in held for k in ('kitti360/0004', 'helipr/Roundabout01', 'mulran/Sejong03')},
            place_check=pc,
            train_totals=totals(tr), heldout_totals=totals(he),
            train_shares_raw=share_block(tr),
            cap=dict(rule=f'site share of training revisit queries <= {SITE_SHARE_CAP:.0%} per epoch; the site budget '
                          'is split equally over its sequences (per-sequence cap); revisit queries are subsampled '
                          'uniformly inside a sequence and redrawn every epoch (no data dropped)',
                     site_budget_queries_per_epoch=None if B is None else int(B),
                     per_sequence_cap={k: (None if B is None else int(B / n_site[site_of(recs_by_key[k])]))
                                       for k in train},
                     sample_fraction={k: round(w[k], 4) for k in train}),
            train_totals_capped=totals(tr, w), train_shares_capped=share_block(tr, w),
        )
    return res


def place_check(train, held, city):
    """후보의 학습×held-out 모든 쌍을 감사 표에서 찾아 가장 가까운 거리를 모은다.
    측정 쌍: 두 감사 표의 min_dist_m. Bridge 하한 쌍: bridge_rotation_free_check.json의 lb_any_route_point_km.
    감사 표에 없는 쌍(논문에서 같은 분할이고 공통 좌표가 없는 쌍)은 도시가 달라야 한다."""
    same = {frozenset((p['a'], p['b'])): float(p['min_dist_m']) for p in csv.DictReader(open(SAME_PAIRS))}
    cross = {frozenset((p['a'], p['b'])): p for p in csv.DictReader(open(CROSS_PAIRS))}
    brf = json.load(open(BRIDGE_RF))['per_sequence']
    meas, bnd, geo_km = [], [], []
    diff_city, unresolved = 0, []
    for a in train:
        for b in held:
            k = frozenset((a, b))
            if k in same:
                meas.append((same[k], a, b))
            elif k in cross:
                p = cross[k]
                if p['comparison'].startswith('measured'):
                    meas.append((float(p['min_dist_m']), a, b))
                elif p['comparison'] == 'bounded':
                    other = b if 'Bridge' in a else a
                    bnd.append((brf[other]['lb_any_route_point_km'] * 1000, a, b))
                else:
                    geo_km.append(float(p['min_dist_lower_bound_km']))
            elif city[a] != city[b]:
                diff_city += 1
            else:
                unresolved.append((a, b))
    mm, bb = min(meas, default=None), min(bnd, default=None)
    return dict(n_pairs=len(train) * len(held), measured_pairs=len(meas),
                measured_min_dist_m=None if mm is None else round(mm[0], 1),
                measured_min_pair=None if mm is None else [mm[1], mm[2]],
                measured_pairs_lt50m=sum(1 for m in meas if m[0] < 50.0),
                bridge_bounded_pairs=len(bnd), bridge_min_lower_bound_m=None if bb is None else round(bb[0], 1),
                bridge_min_pair=None if bb is None else [bb[1], bb[2]],
                geographic_exclusion_pairs=len(geo_km),
                geographic_min_lower_bound_km=min(geo_km, default=None),
                different_city_no_common_frame_pairs=diff_city, unresolved_same_city_pairs=unresolved)


def xsession_check(pos):
    out = []
    for a, b in XSESSION_PAIRS:
        d, _ = cKDTree(pos[a]).query(pos[b])
        p5, p25, p50, p75 = (round(float(x), 2) for x in np.percentile(d, [5, 25, 50, 75]))
        out.append(dict(a=a, b=b, nn_dist_b_to_a_m=dict(p5=p5, p25=p25, median=p50, p75=p75),
                        frac_b_within_5m_of_a=round(float((d <= RADIUS).mean()), 4),
                        frac_b_within_50m_of_a=round(float((d <= 50).mean()), 4),
                        first_position_gap_m=round(float(np.linalg.norm(pos[b][0] - pos[a][0])), 1)))
    return out


def md_tables(recs, units, cands):
    L = []
    L.append('| 시퀀스 | 센서 | 사이트(도시) | 논문 분할 | 키프레임 | 길이 km | 재방문 질의 | 양성 쌍 | 같은/반대/교차 % (쌍, 방향 아는 것 중) | 반대 질의(최근접) | 최근접 거리 중앙/90% m | 시간 간격 중앙 s |')
    L.append('|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|')
    for r in recs:
        sh = r['pair_share_of_known']
        shs = '–' if sh['same'] is None else f"{100*sh['same']:.0f}/{100*sh['reverse']:.0f}/{100*sh['crossing']:.0f}"
        nd = r['nearest_dist_m']
        nds = '–' if nd['median'] is None else f"{nd['median']:.2f}/{nd['p90']:.2f}"
        tg = '–' if r['time_gap_s_median'] is None else f"{r['time_gap_s_median']:.0f}"
        L.append(f"| {r['key']} | {r['sensor']} | {r['site']}({r['city']}) | {r['paper_split']} | {r['n_keyframes']:,} | "
                 f"{r['path_km']:.1f} | {r['revisit_queries']:,} | {r['positive_pairs']:,} | {shs} | "
                 f"{r['queries_by_dir_nearest']['reverse']:,} | {nds} | {tg} |")
    L.append('')
    L.append('| 묶음 | 시퀀스 | 센서 | 재방문 질의 | 반대(최근접) | 같은(최근접) | 양성 쌍 |')
    L.append('|---|---|---|---:|---:|---:|---:|')
    by = {r['key']: r for r in recs}
    for u, ms in units.items():
        rs = [by[k] for k in ms]
        L.append(f"| {u} | {', '.join(ms)} | {', '.join(sorted({r['sensor'] for r in rs}))} | "
                 f"{sum(r['revisit_queries'] for r in rs):,} | {sum(r['queries_by_dir_nearest']['reverse'] for r in rs):,} | "
                 f"{sum(r['queries_by_dir_nearest']['same'] for r in rs):,} | {sum(r['positive_pairs'] for r in rs):,} |")
    L.append('')
    for name, c in cands.items():
        for side, key in (('학습(원본)', 'train_totals'), ('학습(상한 적용 기대값)', 'train_totals_capped'),
                          ('held-out', 'heldout_totals')):
            L.append(f'후보 {name} — {side}')
            L.append('')
            L.append('| 센서 | 시퀀스 | 재방문 질의 | 같은 | 반대 | 교차 | 미정 | 양성 쌍 | 같은 | 반대 | 교차 | 미정 |')
            L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for s, v in c[key].items():
                qd, pd = v['queries_by_dir_nearest'], v['pairs_by_dir']
                L.append(f"| {s} | {v['sequences']} | {v['revisit_queries']:,} | {qd['same']:,} | {qd['reverse']:,} | "
                         f"{qd['crossing']:,} | {qd['unknown']:,} | {v['positive_pairs']:,} | {pd['same']:,} | "
                         f"{pd['reverse']:,} | {pd['crossing']:,} | {pd['unknown']:,} |")
            L.append('')
    return '\n'.join(L)


def main():
    rows = load_roster()
    OUT.mkdir(parents=True, exist_ok=True)
    recs, pos = [], {}
    for r in rows:
        rec, P = survey_one(r)
        rec['cache_sha256_matches_roster'] = sha256(REPO / r['cache']) == r['cache_sha256']
        recs.append(rec)
        pos[r['key']] = P
        print(f"{rec['key']:24s} n={rec['n_keyframes']:6d} RQ={rec['revisit_queries']:6d} pairs={rec['positive_pairs']:7d} "
              f"q_dir={rec['queries_by_dir_nearest']} sha_ok={rec['cache_sha256_matches_roster']}", flush=True)
    by = {r['key']: r for r in recs}
    # 검산 1: 목표값
    sanity = {}
    for k, (tq, tp) in SANITY.items():
        ok = by[k]['revisit_queries'] == tq and (tp is None or by[k]['positive_pairs'] == tp)
        sanity[k] = dict(target_queries=tq, target_pairs=tp, got_queries=by[k]['revisit_queries'],
                         got_pairs=by[k]['positive_pairs'], ok=ok)
    # 검산 2: 무차별 대입
    brute = {}
    for k in BRUTE_CHECK:
        nq, npair = brute_counts(pos[k])
        brute[k] = dict(queries=nq, pairs=npair,
                        ok=(nq == by[k]['revisit_queries'] and npair == by[k]['positive_pairs']))
    print('sanity', json.dumps(sanity), '\nbrute', json.dumps(brute), flush=True)
    if not all(v['ok'] for v in sanity.values()) or not all(v['ok'] for v in brute.values()):
        sys.exit('STOP: sanity/brute-force mismatch')
    bad_sha = [r['key'] for r in recs if not r['cache_sha256_matches_roster']]
    if bad_sha:
        sys.exit(f'STOP: cache sha256 differs from roster: {bad_sha}')
    units, geo_notes = build_units(rows)
    cands = evaluate_candidates(by, units)
    xs = xsession_check(pos)
    out = dict(
        _meta=dict(date='2026-11-14', script='scripts/_invkey_data_survey_20261114.py',
                   roster=str(ROSTER.relative_to(REPO)), protocol=str(PROTOCOL.relative_to(REPO)),
                   geo_audit=str(GEO_AUDIT.relative_to(REPO)), same_split_pairs=str(SAME_PAIRS.relative_to(REPO)),
                   cross_split_pairs=str(CROSS_PAIRS.relative_to(REPO)),
                   definitions=dict(gap=f'j <= q - {GAP}', radius_m=RADIUS, distance='3D Euclidean of GT translation, float64',
                                    heading=f'unit 3D vector P[i+{HALF_WIN}] - P[i-{HALF_WIN}] (clipped at ends); '
                                            f'unknown if |.| < {STILL_M} m',
                                    direction=f'same < {SAME_DEG} deg, reverse > {REV_DEG} deg, else crossing',
                                    per_query='nearest past match (j <= q - 30) for direction, distance and time gap'),
                   cache_sha256_mismatch=bad_sha),
        sanity=sanity, brute_force_check=brute,
        sequences=recs, overlap_units=units, geo_notes=geo_notes,
        candidates=cands, site_share_cap=SITE_SHARE_CAP, cross_session_frame_check=xs,
    )
    with open(OUT / 'data_survey.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(md_tables(recs, units, cands))
    print('geo_notes', json.dumps(geo_notes, ensure_ascii=False))
    print('rotation_crosscheck', json.dumps({r['key']: r['rotation_crosscheck'] for r in recs}))
    for x in xs:
        print('xsession', json.dumps(x))
    for n, c in cands.items():
        print(n, 'train sensors', c['train_sensors'], 'unseen', c['heldout_unseen_sensors'], 'seen',
              c['heldout_seen_sensors'], 'ref', c['reference_sequences_heldout'],
              'B', c['cap']['site_budget_queries_per_epoch'], 'n_train', len(c['train']), 'n_held', len(c['heldout']))
        print('  place', json.dumps(c['place_check']))
        print('  per-seq cap', {k: v for k, v in c['cap']['per_sequence_cap'].items()})
        for t in ('train_shares_raw', 'train_shares_capped'):
            print(' ', t, json.dumps(c[t], ensure_ascii=False))


if __name__ == '__main__':
    main()
