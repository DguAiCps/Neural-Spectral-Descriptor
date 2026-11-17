#!/usr/bin/env python3
"""그림 6과 같은 수락 규칙으로 정방향(<15도)·역방향(>165도) 재방문의 정답 반환·수락 비율을 데이터셋별로 낸다(원장만 읽음)."""
import sys, json, os
sys.path.insert(0, "/home/pkm/Neural-Spectral-Codec/scripts")
import numpy as np
import _paper_heading_figure_v23 as H
ROOT = H.ROOT; MAIN = H.MAIN; BEV = ROOT / "results/pdv2_review_response_20261104/bev/test"
seqs = sorted(os.listdir(MAIN)); angle = {}
for s in seqs:
    n = np.load(MAIN / s / "diff_s1.npz"); q, g = n["query_idx"], n["min_past_gt_dist"]
    src = json.loads((MAIN / s / "reference_validation.json").read_text())
    pos = np.load(ROOT / next(p for p in src["input_sha256"] if "/cache_" in p))["poses_gt"][:, :3, 3]
    idx = np.arange(len(pos)); d = pos - pos[np.maximum(idx - 5, 0)]; ln = np.linalg.norm(d, axis=1)
    head = d / np.maximum(ln[:, None], 1e-12); ok = (idx >= 5) & (ln >= 1.); a = np.full(len(q), np.nan)
    for r in np.flatnonzero(g <= 5):
        t = q[r]; j = int(np.argmin(np.linalg.norm(pos[:t - 29] - pos[t], axis=1)))
        if ok[t] and ok[j]: a[r] = np.degrees(np.arccos(np.clip(head[t] @ head[j], -1, 1)))
    angle[s] = a
def ds(s):
    return "helipr_Town" if s.startswith("helipr_Town") else s.split("_")[0]
V = {"NSD": [(lambda s, i=i: MAIN / s / f"diff_s{i}.npz", "score") for i in (1, 2, 3)],
     "BEVPlace++": [(lambda s, i=i: BEV / f"s{i}" / f"{s}.npz", "score_negl2") for i in (1, 2, 3)]}
for name, runs in V.items():
    tot = {}
    for path, key in runs:
        for s in seqs:
            d = np.load(path(s)); sc, t, g = d[key], d["top1_geo_dist"], d["min_past_gt_dist"]; sc = np.where(np.isfinite(sc), sc, -np.inf)
            pos, neg = g <= 5, g > 25; true = pos & (t <= 5); wrong = (pos | neg) & ~true
            a = true & (sc > (sc[wrong].max() if wrong.any() else -np.inf)); v = pos & np.isfinite(angle[s])
            for lab, m in (("fwd<15", angle[s] < 15), ("rev>165", angle[s] > 165)):
                k = (ds(s), lab); x = tot.setdefault(k, np.zeros(3)); x += ((a & v & m).sum(), (true & v & m).sum(), (v & m).sum())
    print(name)
    for k in sorted(tot):
        x = tot[k]
        if x[2]: print("  %-12s %-8s n=%6d  R@1 %5.1f  accepted %5.1f" % (k[0], k[1], x[2] / len(runs), 100 * x[1] / x[2], 100 * x[0] / x[2]))
