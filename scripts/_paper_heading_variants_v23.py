#!/usr/bin/env python3
"""그림 6과 같은 진행 방향 구간·수락 규칙으로 구성 요소 제거 변형과 BEVPlace++(+시간 증거)의 수락 비율·정답 반환 비율을 낸다(원장만 읽음)."""
import sys, json, os
sys.path.insert(0, "/home/pkm/Neural-Spectral-Codec/scripts")
import numpy as np
from pathlib import Path
import _paper_heading_figure_v23 as H
ROOT = H.ROOT; MAIN = H.MAIN
ABL = ROOT / "results/pdv2_macro_ablation_20261106/test"; BEV = ROOT / "results/pdv2_review_response_20261104/bev/test"
seqs = sorted(os.listdir(MAIN)); angle = {}
for s in seqs:   # 그림 6과 같은 진행 방향 차이
    n = np.load(MAIN / s / "diff_s1.npz"); q, g = n["query_idx"], n["min_past_gt_dist"]
    src = json.loads((MAIN / s / "reference_validation.json").read_text())
    pos = np.load(ROOT / next(p for p in src["input_sha256"] if "/cache_" in p))["poses_gt"][:, :3, 3]
    idx = np.arange(len(pos)); d = pos - pos[np.maximum(idx - 5, 0)]; ln = np.linalg.norm(d, axis=1)
    head = d / np.maximum(ln[:, None], 1e-12); ok = (idx >= 5) & (ln >= 1.); a = np.full(len(q), np.nan)
    for r in np.flatnonzero(g <= 5):
        t = q[r]; j = int(np.argmin(np.linalg.norm(pos[:t - 29] - pos[t], axis=1)))
        if ok[t] and ok[j]: a[r] = np.degrees(np.arccos(np.clip(head[t] @ head[j], -1, 1)))
    angle[s] = a
V = {"NSD": [(lambda s, i=i: MAIN / s / f"diff_s{i}.npz", "score") for i in (1, 2, 3)],
     "No temporal evidence": [(lambda s, i=i: ABL / s / f"no_temporal_s{i}.npz", "score") for i in (1, 2, 3)],
     "No refining (identity)": [(lambda s: MAIN / s / "identity.npz", "score")],
     "Uniform subtraction": [(lambda s: MAIN / s / "fixed.npz", "score")],
     "No alignment": [(lambda s, i=i: ABL / s / f"no_alignment_s{i}.npz", "score") for i in (1, 2, 3)],
     "BEVPlace++": [(lambda s, i=i: BEV / f"s{i}" / f"{s}.npz", "score_negl2") for i in (1, 2, 3)],
     "BEVPlace++ + TE": [(lambda s, i=i: BEV / f"s{i}" / f"{s}.npz", "score") for i in (1, 2, 3)]}
E_ = H.EDGES; out = {}
for name, runs in V.items():
    acc, r1 = [], []
    for path, key in runs:
        hb = np.zeros((len(E_) - 1, 3))
        for s in seqs:
            d = np.load(path(s)); sc, t, g = d[key], d["top1_geo_dist"], d["min_past_gt_dist"]; sc = np.where(np.isfinite(sc), sc, -np.inf)
            assert len(sc) == len(angle[s])
            pos, neg = g <= 5, g > 25; true = pos & (t <= 5); wrong = (pos | neg) & ~true
            a = true & (sc > (sc[wrong].max() if wrong.any() else -np.inf)); b = np.digitize(angle[s], E_[1:-1]); v = pos & np.isfinite(angle[s])
            for i in range(len(E_) - 1): hb[i] += ((a & v & (b == i)).sum(), (true & v & (b == i)).sum(), (v & (b == i)).sum())
        acc.append(100 * hb[:, 0] / hb[:, 2]); r1.append(100 * hb[:, 1] / hb[:, 2])
    out[name] = dict(accepted=np.mean(acc, 0).round(1).tolist(), correct_return=np.mean(r1, 0).round(1).tolist())
    print(f"{name:24s} acc {out[name]['accepted']}  R@1 {out[name]['correct_return']}")
print("n", hb[:, 2].astype(int).tolist())
# 90도 구간의 시퀀스 구성
b90 = {s: int(((np.digitize(angle[s], E_[1:-1]) == 3) & np.isfinite(angle[s])).sum()) for s in seqs}
print(sorted(((v, k) for k, v in b90.items() if v), reverse=True)[:12])
json.dump(out, open(ROOT / "results/paper_heading_figure_20261113/crossing/variants_by_heading.json", "w"), indent=1)
