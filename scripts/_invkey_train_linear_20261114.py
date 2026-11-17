#!/usr/bin/env python3
"""invkey 진단 연구: 고정 서술자 spec(805x16) 위에 선형층 하나를 학습한다 (2026-11-14).

모델: y[k, t] = sum_r W[k, r] * x[r, t] / s[r].  W (K, 805)는 16방향 모두에 같이 쓰고, s[r]는 학습 키프레임의
  행 r 제곱 평균의 제곱근(16방향을 모아서)이다. 편향·비선형은 없다. 같은 사상을 모든 방향에 쓰므로
  "회전 = t축 순환 이동" 성질이 그대로 남는다. 키 비교는 spec과 같다: t축 rfft -> 원형 상관을 nfft 128로
  보간해 최댓값 -> 코사인(scripts/_poolalt_probe_20261114.py의 to_freq·aligned_sim과 같은 식).
초기값: 척도를 맞춘 행의 2차 모멘트(중심을 빼지 않음, 편향 없는 모델과 맞춤) 상위 K 고유벡터 = PCA-K 대조군.

분할: results/invkey_study_20261114/data_survey.json의 candidates["A"]. 학습 쪽 42시퀀스에서 dev 묶음
  (DEV_UNITS, 장소 겹침 묶음 단위)을 떼어 모델 선택에만 쓴다. held-out 21시퀀스는 이 스크립트에서 읽지 않는다.
학습 신호(시퀀스 안):
  양성: 같은 시퀀스, 키프레임 간격 >= 30, GT 거리 <= 5 m. 앵커는 재방문 질의 q, 양성은 q의 과거 짝 j <= q-30 중
        고르게 하나.
  음성: (i) 같은 시퀀스, GT 거리 > 25 m 그리고 간격 > 150, (ii) 도시가 다른 시퀀스의 키프레임.
        같은 시퀀스 5-25 m, 같은 시퀀스 간격 <= 150, 같은 도시 다른 시퀀스는 음성으로 쓰지 않는다(가림).
  균형: 에폭마다 조사 스크립트의 사이트 몫 상한(capped_weights, 25%)으로 시퀀스별 앵커 수를 정해 다시 뽑는다.
  배치: 시퀀스별로 앵커를 64개씩 묶어(CH) 8묶음을 한 배치로 쓴다. 묶음마다 그 시퀀스에서 임의 키프레임 128개를
        더 넣는다(EXTRA). 후보 풀 = 앵커 + 양성 + 추가 키프레임. 손실은 정렬 코사인/온도의 InfoNCE.
        정렬(이동 최댓값)은 손실 안에 있다.
  스캔 단위 증강은 하지 않는다(서술자를 미리 계산해 둠).

실행 (리포 루트, 컨테이너 안):
  # 1) s·PCA를 학습(dev 제외) 키프레임으로 구하고 dev에서 raw/scaled/PCA 대조군을 잰다
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_linear_20261114.py prep
  # 2) dev 격자 탐색. 설정 격자를 한 프로세스에서 돈다(특징을 한 번만 읽는다). 에폭마다 dev를 잰다.
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_linear_20261114.py train --K 16 --tau 0.02 0.05 0.1 \
      --lr 3e-4 1e-3 3e-3 --seeds 0 --epochs 40 --tag sweep
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_linear_20261114.py train --K 64 --tau 0.02 0.05 \
      --lr 1e-3 3e-3 --seeds 0 --epochs 40 --tag sweep
  # 3) 선택 (호스트): dev 지표 최대인 (tau, lr, 에폭) -> selection.json
  python3 scripts/_invkey_train_linear_20261114.py select --K 16 --tag sweep
  python3 scripts/_invkey_train_linear_20261114.py select --K 64 --tag sweep
  # 4) 최종 학습 (2026-11-14 선택값: K16 tau 0.05 lr 3e-3 40에폭, K64 tau 0.02 lr 3e-3 27에폭)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_linear_20261114.py train --K 16 --tau 0.05 --lr 3e-3 --seeds 0 1 2 \
      --epochs 40 --tag final --save
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_linear_20261114.py train --K 64 --tau 0.02 --lr 3e-3 --seeds 0 1 \
      --epochs 27 --tag final --save
출력: results/invkey_study_20261114/linear/ (stats.npz, pca_K*.npz, dev_controls.json, selection.json,
  run_*.json, run_final_*.npz). held-out 평가는 scripts/_invkey_eval_linear_20261114.py.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _invkey_data_survey_20261114 import (  # noqa: E402
    DIRS, capped_weights, classify, headings, positive_pairs, site_of)

SURVEY = REPO / "results/invkey_study_20261114/data_survey.json"
OUTD = REPO / "results/invkey_study_20261114/linear"
FEAT = Path("/workspace/data/invkey_study_20261114/features")
DEV_UNITS = ("U_daejeon_dcc", "U_kitti00_07", "U_kitti_08")
REFS = ("kitti360/0004", "helipr/Roundabout01", "mulran/Sejong03")
EXCLUDE_TAIL = {"mulran/Sejong02": 19068}   # 이 행부터 끝까지 제외(전부 0인 서술자, 원본 0바이트)
R, T = 805, 16
NFFT = 128
GAP, POS_M, NEG_M, NEG_GAP = 30, 5.0, 25.0, 150
CH, NCH, EXTRA = 64, 8, 128
DEV = "cuda"


# ---------------------------------------------------------------- 공통
def chmod_write(path, text=None, arrays=None):
    path = Path(path)
    if text is not None:
        path.write_text(text)
    else:
        np.savez(path, **arrays)
    os.chmod(path, 0o666)


def split():
    """(recs, train, dev, held). train = A 학습 쪽에서 dev 묶음을 뺀 것."""
    d = json.load(open(SURVEY))
    recs = {s["key"]: s for s in d["sequences"]}
    A = d["candidates"]["A"]
    units = d["overlap_units"]
    dev = [k for u in DEV_UNITS for k in units[u]]
    held = list(A["heldout"])
    train = [k for k in A["train"] if k not in dev]
    assert set(dev) <= set(A["train"]) and not set(dev) & set(held)
    assert len(train) + len(dev) == len(A["train"]) == 42 and len(held) == 21
    # 장소 겹침 묶음이 train/dev/held 중 한쪽에만 있는지
    role = {**{k: "train" for k in train}, **{k: "dev" for k in dev}, **{k: "held" for k in held}}
    for u, ms in units.items():
        assert len({role[k] for k in ms}) == 1, (u, ms)
    return recs, train, dev, held


def load_seq(key):
    """(spec float16 (n,805,16), xyz float64 (n,3), scan_ids). 제외 행은 뒤에서 자른다."""
    sensor, seq = key.split("/")
    with np.load(FEAT / f"{sensor}_{seq}.npz") as z:
        spec, xyz, ids = z["spec"], z["xyz"], z["scan_ids"]
    if key in EXCLUDE_TAIL:
        c = EXCLUDE_TAIL[key]
        assert not spec[c:].any(), key
        spec, xyz, ids = spec[:c], xyz[:c], ids[:c]
    return spec, xyz, ids


def to_freq_t(D):
    """_poolalt_probe_20261114.to_freq와 같은 식(입력이 GPU 텐서). D (n, R, 16) -> (9, n, R), (n,)."""
    import torch
    Fd = torch.fft.rfft(D, dim=-1)
    w = torch.tensor([1.0] + [2.0] * (Fd.shape[-1] - 1), device=D.device)
    nrm = torch.sqrt((Fd.abs() ** 2 * w).sum((-1, -2)))
    return Fd.movedim(-1, 0).contiguous(), nrm


def encode(X, s=None, W=None, chunk=2048):
    """X (n,805,16) float16 (numpy 또는 GPU 텐서) -> 키 (n, R', 16) float32 GPU.
    s 없음: raw. s만: 행 척도. s와 W: 선형 키 y = W (x / s)."""
    import torch
    out = []
    for c in range(0, len(X), chunk):
        x = torch.as_tensor(X[c:c + chunk]).to(DEV).float()
        if s is not None:
            x = x / s[None, :, None]
        if W is not None:
            x = torch.einsum("kr,nrt->nkt", W, x)
        out.append(x)
    return torch.cat(out)


def eval_seq(D, xyz64, chunk=16):
    """scripts/_poolalt_probe_20261114.py main()의 검색·PR 회계. 거리는 float64 직접 차.

    D (n, R', 16) float32 GPU 키. 질의 q >= 30, 후보 j <= q-30.
    반환: 지표 dict, 질의별 배열(양성 여부, 1위 정답, 과거 최근접 색인)."""
    import torch
    from _poolalt_probe_20261114 import aligned_sim
    from _pr_eval_lib import classify_queries, pr_sweep
    Fdb, ndb = to_freq_t(D)
    assert bool((ndb > 0).all()), "zero-norm key"
    n = D.shape[0]
    xyz = torch.as_tensor(xyz64, dtype=torch.float64, device=DEV)
    idx = torch.arange(n, device=DEV)
    qs = torch.arange(30, n, device=DEV)
    score, top1d, dmin, jnn, hit100 = [], [], [], [], []
    for s in range(0, len(qs), chunk):
        q = qs[s:s + chunk]
        past = idx[None] <= (q[:, None] - 30)
        Dm = (xyz[q][:, None] - xyz[None]).norm(dim=2).masked_fill(~past, torch.inf)
        S = aligned_sim(Fdb[:, q], ndb[q], Fdb, ndb, NFFT).masked_fill(~past, -torch.inf)
        v, t = S.topk(min(100, n), dim=1)
        ok = t <= (q[:, None] - 30)
        score.append(v[:, 0])
        top1d.append(Dm.gather(1, t[:, :1])[:, 0])
        dm, jm = Dm.min(dim=1)
        dmin.append(dm)
        jnn.append(jm)
        hit100.append(((Dm.gather(1, t) <= POS_M) & ok).any(1))
    score = torch.cat(score).double().cpu().numpy()
    top1d, dmin = (torch.cat(v).cpu().numpy() for v in (top1d, dmin))
    jnn, hit100 = torch.cat(jnn).cpu().numpy(), torch.cat(hit100).cpu().numpy()
    cls = classify_queries(dmin)
    pos = cls["positive"]
    pr = pr_sweep(score, top1d, cls)
    elig = pos | cls["negative"]
    fp10 = elig & (top1d > 10.0)
    thr = score[fp10].max() if fp10.any() else -np.inf
    r100p_10 = float((pos & (top1d <= 10.0) & (score > thr)).sum() / pos.sum()) if pos.any() else 0.0
    hit1 = top1d <= POS_M
    m = {"R@1": float(hit1[pos].mean()) if pos.any() else None,
         "R@100": float(hit100[pos].mean()) if pos.any() else None,
         "AP": float(pr["AP"]), "max_F1": float(pr["max_F1"]),
         "R@100P": float(pr["R@100P"]), "R@100P_10m": r100p_10,
         "n_pos": cls["n_pos"], "n_neg": cls["n_neg"], "n_ambiguous": cls["n_ambiguous"],
         "n_hit1": int(hit1[pos].sum()), "n_map": int(n)}
    q_np = np.arange(30, n)
    return m, {"pos": pos, "hit1": hit1, "q": q_np, "jnn": jnn}


def direction_of(xyz64, q, j):
    """조사 스크립트의 진행 방향 정의로 (q, j) 쌍의 방향 분류 0 same 1 reverse 2 crossing 3 unknown."""
    u, moving = headings(np.asarray(xyz64, dtype=np.float64))
    return classify(u, moving, np.asarray(q), np.asarray(j))


def eval_keys(D, xyz64, chunk=16):
    """eval_seq + 양성 질의의 방향별 1위 정답 수."""
    m, a = eval_seq(D, xyz64, chunk)
    pos = a["pos"]
    dirs = direction_of(xyz64, a["q"][pos], a["jnn"][pos])
    h = a["hit1"][pos]
    m["dir_n"] = {d: int((dirs == i).sum()) for i, d in enumerate(DIRS)}
    m["dir_hit1"] = {d: int(h[dirs == i].sum()) for i, d in enumerate(DIRS)}
    return m


def dev_groups(recs, dev):
    g = {}
    for k in dev:
        g.setdefault(recs[k]["sensor"], []).append(k)
    return g


def dev_summary(per, groups):
    """선택 지표: dev 센서 묶음별로 양성 질의를 모은 R@1의 평균. 참고로 시퀀스 평균 AP."""
    pooled = {}
    for gname, ks in groups.items():
        pooled[gname] = sum(per[k]["n_hit1"] for k in ks) / sum(per[k]["n_pos"] for k in ks)
    rev_n = sum(per[k]["dir_n"]["reverse"] for k in per)
    rev_h = sum(per[k]["dir_hit1"]["reverse"] for k in per)
    same_n = sum(per[k]["dir_n"]["same"] for k in per)
    same_h = sum(per[k]["dir_hit1"]["same"] for k in per)
    return {"sel_R@1_sensor_mean": float(np.mean(list(pooled.values()))),
            "R@1_pooled_by_sensor": pooled,
            "AP_seq_mean": float(np.mean([per[k]["AP"] for k in per])),
            "R@1_reverse_pooled": rev_h / rev_n if rev_n else None,
            "R@1_same_pooled": same_h / same_n if same_n else None}


# ---------------------------------------------------------------- prep: s, PCA, dev 대조군
def prep(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, train, dev, held = split()
    OUTD.mkdir(parents=True, exist_ok=True)
    M0 = torch.zeros(R, R, dtype=torch.float64, device=DEV)
    cnt = 0
    used = []
    t0 = time.time()
    for k in train:
        assert k not in held and k not in dev
        spec, _, _ = load_seq(k)
        for c in range(0, len(spec), 1024):
            x = torch.as_tensor(spec[c:c + 1024]).to(DEV).double()      # (b, 805, 16)
            xr = x.permute(0, 2, 1).reshape(-1, R)                      # 방향을 모은 행 벡터
            M0 += xr.T @ xr
            cnt += xr.shape[0]
        used.append(k)
        print(f"  moments {k} n {len(spec)}  {time.time() - t0:.0f}s", flush=True)
    M0 = M0.cpu().numpy()
    s = np.sqrt(np.diag(M0) / cnt)
    M = M0 / cnt / np.outer(s, s)
    ev, V = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    chmod_write(OUTD / "stats.npz", arrays=dict(
        s=s.astype(np.float32), s64=s, eigvals=ev, eigvecs_top64=V[:, :64], n_vectors=cnt,
        train_keys=np.array(used)))
    for K in (16, 64):
        chmod_write(OUTD / f"pca_K{K}.npz", arrays=dict(W=V[:, :K].T.astype(np.float32),
                                                        s=s.astype(np.float32)))
    print(f"stats: {len(used)} seqs, {cnt} row vectors; top eigvals {ev[:5]}; "
          f"energy K16 {ev[:16].sum() / ev.sum():.4f} K64 {ev[:64].sum() / ev.sum():.4f}", flush=True)

    # dev 대조군
    st = torch.as_tensor(s.astype(np.float32), device=DEV)
    reps = {"raw": (None, None), "scaled": (st, None)}
    for K in (16, 64):
        reps[f"pca{K}"] = (st, torch.as_tensor(V[:, :K].T.astype(np.float32), device=DEV))
    groups = dev_groups(recs, dev)
    out = {"dev": dev, "dev_groups": groups, "reps": {}}
    per = {rep: {} for rep in reps}
    for k in dev:
        spec, xyz, _ = load_seq(k)
        for rep, (ss, W) in reps.items():
            with torch.no_grad():
                D = encode(spec, ss, W)
            per[rep][k] = eval_keys(D, xyz, chunk=16 if W is None else 64)
            del D
            torch.cuda.empty_cache()
        print(f"  dev {k} done {time.time() - t0:.0f}s", flush=True)
    for rep in reps:
        out["reps"][rep] = {"per_seq": per[rep], "summary": dev_summary(per[rep], groups)}
        print(f" dev {rep:7s} {json.dumps(out['reps'][rep]['summary'])}", flush=True)
    out["energy"] = {"K16": float(ev[:16].sum() / ev.sum()), "K64": float(ev[:64].sum() / ev.sum())}
    chmod_write(OUTD / "dev_controls.json", text=json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- 학습
class TrainData:
    """학습(dev 제외) 시퀀스의 특징을 GPU float16 한 텐서에 담고, 시퀀스별 재방문 질의와 양성 짝을 둔다."""

    def __init__(self, recs, keys, forbidden):
        import torch
        assert not set(keys) & set(forbidden)
        self.keys = list(keys)
        sizes = []
        for k in keys:
            n = recs[k]["n_keyframes"] - (recs[k]["n_keyframes"] - EXCLUDE_TAIL[k]
                                          if k in EXCLUDE_TAIL else 0)
            sizes.append(n)
        self.off = np.r_[0, np.cumsum(sizes)]
        N = int(self.off[-1])
        self.X = torch.empty((N, R, T), dtype=torch.float16, device=DEV)
        xyz = np.empty((N, 3), np.float64)
        cities = sorted({recs[k]["city"] for k in keys})
        self.city_names = cities
        seq = np.empty(N, np.int64)
        loc = np.empty(N, np.int64)
        city = np.empty(N, np.int64)
        self.rq, self.ptr, self.js, self.u, self.moving = [], [], [], [], []
        t0 = time.time()
        for i, k in enumerate(keys):
            spec, P, _ = load_seq(k)
            assert len(spec) == sizes[i]
            a, b = self.off[i], self.off[i + 1]
            self.X[a:b] = torch.as_tensor(spec).to(DEV)
            xyz[a:b] = P
            seq[a:b] = i
            loc[a:b] = np.arange(b - a)
            city[a:b] = cities.index(recs[k]["city"])
            q, j, _ = positive_pairs(P)
            o = np.lexsort((j, q))
            q, j = q[o], j[o]
            uq, st = np.unique(q, return_index=True)
            self.rq.append(uq)
            self.ptr.append(np.r_[st, len(q)])
            self.js.append(j)
            u, mv = headings(P)
            self.u.append(u)
            self.moving.append(mv)
            assert len(uq) == recs[k]["revisit_queries"], (k, len(uq), recs[k]["revisit_queries"])
            print(f"  load {k:22s} n {b - a:6d} RQ {len(uq):5d}  {time.time() - t0:.0f}s", flush=True)
        self.xyz = torch.as_tensor(xyz, device=DEV)
        self.seq = torch.as_tensor(seq, device=DEV)
        self.loc = torch.as_tensor(loc, device=DEV)
        self.city = torch.as_tensor(city, device=DEV)
        w, B = capped_weights([recs[k] for k in keys])
        self.w = np.array([w[k] for k in keys])
        self.site_budget = B
        self.sites = [site_of(recs[k]) for k in keys]
        self.sensors = [recs[k]["sensor"] for k in keys]

    def epoch(self, rng):
        """사이트 몫 상한으로 시퀀스별 앵커 수를 정해 뽑는다. 반환: 배치 목록, 시퀀스x방향 앵커 수."""
        chunks = []
        comp = np.zeros((len(self.keys), len(DIRS)), np.int64)
        for i in range(len(self.keys)):
            Q = len(self.rq[i])
            if Q == 0:
                continue
            m = self.w[i] * Q
            k = int(np.floor(m)) + int(rng.random() < m - np.floor(m))
            k = min(k, Q)
            pick = rng.choice(Q, k, replace=False)
            a = self.rq[i][pick]
            lo, hi = self.ptr[i][pick], self.ptr[i][pick + 1]
            p = self.js[i][lo + (rng.random(k) * (hi - lo)).astype(np.int64)]
            comp[i] += np.bincount(classify(self.u[i], self.moving[i], a, p), minlength=len(DIRS))
            for c in range(0, k, CH):
                chunks.append((i, a[c:c + CH], p[c:c + CH]))
        order = rng.permutation(len(chunks))
        batches = [[chunks[o] for o in order[b:b + NCH]] for b in range(0, len(order), NCH)]
        return batches, comp

    def batch_ids(self, batch, rng):
        import torch
        A, P, E = [], [], []
        for i, a, p in batch:
            n = self.off[i + 1] - self.off[i]
            A.append(self.off[i] + a)
            P.append(self.off[i] + p)
            E.append(self.off[i] + rng.integers(0, n, EXTRA))
        f = lambda v: torch.as_tensor(np.concatenate(v), device=DEV)  # noqa: E731
        return f(A), f(P), f(E)


def train_one(td, dev_data, groups, s, W0, tau, lr, epochs, seed, log_every=1, forbidden=()):
    import torch
    from _poolalt_probe_20261114 import aligned_sim
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    W = torch.nn.Parameter(W0.clone())
    opt = torch.optim.Adam([W], lr=lr)
    curve = []
    comp_total = np.zeros((len(td.keys), len(DIRS)), np.int64)
    comp_ep1 = None
    used_seq = np.zeros(len(td.keys), bool)
    negstat = np.zeros(3)   # 같은 시퀀스 음성 합, 다른 도시 음성 합, 앵커 수

    def dev_eval(ep, loss):
        per = {}
        with torch.no_grad():
            for k, (spec16, xyz) in dev_data.items():
                per[k] = eval_keys(encode(spec16, s, W.detach()), xyz, chunk=64)
        summ = dev_summary(per, groups)
        curve.append({"epoch": ep, "loss": loss, "summary": summ,
                      "per_seq": {k: {m: v[m] for m in ("R@1", "R@100", "AP", "max_F1", "R@100P",
                                                         "R@100P_10m", "n_pos")}
                                  for k, v in per.items()}})
        print(f"   ep {ep:3d} loss {loss if loss is None else round(loss, 4)}  "
              f"sel {summ['sel_R@1_sensor_mean']:.4f}  AP {summ['AP_seq_mean']:.4f}  "
              f"by sensor {json.dumps({g: round(v, 4) for g, v in summ['R@1_pooled_by_sensor'].items()})}"
              f"  rev {summ['R@1_reverse_pooled']:.4f}", flush=True)
        return per

    step0 = dev_eval(0, None)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        batches, comp = td.epoch(rng)
        comp_total += comp
        if comp_ep1 is None:
            comp_ep1 = comp.copy()
        losses = []
        for b in batches:
            a_ids, p_ids, e_ids = td.batch_ids(b, rng)
            ids = torch.cat([a_ids, p_ids, e_ids])
            Bn = len(a_ids)
            used_seq[np.unique(td.seq[ids].cpu().numpy())] = True
            x = td.X[ids].float() / s[None, :, None]
            y = torch.einsum("kr,nrt->nkt", W, x)
            F, nrm = to_freq_t(y)
            S = aligned_sim(F[:, :Bn], nrm[:Bn], F, nrm, NFFT)          # (Bn, pool)
            with torch.no_grad():
                same = td.seq[a_ids][:, None] == td.seq[ids][None]
                gap = (td.loc[a_ids][:, None] - td.loc[ids][None]).abs()
                dist = (td.xyz[a_ids][:, None] - td.xyz[ids][None]).norm(dim=2)
                neg_same = same & (dist > NEG_M) & (gap > NEG_GAP)
                neg_city = td.city[a_ids][:, None] != td.city[ids][None]
                allow = neg_same | neg_city
                ar = torch.arange(Bn, device=DEV)
                assert not bool(allow[ar, Bn + ar].any())
                allow[ar, Bn + ar] = True
                negstat += [float(neg_same.sum()), float(neg_city.sum()), Bn]
            logits = (S / tau).masked_fill(~allow, -torch.inf)
            loss = -logits.log_softmax(dim=1)[ar, Bn + ar].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if ep % log_every == 0 or ep == epochs:
            dev_eval(ep, float(np.mean(losses)))
        print(f"   ep {ep} steps {len(batches)} anchors {int(comp.sum())}  {time.time() - t0:.0f}s",
              flush=True)
    names = {td.keys[i] for i in np.flatnonzero(used_seq)}
    leak = sorted(names & set(forbidden))
    assert not leak, leak
    return W.detach(), {
        "curve": curve, "step0_dev": {k: {m: v[m] for m in ("R@1", "AP", "R@100P")}
                                      for k, v in step0.items()},
        "composition_total": {td.keys[i]: dict(zip(DIRS, map(int, comp_total[i])))
                              for i in range(len(td.keys))},
        "composition_epoch1": {td.keys[i]: dict(zip(DIRS, map(int, comp_ep1[i])))
                               for i in range(len(td.keys))},
        "neg_per_anchor": {"same_seq": negstat[0] / negstat[2], "other_city": negstat[1] / negstat[2]},
        "leak_check": {"sequences_touched": sorted(names), "forbidden_touched": leak,
                       "n_forbidden": len(forbidden)},
        "train_s": time.time() - t0}


def train(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, train_keys, dev, held = split()
    forbidden = list(held) + list(dev)
    st = np.load(OUTD / "stats.npz")
    assert sorted(st["train_keys"].tolist()) == sorted(train_keys)
    s = torch.as_tensor(st["s"], device=DEV)
    td = TrainData(recs, train_keys, forbidden)
    dev_data = {}
    for k in dev:
        spec, xyz, _ = load_seq(k)
        dev_data[k] = (torch.as_tensor(spec).to(DEV), xyz)
    groups = dev_groups(recs, dev)
    print(f"train {len(train_keys)} seqs, {len(td.X)} keyframes; dev {len(dev)}; site budget "
          f"{td.site_budget}; w<1: {[(k, round(w, 3)) for k, w in zip(td.keys, td.w) if w < 1]}",
          flush=True)
    for K in a.K:
        W0 = torch.as_tensor(np.load(OUTD / f"pca_K{K}.npz")["W"], device=DEV)
        for tau in a.tau:
            for lr in a.lr:
                for seed in a.seeds:
                    name = f"{a.tag}_K{K}_tau{tau}_lr{lr}_s{seed}_e{a.epochs}"
                    print(f"== {name}", flush=True)
                    W, info = train_one(td, dev_data, groups, s, W0, tau, lr, a.epochs, seed,
                                        a.log_every, forbidden)
                    info.update({"name": name, "K": K, "tau": tau, "lr": lr, "seed": seed,
                                 "epochs": a.epochs, "batch": {"chunk": CH, "chunks": NCH,
                                                               "extra_per_chunk": EXTRA},
                                 "optimizer": "Adam, constant lr, no weight decay",
                                 "init": f"pca_K{K}.npz", "dev": dev, "train": train_keys,
                                 "heldout_not_loaded": held,
                                 "sites": dict(zip(td.keys, td.sites)),
                                 "sensors": dict(zip(td.keys, td.sensors)),
                                 "sample_fraction": dict(zip(td.keys, map(float, td.w)))})
                    chmod_write(OUTD / f"run_{name}.json", text=json.dumps(info, indent=1))
                    if a.save:
                        chmod_write(OUTD / f"run_{name}.npz",
                                    arrays=dict(W=W.cpu().numpy(), s=st["s"]))
    return 0


def select(a):
    """dev 곡선에서 (tau, lr, 에폭)을 고른다. 기준: sel_R@1_sensor_mean 최대, 같으면 AP_seq_mean.
    호스트에서 돈다(numpy만). 결과는 selection.json에 K별로 더한다."""
    path = OUTD / "selection.json"
    sel = json.load(open(path)) if path.is_file() else {}
    cands = []
    for p in sorted(OUTD.glob(f"run_{a.tag}_K{a.K}_*.json")):
        d = json.load(open(p))
        for c in d["curve"]:
            if c["epoch"] == 0:
                continue
            sm = c["summary"]
            cands.append((sm["sel_R@1_sensor_mean"], sm["AP_seq_mean"], d["tau"], d["lr"],
                          c["epoch"], d["name"]))
    cands.sort(reverse=True)
    best = cands[0]
    sel[f"K{a.K}"] = {"criterion": "max dev sel_R@1_sensor_mean (tie: AP_seq_mean)",
                      "tag": a.tag, "tau": best[2], "lr": best[3], "epochs": best[4],
                      "dev_sel": best[0], "dev_AP": best[1], "from_run": best[5],
                      "top5": [dict(zip(("sel", "AP", "tau", "lr", "epoch", "run"), c))
                               for c in cands[:5]]}
    path.write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel[f"K{a.K}"], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    p = sub.add_parser("select")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--tag", default="sweep")
    p = sub.add_parser("train")
    p.add_argument("--K", type=int, nargs="+", default=[16])
    p.add_argument("--tau", type=float, nargs="+", default=[0.05])
    p.add_argument("--lr", type=float, nargs="+", default=[1e-3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--tag", default="run")
    p.add_argument("--save", action="store_true")
    a = ap.parse_args()
    return {"prep": prep, "select": select, "train": train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
