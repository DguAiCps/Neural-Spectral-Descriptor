#!/usr/bin/env python3
"""invkey 진단 연구: 증강 계수를 섞어 K=16 선형층을 다시 학습한다 (2026-11-15).

모델·손실·배치·양성/음성 규칙·사이트 몫 상한·dev 평가는 scripts/_invkey_train_linear_20261114.py를
그대로 import해 쓴다(train_one, TrainData, eval_keys, dev_summary). 바뀐 것은 한 가지다.
  배치에 들어가는 키프레임(앵커·양성·추가 후보 모두)을 각각 독립으로 확률 p로 증강 계수
  (features_aug1, 조건 all, 키프레임마다 1벌)로 바꾼다. 음성 후보도 바꾸므로 "증강 여부"가
  구분 단서가 되지 않는다. 같은 키프레임이 한 배치에 두 번 들어가면 각각 따로 뽑는다.
  구현: TrainData.X 자리에 MixX를 넣는다. train_one이 부르는 td.X[ids]가 섞인 텐서를 돌려준다.
  바꿀지 정하는 난수는 별도 생성기 default_rng([seed, AUG_SALT])에서 나오므로, 키프레임 뽑기
  난수(default_rng(seed))는 p와 무관하게 같다. p = 0이면 td.X[ids]를 그대로 돌려준다(기존 코드 경로).
출발점: s = linear/stats.npz, W0 = linear/pca_K16.npz (깨끗한 학습과 같은 출발점).
고정값: K 16, tau 0.05, lr 3e-3, 40에폭 (linear/selection.json의 K16 선택값), TF32 끔.
증강 계수는 학습 33시퀀스만 있다. 깨끗한 계수와 행 순서·scan_ids가 같은지 시퀀스마다 단언한다.
증강 텐서(약 5.7 GB)는 GPU가 16 GB라 CPU 메모리에 두고, 바꿀 행만 배치마다 GPU로 옮긴다.

실행 (리포 루트):
  # 1) p 격자 (seed 0) + p = 0 재현 점검, 한 프로세스
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_linear_aug_20261115.py train --p 0 0.25 0.5 0.75 --seeds 0 --tag sweep
  # 2) 선택 (호스트): p in {0.25, 0.5, 0.75} 곡선에서 dev 기준 최대 (p, 에폭) -> selection.json
  python3 scripts/_invkey_train_linear_aug_20261115.py select
  # 3) 최종 (선택한 p, 에폭으로 seed 0 1 2, 가중치 저장)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_linear_aug_20261115.py train --p P --seeds 0 1 2 --epochs E --tag final --save
출력: results/invkey_study_20261114/linear_aug/ (run_*.json, run_final_*.npz, selection.json)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _invkey_train_linear_20261114 import (  # noqa: E402
    CH, DEV, EXCLUDE_TAIL, EXTRA, FEAT, NCH, OUTD as LIN, TrainData, chmod_write, dev_groups,
    load_seq, split, train_one)

OUTA = REPO / "results/invkey_study_20261114/linear_aug"
FEAT_AUG = Path("/workspace/data/invkey_study_20261114/features_aug1")
K, TAU, LR, EPOCHS = 16, 0.05, 3e-3, 40
AUG_SALT = 20261115


def load_aug(key):
    """증강 spec float16 (n,805,16)과 scan_ids. 깨끗한 파일과 같은 제외 규칙."""
    sensor, seq = key.split("/")
    with np.load(FEAT_AUG / f"{sensor}_{seq}.npz") as z:
        spec, ids = z["spec"], z["scan_ids"]
    if key in EXCLUDE_TAIL:
        c = EXCLUDE_TAIL[key]
        spec, ids = spec[:c], ids[:c]
    return spec, ids


class MixX:
    """td.X 대신 쓴다. self[ids]: 각 행을 확률 p로 증강 계수로 바꾼 (n, R, T) float16 GPU 텐서."""

    def __init__(self, X, Xa, p, seed):
        import torch
        self.X, self.Xa, self.p = X, Xa, float(p)
        self.rng = np.random.default_rng([int(seed), AUG_SALT])
        self.n_rows = 0
        self.n_aug = 0
        self.torch = torch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ids):
        x = self.X[ids]
        self.n_rows += len(ids)
        if self.p <= 0.0:
            return x
        m = np.flatnonzero(self.rng.random(len(ids)) < self.p)
        if len(m):
            mt = self.torch.as_tensor(m, device=DEV)
            rows = ids[mt].cpu()
            x[mt] = self.Xa[rows].to(DEV)
            self.n_aug += len(m)
        return x


def train(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    OUTA.mkdir(parents=True, exist_ok=True)
    recs, train_keys, dev, held = split()
    forbidden = list(held) + list(dev)
    st = np.load(LIN / "stats.npz")
    assert sorted(st["train_keys"].tolist()) == sorted(train_keys)
    s = torch.as_tensor(st["s"], device=DEV)
    z = np.load(LIN / f"pca_K{K}.npz")
    assert np.array_equal(z["s"], st["s"])
    W0 = torch.as_tensor(z["W"], device=DEV)
    td = TrainData(recs, train_keys, forbidden)
    # 증강 계수 (CPU)
    Xclean = td.X
    Xa = torch.empty(tuple(Xclean.shape), dtype=torch.float16)
    t0 = time.time()
    aug_check = {}
    for i, k in enumerate(td.keys):
        assert k not in forbidden
        spec_a, ids_a = load_aug(k)
        with np.load(FEAT / f"{k.replace('/', '_')}.npz") as zc:
            ids_c = zc["scan_ids"]
        if k in EXCLUDE_TAIL:
            ids_c = ids_c[:EXCLUDE_TAIL[k]]
        assert np.array_equal(ids_a, ids_c), k
        lo, hi = td.off[i], td.off[i + 1]
        assert len(spec_a) == hi - lo, k
        Xa[lo:hi] = torch.from_numpy(spec_a)
        # 행 일치의 추가 점검: 같은 행끼리 깨끗/증강 차이가 0이 아닌 비율 (증강이 실제로 들어갔는지)
        same = (torch.from_numpy(spec_a).to(DEV) == Xclean[lo:hi]).flatten(1).all(1)
        aug_check[k] = {"n": int(hi - lo), "rows_identical_to_clean": int(same.sum())}
        del spec_a
        print(f"  aug {k:22s} n {hi - lo:6d} identical rows {int(same.sum())}  {time.time() - t0:.0f}s",
              flush=True)
    dev_data = {}
    for k in dev:
        spec, xyz, _ = load_seq(k)
        dev_data[k] = (torch.as_tensor(spec).to(DEV), xyz)
    groups = dev_groups(recs, dev)
    print(f"train {len(train_keys)} seqs, {len(Xclean)} keyframes; dev {len(dev)}", flush=True)
    for p in a.p:
        for seed in a.seeds:
            name = f"{a.tag}_K{K}_tau{TAU}_lr{LR}_p{p}_s{seed}_e{a.epochs}"
            print(f"== {name}", flush=True)
            mx = MixX(Xclean, Xa, p, seed)
            td.X = mx
            try:
                W, info = train_one(td, dev_data, groups, s, W0, TAU, LR, a.epochs, seed,
                                    1, forbidden)
            finally:
                td.X = Xclean
            info.update({"name": name, "K": K, "tau": TAU, "lr": LR, "seed": seed, "p_aug": p,
                         "epochs": a.epochs,
                         "batch": {"chunk": CH, "chunks": NCH, "extra_per_chunk": EXTRA},
                         "optimizer": "Adam, constant lr, no weight decay",
                         "init": "linear/pca_K16.npz", "s": "linear/stats.npz",
                         "aug_features": str(FEAT_AUG), "aug_cond": "all (features_aug1)",
                         "aug_rule": "each keyframe in the batch pool (anchor, positive, extra) replaced "
                                     "independently with prob p by its features_aug1 row; mask rng "
                                     f"default_rng([seed, {AUG_SALT}]) separate from sampling rng",
                         "aug_rows_drawn": mx.n_rows, "aug_rows_replaced": mx.n_aug,
                         "aug_rows_replaced_frac": mx.n_aug / mx.n_rows if mx.n_rows else None,
                         "aug_file_check": aug_check,
                         "dev": dev, "train": train_keys, "heldout_not_loaded": held,
                         "sites": dict(zip(td.keys, td.sites)),
                         "sensors": dict(zip(td.keys, td.sensors)),
                         "sample_fraction": dict(zip(td.keys, map(float, td.w)))})
            chmod_write(OUTA / f"run_{name}.json", text=json.dumps(info, indent=1))
            if a.save:
                chmod_write(OUTA / f"run_{name}.npz", arrays=dict(W=W.cpu().numpy(), s=st["s"]))
            print(f"   replaced {mx.n_aug}/{mx.n_rows} rows", flush=True)
    return 0


def select(a):
    """p in --p 의 sweep 곡선에서 dev 기준(sel_R@1_sensor_mean 최대, 같으면 AP_seq_mean)으로 (p, 에폭)."""
    cands = []
    for p in a.p:
        f = OUTA / f"run_sweep_K{K}_tau{TAU}_lr{LR}_p{p}_s0_e{EPOCHS}.json"
        d = json.load(open(f))
        for c in d["curve"]:
            if c["epoch"] == 0:
                continue
            sm = c["summary"]
            cands.append((sm["sel_R@1_sensor_mean"], sm["AP_seq_mean"], p, c["epoch"], d["name"]))
    cands.sort(reverse=True)
    best = cands[0]
    best_per_p = {}
    for c in cands:
        best_per_p.setdefault(c[2], dict(zip(("sel", "AP", "p", "epoch", "run"), c)))
    sel = {"criterion": "max dev sel_R@1_sensor_mean (tie: AP_seq_mean), epochs >= 1, p in "
                        f"{list(a.p)}, seed 0", "tau": TAU, "lr": LR, "K": K,
           "p": best[2], "epochs": best[3], "dev_sel": best[0], "dev_AP": best[1], "from_run": best[4],
           "best_per_p": best_per_p,
           "top10": [dict(zip(("sel", "AP", "p", "epoch", "run"), c)) for c in cands[:10]]}
    (OUTA / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--p", type=float, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--tag", default="sweep")
    p.add_argument("--save", action="store_true")
    p = sub.add_parser("select")
    p.add_argument("--p", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    a = ap.parse_args()
    return {"train": train, "select": select}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
