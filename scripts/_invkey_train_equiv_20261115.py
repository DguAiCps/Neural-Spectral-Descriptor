#!/usr/bin/env python3
"""invkey 진단 연구: 방향 축의 주파수 성분에서 계산하는 등변 신경망을 단계별로 학습한다 (2026-11-15).

입력 x = spec (805, 16) / s[r] (s = linear/stats.npz). 방향 축 rfft: X[r, k], k = 0..8.
스캔 회전(방향 축 이동 phi) = 각 성분에 e^{-ik phi}. 아래 두 연산만 쓰므로 임의 각도(칸 사이 포함)에 등변이다.
  성분별 선형   Y[c, k] = sum_h A_k[c, h] Z[h, k]. A_k 복소, k = 0과 8은 실수(실수 출력).
                방향 축 원형 합성곱과 같다. 시간 영역 점별 비선형은 쓰지 않는다.
  불변량 조절   Y[c, k] <- g * Y[c, k], g = exp(MLP(f)) > 0. f는 |Z[h, k]| (회전 불변)를 스캔 전체 노름
                ||Z|| = sqrt(sum_k w_k sum_h |Z[h, k]|^2) (w = 1, 2, ..., 2; to_freq_t와 같은 가중)으로 나눠
                척도를 없애고 log1p(u / DELTA)로 누른 뒤 학습 키프레임 평균·표준편차(prep, 고정)로 표준화한 64x9 = 576개.
                MLP: 576 -> 128 (ReLU) -> 16 (채널별 g[c], 변형 "c") 또는 144 (채널x주파수별 g[c, k], 변형 "ck").
                비선형은 여기(불변량 위)에만 있다.
  출력          irfft(Y, n=16) -> (16, 16) 실수 키. 비교는 기존과 같은 정렬 코사인(rfft -> nfft 128 원형 상관 최댓값).
단계 (STAGES)
  st1     Z = P x (P 64x805 실수, 방향·주파수 공유; 시간 영역에서 곱한 뒤 rfft, 같은 식), Y = A_k Z (A_k 16x64).
  st2c    st1 + 조절 한 층(g[c]).          st2ck   st1 + 조절 한 층(g[c, k]).
  st3c    조절 블록 두 개 64 -> 64 -> 16:  Y1 = g1 * A1_k Z (A1_k 64x64, g1은 |Z|에서),
          Y = g2 * A2_k Y1 (A2_k 16x64, g2는 |Y1|에서). st3ck는 같은 구조에 g[c, k].
초기값: P = linear/pca_K64.npz의 W (64x805, 앞 16행 = pca_K16의 W), A_k = [I16 0] (st3 첫 블록은 I64),
  MLP 마지막 층 0 -> g = exp(0) = 1. 그러면 Y = Z[:16]이고 에폭 0의 키가 PCA-16 키와 같다(점검 1).
  복소 곱은 실수 einsum 넷으로 계산한다(초기 항등에서 곱 1·z와 0만 더해져 정확히 z).
학습: scripts/_invkey_train_linear_20261114.py의 train_one과 같은 루프를 모델로 바꿔 쓴다. TrainData·epoch·batch_ids
  (양성·음성 규칙, 사이트 몫 상한, 배치 64x8 + 추가 128), 난수 소비 순서가 같아 같은 seed면 선형 학습과 같은 배치를
  본다. 손실 정렬 코사인/tau InfoNCE, Adam 고정 lr, 가중치 감쇠 없음(기본), TF32 끔. 에폭마다 dev 9시퀀스.
  --wd > 0: 과적합 확인용. AdamW 분리 감쇠를 조절 MLP(fc1, fc2)에만 건다(P·A는 감쇠 0).
증강(--p > 0): scripts/_invkey_train_linear_aug_20261115.py의 MixX(배치 풀의 각 키프레임을 확률 p로 features_aug1
  행으로 교체, 마스크 난수 default_rng([seed, 20261115])). 증강 텐서는 CPU에 둔다.

실행 (리포 루트, 컨테이너 안):
  # 0) f 표준화 통계와 초기값 점검 (학습 키프레임 4개마다 1개)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_train_equiv_20261115.py prep
  # 1) dev 격자 (seed 0, 40에폭, lr 1e-3 3e-3)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_train_equiv_20261115.py \
    train --stage st1 st2c st2ck --lr 1e-3 3e-3 --seeds 0 --tag sweep --save
  # 2) 선택 (호스트): 단계마다 dev 최대 (lr, 에폭) -> equiv/selection.json
  python3 scripts/_invkey_train_equiv_20261115.py select --stage st1 st2c st2ck
  # 3) 최종: 선택한 (lr, 에폭)으로 seed 0 1 2
  docker exec ... train --stage ST --lr LR --seeds 0 1 2 --epochs E --tag final --save
  # 4) 가장 나은 단계 (호스트, 최종 seed 평균 dev) -> selection.json["best_stage"]
  python3 scripts/_invkey_train_equiv_20261115.py best --stage st1 st2ck st3ck
  # 5) 가장 나은 단계 + 증강 p 0.25 (같은 lr·에폭)
  docker exec ... train --stage ST --lr LR --p 0.25 --seeds 0 1 2 --epochs E --tag finalaug --save
출력: results/invkey_study_20261114/equiv/ (feat_stats.npz, prep_checks.json, run_*.json, run_*.npz, selection.json)
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
    CH, DEV, DIRS, EXCLUDE_TAIL, EXTRA, FEAT, NCH, NEG_GAP, NEG_M, NFFT, OUTD as LIN, R, T, TrainData,
    chmod_write, dev_groups, dev_summary, encode, eval_keys, load_seq, split, to_freq_t)
from _invkey_train_linear_aug_20261115 import AUG_SALT, FEAT_AUG, MixX, load_aug  # noqa: E402

try:
    import torch
except ImportError:          # 호스트(select·best)에는 torch가 없다
    torch = None

OUTE = REPO / "results/invkey_study_20261114/equiv"
TAU, EPOCHS = 0.05, 40
H, KOUT = 64, 16
NF = T // 2 + 1              # 9
HID = 128
DELTA = 1e-3
WFREQ = [1.0] + [2.0] * (NF - 1)
STAGES = {
    "st1": [(H, KOUT, None)],
    "st2c": [(H, KOUT, "c")],
    "st2ck": [(H, KOUT, "ck")],
    "st3c": [(H, H, "c"), (H, KOUT, "c")],
    "st3ck": [(H, H, "ck"), (H, KOUT, "ck")],
}


# ---------------------------------------------------------------- 모델
def mod_feats(Zr, Zi):
    """(n, h, 9) 실수부·허수부 -> 척도 없앤 log1p 크기 (n, h*9)."""
    m2 = Zr * Zr + Zi * Zi
    w = torch.tensor(WFREQ, dtype=Zr.dtype, device=Zr.device)
    nrm2 = (m2 * w).sum((1, 2), keepdim=True)
    u = torch.sqrt(m2 / nrm2 + 1e-20)
    return torch.log1p(u / DELTA).flatten(1)


if torch is not None:
    class Block(torch.nn.Module):
        """Y[c, k] = g * sum_h A_k[c, h] Z[h, k].  A_k: Ar (9, c, h), Ai (7, c, h) (k = 1..7)."""

        def __init__(self, hin, cout, mod, mu=None, sd=None):
            super().__init__()
            self.hin, self.cout, self.mod = hin, cout, mod
            Ar = torch.zeros(NF, cout, hin)
            Ar[:, :, :cout] = torch.eye(cout)
            self.Ar = torch.nn.Parameter(Ar)
            self.Ai = torch.nn.Parameter(torch.zeros(NF - 2, cout, hin))
            if mod:
                self.fc1 = torch.nn.Linear(hin * NF, HID)
                self.fc2 = torch.nn.Linear(HID, cout * (NF if mod == "ck" else 1))
                torch.nn.init.zeros_(self.fc2.weight)
                torch.nn.init.zeros_(self.fc2.bias)
                self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32).clone())
                self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32).clone())

        def forward(self, Zr, Zi):
            z0 = self.Ai.new_zeros(1, self.cout, self.hin)
            Ai = torch.cat([z0, self.Ai, z0])
            e = "kch,nhk->nck"
            Yr = torch.einsum(e, self.Ar, Zr) - torch.einsum(e, Ai, Zi)
            Yi = torch.einsum(e, self.Ar, Zi) + torch.einsum(e, Ai, Zr)
            logg = None
            if self.mod:
                f = (mod_feats(Zr, Zi) - self.mu) / self.sd
                logg = self.fc2(torch.relu(self.fc1(f))).view(len(f), self.cout, -1)
                g = torch.exp(logg)
                Yr, Yi = Yr * g, Yi * g
            return Yr, Yi, logg

    class EquivNet(torch.nn.Module):
        def __init__(self, stage, P0, s, mu=None, sd=None):
            super().__init__()
            self.stage = stage
            self.register_buffer("s", torch.as_tensor(s, dtype=torch.float32).clone())
            self.P = torch.nn.Parameter(torch.as_tensor(P0, dtype=torch.float32).clone())
            self.blocks = torch.nn.ModuleList([Block(hin, cout, mod, mu, sd)
                                               for hin, cout, mod in STAGES[stage]])

        def core(self, Zr, Zi):
            loggs = []
            for b in self.blocks:
                Zr, Zi, lg = b(Zr, Zi)
                loggs.append(lg)
            return Zr, Zi, loggs

        def first(self, x):
            """x (n, 805, 16) float32 spec -> Z (실수부, 허수부) (n, 64, 9). 시간 영역 P 곱 후 rfft."""
            z = torch.einsum("hr,nrt->nht", self.P, x / self.s[None, :, None])
            Z = torch.fft.rfft(z, dim=-1)
            return Z.real.contiguous(), Z.imag.contiguous()

        def forward_freq(self, Xr, Xi):
            """주파수 입력 X = rfft(spec) (n, 805, 9)에서 바로 계산(점검 3b용). -> Y 실수부·허수부 (n, 16, 9)."""
            sc = self.s[None, :, None]
            Zr = torch.einsum("hr,nrk->nhk", self.P, Xr / sc)
            Zi = torch.einsum("hr,nrk->nhk", self.P, Xi / sc)
            return self.core(Zr, Zi)

        def forward(self, x):
            Zr, Zi = self.first(x)
            Yr, Yi, loggs = self.core(Zr, Zi)
            return torch.fft.irfft(torch.complex(Yr, Yi), n=T, dim=-1), loggs


def n_params(model):
    out = {"P": int(model.P.numel())}
    for i, b in enumerate(model.blocks):
        out[f"A{i + 1}"] = int(b.Ar.numel() + b.Ai.numel())
        if b.mod:
            out[f"mlp{i + 1}"] = int(sum(p.numel() for n, p in b.named_parameters() if n.startswith("fc")))
    out["total"] = int(sum(p.numel() for p in model.parameters()))
    assert out["total"] == sum(v for k, v in out.items() if k != "total")
    return out


def model_encode(model, X, chunk=2048):
    """X (n, 805, 16) float16 (numpy 또는 GPU 텐서) -> 키 (n, 16, 16) float32 GPU. encode()와 같은 묶음 처리."""
    out = []
    for c in range(0, len(X), chunk):
        x = torch.as_tensor(X[c:c + chunk]).to(DEV).float()
        out.append(model(x)[0])
    return torch.cat(out)


def base_tensors():
    """s, P0 (pca_K64 W), mu, sd (feat_stats) GPU 텐서."""
    st = np.load(LIN / "stats.npz")
    z64 = np.load(LIN / "pca_K64.npz")
    assert np.array_equal(z64["s"], st["s"])
    s = torch.as_tensor(st["s"], device=DEV)
    P0 = torch.as_tensor(z64["W"], device=DEV)
    fs = np.load(OUTE / "feat_stats.npz") if (OUTE / "feat_stats.npz").is_file() else None
    mu = torch.as_tensor(fs["mu"], device=DEV) if fs is not None else None
    sd = torch.as_tensor(fs["sd"], device=DEV) if fs is not None else None
    return s, P0, mu, sd


def save_model(path, model, cfg):
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    arrays["cfg"] = np.array(json.dumps(cfg))
    chmod_write(path, arrays=arrays)


def load_model(path):
    z = np.load(path)
    cfg = json.loads(str(z["cfg"]))
    s, P0, mu, sd = base_tensors()
    m = EquivNet(cfg["stage"], P0, s, mu, sd).to(DEV)
    sdict = {k: torch.as_tensor(z[k], device=DEV) for k in z.files if k != "cfg"}
    m.load_state_dict(sdict)
    m.eval()
    return m, cfg


# ---------------------------------------------------------------- prep
def prep(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUTE.mkdir(parents=True, exist_ok=True)
    recs, train, dev, held = split()
    st = np.load(LIN / "stats.npz")
    assert sorted(st["train_keys"].tolist()) == sorted(train)
    z16, z64 = np.load(LIN / "pca_K16.npz"), np.load(LIN / "pca_K64.npz")
    assert np.array_equal(z16["W"], z64["W"][:16]) and np.array_equal(z16["s"], z64["s"])
    s, P0, _, _ = base_tensors()
    m0 = EquivNet("st1", P0, s).to(DEV)
    t0 = time.time()
    S1 = torch.zeros(H * NF, dtype=torch.float64, device=DEV)
    S2 = torch.zeros(H * NF, dtype=torch.float64, device=DEV)
    U = []
    cnt = 0
    for k in train:
        assert k not in held and k not in dev
        spec, _, _ = load_seq(k)
        sub = spec[::a.every]
        with torch.no_grad():
            for c in range(0, len(sub), 2048):
                Zr, Zi = m0.first(torch.as_tensor(sub[c:c + 2048]).to(DEV).float())
                f = mod_feats(Zr, Zi).double()
                S1 += f.sum(0)
                S2 += (f * f).sum(0)
                cnt += len(f)
                m2 = Zr * Zr + Zi * Zi
                nrm2 = (m2 * torch.tensor(WFREQ, device=DEV)).sum((1, 2), keepdim=True)
                U.append(torch.sqrt(m2 / nrm2)[::16].cpu().numpy())
        print(f"  feat {k:22s} n {len(sub)}  {time.time() - t0:.0f}s", flush=True)
    mu = (S1 / cnt).cpu().numpy()
    sd = np.sqrt(np.maximum((S2 / cnt).cpu().numpy() - mu ** 2, 0.0))
    U = np.concatenate(U)                                  # (m, 64, 9)
    upct = {f"p{q}": np.percentile(U, q, axis=0).tolist() for q in (5, 50, 95)}
    chmod_write(OUTE / "feat_stats.npz", arrays=dict(
        mu=mu.astype(np.float32), sd=np.maximum(sd, 1e-6).astype(np.float32), delta=np.float64(DELTA),
        n=np.int64(cnt), every=np.int64(a.every), train_keys=np.array(train)))
    print(f"feat stats from {cnt} keyframes; sd min {sd.min():.4g} max {sd.max():.4g}; "
          f"u median by k (h-mean) {np.median(U, 0).mean(0).round(4).tolist()}", flush=True)

    # 초기값 점검: 모든 단계의 에폭 0 키 vs PCA-16 키 (dev 첫 두 시퀀스)
    s, P0, mu, sd_t = base_tensors()
    W16 = torch.as_tensor(z16["W"], device=DEV)
    W64 = torch.as_tensor(z64["W"], device=DEV)
    chk = {"u_percentiles_by_h_k": upct, "init": {}}
    for k in dev[:2]:
        spec, _, _ = load_seq(k)
        with torch.no_grad():
            Dp = encode(spec, s, W16)
            D64 = encode(spec, s, W64)
            chk["init"][k] = {"einsum_P64_first16_bitequal_W16": bool(torch.equal(D64[:, :16], Dp))}
            for stg in STAGES:
                m = EquivNet(stg, P0, s, mu, sd_t).to(DEV)
                Dm = model_encode(m, spec)
                d = (Dm - Dp).abs()
                chk["init"][k][stg] = {"max_abs_diff": float(d.max()),
                                       "max_rel_diff": float(d.max() / Dp.abs().max()),
                                       "bitequal": bool(torch.equal(Dm, Dp)), "n_params": n_params(m)}
        print(k, json.dumps(chk["init"][k]), flush=True)
    chmod_write(OUTE / "prep_checks.json", text=json.dumps(chk, indent=1))
    return 0


# ---------------------------------------------------------------- 학습
def load_aug_cpu(td, forbidden):
    """scripts/_invkey_train_linear_aug_20261115.py train()과 같은 증강 텐서 적재·점검 (CPU float16)."""
    Xclean = td.X
    Xa = torch.empty(tuple(Xclean.shape), dtype=torch.float16)
    aug_check = {}
    t0 = time.time()
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
        same = (torch.from_numpy(spec_a).to(DEV) == Xclean[lo:hi]).flatten(1).all(1)
        aug_check[k] = {"n": int(hi - lo), "rows_identical_to_clean": int(same.sum())}
        del spec_a
        print(f"  aug {k:22s} n {hi - lo:6d} identical rows {int(same.sum())}  {time.time() - t0:.0f}s",
              flush=True)
    return Xa, aug_check


def _gstats(loggs):
    out = {}
    for i, lg in enumerate(loggs):
        if lg is not None:
            v = lg.detach()
            out[f"block{i + 1}"] = {"mean": float(v.mean()), "std": float(v.std()),
                                    "min": float(v.min()), "max": float(v.max()),
                                    "std_over_scans": float(v.std(0).mean())}
    return out


def train_model(td, dev_data, groups, stage, base, tau, lr, epochs, seed, forbidden, wd=0.0):
    """_invkey_train_linear_20261114.train_one과 같은 루프(난수 소비 순서 같음), 선형층 대신 EquivNet."""
    from _poolalt_probe_20261114 import aligned_sim
    s, P0, mu, sd = base
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = EquivNet(stage, P0, s, mu, sd).to(DEV)
    if wd == 0.0:
        opt = torch.optim.Adam(list(model.parameters()), lr=lr)
    else:
        # 감쇠는 조절 MLP(fc*)에만: P·A는 척도 불변 경로라 감쇠가 유효 학습률만 바꾼다
        mlp = [p for n, p in model.named_parameters() if ".fc" in n]
        rest = [p for n, p in model.named_parameters() if ".fc" not in n]
        assert mlp, "weight decay needs a modulated stage"
        opt = torch.optim.AdamW([{"params": rest, "weight_decay": 0.0},
                                 {"params": mlp, "weight_decay": wd}], lr=lr)
    curve = []
    comp_total = np.zeros((len(td.keys), len(DIRS)), np.int64)
    used_seq = np.zeros(len(td.keys), bool)
    negstat = np.zeros(3)
    tim = {"train": 0.0, "dev": 0.0}

    def dev_eval(ep, loss, extra):
        tt = time.time()
        per = {}
        with torch.no_grad():
            for k, (spec16, xyz) in dev_data.items():
                per[k] = eval_keys(model_encode(model, spec16), xyz, chunk=64)
        summ = dev_summary(per, groups)
        curve.append({"epoch": ep, "loss": loss, "summary": summ,
                      "per_seq": {k: {m: v[m] for m in ("R@1", "R@100", "AP", "max_F1", "R@100P",
                                                         "R@100P_10m", "n_pos")}
                                  for k, v in per.items()}, **extra})
        tim["dev"] += time.time() - tt
        print(f"   ep {ep:3d} loss {loss if loss is None else round(loss, 4)}  "
              f"sel {summ['sel_R@1_sensor_mean']:.4f}  AP {summ['AP_seq_mean']:.4f}  "
              f"by sensor {json.dumps({g: round(v, 4) for g, v in summ['R@1_pooled_by_sensor'].items()})}"
              f"  rev {summ['R@1_reverse_pooled']:.4f}"
              + (f"  logg {json.dumps({b: round(v['std'], 3) for b, v in extra['logg'].items()})}"
                 if extra.get("logg") else ""), flush=True)
        return per

    step0 = dev_eval(0, None, {})
    t0 = time.time()
    for ep in range(1, epochs + 1):
        te = time.time()
        batches, comp = td.epoch(rng)
        comp_total += comp
        losses = []
        loggs = []
        for b in batches:
            a_ids, p_ids, e_ids = td.batch_ids(b, rng)
            ids = torch.cat([a_ids, p_ids, e_ids])
            Bn = len(a_ids)
            used_seq[np.unique(td.seq[ids].cpu().numpy())] = True
            x = td.X[ids].float()
            y, loggs = model(x)
            F, nrm = to_freq_t(y)
            S = aligned_sim(F[:, :Bn], nrm[:Bn], F, nrm, NFFT)
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
        assert np.isfinite(losses).all(), "non-finite loss"
        tim["train"] += time.time() - te
        dev_eval(ep, float(np.mean(losses)), {"logg": _gstats(loggs)})
        print(f"   ep {ep} steps {len(batches)} anchors {int(comp.sum())}  {time.time() - t0:.0f}s",
              flush=True)
    names = {td.keys[i] for i in np.flatnonzero(used_seq)}
    leak = sorted(names & set(forbidden))
    assert not leak, leak
    return model, {
        "curve": curve, "step0_dev": {k: {m: v[m] for m in ("R@1", "AP", "R@100P")}
                                      for k, v in step0.items()},
        "composition_total": {td.keys[i]: dict(zip(DIRS, map(int, comp_total[i])))
                              for i in range(len(td.keys))},
        "neg_per_anchor": {"same_seq": negstat[0] / negstat[2], "other_city": negstat[1] / negstat[2]},
        "leak_check": {"sequences_touched": sorted(names), "forbidden_touched": leak,
                       "n_forbidden": len(forbidden)},
        "train_s": time.time() - t0, "train_only_s": tim["train"], "dev_eval_s": tim["dev"],
        "n_params": n_params(model)}


def run_name(tag, stage, lr, p, wd, seed, epochs):
    return (f"{tag}_{stage}_tau{TAU}_lr{lr}" + (f"_p{p}" if p > 0 else "") + (f"_wd{wd}" if wd else "")
            + f"_s{seed}_e{epochs}")


def train(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recs, train_keys, dev, held = split()
    forbidden = list(held) + list(dev)
    st = np.load(LIN / "stats.npz")
    assert sorted(st["train_keys"].tolist()) == sorted(train_keys)
    fs = np.load(OUTE / "feat_stats.npz")
    assert sorted(fs["train_keys"].tolist()) == sorted(train_keys)
    base = base_tensors()
    td = TrainData(recs, train_keys, forbidden)
    Xclean = td.X
    Xa, aug_check = load_aug_cpu(td, forbidden) if max(a.p) > 0 else (None, None)
    dev_data = {}
    for k in dev:
        spec, xyz, _ = load_seq(k)
        dev_data[k] = (torch.as_tensor(spec).to(DEV), xyz)
    groups = dev_groups(recs, dev)
    print(f"train {len(train_keys)} seqs, {len(Xclean)} keyframes; dev {len(dev)}", flush=True)
    for stage in a.stage:
        for lr in a.lr:
            for p in a.p:
                for wd in a.wd:
                    for seed in a.seeds:
                        name = run_name(a.tag, stage, lr, p, wd, seed, a.epochs)
                        print(f"== {name}", flush=True)
                        mx = None
                        if p > 0:
                            mx = MixX(Xclean, Xa, p, seed)
                            td.X = mx
                        try:
                            model, info = train_model(td, dev_data, groups, stage, base, TAU, lr, a.epochs,
                                                      seed, forbidden, wd)
                        finally:
                            td.X = Xclean
                        info.update({
                            "name": name, "stage": stage, "blocks": STAGES[stage], "tau": TAU, "lr": lr,
                            "seed": seed, "epochs": a.epochs, "p_aug": p, "weight_decay": wd,
                            "batch": {"chunk": CH, "chunks": NCH, "extra_per_chunk": EXTRA},
                            "optimizer": ("Adam, constant lr, no weight decay" if not wd else
                                          f"AdamW, constant lr, decoupled weight decay {wd} on modulation "
                                          "MLP (fc1, fc2) only"),
                            "init": "P = linear/pca_K64.npz W; A_k = [I 0]; last MLP layer 0 (g = 1)",
                            "mod_feature": f"log1p((|Z|/||Z||)/{DELTA}) standardized by equiv/feat_stats.npz",
                            "mlp_hidden": HID, "tf32": False,
                            "dev": dev, "train": train_keys, "heldout_not_loaded": held,
                            "sites": dict(zip(td.keys, td.sites)),
                            "sensors": dict(zip(td.keys, td.sensors)),
                            "sample_fraction": dict(zip(td.keys, map(float, td.w)))})
                        if mx is not None:
                            info.update({
                                "aug_features": str(FEAT_AUG), "aug_cond": "all (features_aug1)",
                                "aug_rule": "each keyframe in the batch pool (anchor, positive, extra) replaced "
                                            "independently with prob p by its features_aug1 row; mask rng "
                                            f"default_rng([seed, {AUG_SALT}]) separate from sampling rng",
                                "aug_rows_drawn": mx.n_rows, "aug_rows_replaced": mx.n_aug,
                                "aug_rows_replaced_frac": mx.n_aug / mx.n_rows if mx.n_rows else None,
                                "aug_file_check": aug_check})
                        chmod_write(OUTE / f"run_{name}.json", text=json.dumps(info, indent=1))
                        if a.save:
                            save_model(OUTE / f"run_{name}.npz", model,
                                       {"stage": stage, "name": name, "n_params": info["n_params"]})
                        del model
                        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- 선택 (호스트)
def _cands(path_glob):
    cands = []
    for p in sorted(OUTE.glob(path_glob)):
        d = json.load(open(p))
        for c in d["curve"]:
            if c["epoch"] == 0:
                continue
            sm = c["summary"]
            cands.append((sm["sel_R@1_sensor_mean"], sm["AP_seq_mean"], d["lr"], c["epoch"], d["name"]))
    cands.sort(reverse=True)
    return cands


def select(a):
    """단계마다 sweep 곡선(lr x 에폭 1..40)에서 dev 기준 최대(sel_R@1_sensor_mean, 같으면 AP_seq_mean)."""
    path = OUTE / "selection.json"
    sel = json.load(open(path)) if path.is_file() else {}
    for stage in a.stage:
        cands = _cands(f"run_{a.tag}_{stage}_tau{TAU}_lr*_s0_e{EPOCHS}.json")
        best = cands[0]
        per_lr = {}
        for c in cands:
            per_lr.setdefault(str(c[2]), dict(zip(("sel", "AP", "lr", "epoch", "run"), c)))
        sel[stage] = {"criterion": "max dev sel_R@1_sensor_mean (tie: AP_seq_mean), epochs >= 1, seed 0",
                      "tag": a.tag, "lr": best[2], "epochs": best[3], "dev_sel": best[0], "dev_AP": best[1],
                      "from_run": best[4], "best_per_lr": per_lr,
                      "top5": [dict(zip(("sel", "AP", "lr", "epoch", "run"), c)) for c in cands[:5]]}
        print(stage, json.dumps({k: v for k, v in sel[stage].items() if k != "top5"}))
    path.write_text(json.dumps(sel, indent=1))
    return 0


def best(a):
    """최종 seed 0 1 2의 마지막 에폭 dev 기준 평균이 가장 높은 단계."""
    path = OUTE / "selection.json"
    sel = json.load(open(path))
    rows = {}
    for stage in a.stage:
        s = sel[stage]
        vals = []
        for seed in (0, 1, 2):
            d = json.load(open(OUTE / f"run_final_{stage}_tau{TAU}_lr{s['lr']}_s{seed}_e{s['epochs']}.json"))
            sm = d["curve"][-1]["summary"]
            assert d["curve"][-1]["epoch"] == s["epochs"]
            vals.append((sm["sel_R@1_sensor_mean"], sm["AP_seq_mean"]))
        v = np.array(vals)
        rows[stage] = {"sel_mean": float(v[:, 0].mean()), "sel_min": float(v[:, 0].min()),
                       "sel_max": float(v[:, 0].max()), "AP_mean": float(v[:, 1].mean()),
                       "per_seed": vals}
    lin = []
    for seed in (0, 1, 2):
        d = json.load(open(LIN / f"run_final_K16_tau0.05_lr0.003_s{seed}_e40.json"))
        sm = d["curve"][-1]["summary"]
        lin.append((sm["sel_R@1_sensor_mean"], sm["AP_seq_mean"]))
    order = sorted(rows, key=lambda k: (rows[k]["sel_mean"], rows[k]["AP_mean"]), reverse=True)
    sel["best_stage"] = {"criterion": "max mean over seeds 0-2 of final-epoch dev sel_R@1_sensor_mean "
                                      "(tie: AP_seq_mean)", "stage": order[0], "rows": rows,
                         "lin16_final_dev": {"sel_mean": float(np.mean([x[0] for x in lin])),
                                             "AP_mean": float(np.mean([x[1] for x in lin])), "per_seed": lin}}
    path.write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel["best_stage"], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--every", type=int, default=4)
    p = sub.add_parser("train")
    p.add_argument("--stage", nargs="+", required=True, choices=list(STAGES))
    p.add_argument("--lr", type=float, nargs="+", default=[1e-3])
    p.add_argument("--p", type=float, nargs="+", default=[0.0])
    p.add_argument("--wd", type=float, nargs="+", default=[0.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--tag", default="sweep")
    p.add_argument("--save", action="store_true")
    for c in ("select", "best"):
        p = sub.add_parser(c)
        p.add_argument("--stage", nargs="+", required=True)
        p.add_argument("--tag", default="sweep")
    a = ap.parse_args()
    return {"prep": prep, "train": train, "select": select, "best": best}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
