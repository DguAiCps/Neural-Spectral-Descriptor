#!/usr/bin/env python3
"""차이 K/V 입력 변형 1차 탐색(시드 1~3): 차이 입력 계열이 절대 K/V(abskv) 이상이 되는 구성 찾기.

학습·검색 경로는 scripts/_report_model_controls_20261116.py(C)의 train / pipe를 그대로 쓰고,
C의 전역(RUNS·LEDGERS·OUT·VARIANTS·build)만 이 스크립트의 것으로 바꾼다. 데이터 분할·표본·손실 보정·
고정 설정(융합·lambda)·CFG·난수 규약은 C(대칭 조건)와 같다.

변형(모두 대칭 창, 게이트 고정):
  sym        차이 K, 차이 V (= NSD; 기존 ControlledRelationConv "diff")
  abskv      절대 K, 절대 V (기존 "abs_kv")
  absk       절대 K, 차이 V (기존 "abs_k")
  diffk_absv 차이 K, 절대 V
  diff_ln    차이에 층별 LayerNorm(128, affine) 뒤 W_k/W_v
  diff_scale 차이에 층별 학습 스칼라 exp(s), s=0에서 시작 / diff_scale4: s=log 4에서 시작
  diff_unit  차이 / (L2 노름 + 1e-6) * sqrt(128)
  sum_kv     K = tanh(W_k d + W_k2 x_src), V = tanh(W_v d + W_v2 x_src) (W_k2/W_v2 편향 없음, xavier_uniform)
  cat_kv     [d, x_src] -> 새 W_k/W_v 256->128 (편향 없음, xavier_uniform)
  diff_absmag [d, |d|] -> 새 W_k/W_v 256->128
  difflin_k  (추가) 차이 K에서 tanh 제거(차이 V는 그대로). tanh가 없으면 차이 K와 절대 K는 softmax 뒤 같다.
새 모듈은 기존 매개변수를 모두 만든 뒤에 만들므로 같은 시드에서 공유 매개변수의 초기값은 sym과 같다.
W_q·기하 MLP 최종층이 0이라 초기 주의는 균등, 게이트 고정이라 초기 출력은 모든 변형에서 균등 감산과 같다(C.train이 검사).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kvdiff_variants_20261118.py <action>
  train --variant V --seed K | repro | pipe --role R --sequence S --models FILE | pipe_campaign --role R --models FILE | summary
"""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _report_model_controls_20261116 as C  # noqa: E402
from gnn.contrast_spectral_difference import ContrastSpectralDifference  # noqa: E402
from gnn.geometry_interaction import combine_scores  # noqa: E402
from gnn.learned_spectral_difference import ControlledRelationConv  # noqa: E402
from torch_geometric.utils import softmax  # noqa: E402

S, P = C.S, C.P
run, prior = P.run, P.prior
ROOT, CFG, U = C.ROOT, C.CFG, C.U
OUT = ROOT / "results/kv_diff_improve_20261118/sweep1"
BASE = {"sym": "diff", "abskv": "abs_kv", "absk": "abs_k"}
NEW = ("diffk_absv", "diff_ln", "diff_scale", "diff_scale4", "diff_unit", "sum_kv", "cat_kv", "diff_absmag", "difflin_k")
ALL = tuple(BASE) + NEW
SEEDS = (1, 2, 3)

# C의 전역을 이 실험 폴더·변형으로 돌린다(C.train / C.load / C.pipe가 호출 시점에 읽는다).
C.OUT, C.RUNS, C.LEDGERS = OUT, OUT / "runs", OUT / "ledgers"
C.VARIANTS = {v: dict(kind=BASE.get(v, v), window="sym") for v in ALL}


class KVConv(ControlledRelationConv):
    """ControlledRelationConv.forward와 같은 계산에서 K/V 입력(과 K의 tanh)만 변형에 따라 바꾼다."""

    def __init__(self, reference: ControlledRelationConv, variant: str) -> None:
        super().__init__(reference, "diff")
        self.variant = variant
        d = self.channels
        if variant == "diff_ln":
            self.diff_norm = nn.LayerNorm(d)
        elif variant in ("diff_scale", "diff_scale4"):
            self.log_scale = nn.Parameter(torch.tensor(math.log(4.0) if variant == "diff_scale4" else 0.0))
        elif variant == "sum_kv":
            self.W_k2, self.W_v2 = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
            nn.init.xavier_uniform_(self.W_k2.weight)
            nn.init.xavier_uniform_(self.W_v2.weight)
        elif variant in ("cat_kv", "diff_absmag"):
            self.W_k, self.W_v = nn.Linear(2 * d, d, bias=False), nn.Linear(2 * d, d, bias=False)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.xavier_uniform_(self.W_v.weight)
        else:
            assert variant in ("diffk_absv", "diff_unit", "difflin_k"), variant

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        src, tgt = edge_index
        center, neighbor = x[tgt], x[src]
        d = neighbor - center
        v = self.variant
        if v == "diff_ln":
            d = self.diff_norm(d)
        elif v in ("diff_scale", "diff_scale4"):
            d = d * self.log_scale.exp()
        elif v == "diff_unit":
            d = d / (d.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(d.shape[-1])
        if v == "cat_kv":
            d = torch.cat((d, neighbor), -1)
        elif v == "diff_absmag":
            d = torch.cat((d, d.abs()), -1)
        key_in, value_in = d, (neighbor if v == "diffk_absv" else d)
        k = self.W_k(key_in)
        val = self.W_v(value_in)
        if v == "sum_kv":
            k = k + self.W_k2(neighbor)
            val = val + self.W_v2(neighbor)
        if v != "difflin_k":
            k = k.tanh()
        q = self.W_q(center).view(-1, self.heads, self.head_dim)
        content = (q * k.view(-1, self.heads, self.head_dim)).sum(-1) / self.scale
        logits = (combine_scores(content, self.bias_mlp(edge_attr), tgt, len(x), self.geometry_mode)
                  if edge_attr is not None and self.bias_mlp is not None else content)
        self.attention = self.dropout(softmax(logits, tgt, num_nodes=len(x)))
        messages = (val.tanh().view(-1, self.heads, self.head_dim) * self.attention[..., None]).flatten(1)
        out = torch.zeros(len(x), self.channels, dtype=x.dtype, device=x.device)
        return out.index_add(0, tgt, messages)


def build(variant: str, seed: int, stats: dict = None) -> nn.Module:
    """C.build와 같은 순서(시드 -> factory -> ContrastSpectralDifference -> LayerNorm 3개 -> log1p 입력) 뒤 새 conv로 감싼다."""
    cfg = S.parse(CFG)
    torch.manual_seed(seed)
    ref = run.source.factory(288)
    net = ContrastSpectralDifference(ref, BASE.get(variant, "diff")).cuda().eval()
    net.input_norm = nn.LayerNorm(128)
    net.batch_norms = nn.ModuleList([nn.LayerNorm(128), nn.LayerNorm(128)])
    if stats is None:
        z = torch.zeros(288)
        stats = dict(lin=dict(mean=z, std=z + 1), log=dict(mean=z, std=z + 1))
    st = stats["log"]
    net.input_proj = nn.Sequential(S.XNorm(cfg["x"], st["mean"], st["std"]), net.input_proj)
    if variant in NEW:
        net.convs = nn.ModuleList(KVConv(c, variant) for c in net.convs)
    return net.cuda().eval()


C.build = build


# ----------------------------------------------------------------- 재현 검사
def repro() -> None:
    """새 sym·abskv 시드 4 체크포인트가 기존 controls 체크포인트와 비트 단위로 같은지."""
    old_root = ROOT / "results/pdv2_report_model_20261116/controls/runs"
    rec = {}
    for v in ("sym", "abskv"):
        new = torch.load(C.RUNS / v / "s4" / f"u{U}.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        old = torch.load(old_root / v / "s4" / f"u{U}.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        rec[v] = dict(same_keys=list(new) == list(old),
                      bitwise_equal=all(torch.equal(t, old[k]) for k, t in new.items()),
                      max_abs=max(float((t.float() - old[k].float()).abs().max()) for k, t in new.items()),
                      new_sha256=run.HASH(C.RUNS / v / "s4" / f"u{U}.pth"), old_sha256=run.HASH(old_root / v / "s4" / f"u{U}.pth"))
        print("REPRO", v, rec[v], flush=True)
    P.write(OUT / "repro_s4.json", rec)


# ----------------------------------------------------------------- 실제 검색(시퀀스별 프로세스)
def pipe_campaign(role: str, models_file: str, workers: int, nprefix: int) -> None:
    seqs = C.VAL if role == "val" else C.TESTSEQ
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def call(s):
        with (logs / f"pipe_{role}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--role", role, "--sequence", s, "--models", models_file,
                            "--prefix", str(nprefix)], stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", role, i, len(seqs), round(time.time() - tick, 1), flush=True)
    P.write(logs / f"pipe_campaign_{role}_{Path(models_file).stem}.done.json", dict(seconds=time.time() - tick, workers=workers))


# ----------------------------------------------------------------- 집계 (C.summary의 "all" 규약)
def tags(arm: str) -> list:
    if arm == "uniform":
        return ["uniform"]
    return [f"{arm}@s{k}@u{U}" for k in SEEDS]


def ledger(role: str, name: str, tag: str) -> Path:
    if tag == "uniform":
        return S.LEDGERS / role / name / "uniform.npz"
    return C.LEDGERS / role / name / (tag + ".npz")


def summary() -> None:
    import _pdv2_scratch_search_compare_20261116 as K

    keys, metrics = K.KEYS, C.METRICS
    arms = {a: tags(a) for a in ALL + ("uniform",)}
    out = dict(config=CFG, updates=U, seeds=list(SEEDS), fixed=dict(weights=P.selected()[0].tolist(), lam=P.selected()[1]), arms=arms)
    val = {}
    for arm, ts in arms.items():
        per = {t: S.macro("val", {s: run.READ(ledger("val", s.replace("/", "_"), t).with_suffix(".json")) for s in C.VAL}) for t in ts}
        val[arm] = dict(per_model=per, **{k: dict(mean=float(np.mean([per[t][k] for t in ts])),
                                                   sd=float(np.std([per[t][k] for t in ts], ddof=1)) if len(ts) > 1 else None,
                                                   values=[per[t][k] for t in ts]) for k in metrics})
    out["val"] = val
    rows = {}
    for arm, ts in arms.items():
        for t in ts:
            for seqs in K.CITY.values():
                for s in seqs:
                    name = s.replace("/", "_")
                    d = dict(np.load(K.DIRS / (name + ".npz")))
                    with np.load(ledger("test", name, t)) as z:
                        np.testing.assert_array_equal(z["query_idx"], d["query_idx"])
                        rows.setdefault(t, {})[s] = K.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"])

    def macro(t):
        cities = []
        for seqs in K.CITY.values():
            v = [rows[t][s] for s in seqs if rows[t][s] is not None]
            if v:
                cities.append({k: np.mean([x[k] for x in v]) for k in keys})
        return {k: float(np.mean([c[k] for c in cities])) for k in keys}
    per_model = {t: macro(t) for ts in arms.values() for t in ts}
    stats = {arm: {k: dict(mean=float(np.mean([per_model[t][k] for t in ts])),
                           sd=float(np.std([per_model[t][k] for t in ts], ddof=1)) if len(ts) > 1 else None,
                           values=[per_model[t][k] for t in ts]) for k in keys} for arm, ts in arms.items()}
    comp = {}
    for a in ALL + ("uniform",):
        for b in ("abskv", "sym"):
            if a == b:
                continue
            rng = np.random.default_rng(20261104)
            draws, city = [], {}
            for cname, seqs in K.CITY.items():
                avail = [s for s in seqs if rows[arms[a][0]][s] is not None]
                if not avail:
                    continue
                mat = np.array([[100 * (np.mean([rows[t][s][k] for t in arms[a]]) - np.mean([rows[t][s][k] for t in arms[b]]))
                                 for k in keys] for s in avail])
                city[cname] = dict(zip(keys, mat.mean(0).tolist()))
                idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                draws.append(mat[idx].mean(1))
            boot = np.stack(draws).mean(0)
            comp[f"{a}-{b}"] = dict(point_pp={k: 100 * (stats[a][k]["mean"] - stats[b][k]["mean"]) for k in keys},
                                    conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)},
                                    val9_mean_diff_pp={k: 100 * (val[a][k]["mean"] - val[b][k]["mean"]) for k in metrics},
                                    city_delta_pp=city)
    out["test"] = dict(per_model=per_model, arms=stats, differences=comp)
    # 검산: 시험46 파이프라인 집계(S.macro)와 compare 경로
    out["metric_checks"] = {t: max(abs(S.macro("test", {s: run.READ(ledger("test", s.replace("/", "_"), t).with_suffix(".json"))
                                                       for s in C.TESTSEQ})[k] - per_model[t][k]) for k in keys)
                            for ts in arms.values() for t in ts}
    # sym 시드 1~3 원장 = 기존 탐색(S.LEDGERS) NSD 원장인지
    same = {}
    for role, seqs in (("val", C.VAL), ("test", C.TESTSEQ)):
        for k in SEEDS:
            res = []
            for s in seqs:
                name = s.replace("/", "_")
                a = dict(np.load(ledger(role, name, f"sym@s{k}@u{U}")))
                b = dict(np.load(S.LEDGERS / role / name / f"{S.canon(CFG)}@s{k}@u{U}.npz"))
                res.append(all(np.array_equal(a[f], b[f]) for f in ("query_idx", "j_star", "score")))
            same[f"{role}/sym@s{k}"] = dict(all_sequences_bitwise_equal=all(res), sequences=len(res))
    out["sym_ledger_vs_scratch_search"] = same
    tr = {}
    for v in ALL:
        for k in SEEDS:
            r = run.READ(C.RUNS / v / f"s{k}" / "training_complete.json")
            tr[f"{v}@s{k}"] = dict(sha256=r["sha256"], trainable_parameters=r["trainable_parameters"], total_parameters=r["total_parameters"],
                                   init_bitwise_equal_uniform=all(r["init_bitwise_equal_uniform"].values()),
                                   init_max_abs=max(r["init_max_abs_vs_uniform"].values()), seconds=r["seconds"], gpu=r["gpu"],
                                   final_loss=float(np.mean([x["loss"] for x in r["records"][-8:]])),
                                   reproduction=r["reproduction_vs_scratch_search"], val_fixed_AP=r["val_fixed"]["macro"]["AP"])
    out["training"] = tr
    out["training_by_variant"] = {v: dict(trainable_parameters=tr[f"{v}@s1"]["trainable_parameters"],
                                          final_loss_mean=float(np.mean([tr[f"{v}@s{k}"]["final_loss"] for k in SEEDS])),
                                          val_fixed_AP_mean=float(np.mean([tr[f"{v}@s{k}"]["val_fixed_AP"] for k in SEEDS])))
                                  for v in ALL}
    rp = OUT / "repro_s4.json"
    out["repro_s4"] = run.READ(rp) if rp.exists() else None
    out["conventions"] = dict(
        val9="S.macro('val') 파이프라인 집계, 시드 평균, SD ddof=1",
        test46="KITTI 07(양성 없음) 제외 45시퀀스 -> 6도시 평균 -> 시드 평균(compare 스크립트 metric = _pr_eval_lib.pr_sweep), SD ddof=1",
        intervals="a-b: 시퀀스별 (a 시드 평균 - b 시드 평균), 도시 안 시퀀스 재표집 10,000회, 도시 평균, 2.5/97.5% 분위, 쌍마다 난수 20261104",
        prefix="원장 접두 검사 생략(--prefix 0)", units="0~1 비율(차이·구간은 %p)", uniform="기존 탐색 원장(S.LEDGERS) uniform")
    P.write(OUT / "summary.json", out)
    for arm in arms:
        print(f"{arm:12s} val", {k: round(100 * val[arm][k]["mean"], 3) for k in metrics},
              "test", {k: round(100 * stats[arm][k]["mean"], 3) for k in keys})
    for c, d in comp.items():
        print(f"  {c:20s}", {k: (round(d["point_pp"][k], 3), [round(x, 3) for x in d["conditional_95_pp"][k]]) for k in keys})
    print("CHECKS", max(out["metric_checks"].values()), same)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--variant", choices=list(ALL))
    p.add_argument("--seed", type=int)
    p.add_argument("--role", default="val")
    p.add_argument("--sequence")
    p.add_argument("--models")
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    P.e.configure()
    torch.set_num_threads(4)
    if a.action == "train":
        C.train(a.variant, a.seed)
    elif a.action == "repro":
        repro()
    elif a.action == "pipe":
        C.pipe(a.role, a.sequence, Path(a.models).read_text().split(), a.prefix)
    elif a.action == "pipe_campaign":
        pipe_campaign(a.role, a.models, a.workers, a.prefix)
    elif a.action == "summary":
        summary()
    else:
        raise SystemExit(a.action)
