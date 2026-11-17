#!/usr/bin/env python3
"""단일 임계값 운영점 확장: (A) 검증 고정 임계값, (B) 시험 전체에 임계값 하나를 쓴 오수락 예산 비교.

기존 결과(NSD·BEVPlace++ 두 방법)를 그대로 재현한 뒤 아래를 더한다. 새 학습·추론은 없고 원장만 읽는다.

(A) 검증 고정 임계값(배치 가능한 운영점)
  - 방법: NSD, BEVPlace++, BEVPlace++ + 거리 대응 시간 증거(같은 β·지연, 반환 장소 동일).
  - 임계값 τ는 방법·시작점마다 검증 9시퀀스 합산 점수에서만 고른다(P99·P100·F1, 기존 freeze.json 규칙 그대로).
  - 시험 46시퀀스에 그대로 적용해 도시 재현율·합산 재현율·합산 정밀도·TP·오수락을 낸다.
  - 짝지은 구간: τ를 고정한 채 도시 안 양성 시퀀스 복원추출 10,000회(시드 20261104, 원고 구간과 같은 절차).
  - 민감도: 검증 시퀀스를 하나씩 뺀 9가지 τ로 시험 결과의 범위를 낸다(τ 선택이 검증 구성에 얼마나 기대는지).
(B) 시험 전체 단일 임계값의 오수락 예산(사후 비교, 배치 가능한 운영점이 아님)
  - 방법: NSD, BEVPlace++ + 시간 증거, 표 2 비교 방법 10종. 방법·실행마다 46시퀀스를 합쳐 임계값 하나.
  - 합산 오수락 ≤ k (k = 0, 5, 10, 30, 100)인 임계값 가운데 정답 수락이 가장 많은 것(같은 점수는 함께 수락).
  - 짝지은 구간: 표본마다 임계값을 다시 고르는 재표집(기존 _paper_matched_false_accepts.py의 reselect와 같은 절차).
검산(하나라도 어긋나면 멈춘다)
  - (A) NSD·BEVPlace++의 τ 18개가 기존 freeze.json과 비트 단위로 같고, 시험 수치가 기존 summary.json과 같다.
  - (B) NSD·BEVPlace++의 TP·도시 재현율·차이 구간이 기존 result.json과 같다. 가중치 1 표본이 전체 시험 선택과 같다.
  - 비교 방법 10종의 6도시 매크로 R@100P(시퀀스별 임계값)가 표 2 채택 요약과 같다(원장이 표 2 원장임을 확인).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import _review_v23_fixed_threshold_20261107 as F
from _pdv2_pilot import CITY, TEST, VAL
from _pr_eval_lib import classify_queries, pr_sweep

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_single_threshold_all_20261115"
NSD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104/test"
NSD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104/val"
BEV = ROOT / "results/pdv2_review_response_20261104/bev"
FIXED = ROOT / "results/pdv2_review_closure_20261107/threshold"
MATCHED = Path("/mnt/d/NSD_datasets/nsd_results/pdv2_matched_false_accepts_20261112/result.json")
SUMMARY = ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
PREFIX = dict(scpp="dump_scpp", fresco="dump_fresco", ring="dump_ring", ringpp="dump_ringpp_i2", ringsharp="dump_ringsharp",
              deepring="dump_deepring", disco="dump_disco", helios="dump_helios", seqot="dump_seqotl2", bevplace="dump_bevplace")
LABEL = dict(nsd="NSD", bevt="BEVPlace++ + TE", bevplace="BEVPlace++", scpp="SC++", fresco="FreSCo", ring="RING", ringpp="RING++",
             ringsharp="RING#-L", deepring="DeepRING", disco="DiSCO", helios="HeLiOS", seqot="SeqOT")
SEEDS = (1, 2, 3)
POINTS = ("P99", "P100", "F1")
BUDGETS = (0, 5, 10, 30, 100)
B, BOOT_SEED = 10000, 20261104
SEQS = [f"{s}/{q}" for s, q in TEST]
POS_SEQS = None   # 양성 있는 시퀀스(재표집 대상)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name: str, value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def ms(values: list) -> dict:
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None, values=[float(v) for v in values])


# ------------------------------------------------------------------ 원장

def val_rows(method: str, seed: int, drop: str = None) -> list:
    """검증 9시퀀스의 점수·반환 거리·라벨. method: nsd | bevplace | bevt."""
    rows = []
    for sensor, seq in VAL:
        name = f"{sensor}_{seq}"
        if f"{sensor}/{seq}" == drop:
            continue
        ref = np.load(NSD_VAL / name / "identity.npz")
        if method == "nsd":
            path = NSD_VAL / name / f"diff_s{seed}.npz"; key = "score"
        else:
            path = BEV / "val" / f"s{seed}" / f"{name}.npz"; key = "score_negl2" if method == "bevplace" else "score"
        record = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == record["sha256"], path
        d = np.load(path)
        for k in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(d[k], ref[k])
        rows.append(dict(sequence=f"{sensor}/{seq}", score=d[key].astype(np.float64), dist=d["top1_geo_dist"], mpgd=d["min_past_gt_dist"]))
    return rows


def test_rows(method: str, run, source: str = None, key: str = None) -> dict:
    """시험 46시퀀스의 점수·반환 거리·라벨(공통 질의로 정렬)."""
    rows = {}
    for sensor, seq in TEST:
        name = f"{sensor}_{seq}"
        ref = np.load(NSD_TEST / name / "identity.npz")
        if method == "nsd":
            d = np.load(NSD_TEST / name / f"diff_s{run}.npz"); k = "score"
        elif method in ("bevplace_ledger", "bevt"):
            d = np.load(BEV / "test" / f"s{run}" / f"{name}.npz"); k = "score_negl2" if method == "bevplace_ledger" else "score"
        else:
            d = np.load(ROOT / source / f"{PREFIX[method]}_{name}.npz"); k = key
        ix = np.searchsorted(d["query_idx"], ref["query_idx"])
        np.testing.assert_array_equal(d["query_idx"][ix], ref["query_idx"])
        np.testing.assert_array_equal(d["min_past_gt_dist"][ix], ref["min_past_gt_dist"])
        assert np.all(d["j_star"][ix] <= ref["query_idx"] - 30)
        rows[f"{sensor}/{seq}"] = dict(score=d[k][ix].astype(np.float64), dist=d["top1_geo_dist"][ix].astype(np.float64),
                                       qc=classify_queries(ref["min_past_gt_dist"]))
    assert sum(len(r["score"]) for r in rows.values()) == 195726
    assert sum(r["qc"]["n_pos"] for r in rows.values()) == 61682
    return rows


def table2_r100(rows: dict) -> float:
    """시퀀스별 임계값의 6도시 매크로 R@100P(표 2 지표). 원장 확인용."""
    per = {s: pr_sweep(r["score"], r["dist"], r["qc"])["R@100P"] for s, r in rows.items() if r["qc"]["n_pos"]}
    return F.city_mean(per)[0]


# ------------------------------------------------------------------ 재표집

def draws() -> np.ndarray:
    """원고 짝지은 구간과 같은 표본: 도시마다 양성 시퀀스를 같은 수만큼 복원추출. 반환 (B, 46) 가중치."""
    rng = np.random.default_rng(BOOT_SEED)
    w = np.zeros((B, len(SEQS)), np.int16)
    for names in CITY.values():
        avail = [s for s in names if s in POS_SEQS]
        idx = rng.integers(0, len(avail), size=(B, len(avail)))
        cols = np.array([SEQS.index(s) for s in avail])
        for j in range(len(avail)):
            np.add.at(w, (np.arange(B), cols[idx[:, j]]), 1)
    for s in SEQS:   # 양성 없는 시퀀스(KITTI 07)는 오수락 계산에 한 번씩 고정 포함
        if s not in POS_SEQS:
            w[:, SEQS.index(s)] = 1
    return w


def city_recall(w: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """표본별 도시 재현율: 도시마다 뽑힌 시퀀스 재현율 평균 → 6도시 평균. recall은 (46,) 또는 (B, 46)."""
    out = []
    for names in CITY.values():
        cols = [SEQS.index(s) for s in names if s in POS_SEQS]
        out.append((w[:, cols] * (recall[..., cols])).sum(1) / len(cols))
    return np.mean(out, 0)


def q95(x: np.ndarray) -> list:
    return [float(v) for v in np.quantile(x, [.025, .975])]


# ------------------------------------------------------------------ (A)

def part_a(w: np.ndarray) -> dict:
    freeze = json.loads((FIXED / "freeze.json").read_text())["thresholds"]
    old = json.loads((FIXED / "summary.json").read_text())["methods"]
    out, cells = {}, {}
    for m, test_name in (("nsd", "nsd"), ("bevplace", "bevplace_ledger"), ("bevt", "bevt")):
        out[m] = {p: dict(per_seed={}) for p in POINTS}
        for s in SEEDS:
            chosen = F.choose(val_rows(m, s))
            if m in freeze:   # 기존 τ와 비트 단위로 같아야 한다
                for p in POINTS:
                    assert float(chosen[p]["tau"]).hex() == freeze[m][str(s)][p]["tau_hex"], (m, s, p)
            loso = {}
            for sensor, seq in VAL:
                c = F.choose(val_rows(m, s, drop=f"{sensor}/{seq}"))
                loso[f"{sensor}/{seq}"] = {p: c[p]["tau"] for p in POINTS}
            rows = test_rows(test_name, s)
            for p in POINTS:
                res = F.fixed(rows, chosen[p]["tau"])
                seqv = res["sequence"]
                cells[(m, s, p)] = dict(TP=np.array([seqv[q]["TP"] for q in SEQS]), FP=np.array([seqv[q]["FP"] for q in SEQS]),
                                        npos=np.array([seqv[q]["n_pos"] for q in SEQS]))
                sens = {}
                for left, taus in loso.items():
                    r = F.fixed(rows, taus[p])
                    sens[left] = dict(tau=taus[p], city_macro_recall=r["city_macro_recall"], recall=r["pooled"]["recall"],
                                      precision=r["pooled"]["precision"], TP=r["pooled"]["TP"], FP=r["pooled"]["FP"])
                out[m][p]["per_seed"][str(s)] = dict(
                    tau=chosen[p]["tau"], val_precision=chosen[p]["val_precision"], val_recall=chosen[p]["val_recall"],
                    val_TP=chosen[p]["val_TP"], val_FP=chosen[p]["val_FP"], city_macro_recall=res["city_macro_recall"],
                    city_recall=res["city_recall"], recall=res["pooled"]["recall"], precision=res["pooled"]["precision"],
                    TP=res["pooled"]["TP"], FP=res["pooled"]["FP"], FP_positive_query=res["pooled"]["FP_positive_query"],
                    FP_negative_query=res["pooled"]["FP_negative_query"],
                    city_FP={c: v["FP"] for c, v in res["city"].items()}, leave_one_validation_sequence_out=sens)
            print("A", m, s, {p: (round(100 * out[m][p]["per_seed"][str(s)]["city_macro_recall"], 2), out[m][p]["per_seed"][str(s)]["FP"]) for p in POINTS}, flush=True)
        for p in POINTS:
            ps = out[m][p]["per_seed"]
            out[m][p]["mean_sd"] = {k: ms([ps[str(s)][k] for s in SEEDS]) for k in
                                    ("val_precision", "val_recall", "city_macro_recall", "recall", "precision", "TP", "FP", "FP_positive_query", "FP_negative_query")}
            out[m][p]["city_recall_mean"] = {c: float(np.mean([ps[str(s)]["city_recall"][c] for s in SEEDS])) for c in CITY}
            out[m][p]["city_FP_mean"] = {c: float(np.mean([ps[str(s)]["city_FP"][c] for s in SEEDS])) for c in CITY}
            lo = [[ps[str(s)]["leave_one_validation_sequence_out"][f"{a}/{b}"] for s in SEEDS] for a, b in VAL]
            out[m][p]["leave_one_validation_sequence_out"] = {
                f"{a}/{b}": {k: float(np.mean([x[k] for x in lo[i]])) for k in ("city_macro_recall", "recall", "precision", "TP", "FP")}
                for i, (a, b) in enumerate(VAL)}
            if m in old:   # 기존 요약과 대조
                for k in ("city_macro_recall", "precision", "recall"):
                    assert abs(out[m][p]["mean_sd"][k]["mean"] - old[m][p]["mean_sd"][k]["mean"]) < 1e-12, (m, p, k)
    # 짝지은 구간(τ 고정)
    rep = {}
    for m in out:
        for p in POINTS:
            c = [cells[(m, s, p)] for s in SEEDS]
            rep[(m, p)] = dict(
                city=np.mean([city_recall(w, np.divide(x["TP"], x["npos"], out=np.zeros(len(SEQS)), where=x["npos"] > 0)) for x in c], 0),
                pooled=np.mean([(w * x["TP"]).sum(1) / (w * x["npos"]).sum(1) for x in c], 0),
                FP=np.mean([(w * x["FP"]).sum(1) for x in c], 0))
            ident = np.ones((1, len(SEQS)), np.int16)
            for s, x in zip(SEEDS, c):   # 가중치 1 표본 = 전체 시험
                r = np.divide(x["TP"], x["npos"], out=np.zeros(len(SEQS)), where=x["npos"] > 0)
                assert abs(city_recall(ident, r)[0] - out[m][p]["per_seed"][str(s)]["city_macro_recall"]) < 1e-12
    comparisons = {}
    for a, b in (("nsd", "bevplace"), ("nsd", "bevt"), ("bevt", "bevplace")):
        comparisons[f"{a}-{b}"] = {p: dict(
            city_macro_recall_pp=100 * (out[a][p]["mean_sd"]["city_macro_recall"]["mean"] - out[b][p]["mean_sd"]["city_macro_recall"]["mean"]),
            city_macro_recall_95_pp=q95(100 * (rep[(a, p)]["city"] - rep[(b, p)]["city"])),
            pooled_recall_pp=100 * (out[a][p]["mean_sd"]["recall"]["mean"] - out[b][p]["mean_sd"]["recall"]["mean"]),
            pooled_recall_95_pp=q95(100 * (rep[(a, p)]["pooled"] - rep[(b, p)]["pooled"])),
            FP_difference=out[a][p]["mean_sd"]["FP"]["mean"] - out[b][p]["mean_sd"]["FP"]["mean"],
            FP_difference_95=q95(rep[(a, p)]["FP"] - rep[(b, p)]["FP"])) for p in POINTS}
    return dict(methods=out, comparisons=comparisons,
                replicate_FP_95={f"{m}/{p}": q95(rep[(m, p)]["FP"]) for m in out for p in POINTS})


# ------------------------------------------------------------------ (B)

def build(rows: dict) -> dict:
    """합산 sweep 재료: 오수락 점수(내림차순)와 그 시퀀스, 시퀀스별 정답 수락 점수(오름차순)."""
    fp_score, fp_seq, tp_sorted, npos = [], [], [], []
    for i, s in enumerate(SEQS):
        r = rows[s]; qc = r["qc"]
        finite = np.isfinite(r["score"]); correct = np.isfinite(r["dist"]) & (r["dist"] <= 5)
        tp = finite & qc["positive"] & correct
        fp = finite & (qc["negative"] | (qc["positive"] & ~correct))
        fp_score.append(r["score"][fp]); fp_seq.append(np.full(int(fp.sum()), i))
        tp_sorted.append(np.sort(r["score"][tp])); npos.append(qc["n_pos"])
    fs, fq = np.concatenate(fp_score), np.concatenate(fp_seq)
    order = np.argsort(-fs, kind="stable")
    return dict(fp_score=fs[order], fp_seq=fq[order], tp_sorted=tp_sorted, npos=np.array(npos))


def select(b: dict, w: np.ndarray) -> dict:
    """표본마다 예산별 임계값을 다시 고른다. 반환: 예산 -> 시퀀스별 TP (R, 46)와 배제 임계값 (R,).

    배제 임계값은 가중 오수락 누적이 처음 예산을 넘는 오수락의 점수이고, 그보다 큰 점수만 수락한다
    (같은 점수는 함께 판정하므로 그 점수의 정답도 빠진다). 오수락 전체가 예산 안이면 유한 점수를 모두 수락한다.
    """
    R, top = len(w), min(4000, len(b["fp_score"]))
    excl = {k: np.empty(R) for k in BUDGETS}
    for start in range(0, R, 500):
        ww = w[start:start + 500].astype(np.int32)
        cum = np.cumsum(ww[:, b["fp_seq"][:top]], 1)
        # 앞 4000개로 가장 큰 예산을 넘지 못한 표본은 전체 오수락으로 다시 센다
        short = np.flatnonzero(cum[:, -1] <= max(BUDGETS)) if 0 < top < len(b["fp_score"]) else []
        for k in BUDGETS:
            over = cum > k
            tau = np.where(over.any(1), b["fp_score"][:top][over.argmax(1)], -np.inf) if top else np.full(len(ww), -np.inf)
            for i in short:
                o = np.cumsum(ww[i, b["fp_seq"]]) > k
                tau[i] = b["fp_score"][int(o.argmax())] if o.any() else -np.inf
            excl[k][start:start + 500] = tau
    return {k: dict(TP=np.stack([len(t) - np.searchsorted(t, excl[k], side="right") for t in b["tp_sorted"]], 1), tau=excl[k])
            for k in BUDGETS}


def exact_fp(b: dict, tau: float) -> int:
    return int((b["fp_score"] > tau).sum())


def part_b(w: np.ndarray) -> dict:
    summ = json.loads(SUMMARY.read_text())["methods"]
    runs = {"nsd": [("nsd", s, None, None) for s in SEEDS], "bevt": [("bevt", s, None, None) for s in SEEDS]}
    for m in PREFIX:
        runs[m] = [(m, seed, r["source"], summ[m]["score"]) for seed, r in summ[m]["per_seed"].items()]
    ident = np.ones((1, len(SEQS)), np.int16)
    full, rep = {}, {}
    for m, rr in runs.items():
        per_run = []
        for name, run, source, key in rr:
            rows = test_rows(name, run, source, key)
            if m in summ:
                assert abs(table2_r100(rows) - summ[m]["per_seed"][str(run)]["macro"]["R@100P"]) < 1e-12, (m, run)
            b = build(rows)
            one = select(b, ident)
            res = select(b, w)
            per_run.append((b, one, res))
        full[m] = {}
        for k in BUDGETS:
            vals = dict(city=[], pooled=[], TP=[], FP=[])
            for b, one, _ in per_run:
                tp = one[k]["TP"][0]
                recall = np.divide(tp, b["npos"], out=np.zeros(len(SEQS)), where=b["npos"] > 0)
                vals["city"].append(F.city_mean({s: recall[i] for i, s in enumerate(SEQS) if b["npos"][i]})[0])
                vals["pooled"].append(tp.sum() / b["npos"].sum()); vals["TP"].append(int(tp.sum()))
                fp = exact_fp(b, one[k]["tau"][0]); assert fp <= k, (m, k, fp)
                # 예산 규칙의 최적성: 임계값을 한 단계 낮추면(배제 임계값의 점수까지 수락) 오수락이 예산을 넘는다
                if np.isfinite(one[k]["tau"][0]):
                    assert int((b["fp_score"] >= one[k]["tau"][0]).sum()) > k
                vals["FP"].append(fp)
            full[m][k] = dict(city_macro_recall=ms(vals["city"]), recall=ms(vals["pooled"]), TP=ms(vals["TP"]), FP=ms(vals["FP"]), runs=len(per_run))
            cr, pr = [], []
            for b, _, res in per_run:
                tp = res[k]["TP"]
                cr.append(city_recall(w, np.divide(tp, b["npos"], out=np.zeros(tp.shape), where=b["npos"] > 0)))
                pr.append((w * tp).sum(1) / (w * b["npos"]).sum(1))
            rep[(m, k)] = dict(city=np.mean(cr, 0), pooled=np.mean(pr, 0))
        print("B", m, {k: (round(100 * full[m][k]["city_macro_recall"]["mean"], 2), round(full[m][k]["TP"]["mean"])) for k in BUDGETS}, flush=True)
    # 기존 결과와 대조(NSD·BEVPlace++)
    old = json.loads(MATCHED.read_text())["criteria"]
    for k in BUDGETS:
        c = old[f"FA<={k}"]
        for m, om in (("nsd", "nsd"), ("bevplace", "bevplace")):
            for a, bkey in (("TP", "TP"), ("city_macro_recall", "city_macro_recall"), ("recall", "recall")):
                assert abs(full[m][k][a]["mean"] - c["methods"][om]["mean_sd"][bkey]["mean"]) < 1e-9, (m, k, a)
        new = q95(100 * (rep[("nsd", k)]["city"] - rep[("bevplace", k)]["city"]))
        ref = c["intervals_95_pp"]["reselect"]["city_macro_recall"]
        assert max(abs(new[0] - ref[0]), abs(new[1] - ref[1])) < 1e-9, (k, new, ref)
    comparisons = {}
    for m in runs:
        if m == "nsd":
            continue
        comparisons[f"nsd-{m}"] = {str(k): dict(
            city_macro_recall_pp=100 * (full["nsd"][k]["city_macro_recall"]["mean"] - full[m][k]["city_macro_recall"]["mean"]),
            city_macro_recall_95_pp=q95(100 * (rep[("nsd", k)]["city"] - rep[(m, k)]["city"])),
            pooled_recall_pp=100 * (full["nsd"][k]["recall"]["mean"] - full[m][k]["recall"]["mean"]),
            pooled_recall_95_pp=q95(100 * (rep[("nsd", k)]["pooled"] - rep[(m, k)]["pooled"]))) for k in BUDGETS}
    comparisons["bevt-bevplace"] = {str(k): dict(
        city_macro_recall_pp=100 * (full["bevt"][k]["city_macro_recall"]["mean"] - full["bevplace"][k]["city_macro_recall"]["mean"]),
        city_macro_recall_95_pp=q95(100 * (rep[("bevt", k)]["city"] - rep[("bevplace", k)]["city"]))) for k in BUDGETS}
    return dict(methods={m: {str(k): v for k, v in d.items()} for m, d in full.items()}, comparisons=comparisons)


def main() -> None:
    global POS_SEQS
    t0 = time.time()
    POS_SEQS = {s for (a, b), s in zip(TEST, SEQS) if classify_queries(np.load(NSD_TEST / f"{a}_{b}" / "identity.npz")["min_past_gt_dist"])["n_pos"]}
    assert len(POS_SEQS) == 45
    w = draws()
    parts = sys.argv[1:] or ["a", "b"]
    common = dict(script_sha256=sha(Path(__file__)), replicates=B, seed=BOOT_SEED, labels=LABEL,
                  disclosure="NSD·BEVPlace++의 기존 결과(2026-11-07, 11-12)를 본 뒤에 한 확장이다. 규칙(운영점·예산·재표집)은 기존 고정 규칙을 그대로 쓰고 새로 고르지 않았다.")
    if "a" in parts:
        write("validation_fixed.json", dict(**common, **part_a(w), threshold_source="validation", test_used_for_selection=False))
    if "b" in parts:
        write("test_single_threshold_budgets.json", dict(**common, **part_b(w), threshold_source="test (사후 비교)", budgets=list(BUDGETS)))
    print("DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
