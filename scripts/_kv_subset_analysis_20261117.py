#!/usr/bin/env python3
"""차이 K/V(NSD) 대 절대 K/V(abskv) 기제 가설 4개 확인(2026-11-17). 원장만 읽는 CPU 분석, 재학습 없음.

python3 scripts/_kv_subset_analysis_20261117.py      (호스트 numpy)
출력: results/pdv2_report_model_20261116/analysis/kv_subsets/{summary.json, tables.csv}

==================================================================== 사전 고정(결과 보기 전에 적음)
비교: NSD(새 보고모델 CFG@s{4,5,6}@u512, results/pdv2_scratch_search_20261116/ledgers)
      − abskv(abskv@s{4,5,6}@u512, results/pdv2_report_model_20261116/controls/ledgers). 균등 감산(uniform)은 참고 열.
집계: 시퀀스 지표(_pr_eval_lib.pr_sweep, R@1=양성 top1<=5 m) → 양성 있는 시퀀스를 묶음 안 평균 → 묶음 동일 가중,
      시드 평균. 묶음 = 시험은 6도시(city.json), 검증은 기존 val 규약의 3계열(kitti[KITTI-360 포함]·helipr·mulran).
구간: 시퀀스별 (NSD 시드 평균 − abs 시드 평균) → 묶음 안 시퀀스 복원추출 10,000회 → 묶음 평균, 2.5/97.5%.
      단일 부분집합 구간은 기존 코드 규약 그대로(rng=default_rng(20261104)를 비교마다 새로, 묶음 순서대로).
      차이의 차이 D = (NSD−abs)_T − (NSD−abs)_C 구간은 rng=default_rng(20261117):
        - T·C가 서로 다른 시퀀스(H1): 각 쪽을 독립 재표집.
        - T·C가 같은 시퀀스의 질의 부분집합(H2·H3): 묶음 안 시퀀스 색인을 두 쪽에 공유(짝지은 재표집).
판정(모든 가설 공통, "차이 입력이 T에서 상대적으로 더 강한가"):
  지지     : AP의 D 95% 구간 하한 > 0.
  부분 지지: AP는 아니지만 F1·R@100P·R@1 중 하나 이상의 D 구간 하한 > 0.
  기각     : 넷 다 하한 <= 0. (D 구간 상한 < 0이면 "반대 방향"으로 따로 적는다.)
  덧붙여 T 안에서 NSD−abs 자체의 구간 하한 > 0이면 "T에서 NSD가 abs보다 실제로 높음"으로 적는다.

H1 센서 도메인. T = 학습에 없던 센서, C = 학습 센서 계열.
   학습 = KITTI 01,02,05,06(Velodyne HDL-64E) + HeLiPR Bridge01–04(원장 입력은 VLP-16만, docs/데이터와 분할.md).
   KITTI-360은 HDL-64E이고 KITTI와 같은 캐시 계열(SENSOR_FAMILY) → 주 정의에서 C(학습 계열).
   주 정의: C = kitti 00,07,08 + kitti360 0000 + helipr 시험 12, T = mulran 9(OS1-64 1세대) + nclt 9(HDL-32E) + mcd 12(OS1-64 2세대).
   민감도(판정에 안 씀): KITTI-360을 T로 옮긴 정의.
H2 정지·저속. 키프레임이 거리 기준(인접 최소 간격 ~0.85 m)이라 ±5 키프레임 주행 거리는 정지를 담지 못한다
   (모든 시험 시퀀스 하위 5%가 8–9 m; 결과 보기 전 공변량만 확인). 그래서 주 정의는 속도:
   v_q = (travel[q+5] − travel[q−5]) / (t[q+5] − t[q−5]), 색인은 시퀀스 끝에서 자름. travel = 캐시 poses(방법이 쓰는
   오도메트리) 누적 거리, t = 캐시 timestamps. T = 시퀀스 안 v_q 하위 20%(질의 기준 분위), C = 나머지.
   부분집합 지표는 그 부분집합 질의만(양성·음성 모두 부분집합으로 제한)으로 시퀀스별 임계 스윕.
   민감도(판정에 안 씀): (a) 요청 원문 정의 = ±5 키프레임 주행 거리 하위 20%, (b) 주 정의에서 양성만 제한하고 음성은
   전체를 두는 방향 부분집합식 계산.
H3 방향. T = 순방향 전용, C = 역방향 전용(results/pdv2_review_direction_yaw_20261105/directions의 group 0/1,
   양성만 제한·음성 전체 = 기존 방향 부분집합 정의). 기제 가정: 순방향 재방문은 시간 이웃 순서가 같아 Δh 배열이 질의·후보에서
   맞고, 역방향은 순서가 뒤집혀 Δh 부호가 바뀐다 → 차이 입력이 순방향에서 상대적으로 유리할 것.
   주의: 방향별 NSD−abs 점추정·단일 구간은 controls/summary.json(test.forward_only/reverse_only, "nsd-abskv")에 이미 있고,
   이 기준을 적기 전에 그 값을 보았다(역방향에서 abs 우위가 더 큼). 방향 T 선택은 위 기제 가정에 따른 것이며 사전 등록이 아니다.
H4 검증9. 실제 검색 원장(ledgers/val)으로 같은 전체 비교. T = 검증9 전체, C 없음.
   판정: NSD−abs의 AP 구간 하한 > 0이면 "검증에서는 차이 입력이 강함"(지지), 상한 < 0이면 반대, 그 외 차이 없음.
   주의: NSD 구성(CFG·갱신 수)은 검증9로 골랐고(시드 1–3) abs는 그 구성에 K/V만 바꿨으므로 검증 비교는 NSD 쪽에 선택 이득이 있을 수 있다.
부록: 도시별·데이터셋별 NSD−abs(판정 없음).
====================================================================
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _report_model_common_20261116 as C  # noqa: E402
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402

KEYS = C.METRICS
OUT = ROOT / "results/pdv2_report_model_20261116/analysis/kv_subsets"
CTRL = ROOT / "results/pdv2_report_model_20261116/controls/ledgers"
ARMS = {"nsd": [f"new:{s}" for s in C.NEW_SEEDS], "abs": [f"abskv@s{s}@u512" for s in C.NEW_SEEDS], "uniform": ["uniform"]}
EXPECT = {"nsd": (89.879, 90.389, 72.174, 91.742), "abs": (90.509, 90.944, 73.367, 92.220)}
VAL_FAMILY_ORDER = ("helipr", "kitti", "mulran")
SEEN_TEST = [s for s in C.SEQS if s.split("/")[0] in ("kitti", "kitti360", "helipr")]
UNSEEN_TEST = [s for s in C.SEQS if s.split("/")[0] in ("mulran", "nclt", "mcd")]


def load(tag: str, seq: str, split: str) -> dict:
    name = seq.replace("/", "_")
    if tag.startswith("abskv@"):
        with np.load(CTRL / split / name / f"{tag}.npz") as z:
            return dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], q=z["query_idx"])
    return C.load(tag, seq, split)


def metric(d: dict, pos_mask=None, neg_mask=None) -> dict | None:
    qc = classify_queries(d["gt"])
    if pos_mask is not None:
        qc["positive"] &= pos_mask
    if neg_mask is not None:
        qc["negative"] &= neg_mask
    qc["ambiguous"] = ~(qc["positive"] | qc["negative"])
    qc["n_pos"], qc["n_neg"] = int(qc["positive"].sum()), int(qc["negative"].sum())
    if not qc["n_pos"]:
        return None
    v = pr_sweep(d["score"], d["top1"], qc)
    out = {k: float(v[k]) for k in KEYS[:3]}
    out["R@1"] = float(np.mean(d["top1"][qc["positive"]] <= 5))
    out["n_pos"], out["n_neg"] = qc["n_pos"], qc["n_neg"]
    return out


def macro(rows: dict, groups: dict) -> dict:
    vals = []
    for seqs in groups.values():
        v = [rows[s] for s in seqs if rows.get(s) is not None]
        if v:
            vals.append({k: np.mean([x[k] for x in v]) for k in KEYS})
    return {k: float(np.mean([g[k] for g in vals])) for k in KEYS}


def seq_delta(A: list, B: list, s: str) -> np.ndarray:
    return np.array([100 * (np.mean([r[s][k] for r in A]) - np.mean([r[s][k] for r in B])) for k in KEYS])


def boot_single(A: list, B: list, groups: dict, rng) -> tuple:
    """기존 규약: 묶음 순서대로, 가용 = A[0] 기준."""
    draws, point = [], []
    for seqs in groups.values():
        avail = [s for s in seqs if A[0].get(s) is not None]
        if not avail:
            continue
        mat = np.array([seq_delta(A, B, s) for s in avail])
        point.append(mat.mean(0))
        idx = rng.integers(0, len(avail), size=(C.BOOT, len(avail)))
        draws.append(mat[idx].mean(1))
    return np.mean(point, 0), np.stack(draws).mean(0)


def boot_joint(AT, BT, AC, BC, groups: dict, rng) -> np.ndarray:
    """같은 시퀀스의 두 질의 부분집합: 묶음 안 시퀀스 색인 공유. 반환 D 재표집 (n, 4)."""
    dT, dC = [], []
    for seqs in groups.values():
        avail = [s for s in seqs if AT[0].get(s) is not None or AC[0].get(s) is not None]
        if not avail:
            continue
        nan = np.full(len(KEYS), np.nan)
        mT = np.array([seq_delta(AT, BT, s) if AT[0].get(s) is not None else nan for s in avail])
        mC = np.array([seq_delta(AC, BC, s) if AC[0].get(s) is not None else nan for s in avail])
        idx = rng.integers(0, len(avail), size=(C.BOOT, len(avail)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dT.append(np.nanmean(mT[idx], 1))
            dC.append(np.nanmean(mC[idx], 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(dT), 0) - np.nanmean(np.stack(dC), 0)


def ci(draws: np.ndarray) -> dict:
    return {k: np.quantile(draws[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(KEYS)}


def subset_report(rows: dict, groups: dict) -> dict:
    """rows: arm -> [시드별 {seq: metric|None}]. 셋 팔의 매크로와 NSD−abs, 부분집합 크기."""
    per = {a: [macro(r, groups) for r in rs] for a, rs in rows.items()}
    means = {a: {k: 100 * float(np.mean([p[k] for p in ps])) for k in KEYS} for a, ps in per.items()}
    point, draws = boot_single(rows["nsd"], rows["abs"], groups, np.random.default_rng(C.BOOT_SEED))
    seqs = [s for v in groups.values() for s in v if rows["nsd"][0].get(s) is not None]
    first = rows["nsd"][0]
    return dict(mean_pp=means, nsd_minus_abs_pp=dict(zip(KEYS, point.tolist())), ci95_pp=ci(draws),
                n_sequences_with_pos=len(seqs), n_pos=int(sum(first[s]["n_pos"] for s in seqs)),
                n_neg=int(sum(first[s]["n_neg"] for s in seqs)), _draws=draws)


def verdict(dod_ci: dict, t_ci: dict) -> str:
    lo = {k: dod_ci[k][0] for k in KEYS}
    if lo["AP"] > 0:
        v = "지지"
    elif any(lo[k] > 0 for k in KEYS[1:]):
        v = "부분 지지(" + ",".join(k for k in KEYS[1:] if lo[k] > 0) + ")"
    else:
        v = "기각"
        rev = [k for k in KEYS if dod_ci[k][1] < 0]
        if rev:
            v += "; 반대 방향(" + ",".join(rev) + ")"
    better = [k for k in KEYS if t_ci[k][0] > 0]
    if better:
        v += "; T에서 NSD>abs(" + ",".join(better) + ")"
    return v


def restrict(groups: dict, seqs: list) -> dict:
    return {c: [s for s in v if s in seqs] for c, v in groups.items() if any(s in seqs for s in v)}


def rows_for(data: dict, seqs: list, pos=None, neg=None) -> dict:
    """data: arm -> [시드별 {seq: ledger}]. pos/neg: {seq: mask} 또는 None."""
    return {a: [{s: metric(d[s], None if pos is None else pos[s], None if neg is None else neg[s]) for s in seqs} for d in ds]
            for a, ds in data.items()}


def motion(seq: str, q: np.ndarray) -> tuple:
    name = seq.replace("/", "_")
    audit = json.loads((C.OLD / "test" / name / "summary.json").read_text())["input_audit"]
    (cache,) = [k for k in audit["inputs"] if "/cache_" in k]
    with np.load(ROOT / cache, allow_pickle=False) as z:
        xyz, t = z["poses"][:, :3, 3], z["timestamps"].astype(np.float64)
    n = len(xyz)
    assert audit["nodes"] == n
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    lo, hi = np.maximum(q - 5, 0), np.minimum(q + 5, n - 1)
    dist, span = travel[hi] - travel[lo], t[hi] - t[lo]
    assert (span > 0).all(), seq
    return dist / span, dist


def main() -> None:
    tick = time.time()
    out = dict(conventions=__doc__.split("====================================================================")[1].strip())
    # ---------------------------------------------------------------- 시험 원장
    data = {a: [{s: load(t, s, "test") for s in C.SEQS} for t in ts] for a, ts in ARMS.items()}
    dirs = {}
    for s in C.SEQS:
        q = data["nsd"][0][s]["q"]
        for a, ds in data.items():
            for d in ds:
                np.testing.assert_array_equal(d[s]["q"], q)
                np.testing.assert_array_equal(d[s]["gt"], data["nsd"][0][s]["gt"])
        with np.load(C.DIRS / (s.replace("/", "_") + ".npz")) as z:
            np.testing.assert_array_equal(z["query_idx"], q)
            dirs[s] = z["group"].copy()
    # ---------------------------------------------------------------- 검증: 전체 시험 지표 재현
    full = rows_for(data, C.SEQS)
    allrep = subset_report(full, C.CITY)
    for a in ("nsd", "abs"):
        got = tuple(round(allrep["mean_pp"][a][k], 3) for k in KEYS)
        assert got == EXPECT[a], (a, got)
    ctrl = json.loads((ROOT / "results/pdv2_report_model_20261116/controls/summary.json").read_text())["test"]
    for g in ("all", "forward_only", "reverse_only"):
        ref = ctrl[g]["differences"]["nsd-abskv"]
        if g == "all":
            for k in KEYS:
                assert abs(ref["point_pp"][k] - allrep["nsd_minus_abs_pp"][k]) < 1e-9
                assert np.allclose(ref["conditional_95_pp"][k], allrep["ci95_pp"][k], atol=1e-9)
    out["check"] = dict(test_all_matches_expected=True, test_all_ci_matches_controls_summary=True)
    out["test_all"] = allrep
    # ---------------------------------------------------------------- H1
    def h1(seen, unseen, label):
        rs = {n: (restrict(C.CITY, v), rows_for(data, v)) for n, v in (("seen", seen), ("unseen", unseen))}
        rep = {n: subset_report(r, g) for n, (g, r) in rs.items()}
        rng = np.random.default_rng(20261117)
        _, dT = boot_single(rs["unseen"][1]["nsd"], rs["unseen"][1]["abs"], rs["unseen"][0], rng)
        _, dC = boot_single(rs["seen"][1]["nsd"], rs["seen"][1]["abs"], rs["seen"][0], rng)
        dod_pt = {k: rep["unseen"]["nsd_minus_abs_pp"][k] - rep["seen"]["nsd_minus_abs_pp"][k] for k in KEYS}
        dod = ci(dT - dC)
        return dict(definition=label, T="unseen", C="seen", subsets=rep, D_pp=dod_pt, D_ci95_pp=dod,
                    verdict=verdict(dod, rep["unseen"]["ci95_pp"]))
    out["H1"] = h1(SEEN_TEST, UNSEEN_TEST, "주: KITTI-360=학습 계열(HDL-64E)")
    out["H1_sens_kitti360_unseen"] = h1([s for s in SEEN_TEST if not s.startswith("kitti360")],
                                        UNSEEN_TEST + ["kitti360/0000"], "민감도: KITTI-360=미학습")
    # ---------------------------------------------------------------- H2 / H3 (같은 시퀀스의 두 부분집합)
    def paired_subsets(maskT, maskC, label, neg_restrict=True):
        rT = rows_for(data, C.SEQS, maskT, maskT if neg_restrict else None)
        rC = rows_for(data, C.SEQS, maskC, maskC if neg_restrict else None)
        rep = dict(T=subset_report(rT, C.CITY), C=subset_report(rC, C.CITY))
        d = boot_joint(rT["nsd"], rT["abs"], rC["nsd"], rC["abs"], C.CITY, np.random.default_rng(20261117))
        dod = ci(d)
        dod_pt = {k: rep["T"]["nsd_minus_abs_pp"][k] - rep["C"]["nsd_minus_abs_pp"][k] for k in KEYS}
        return dict(definition=label, subsets=rep, D_pp=dod_pt, D_ci95_pp=dod, verdict=verdict(dod, rep["T"]["ci95_pp"]))

    speed, dist = {}, {}
    for s in C.SEQS:
        speed[s], dist[s] = motion(s, data["nsd"][0][s]["q"])
    low = lambda m: {s: m[s] <= np.quantile(m[s], 0.2) for s in C.SEQS}  # noqa: E731
    slow, short = low(speed), low(dist)
    out["H2"] = paired_subsets(slow, {s: ~v for s, v in slow.items()}, "주: ±5 키프레임 평균 속도 하위 20%(시퀀스 안), 양성·음성 모두 제한")
    out["H2"]["speed_threshold_mps"] = {s: float(np.quantile(speed[s], 0.2)) for s in C.SEQS}
    out["H2_sens_distance"] = paired_subsets(short, {s: ~v for s, v in short.items()}, "민감도(a): ±5 키프레임 주행 거리 하위 20%")
    out["H2_sens_posonly"] = paired_subsets(slow, {s: ~v for s, v in slow.items()}, "민감도(b): 속도 하위 20%, 양성만 제한", neg_restrict=False)
    fwd, rev = {s: dirs[s] == 0 for s in C.SEQS}, {s: dirs[s] == 1 for s in C.SEQS}
    out["H3"] = paired_subsets(fwd, rev, "순방향 전용(T) 대 역방향 전용(C), 양성만 제한", neg_restrict=False)
    for g, k in (("forward_only", "T"), ("reverse_only", "C")):
        ref = ctrl[g]["differences"]["nsd-abskv"]
        mine = out["H3"]["subsets"][k]
        assert all(abs(ref["point_pp"][m] - mine["nsd_minus_abs_pp"][m]) < 1e-9 for m in KEYS), g
        assert all(np.allclose(ref["conditional_95_pp"][m], mine["ci95_pp"][m], atol=1e-9) for m in KEYS), g
    out["H3"]["reused_from"] = "단일 부분집합 점추정·구간은 results/pdv2_report_model_20261116/controls/summary.json test.{forward_only,reverse_only}.differences.nsd-abskv와 일치(검산); D만 새로 계산"
    # ---------------------------------------------------------------- H4 검증9
    vdir = C.SEARCH / "ledgers" / "val"
    vseqs = sorted(p.name.replace("_", "/", 1) for p in vdir.iterdir() if p.is_dir())
    vfam = {f: [s for s in vseqs if ("kitti" if s.startswith("kitti") else s.split("/")[0]) == f] for f in VAL_FAMILY_ORDER}
    vdata = {a: [{s: load(t, s, "val") for s in vseqs} for t in ts] for a, ts in ARMS.items()}
    vrows = rows_for(vdata, vseqs)
    h4 = subset_report(vrows, vfam)
    vctrl = json.loads((ROOT / "results/pdv2_report_model_20261116/controls/summary.json").read_text())["val"]
    for a, ca in (("nsd", "nsd"), ("abs", "abskv"), ("uniform", "uniform")):
        for k in KEYS:
            assert abs(h4["mean_pp"][a][k] - 100 * vctrl[ca][k]["mean"]) < 1e-6, (a, k)
    lo, hi = h4["ci95_pp"]["AP"]
    h4["verdict"] = "지지(검증에서 차이 입력이 강함)" if lo > 0 else ("반대" if hi < 0 else "차이 없음(AP 구간이 0 포함)")
    h4["families"] = vfam
    h4["check"] = "검증 매크로가 controls/summary.json val(파이프라인 JSON 집계)과 1e-6 안에서 일치"
    out["H4"] = h4
    # ---------------------------------------------------------------- 부록
    app = {}
    for c, v in C.CITY.items():
        app["city:" + c] = subset_report(rows_for(data, v), {c: v})
    for ds in ("kitti", "kitti360", "helipr", "mulran", "nclt", "mcd"):
        v = [s for s in C.SEQS if s.split("/")[0] == ds]
        app["dataset:" + ds] = subset_report(rows_for(data, v), {ds: v})
    out["appendix"] = app
    out["seconds"] = time.time() - tick
    # ---------------------------------------------------------------- 쓰기
    lines = ["block,subset,n_seq,n_pos,n_neg," + ",".join(f"{a}_{k}" for a in ARMS for k in KEYS) + ","
             + ",".join(f"d_{k},lo_{k},hi_{k}" for k in KEYS)]

    def line(block, name, r):
        lines.append(",".join([block, name, str(r["n_sequences_with_pos"]), str(r["n_pos"]), str(r["n_neg"])]
                              + [f"{r['mean_pp'][a][k]:.3f}" for a in ARMS for k in KEYS]
                              + [f"{x:.3f}" for k in KEYS for x in (r["nsd_minus_abs_pp"][k], *r["ci95_pp"][k])]))

    def dline(block, h):
        lines.append(",".join([block, "D=T-C", "", "", ""] + [""] * (len(ARMS) * len(KEYS))
                              + [f"{x:.3f}" for k in KEYS for x in (h["D_pp"][k], *h["D_ci95_pp"][k])]))

    line("test_all", "all", allrep)
    for key in ("H1", "H1_sens_kitti360_unseen"):
        line(key, "seen(C)", out[key]["subsets"]["seen"])
        line(key, "unseen(T)", out[key]["subsets"]["unseen"])
        dline(key, out[key])
    for key in ("H2", "H2_sens_distance", "H2_sens_posonly", "H3"):
        line(key, "T", out[key]["subsets"]["T"])
        line(key, "C", out[key]["subsets"]["C"])
        dline(key, out[key])
    line("H4", "val9", h4)
    for k, r in app.items():
        line("appendix", k, r)

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "_draws"}
        return o
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.csv").write_text("\n".join(lines) + "\n")
    C.write(OUT / "summary.json", strip(out))
    for key in ("H1", "H1_sens_kitti360_unseen", "H2", "H2_sens_distance", "H2_sens_posonly", "H3"):
        print(key, out[key]["verdict"], {k: (round(out[key]["D_pp"][k], 3), [round(x, 3) for x in out[key]["D_ci95_pp"][k]]) for k in KEYS})
    print("H4", h4["verdict"], {k: (round(h4["nsd_minus_abs_pp"][k], 3), [round(x, 3) for x in h4["ci95_pp"][k]]) for k in KEYS})
    print("seconds", round(out["seconds"], 1))


if __name__ == "__main__":
    main()
