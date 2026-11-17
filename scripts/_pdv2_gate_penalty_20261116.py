#!/usr/bin/env python3
"""변형 (c): 게이트는 원래대로(η=−sigmoid(g), −1<η<0) 두고, 학습 손실에 η<−0.5일 때만 벌점을 더한 `diff` 재학습.

손실 = BCE + λ · mean_{층 2개, 노드}(relu(−0.5 − η)²). 초기 η=−0.5에서 벌점과 그 기울기가 정확히 0이라
초기 출력·첫 갱신은 원래 `diff`와 같다. λ는 {1, 10}에서 검증 9만으로 고른다(protocol.json에 학습 전 고정).
  - 척도 근거: 마지막 에포크 BCE는 약 0.21–0.23. 강화 상태(η≈−0.7, 모든 노드)에서 벌점 평균은 0.04라서
    λ=1이면 BCE의 약 20%, λ=10이면 약 2배다. 초기 이탈(η≈−0.55)에서는 0.0025로 λ=1은 약 1%, λ=10은 약 12%다.
  - 선택 규칙: 두 λ 모두 시작점 3 × 실행 시드 r0..r5 학습 → 검증 9 평가 → r1..r5 15회의 평균 검증 AP 이득
    (−fixed)이 큰 λ. 같으면(차이 < 0.01%p) 검증 AP 이득이 음수인 실행 수가 적은 쪽, 그래도 같으면 λ=1.
  - 시험 46은 고른 λ 하나만 평가한다.
그 밖의 학습·평가는 `_pdv2_bounded_gate_20261116.py`와 같다(독립 반복 train 함수, 기존 검증·시험 실행기).

사용법(컨테이너 ring_baseline):
  python3 scripts/_pdv2_gate_penalty_20261116.py preflight
  python3 scripts/_pdv2_gate_penalty_20261116.py train_all --lam 1   (그리고 --lam 10)
  python3 scripts/_pdv2_gate_penalty_20261116.py evaluate --lam 1 --role val
  python3 scripts/_pdv2_gate_penalty_20261116.py select
  python3 scripts/_pdv2_gate_penalty_20261116.py evaluate --lam <선택> --role test
  python3 scripts/_pdv2_gate_penalty_20261116.py analyze
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import _pdv2_bounded_gate_20261116 as G  # noqa: E402

READ, WRITE, HASH = G.READ, G.WRITE, G.HASH
OUT = G.OUT / "penalty"
LAMS = (1.0, 10.0)
STARTS, RUNS, ALL_RUNS, METRICS = G.STARTS, G.RUNS, G.ALL_RUNS, G.METRICS
TRAIN_WORKERS = int(os.environ.get("NSD_TRAIN_WORKERS", 1))
EVAL_WORKERS = int(os.environ.get("NSD_EVAL_WORKERS", 1))
STATE = {}


def lam_dir(lam):
    return OUT / f"lam{lam:g}"


def run_folder(lam, k, r):
    return lam_dir(lam) / "runs" / f"s{k}" / f"r{r}"


def penalty_of(eta):
    import torch.nn.functional as F

    return (F.relu(-0.5 - eta) ** 2).mean()


def patched(lam):
    """독립 반복 학습 모듈에 'penalty' 조건을 더하고, BCE 호출마다 직전 forward의 η 벌점을 더한다."""
    import torch.nn.functional as F
    import _pdv2_diffattn_independent_runs as I

    if "penalty" not in I.ARMS:
        I.ARMS["penalty"] = I.ARMS["diff"]
        original = I.build

        def build(torch, e, start, arm):
            net = original(torch, e, start, "diff" if arm == "penalty" else arm)
            STATE["net"] = net
            return net

        I.build = build
        bce = F.binary_cross_entropy_with_logits

        def loss(*a, **k):
            base = bce(*a, **k)
            if STATE.get("active"):
                pen = penalty_of(STATE["net"].last_strengths)
                STATE["log"].append((float(base.detach()), float(pen.detach())))
                return base + STATE["lam"] * pen
            return base

        F.binary_cross_entropy_with_logits = loss
    STATE.update(lam=float(lam), log=[])
    return I


def protocol():
    path = OUT / "protocol.json"
    if path.exists():
        return READ(path)
    spec = dict(
        created_unix=time.time(),
        variant="(c) 원래 게이트 η=−sigmoid(g) + 손실 벌점 λ·mean(relu(−0.5−η)²) (두 전달 층, 모든 노드)",
        lambdas=list(LAMS),
        lambda_rationale=__doc__.split("  - 척도 근거: ")[1].split("  - 선택 규칙")[0].strip(),
        selection_rule="두 λ 모두 18회 학습·검증 9 평가. r1..r5 15회 평균 검증 AP 이득(−fixed, 같은 GPU 평가)이 큰 λ. "
                       "차이<0.01%p면 검증 AP 이득<0 실행 수가 적은 쪽, 그래도 같으면 λ=1. 시험은 고른 λ만.",
        unchanged="시작점 3, r0·r1..r5, 512갱신, AdamW lr 1e-4, 재보정 손실, epoch8, matching_only. 학습은 "
                  "_pdv2_diffattn_independent_runs.train을 그대로 부르고 BCE에 벌점만 더함.",
        sources={str(Path(p).relative_to(ROOT)): HASH(p) for p in (Path(__file__), Path(G.__file__))},
        test_used_for_selection=False,
    )
    WRITE(path, spec)
    return spec


def preflight():
    """초기 벌점=0, 벌점 기울기=0, 첫 기울기가 원래 diff와 비트 단위로 같은지 확인한다."""
    I = patched(10.0)
    torch, R, e = I.trainer()
    import torch.nn.functional as F

    spec = I.prepare()
    protocol()
    name = spec["train"][0].replace("/", "_")
    graph = torch.load(I.TRAIN_INPUTS / (name + ".pt"), map_location="cuda", weights_only=False)
    data = e.run.ARR(e.prior.PARENT / "pools/train" / name / "pool.npz")
    plan = {k: torch.from_numpy(data["plan_" + k]).cuda() for k in ("query", "candidate", "weight")}
    ix = torch.from_numpy(np.flatnonzero(data["eligible"])[:64]).cuda()
    label = torch.from_numpy(data["correct"][ix.cpu().numpy()]).cuda().float()
    rows = {}
    for k in STARTS:
        grads, outs = {}, {}
        for arm in ("diff", "penalty"):
            STATE["active"] = arm == "penalty"
            net = I.build(torch, e, k, arm)
            net.zero_grad(set_to_none=True)
            z = net(graph)
            loss = F.binary_cross_entropy_with_logits(e.prior.score(z, {a: v[ix] for a, v in plan.items()}), label)
            loss.backward()
            outs[arm] = z.detach()
            grads[arm] = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}
        STATE["active"] = False
        rows[f"s{k}"] = dict(
            initial_penalty=STATE["log"][-1][1],
            output_bit_identical_to_diff=bool(torch.equal(outs["diff"], outs["penalty"])),
            first_gradient_bit_identical_to_diff=all(torch.equal(grads["diff"][n], grads["penalty"][n])
                                                    for n in grads["diff"]),
        )
        print("PREFLIGHT", k, rows[f"s{k}"], flush=True)
    eta = torch.tensor([-0.7, -0.5, -0.3], device="cuda", requires_grad=True)
    p = (F.relu(-0.5 - eta) ** 2).sum()
    p.backward()
    rows["unit"] = dict(eta=[-0.7, -0.5, -0.3], grad=eta.grad.tolist())
    ok = all(r["initial_penalty"] == 0 and r["output_bit_identical_to_diff"] and r["first_gradient_bit_identical_to_diff"]
             for kk, r in rows.items() if kk.startswith("s"))
    rows["status"] = "passed" if ok else "failed"
    WRITE(OUT / "preflight.json", rows)
    assert ok


def train_one(lam, k, r):
    assert READ(OUT / "preflight.json")["status"] == "passed"
    protocol()
    I = patched(lam)
    STATE["active"] = True
    folder = run_folder(lam, k, r)
    I.train(k, r, "penalty", folder)
    if STATE["log"]:
        log = np.array(STATE["log"])
        WRITE(folder / "penalty_log.json", dict(lam=lam, updates=len(log), bce=log[:, 0].tolist(),
                                                penalty=log[:, 1].tolist(),
                                                note="records의 loss는 BCE+λ·벌점. 여기 bce·penalty는 갱신별 값."))


def train_all(lam):
    jobs = [(k, r) for k in STARTS for r in ALL_RUNS]
    logs = lam_dir(lam) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(job):
        k, r = job
        if (run_folder(lam, k, r) / "training_complete.json").exists():
            return
        with (logs / f"train_s{k}_r{r}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "train", "--lam", str(lam), "--start", str(k), "--run", str(r)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=TRAIN_WORKERS) as pool:
        for i, f in enumerate(concurrent.futures.as_completed([pool.submit(call, j) for j in jobs]), 1):
            f.result()
            print("TRAIN_DONE", lam, i, len(jobs), flush=True)


# ---------------------------------------------------------------- 평가
def eval_spec(lam, role):
    path = lam_dir(lam) / f"{role}_eval/protocol.json"
    if path.exists():
        return READ(path)
    base = G.OLD_TEST if role == "test" else G.OLD_VAL
    spec = dict(READ(base / "protocol.json"))
    cases = {"fixed": dict(kind="fixed", seed=0, checkpoint=None)}
    for k in STARTS:
        for r in ALL_RUNS:
            p = run_folder(lam, k, r) / "epoch8.pth"
            h = HASH(p)
            assert READ(p.parent / "training_complete.json")["sha256"] == h
            # 학습 중 벌점만 다르고 추론 구조는 원래 diff와 같다. seed=100+k로 기존 원장 대조 분기를 피한다.
            cases[f"penalty_s{k}_r{r}"] = dict(kind="diff", seed=100 + k, checkpoint=str(p.relative_to(ROOT)),
                                               checkpoint_sha256=h)
    spec.update(created_unix=time.time(), cases=cases, parent_protocol_sha256=HASH(OUT / "protocol.json"),
                test_policy="게이트 벌점 변형의 고정 평가. λ 선택은 검증 9만.")
    WRITE(path, spec)
    return spec


def eval_worker(lam, role, sequence):
    import torch
    import _pdv2_contrast_pipeline as P

    P.e.configure()
    folder = lam_dir(lam) / f"{role}_eval" / role / sequence.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    src = (G.OLD_TEST / "test" if role == "test" else G.OLD_VAL / "val") / sequence.replace("/", "_")
    for suffix in (".npz", ".json"):
        t = folder / ("candidate_pool" + suffix)
        if not t.exists():
            t.symlink_to(os.path.relpath(src / ("candidate_pool" + suffix), folder))
    if role == "test":
        import _pdv2_contrast_pipeline_test as T

        T.OUT = lam_dir(lam) / "test_eval"
        T.prepare = lambda: eval_spec(lam, "test")
        T.worker("test", sequence)
    else:
        P.OUT = lam_dir(lam) / "val_eval"
        P.run.OUT = P.OUT
        P.prepare = lambda: eval_spec(lam, "val")
        P.worker("val", sequence)
    name = sequence.replace("/", "_")
    ref = (G.OLD_TEST / "test" / name / "fixed.npz") if role == "test" else (G.OLD_VAL / "val" / name / "fixed.npz")
    a, b = np.load(folder / "fixed.npz"), np.load(ref)
    np.testing.assert_array_equal(a["j_star"], b["j_star"])
    err = float(np.abs(a["score"] - b["score"]).max())
    assert err < 8e-6, err
    graph, _, _ = P.e.prior.graph_inputs(role, sequence)
    arrays = {}
    with torch.no_grad():
        for tag, case in eval_spec(lam, role)["cases"].items():
            if case["checkpoint"] is None:
                continue
            net = P.load_model(case)
            net(graph)
            arrays[tag] = net.last_strengths.squeeze(-1).float().cpu().numpy()
            del net
    np.savez_compressed(folder / "strengths.npz", **arrays)
    WRITE(folder / "reproduction_check.json", dict(status="passed", checks=dict(
        fixed=dict(reference=str(ref), winner_agreement=1.0, score_max_abs=err))))
    print("EVAL_DONE", lam, role, sequence, flush=True)


def evaluate(lam, role):
    if role == "test":
        assert READ(OUT / "selection.json")["chosen_lambda"] == lam, "시험은 검증으로 고른 λ만"
    eval_spec(lam, role)
    seqs = READ((G.OLD_TEST if role == "test" else G.OLD_VAL) / "protocol.json")[role]
    logs = lam_dir(lam) / f"{role}_eval/logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(s):
        if (lam_dir(lam) / f"{role}_eval" / role / s.replace("/", "_") / "reproduction_check.json").exists():
            return
        with (logs / (s.replace("/", "_") + ".log")).open("a") as f:
            subprocess.run([sys.executable, __file__, "eval_worker", "--lam", str(lam), "--role", role,
                            "--sequence", s], stdout=f, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=EVAL_WORKERS) as pool:
        futs = [pool.submit(call, s) for s in seqs]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            f.result()
            print("EVAL", lam, role, i, len(futs), flush=True)


# ---------------------------------------------------------------- 선택과 분석
def val_table(lam):
    import _pdv2_contrast_pipeline as P

    fam = P.run.base.macro.__globals__["family"]  # 기존 검증 매크로가 쓰는 계열 함수
    vals = READ(G.OLD_VAL / "protocol.json")["val"]
    vper = {s: READ(lam_dir(lam) / "val_eval/val" / s.replace("/", "_") / "summary.json")["metrics"] for s in vals}
    fams = sorted({fam(s) for s in vals if vper[s]["fixed"]["n_pos"]})
    vorder = [s for g in fams for s in vals if fam(s) == g and vper[s]["fixed"]["n_pos"]]
    vidx = [np.array([vorder.index(s) for s in vorder if fam(s) == g]) for g in fams]

    def vmac(v, p=None):
        return float(np.mean([v[ix if p is None else ix[p[c]]].mean() for c, ix in enumerate(vidx)]))

    ref = P.run.base.macro({s: vper[s]["penalty_s1_r0"] for s in vals})["macro"]
    assert all(abs(vmac(np.array([vper[s]["penalty_s1_r0"][m] for s in vorder])) - ref[m]) < 1e-12 for m in METRICS)
    res = {}
    for m in METRICS:
        D = np.array([[[100 * (vper[s][f"penalty_s{k}_r{r}"][m] - vper[s]["fixed"][m]) for s in vorder]
                       for r in RUNS] for k in STARTS])
        runs = np.array([[vmac(D[i, j]) for j in range(5)] for i in range(3)])
        mean, iv, by_seed = G.crossed_interval(D, vmac, vidx)
        r0 = [vmac(np.array([100 * (vper[s][f"penalty_s{k}_r0"][m] - vper[s]["fixed"][m]) for s in vorder]))
              for k in STARTS]
        res[m] = dict(mean_pp=mean, crossed_95=iv, positive_runs=int((runs > 0).sum()), per_run=runs.tolist(),
                      per_seed_mean=by_seed, per_start_mean=runs.mean(1).tolist(), sd_runs=float(runs.std(ddof=1)),
                      r0_per_start=r0, r0_mean=float(np.mean(r0)))
    return res


def select():
    tables = {f"{lam:g}": val_table(lam) for lam in LAMS}
    a, b = (tables[f"{lam:g}"]["AP"] for lam in LAMS)
    if abs(a["mean_pp"] - b["mean_pp"]) >= 0.01:
        chosen = LAMS[0] if a["mean_pp"] > b["mean_pp"] else LAMS[1]
        why = "평균 검증 AP 이득"
    else:
        na, nb = (15 - t["positive_runs"] for t in (a, b))
        chosen = LAMS[1] if nb < na else LAMS[0]
        why = "동률 → 음수 실행 수/λ=1"
    rec = dict(chosen_lambda=chosen, reason=why, rule=protocol()["selection_rule"], val9=tables,
               test_used_for_selection=False)
    WRITE(OUT / "selection.json", rec)
    print("SELECT", chosen, why, {k: (round(v["AP"]["mean_pp"], 3), v["AP"]["positive_runs"]) for k, v in tables.items()},
          flush=True)


def strengths(lam, role, seqs):
    data = [np.load(lam_dir(lam) / f"{role}_eval" / role / s.replace("/", "_") / "strengths.npz") for s in seqs]
    out = {}
    for k in STARTS:
        for r in ALL_RUNS:
            tag = f"penalty_s{k}_r{r}"
            x = np.concatenate([d[tag] for d in data], axis=1)
            row = {}
            for layer in (0, 1):
                v = x[layer]
                row[f"{role}_eta{layer}_mean"] = float(v.mean())
                for q in (5, 25, 50, 75, 95):
                    row[f"{role}_eta{layer}_p{q:02d}"] = float(np.percentile(v, q))
                row[f"{role}_eta{layer}_frac_weaker"] = float((v > -0.5).mean())
                row[f"{role}_eta{layer}_frac_below_0.6"] = float((v < -0.6).mean())
            folder = run_folder(lam, k, r)
            rec = READ(folder / "training_complete.json")["records"]
            row["train_strength_by_epoch"] = [float(np.mean([z["strength_mean"] for z in rec if z["epoch"] == ep]))
                                              for ep in range(1, 9)]
            pl = READ(folder / "penalty_log.json")
            row["train_last_bce"] = float(np.mean(pl["bce"][-64:]))
            row["train_last_penalty"] = float(np.mean(pl["penalty"][-64:]))
            row["train_max_penalty"] = float(np.max(pl["penalty"]))
            out[tag] = row
    return out


def analyze():
    import _pdv2_pilot as pilot
    from _pr_eval_lib import classify_queries, pr_sweep

    sel = READ(OUT / "selection.json")
    lam = sel["chosen_lambda"]
    tests = READ(G.OLD_TEST / "protocol.json")["test"]
    vals = READ(G.OLD_VAL / "protocol.json")["val"]
    per, prefix = {}, []
    for s in tests:
        d = READ(lam_dir(lam) / "test_eval/test" / s.replace("/", "_") / "summary.json")
        assert READ(lam_dir(lam) / "test_eval/test" / s.replace("/", "_") / "reproduction_check.json")["status"] == "passed"
        per[s] = d["metrics"]
        prefix += d["prefix_checks"]
    cities = {c: [s for s in q if per[s]["fixed"]["n_pos"]] for c, q in pilot.CITY.items()}
    order = [s for q in cities.values() for s in q]
    cidx = [np.array([order.index(s) for s in q]) for q in cities.values()]

    def tmac(v, p=None):
        return float(np.mean([v[ix if p is None else ix[p[c]]].mean() for c, ix in enumerate(cidx)]))

    old_sum = READ(G.OLD_TEST / "summary.json")
    for m in METRICS:
        assert abs(tmac(np.array([per[s]["fixed"][m] for s in order])) - old_sum["mean"]["fixed"][m]) < 1e-12
    worst, n = 0.0, 0
    for s in tests:
        for tag in per[s]:
            d = np.load(lam_dir(lam) / "test_eval/test" / s.replace("/", "_") / f"{tag}.npz")
            if not (d["min_past_gt_dist"] <= 5).any():
                continue
            rr = pr_sweep(d["score"], d["top1_geo_dist"], classify_queries(d["min_past_gt_dist"]))
            worst = max(worst, max(abs(rr[m] - per[s][tag][m]) for m in ("AP", "max_F1", "R@100P")))
            n += 1
    assert worst < 1e-9
    test = {}
    for m in METRICS:
        D = np.array([[[100 * (per[s][f"penalty_s{k}_r{r}"][m] - per[s]["fixed"][m]) for s in order]
                       for r in RUNS] for k in STARTS])
        runs = np.array([[tmac(D[i, j]) for j in range(5)] for i in range(3)])
        mean, iv, by_seed = G.crossed_interval(D, tmac, cidx)
        r0 = [tmac(np.array([100 * (per[s][f"penalty_s{k}_r0"][m] - per[s]["fixed"][m]) for s in order]))
              for k in STARTS]
        test[m] = dict(mean_pp=mean, crossed_95=iv, positive_runs=int((runs > 0).sum()), per_run=runs.tolist(),
                       per_seed_mean=by_seed, per_start_mean=runs.mean(1).tolist(), sd_runs=float(runs.std(ddof=1)),
                       r0_per_start=r0, r0_mean=float(np.mean(r0)))
    # 주 결과표 방식 r0 세 시작점 평균과 BEVPlace++ (clamp 분석과 같은 절차)
    per_case = old_sum["per_case"]
    mac3 = {k: {m: tmac(np.array([per[s][f"penalty_s{k}_r0"][m] for s in order])) for m in METRICS} for k in STARTS}
    pmean = {m: float(np.mean([mac3[k][m] for k in STARTS])) for m in METRICS}

    def paired(b, seed=20261104, reps=10000):
        rng = np.random.default_rng(seed)
        draws = []
        for city, seqs in pilot.CITY.items():
            avail = [q for q in seqs if per_case["fixed"]["sequences"][q]["n_pos"]]
            mat = np.array([[np.mean([100 * (per[q][f"penalty_s{k}_r0"][m]
                                             - (per_case["fixed"] if b == "fixed" else per_case[f"{b}_s{k}"])["sequences"][q][m])
                                      for k in STARTS]) for m in METRICS] for q in avail])
            draws.append(mat[rng.integers(0, len(avail), size=(reps, len(avail)))].mean(1))
        boot = np.stack(draws).mean(0)
        return {m: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, m in enumerate(METRICS)}

    table = dict(penalty_r0_mean_percent={m: 100 * v for m, v in pmean.items()},
                 penalty_r0_sd_percent={m: 100 * float(np.std([mac3[k][m] for k in STARTS], ddof=1)) for m in METRICS},
                 penalty_r0_per_start_percent={str(k): {m: 100 * v for m, v in mac3[k].items()} for k in STARTS})
    for b in ("bev_native", "diff", "fixed"):
        table[f"penalty_minus_{b}_pp"] = {m: 100 * (pmean[m] - old_sum["mean"][b][m]) for m in METRICS}
        table[f"penalty_minus_{b}_conditional_95_pp"] = paired(b)
    out = dict(
        chosen_lambda=lam, selection=sel, test46_gain_pp=test, main_table_r0=table,
        strengths={f"{l:g}": dict(val=strengths(l, "val", vals)) for l in LAMS},
        prefix_q15=dict(checks=len(prefix), max_score_error=max(c["score_error"] for c in prefix),
                        r0_checks=sum(c["condition"].endswith("_r0") for c in prefix)),
        independent_metric_recompute=dict(ledgers=n, max_abs=worst),
        test_used_for_selection=False,
    )
    out["strengths"][f"{lam:g}"]["test"] = strengths(lam, "test", tests)
    WRITE(OUT / "result.json", out)
    print("TEST", {m: (round(v["mean_pp"], 3), np.round(v["crossed_95"], 3).tolist(), v["positive_runs"],
                       round(v["r0_mean"], 3)) for m, v in test.items()}, flush=True)
    print("TABLE", json.dumps(table), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("preflight", "train", "train_all", "evaluate", "eval_worker", "select", "analyze"))
    p.add_argument("--lam", type=float)
    p.add_argument("--start", type=int)
    p.add_argument("--run", type=int)
    p.add_argument("--role", choices=("test", "val"), default="val")
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "train":
        train_one(a.lam, a.start, a.run)
    elif a.action == "train_all":
        train_all(a.lam)
    elif a.action == "evaluate":
        evaluate(a.lam, a.role)
    elif a.action == "eval_worker":
        eval_worker(a.lam, a.role, a.sequence)
    else:
        globals()[a.action]()
