#!/usr/bin/env python3
"""전체 학습 `diff`의 학습 이득이 작은 이유 진단(재학습 없음, CPU 전용).

기존 체크포인트(원래 세 시작점 + 독립 반복 시작점3×실행5)와 기존 시험 원장만 쓴다.
  stats      검증9·시험46 노드에서 전달 세기 η(0), η(1), 균등 대비 주의 총변동거리,
             고정 감산 출력과의 거리
  drift      시작 상태(균등 감산 −0.5와 같은 초기화) 대비 매개변수 묶음별 변화
  val_worker 독립 반복 30개의 검증9 실제 검색 평가(기존 검증 실행기, CPU)
  analyze    실행별 이득과 위 특징의 상관, 시작점별 검증·시험 이득

GPU를 쓰지 않는다: CUDA_VISIBLE_DEVICES=''로 띄우고 .cuda()/map_location을 CPU로 바꾼다.
시험셋은 진단용이며 설정 선택에 쓰지 않는다.

사용법(컨테이너 ring_baseline, 작업 폴더 /workspace/Neural-Spectral-Codec):
  CUDA_VISIBLE_DEVICES= python3 scripts/_pdv2_diff_gain_diagnosis_20261116.py run
  CUDA_VISIBLE_DEVICES= python3 scripts/_pdv2_diff_gain_diagnosis_20261116.py analyze
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

OUT = ROOT / "results/pdv2_diff_gain_diagnosis_20261116"
OLD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
INDEP = Path("/workspace/data/nsd_results/pdv2_diffattn_independent_20261108")
STARTS = (1, 2, 3)
RUNS = (1, 2, 3, 4, 5)
METRICS = ("AP", "max_F1", "R@100P", "R@1")
WORKERS = 4


def READ(p):
    return json.loads(Path(p).read_text())


def WRITE(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    tmp.replace(p)


def cpu_patch():
    """GPU를 건드리지 않도록 .cuda()와 torch.load 위치를 CPU로 고정한다."""
    import torch

    assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CUDA_VISIBLE_DEVICES='' 로 실행"
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    load = torch.load

    def cpu_load(*a, **k):
        k["map_location"] = "cpu"
        return load(*a, **k)

    torch.load = cpu_load
    torch.set_num_threads(4)
    return torch


# ---------------------------------------------------------------- 사례
def models():
    """진단 대상 체크포인트. 원래 세 시작점은 o{k}, 독립 반복은 s{k}_r{r}."""
    old = READ(OLD_TEST / "protocol.json")["cases"]
    out = {}
    for kind in ("diff", "convs_only", "other_only"):
        for k in STARTS:
            c = old[f"{kind}_s{k}"]
            out[f"{kind}_o{k}"] = dict(kind=kind, start=k, run=0, checkpoint=c["checkpoint"],
                                       checkpoint_sha256=c["checkpoint_sha256"])
    indep = READ(INDEP / "test_eval/protocol.json")["cases"]
    for kind in ("diff", "convs_only"):
        for k in STARTS:
            for r in RUNS:
                c = indep[f"{kind}_s{k}_r{r}"]
                out[f"{kind}_s{k}_r{r}"] = dict(kind=kind, start=k, run=r, checkpoint=c["checkpoint"],
                                                checkpoint_sha256=c["checkpoint_sha256"])
    return out


def val_spec():
    """기존 검증 명세에서 사례만 바꾼다. seed는 기존 재현 분기(시드1 원장 대조)를 피하려 100+로 둔다."""
    path = OUT / "val_eval/protocol.json"
    if path.exists():
        return READ(path)
    spec = dict(READ(OLD_VAL / "protocol.json"))
    cases = {"fixed_cpu": dict(kind="fixed", seed=0, checkpoint=None)}
    for tag, c in models().items():
        if c["kind"] == "other_only":
            continue
        cases[tag] = dict(kind=c["kind"], seed=100 + c["start"], checkpoint=c["checkpoint"],
                          checkpoint_sha256=c["checkpoint_sha256"])
    spec.update(cases=cases, scope="diff 학습 이득 진단(CPU 재평가). 설정 선택 없음.",
                device="cpu")
    WRITE(path, spec)
    return spec


# ---------------------------------------------------------------- 검증 실제 평가
def val_worker(sequence):
    cpu_patch()
    import _pdv2_contrast_pipeline as P

    P.e.configure()
    P.OUT = OUT / "val_eval"
    P.run.OUT = P.OUT
    P.prepare = val_spec
    name = sequence.replace("/", "_")
    folder = P.OUT / "val" / name
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        target = folder / ("candidate_pool" + suffix)
        if not target.exists():
            target.symlink_to(os.path.relpath(OLD_VAL / "val" / name / ("candidate_pool" + suffix), folder))
    P.worker("val", sequence)


# ---------------------------------------------------------------- 세기·주의 통계
def stats_worker(role, sequence):
    torch = cpu_patch()
    import _pdv2_contrast_pipeline as P

    P.e.configure()
    if role == "test":
        import _pdv2_contrast_pipeline_test as T

        T.activate()  # 기존 시험 명세를 읽기만 한다(존재하므로 쓰지 않음)
    name = sequence.replace("/", "_")
    target = OUT / "stats" / role / f"{name}.npz"
    if target.exists():
        return
    graph, _, _ = P.e.prior.graph_inputs(role, sequence)
    tgt = graph.edge_index[1]
    n = len(graph.x)
    deg = torch.bincount(tgt, minlength=n).clamp_min(1)
    uniform = (1.0 / deg.double())[tgt][:, None]
    arrays = {}
    with torch.no_grad():
        z_fixed = P.e.prior.filtered(graph, -0.5)
        base = float((z_fixed - graph.spectrum).abs().pow(2).sum().sqrt())
        for tag, case in models().items():
            net = P.load_model(dict(case, kind="diff"))
            z = net(graph)
            eta = net.last_strengths.squeeze(-1).double()  # [2, N]
            tv = []
            for conv in net.convs:
                d = (conv.attention.double() - uniform).abs()  # [E, H]
                acc = torch.zeros(n, d.shape[1], dtype=torch.float64)
                acc.index_add_(0, tgt, d)
                tv.append((0.5 * acc).mean(1))  # 헤드 평균
            arrays[f"{tag}__eta"] = eta.numpy().astype(np.float32)
            arrays[f"{tag}__tv"] = torch.stack(tv).numpy().astype(np.float32)
            arrays[f"{tag}__dev"] = np.array(
                float((z - z_fixed).abs().pow(2).sum().sqrt()) / base)
            del net, z
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, n=np.array(n), **arrays)
    print("STATS", role, sequence, flush=True)


# ---------------------------------------------------------------- 매개변수 변화
GROUPS = (("input_proj", "input_proj."), ("input_norm", "input_norm."), ("batch_norms", "batch_norms."),
          ("gate.0", "gate.0."), ("gate.2", "gate.2."), ("edge_encoder", "edge_encoder."),
          ("convs.qkv", ("convs.0.W_", "convs.1.W_")), ("convs.bias_mlp", ("convs.0.bias_mlp", "convs.1.bias_mlp")))


def drift():
    torch = cpu_patch()
    import _pdv2_diffattn_independent_runs as I

    torch_, R, e = I.trainer()
    spec = READ(INDEP / "protocol.json")
    graph0 = torch.load(I.TRAIN_INPUTS / (spec["train"][0].replace("/", "_") + ".pt"), weights_only=False)
    init = {}
    for k in STARTS:
        net = I.build(torch, e, k, "diff")
        with torch.no_grad():
            err = float((net(graph0) - e.prior.filtered(graph0, -0.5)).abs().max())
        assert err < 4e-6, err
        init[k] = {n: v.double().clone() for n, v in net.state_dict().items()}
    result = {}
    for tag, case in models().items():
        p = Path(case["checkpoint"])
        p = p if p.is_absolute() else ROOT / p
        state = {n: v.double() for n, v in torch.load(p, weights_only=False)["model_state_dict"].items()}
        ref = init[case["start"]]
        assert set(state) == set(ref)
        row = {}
        for gname, prefix in GROUPS:
            keys = [n for n in ref if n.startswith(prefix) and not n.endswith(("running_mean", "running_var",
                                                                              "num_batches_tracked"))]
            d = float(sum((state[n] - ref[n]).pow(2).sum() for n in keys).sqrt())
            b = float(sum(ref[n].pow(2).sum() for n in keys).sqrt())
            row[gname] = dict(abs=d, rel=(d / b if b > 0 else None), init_norm=b)
        buffers = [n for n in ref if n.endswith(("running_mean", "running_var"))]
        row["bn_running_stats_changed"] = bool(any(not torch.equal(state[n], ref[n]) for n in buffers))
        result[tag] = row
    WRITE(OUT / "drift.json", dict(groups=[g for g, _ in GROUPS], init_check="train graph0 출력 = 고정 −0.5 (<4e-6)",
                                   rows=result))
    print("DRIFT done", flush=True)


# ---------------------------------------------------------------- 실행
def run_all():
    val_spec()
    vals = READ(OLD_VAL / "protocol.json")["val"]
    tests = READ(OLD_TEST / "protocol.json")["test"]
    jobs = ([("drift", None, None)] + [("val_worker", "val", s) for s in vals]
            + [("stats_worker", "val", s) for s in vals] + [("stats_worker", "test", s) for s in tests])
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(job):
        action, role, seq = job
        args = [sys.executable, __file__, action]
        if role:
            args += ["--role", role, "--sequence", seq]
        log = logs / f"{action}_{role}_{(seq or '').replace('/', '_')}.log"
        with log.open("a") as f:
            subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(call, j) for j in jobs]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            f.result()
            WRITE(OUT / "status.json", dict(done=i, total=len(futs)))
            print("DONE", i, len(futs), flush=True)


# ---------------------------------------------------------------- 분석
def spearman_perm(x, y, reps=20000, seed=20261116):
    from scipy.stats import rankdata

    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(reps)])
    return dict(rho=rho, p_perm=float((np.abs(null) >= abs(rho) - 1e-12).mean()),
                pearson=float(np.corrcoef(x, y)[0, 1]), n=len(x))


def analyze():
    import _pdv2_contrast_pipeline as P
    import _pdv2_pilot as pilot

    cases = models()
    macro_fam = P.run.base.macro
    # ---- 시험 46: 기존 원장 요약(GPU)
    tests = READ(OLD_TEST / "protocol.json")["test"]
    per = {}
    for s in tests:
        name = s.replace("/", "_")
        a = READ(INDEP / "test_eval/test" / name / "summary.json")["metrics"]
        b = READ(OLD_TEST / "test" / name / "summary.json")["metrics"]
        row = dict(a)
        for kind in ("diff", "convs_only", "other_only"):
            for k in STARTS:
                row[f"{kind}_o{k}"] = b[f"{kind}_s{k}"]
        assert abs(a["fixed"]["AP"] - b["fixed"]["AP"]) < 1e-12
        per[s] = row
    cities = {c: [s for s in seqs if per[s]["fixed"]["n_pos"]] for c, seqs in pilot.CITY.items()}
    assert sum(len(v) for v in cities.values()) == 45

    def tmacro(tag, m):
        return float(np.mean([np.mean([per[s][tag][m] for s in seqs]) for seqs in cities.values()]))

    test_gain = {t: {m: 100 * (tmacro(t, m) - tmacro("fixed", m)) for m in METRICS} for t in cases}
    # ---- 검증 9: CPU 재평가 + 기존 GPU 원장 대조
    vals = READ(OLD_VAL / "protocol.json")["val"]
    vrows = {s: READ(OUT / "val_eval/val" / s.replace("/", "_") / "summary.json")["metrics"] for s in vals}
    vmac = {t: macro_fam({s: r[t] for s, r in vrows.items()})["macro"] for t in vrows[vals[0]]}
    gpu = {s: READ(OLD_VAL / "val" / s.replace("/", "_") / "summary.json")["metrics"] for s in vals}
    gmap = {"fixed_cpu": "fixed", **{f"{k}_o{s}": f"{k}_s{s}" for k in ("diff", "convs_only") for s in STARTS}}
    gmac = {t: macro_fam({s: r[g] for s, r in gpu.items()})["macro"] for t, g in gmap.items()}
    cpu_vs_gpu = {t: {m: 100 * (vmac[t][m] - gmac[t][m]) for m in METRICS} for t in gmap}
    # 승자 일치율
    agree = {}
    for t, g in gmap.items():
        same = tot = 0
        for s in vals:
            nm = s.replace("/", "_")
            a = np.load(OUT / "val_eval/val" / nm / f"{t}.npz")["j_star"]
            b = np.load(OLD_VAL / "val" / nm / f"{g}.npz")["j_star"]
            same += int((a == b).sum())
            tot += len(a)
        agree[t] = same / tot
    val_gain = {t: {m: 100 * (vmac[t][m] - vmac["fixed_cpu"][m]) for m in METRICS}
                for t in cases if t in vmac}
    gpu_val_gain_orig = {t: {m: 100 * (gmac[t][m] - gmac["fixed_cpu"][m]) for m in METRICS}
                         for t in gmap if t != "fixed_cpu"}

    # ---- 세기·주의 특징
    feats = {}
    qs = (5, 25, 50, 75, 95)
    for role, seqs in (("val", vals), ("test", tests)):
        data = [np.load(OUT / "stats" / role / (s.replace("/", "_") + ".npz")) for s in seqs]
        for t in cases:
            eta = np.concatenate([d[f"{t}__eta"] for d in data], axis=1)  # 노드 풀링
            tv = np.concatenate([d[f"{t}__tv"] for d in data], axis=1)
            f = feats.setdefault(t, {})
            for layer in (0, 1):
                f[f"{role}_eta{layer}_mean"] = float(eta[layer].mean())
                f[f"{role}_eta{layer}_std"] = float(eta[layer].std())
                for q in qs:
                    f[f"{role}_eta{layer}_p{q:02d}"] = float(np.percentile(eta[layer], q))
                f[f"{role}_eta{layer}_frac_weaker"] = float((eta[layer] > -0.5).mean())
                f[f"{role}_tv{layer}_mean"] = float(tv[layer].mean())
                f[f"{role}_tv{layer}_p95"] = float(np.percentile(tv[layer], 95))
            f[f"{role}_dev_from_fixed"] = float(np.mean([float(d[f"{t}__dev"]) for d in data]))
    dr = READ(OUT / "drift.json")["rows"]
    for t in cases:
        for g, v in dr[t].items():
            if isinstance(v, dict):
                feats[t][f"drift_{g}"] = v["rel"] if v["rel"] is not None else v["abs"]
    # 학습 기록(독립 반복만): 마지막 에폭 손실·세기
    for t, c in cases.items():
        if c["run"]:
            rec = READ(Path(c["checkpoint"]).parent / "training_complete.json")["records"]
            last = [r for r in rec if r["epoch"] == 8]
            feats[t]["train_last_loss"] = float(np.mean([r["loss"] for r in last]))
            feats[t]["train_last_strength"] = float(np.mean([r["strength_mean"] for r in last]))

    # ---- 상관(독립 반복 diff 15개)
    keys = sorted(feats["diff_s1_r1"])
    diff15 = [f"diff_s{k}_r{r}" for k in STARTS for r in RUNS]
    conv15 = [f"convs_only_s{k}_r{r}" for k in STARTS for r in RUNS]
    corr = {}
    for target, gains in (("test_AP", test_gain), ("val_AP", val_gain)):
        m = target.split("_")[1]
        y = [gains[t][m] for t in diff15]
        corr[target] = {k: spearman_perm([feats[t][k] for t in diff15], y) for k in keys
                        if np.std([feats[t][k] for t in diff15]) > 0}
    corr["val_AP_vs_test_AP_diff15"] = spearman_perm([val_gain[t]["AP"] for t in diff15],
                                                     [test_gain[t]["AP"] for t in diff15])
    corr["val_AP_vs_test_AP_convs15"] = spearman_perm([val_gain[t]["AP"] for t in conv15],
                                                      [test_gain[t]["AP"] for t in conv15])

    # ---- 시작점·실행 시드별 요약
    def group(tags_of, gains, m="AP"):
        return {str(g): float(np.mean([gains[t][m] for t in tags])) for g, tags in tags_of.items()}

    by = {}
    for kind in ("diff", "convs_only"):
        st = {k: [f"{kind}_s{k}_r{r}" for r in RUNS] for k in STARTS}
        rs = {r: [f"{kind}_s{k}_r{r}" for k in STARTS] for r in RUNS}
        by[kind] = dict(
            test_by_start=group(st, test_gain), val_by_start=group(st, val_gain),
            test_by_run=group(rs, test_gain), val_by_run=group(rs, val_gain),
            orig_test={str(k): test_gain[f"{kind}_o{k}"]["AP"] for k in STARTS},
            orig_val_cpu={str(k): val_gain[f"{kind}_o{k}"]["AP"] for k in STARTS},
            orig_val_gpu={str(k): gpu_val_gain_orig[f"{kind}_o{k}"]["AP"] for k in STARTS},
            mean15_test={m: float(np.mean([test_gain[t][m] for t in (st[1] + st[2] + st[3])])) for m in METRICS},
            mean15_val={m: float(np.mean([val_gain[t][m] for t in (st[1] + st[2] + st[3])])) for m in METRICS},
            sd15_test_AP=float(np.std([test_gain[t]["AP"] for t in (st[1] + st[2] + st[3])], ddof=1)),
            sd15_val_AP=float(np.std([val_gain[t]["AP"] for t in (st[1] + st[2] + st[3])], ddof=1)),
        )
    summary_feats = {}
    for kind, tags in (("diff", diff15), ("convs_only", conv15)):
        summary_feats[kind] = {k: dict(mean=float(np.mean([feats[t][k] for t in tags])),
                                       sd=float(np.std([feats[t][k] for t in tags], ddof=1)),
                                       min=float(min(feats[t][k] for t in tags)),
                                       max=float(max(feats[t][k] for t in tags)))
                               for k in keys if all(k in feats[t] for t in tags)}
    summary_feats["by_run_seed_diff"] = {str(r): {k: float(np.mean([feats[f"diff_s{s}_r{r}"][k] for s in STARTS]))
                                                  for k in ("test_eta0_mean", "test_eta1_mean", "test_tv0_mean",
                                                            "test_tv1_mean", "test_dev_from_fixed",
                                                            "drift_input_proj", "drift_batch_norms", "drift_gate.2",
                                                            "drift_edge_encoder", "drift_convs.qkv")}
                                         for r in RUNS}
    report = dict(
        test_gain_pp=test_gain, val_gain_pp=val_gain, features=feats, correlations_diff15=corr,
        groups=by, feature_summary=summary_feats,
        val_cpu_vs_gpu=dict(gain_pp_gpu_originals=gpu_val_gain_orig, metric_diff_pp=cpu_vs_gpu,
                            winner_agreement=agree),
        notes=dict(
            test="기존 GPU 원장(독립 반복 test_eval, 원 시험 폴더). 45시퀀스→6도시 균등. 이득=조건−fixed(%p).",
            val="CPU 재평가(TF32 없음). 원래 세 시작점은 기존 GPU 원장도 함께 보고. 9시퀀스 계열 매크로.",
            eta="last_strengths: 층별 η=−sigmoid(g). 노드 풀링 분위수. frac_weaker=η>−0.5 비율.",
            tv="0.5·Σ_j|a_ij−1/deg_i|, 헤드 평균, 노드 평균.",
            dev="‖z′−z_fixed‖/‖z_fixed−z‖ 시퀀스 평균.",
            drift="‖θ−θ0‖/‖θ0‖(θ0=0인 gate.2는 절대값). θ0=같은 시작점의 균등 감산 초기화.",
            corr="독립 반복 diff 15개 Spearman, 순열 p(20000).",
        ),
        test_used_for_selection=False,
    )
    WRITE(OUT / "result.json", report)
    print(json.dumps(dict(groups=by, cpu_vs_gpu=cpu_vs_gpu, agree=agree), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "val_worker", "stats_worker", "drift", "analyze"))
    p.add_argument("--role", choices=("val", "test"))
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "val_worker":
        val_worker(a.sequence)
    elif a.action == "stats_worker":
        stats_worker(a.role, a.sequence)
    elif a.action == "drift":
        drift()
    elif a.action == "run":
        run_all()
    else:
        analyze()
