#!/usr/bin/env python3
"""전달 세기를 균등 감산 값 −0.5보다 강해지지 못하게 묶은 `diff` 재학습(독립 반복 15회 + 원래 표본 순서 3회).

게이트: η = max(−sigmoid(g), −0.5) = −min(sigmoid(g), 0.5). 게이트 최종층이 0에서 시작하므로 g=0,
sigmoid(0)=0.5가 정확히 경계 위에 있고 초기 출력은 원래 `diff`와 비트 단위로 같다. clamp의 기울기는 경계에서
통과(포함)하므로 첫 갱신도 원래 `diff`와 같다. 학습이 g>0으로 밀면 그 노드의 η는 −0.5에 머물고 기울기가 0이며,
g<0이면 η가 0 쪽(감산 약화)으로만 움직인다.

그 밖의 모든 것은 2026-11-08 독립 반복의 `diff` 조건과 같다: 시작점 3(사전학습 시드 1–3), 실행 시드 r1..r5
(numpy default_rng([r, 45000+1000·epoch+순번])), r0(기존 공식), 512갱신, AdamW lr 1e-4, 재보정 손실, epoch8 사용.
학습 코드는 `_pdv2_diffattn_independent_runs.train`을 그대로 부르고 모델 클래스만 바꾼다.
평가는 기존 시험 실행기(`_pdv2_contrast_pipeline_test`)와 검증 실행기(`_pdv2_contrast_pipeline`)를 그대로 쓴다.

사용법(컨테이너 ring_baseline, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python3 scripts/_pdv2_bounded_gate_20261116.py preflight   # 초기 출력·첫 기울기 동일성, 경계 확인
  python3 scripts/_pdv2_bounded_gate_20261116.py reproduce   # 같은 사슬로 기존 diff s1_r3 체크포인트 비트 재현
  python3 scripts/_pdv2_bounded_gate_20261116.py train_all [--only s1_r1,s1_r3]
  python3 scripts/_pdv2_bounded_gate_20261116.py evaluate --role test|val
  python3 scripts/_pdv2_bounded_gate_20261116.py analyze
시험셋은 어떤 설정 선택에도 쓰지 않는다.
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

OUT = ROOT / "results/pdv2_bounded_gate_20261116"
INDEP = Path(os.environ.get("NSD_INDEP_OUT", "/workspace/data/nsd_results/pdv2_diffattn_independent_20261108"))
DIAG = ROOT / "results/pdv2_diff_gain_diagnosis_20261116"
OLD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
STARTS = (1, 2, 3)
RUNS = (1, 2, 3, 4, 5)
ALL_RUNS = (0,) + RUNS
METRICS = ("AP", "max_F1", "R@100P", "R@1")
REPRO = (1, 3)  # 기존 diff 독립 반복 중 강화 상태 실행 하나를 재현 대상으로 쓴다
TRAIN_WORKERS = int(os.environ.get("NSD_TRAIN_WORKERS", 1))
EVAL_WORKERS = int(os.environ.get("NSD_EVAL_WORKERS", 1))


def READ(p):
    return json.loads(Path(p).read_text())


def WRITE(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    tmp.replace(p)


def HASH(p):
    import hashlib

    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 모델
def bounded_class():
    import torch
    from torch.nn import functional as F
    from gnn.contrast_spectral_difference import ContrastSpectralDifference
    from gnn.spectral_difference import transport_difference

    class BoundedContrastSpectralDifference(ContrastSpectralDifference):
        """ContrastSpectralDifference.forward와 같고 세기만 η=max(−sigmoid(g), −0.5)로 묶는다."""

        def forward(self, graph):
            h = F.relu(self.input_norm(self.input_proj(graph.x)))
            first = h
            attr = (self.edge_encoder(graph.edge_attr, graph.edge_type)
                    if self.edge_encoder is not None else None)
            weights = []
            src, tgt = graph.edge_index
            for layer, (conv, norm) in enumerate(zip(self.convs, self.batch_norms)):
                previous = h
                h = norm(conv(h, graph.edge_index, edge_attr=attr))
                if layer == 0:
                    h = F.relu(h)
                h = h + previous
                if self.kind != "point_gate":
                    dense = torch.zeros(len(h), 11, 4, dtype=h.dtype, device=h.device)
                    dense[tgt, src - tgt + 5] = conv.attention
                    weights.append(dense)
            # 유일한 변경: −sigmoid(g)를 −0.5 아래로 내려가지 못하게 한다(clamp 기울기는 경계 포함 통과).
            strengths = -torch.stack((self.gate(first).sigmoid(), self.gate(h).sigmoid())).clamp(max=0.5)
            self.last_strengths = strengths
            attention = torch.stack(weights) if weights else None
            return transport_difference(graph.spectrum, graph.transport_indices, graph.transport_mask,
                                        graph.transport_phase, strengths, attention)

    return BoundedContrastSpectralDifference


def patched_independent():
    """독립 반복 학습 모듈에 'bounded' 조건을 더한다(lr·학습 범위는 diff와 같음)."""
    import _pdv2_diffattn_independent_runs as I

    if "bounded" not in I.ARMS:
        I.ARMS["bounded"] = I.ARMS["diff"]
        original = I.build
        cls = bounded_class()

        def build(torch, e, start, arm):
            net = original(torch, e, start, "diff" if arm == "bounded" else arm)
            if arm == "bounded":
                net.__class__ = cls
            return net

        I.build = build
    return I


def run_folder(k, r):
    return OUT / "runs" / f"s{k}" / f"r{r}"


# ---------------------------------------------------------------- 사전 확인
def protocol():
    path = OUT / "protocol.json"
    if path.exists():
        return READ(path)
    I = patched_independent()
    spec = dict(
        created_unix=time.time(),
        purpose="게이트 쌍안정성(2026-11-16 진단) 가설 시험: 세기를 −0.5보다 강해지지 못하게 묶으면 diff 학습 이득이 안정되는가.",
        gate="η = max(−sigmoid(g), −0.5). 초기 g=0에서 경계와 정확히 일치. torch.clamp(max=0.5) 기울기는 경계 포함 통과.",
        unchanged="시작점 3, 실행 시드 r0(기존 공식)·r1..r5(default_rng([r, base])), 512갱신, AdamW lr 1e-4·감쇠 1e-5·절단 1, 재보정 손실(독립 반복 protocol.json의 loss_fit), epoch8, matching_only 검색·수락. 학습은 _pdv2_diffattn_independent_runs.train을 그대로 부름.",
        starts=list(STARTS), runs=list(ALL_RUNS),
        independent_protocol_sha256=HASH(INDEP / "protocol.json"),
        sources={str(Path(p).relative_to(ROOT)): HASH(p) for p in (
            Path(__file__), Path(I.__file__), ROOT / "src/gnn/contrast_spectral_difference.py",
            ROOT / "src/gnn/learned_spectral_difference.py", ROOT / "src/gnn/spectral_difference.py",
            ROOT / "scripts/_pdv2_contrast_pipeline.py", ROOT / "scripts/_pdv2_contrast_pipeline_test.py")},
        test_used_for_selection=False,
    )
    WRITE(path, spec)
    return spec


def preflight():
    """시작점마다 초기 출력·세기·첫 기울기가 원래 diff와 비트 단위로 같은지, 고정 감산과의 차이를 잰다."""
    I = patched_independent()
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
        out, grads, strengths = {}, {}, {}
        for arm in ("diff", "bounded"):
            net = I.build(torch, e, k, arm)
            net.zero_grad(set_to_none=True)
            z = net(graph)
            loss = F.binary_cross_entropy_with_logits(
                e.prior.score(z, {a: v[ix] for a, v in plan.items()}), label)
            loss.backward()
            out[arm] = z.detach()
            strengths[arm] = net.last_strengths.detach()
            grads[arm] = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}
        with torch.no_grad():
            fixed = e.prior.filtered(graph, -0.5)
        same_grad = grads["diff"].keys() == grads["bounded"].keys() and all(
            torch.equal(grads["diff"][n], grads["bounded"][n]) for n in grads["diff"])
        rows[f"s{k}"] = dict(
            output_bit_identical_to_diff=bool(torch.equal(out["diff"], out["bounded"])),
            strengths_all_exactly_minus_half=bool((strengths["bounded"] == -0.5).all()),
            first_gradient_bit_identical_to_diff=bool(same_grad),
            gate_final_grad_norm=float(sum(v.pow(2).sum() for n, v in grads["bounded"].items()
                                           if n.startswith("gate.2")).sqrt()),
            max_abs_vs_fixed_bounded=float((out["bounded"] - fixed).abs().max()),
            max_abs_vs_fixed_diff=float((out["diff"] - fixed).abs().max()),
            bit_identical_to_fixed=bool(torch.equal(out["bounded"], fixed)),
        )
        print("PREFLIGHT", k, rows[f"s{k}"], flush=True)
    # 경계 동작 확인: g>0이면 η=−0.5이고 기울기 0, g<0이면 η=−sigmoid(g)이고 기울기 통과
    g = torch.tensor([-1.0, 0.0, 1.0], device="cuda", requires_grad=True)
    eta = -g.sigmoid().clamp(max=0.5)
    eta.sum().backward()
    rows["boundary_unit"] = dict(g=[-1, 0, 1], eta=eta.tolist(), grad=g.grad.tolist())
    ok = all(r["output_bit_identical_to_diff"] and r["strengths_all_exactly_minus_half"]
             and r["first_gradient_bit_identical_to_diff"] and r["max_abs_vs_fixed_bounded"] < 4e-6
             for k, r in rows.items() if k.startswith("s"))
    rows["status"] = "passed" if ok else "failed"
    rows["graph"] = spec["train"][0]
    WRITE(OUT / "preflight.json", rows)
    print(json.dumps(rows, indent=1), flush=True)
    assert ok


def reproduce():
    """같은 학습 사슬(패치된 모듈)로 기존 독립 반복 diff 체크포인트 하나를 비트 단위로 재현한다."""
    I = patched_independent()
    import torch

    k, r = REPRO
    new = I.train(k, r, "diff", OUT / "repro" / f"diff_s{k}_r{r}")
    old = INDEP / "runs" / f"s{k}" / f"r{r}" / "diff" / "epoch8.pth"
    a = torch.load(new, map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(old, map_location="cpu", weights_only=False)["model_state_dict"]
    exact = a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)
    rec = dict(status="passed" if exact else "failed", case=f"diff_s{k}_r{r}", new=str(new.relative_to(ROOT)),
               old=str(old), old_sha256=HASH(old), new_sha256=HASH(new),
               strength_records_equal=READ(new.parent / "training_complete.json")["records"]
               == READ(old.parent / "training_complete.json")["records"])
    WRITE(OUT / "reproduction_train.json", rec)
    print(rec, flush=True)
    assert exact


def train_one(k, r):
    assert READ(OUT / "preflight.json")["status"] == "passed"
    assert READ(OUT / "reproduction_train.json")["status"] == "passed"
    I = patched_independent()
    I.train(k, r, "bounded", run_folder(k, r))


def train_all(only=None):
    jobs = [(k, r) for k in STARTS for r in ALL_RUNS]
    if only:
        want = set(only.split(","))
        jobs = [j for j in jobs if f"s{j[0]}_r{j[1]}" in want]
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(job):
        k, r = job
        if (run_folder(k, r) / "training_complete.json").exists():
            return
        with (logs / f"train_s{k}_r{r}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "train", "--start", str(k), "--run", str(r)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=TRAIN_WORKERS) as pool:
        for i, f in enumerate(concurrent.futures.as_completed([pool.submit(call, j) for j in jobs]), 1):
            f.result()
            print("TRAIN_DONE", i, len(jobs), flush=True)


# ---------------------------------------------------------------- 평가
def cases(role):
    out = {"fixed": dict(kind="fixed", seed=0, checkpoint=None)}
    k, r = REPRO
    rp = OUT / "repro" / f"diff_s{k}_r{r}" / "epoch8.pth"
    # 검증의 seed=1·diff 분기(기존 원장 대조)를 피하려 seed=100+k로 둔다.
    out[f"diff_s{k}_r{r}_repro"] = dict(kind="diff", seed=100 + k, checkpoint=str(rp.relative_to(ROOT)),
                                        checkpoint_sha256=HASH(rp))
    for s in STARTS:
        for rr in ALL_RUNS:
            p = run_folder(s, rr) / "epoch8.pth"
            h = HASH(p)
            assert READ(p.parent / "training_complete.json")["sha256"] == h
            out[f"bounded_s{s}_r{rr}"] = dict(kind="bounded", seed=100 + s, checkpoint=str(p.relative_to(ROOT)),
                                              checkpoint_sha256=h)
    return out


def eval_spec(role):
    path = OUT / f"{role}_eval/protocol.json"
    if path.exists():
        return READ(path)
    base = OLD_TEST if role == "test" else OLD_VAL
    spec = dict(READ(base / "protocol.json"))
    spec.update(created_unix=time.time(), cases=cases(role), parent_protocol_sha256=HASH(OUT / "protocol.json"),
                test_policy="게이트 세기 제한 재학습의 고정 평가. 설정 선택 없음.")
    WRITE(path, spec)
    return spec


def eval_worker(role, sequence):
    import torch
    cls = bounded_class()
    import _pdv2_contrast_pipeline as P

    P.e.configure()
    original_load = P.load_model

    def load_model(case):
        if case["kind"] == "bounded":
            net = original_load(dict(case, kind="diff"))
            net.__class__ = cls
            return net
        return original_load(case)

    P.load_model = load_model
    folder = OUT / f"{role}_eval" / role / sequence.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    src = (OLD_TEST / "test" if role == "test" else OLD_VAL / "val") / sequence.replace("/", "_")
    for suffix in (".npz", ".json"):
        t = folder / ("candidate_pool" + suffix)
        if not t.exists():
            t.symlink_to(os.path.relpath(src / ("candidate_pool" + suffix), folder))
    if role == "test":
        import _pdv2_contrast_pipeline_test as T

        T.OUT = OUT / "test_eval"
        T.prepare = lambda: eval_spec("test")
        T.worker("test", sequence)
    else:
        P.OUT = OUT / "val_eval"
        P.run.OUT = P.OUT
        P.prepare = lambda: eval_spec("val")
        P.worker("val", sequence)
    # 기존 원장 재현: 균등 감산(기존 시험·검증 원장)과 원래 diff 한 실행(독립 반복 시험 GPU 원장 / 진단 검증 CPU 원장)
    name = sequence.replace("/", "_")
    k, r = REPRO
    refs = {"fixed": (OLD_TEST / "test" / name / "fixed.npz") if role == "test"
            else (OLD_VAL / "val" / name / "fixed.npz"),
            f"diff_s{k}_r{r}_repro": (INDEP / "test_eval/test" / name / f"diff_s{k}_r{r}.npz") if role == "test"
            else (DIAG / "val_eval/val" / name / f"diff_s{k}_r{r}.npz")}
    checks = {}
    for tag, ref in refs.items():
        a, b = np.load(folder / f"{tag}.npz"), np.load(ref)
        np.testing.assert_array_equal(a["query_idx"], b["query_idx"])
        agree = float((a["j_star"] == b["j_star"]).mean())
        err = float(np.abs(a["score"] - b["score"]).max())
        exact_gpu = role == "test" or tag == "fixed"  # 검증 diff 원장은 CPU 재평가라 허용폭만 본다
        if exact_gpu:
            np.testing.assert_array_equal(a["j_star"], b["j_star"])
            assert err < 8e-6, (tag, err)
        checks[tag] = dict(reference=str(ref), winner_agreement=agree, score_max_abs=err,
                           exact_required=exact_gpu)
    # 노드별 전달 세기(묶인 게이트 모델 18개)
    from _pdv2_contrast_pipeline import e as E

    graph, _, _ = E.prior.graph_inputs(role, sequence)
    spec = eval_spec(role)
    arrays = {}
    with torch.no_grad():
        for tag, case in spec["cases"].items():
            if case["kind"] != "bounded":
                continue
            net = load_model(case)
            net(graph)
            arrays[tag] = net.last_strengths.squeeze(-1).float().cpu().numpy()
            del net
    np.savez_compressed(folder / "strengths.npz", **arrays)
    WRITE(folder / "reproduction_check.json", dict(status="passed", checks=checks))
    print("EVAL_DONE", role, sequence, flush=True)


def evaluate(role):
    eval_spec(role)
    seqs = READ((OLD_TEST if role == "test" else OLD_VAL) / "protocol.json")[role]
    logs = OUT / f"{role}_eval/logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(s):
        if (OUT / f"{role}_eval" / role / s.replace("/", "_") / "reproduction_check.json").exists():
            return
        with (logs / (s.replace("/", "_") + ".log")).open("a") as f:
            subprocess.run([sys.executable, __file__, "eval_worker", "--role", role, "--sequence", s],
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=EVAL_WORKERS) as pool:
        futs = [pool.submit(call, s) for s in seqs]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            f.result()
            WRITE(OUT / f"{role}_eval/status.json", dict(done=i, total=len(futs)))
            print("EVAL", role, i, len(futs), flush=True)


# ---------------------------------------------------------------- 분석
def crossed_interval(D, macro_arr, groups, seed=20261109, B=10000):
    """posthoc_crossed.py와 같은 재표집: 실행 시드 5개를 세 시작점 공동으로 복원추출 + 묶음 안 시퀀스 복원추출.
    D[k, r, seq]. 같은 난수 흐름(시퀀스 먼저, 다음 시드)을 쓴다."""
    rng = np.random.default_rng(seed)
    bseq = [[rng.integers(0, len(ix), len(ix)) for ix in groups] for _ in range(B)]
    brun = rng.integers(0, 5, (B, 5))
    y = np.array([np.mean([macro_arr(D[k, r]) for k in range(D.shape[0])]) for r in range(D.shape[1])])
    bb = np.array([macro_arr(D[:, brun[b]].mean(axis=(0, 1)), bseq[b]) for b in range(B)])
    return float(y.mean()), [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))], y.tolist()


def analyze():
    import _pdv2_contrast_pipeline as P
    import _pdv2_pilot as pilot
    from _pr_eval_lib import classify_queries, pr_sweep

    fam = P.run.base.macro.__globals__["family"]  # 기존 검증 매크로가 쓰는 계열 함수
    out = dict(notes={})
    # ---------------- 시험 46
    tests = READ(OLD_TEST / "protocol.json")["test"]
    per, prefix, repro = {}, [], {}
    for s in tests:
        nm = s.replace("/", "_")
        mine = READ(OUT / "test_eval/test" / nm / "summary.json")
        rc = READ(OUT / "test_eval/test" / nm / "reproduction_check.json")
        assert rc["status"] == "passed"
        repro[s] = rc["checks"]
        prefix += [dict(c, sequence=s) for c in mine["prefix_checks"]]
        indep = READ(INDEP / "test_eval/test" / nm / "summary.json")["metrics"]
        old = READ(OLD_TEST / "test" / nm / "summary.json")["metrics"]
        row = dict(mine["metrics"])
        for kind in ("diff", "convs_only"):
            for k in STARTS:
                row[f"{kind}_s{k}_r0"] = old[f"{kind}_s{k}"]
                for r in RUNS:
                    row[f"{kind}_s{k}_r{r}"] = indep[f"{kind}_s{k}_r{r}"]
        per[s] = row
    cities = {c: [s for s in q if per[s]["fixed"]["n_pos"]] for c, q in pilot.CITY.items()}
    assert sum(len(v) for v in cities.values()) == 45
    order = [s for q in cities.values() for s in q]
    cidx = [np.array([order.index(s) for s in q]) for q in cities.values()]

    def tmac(v, p=None):
        return float(np.mean([v[ix if p is None else ix[p[c]]].mean() for c, ix in enumerate(cidx)]))

    old_sum = READ(OLD_TEST / "summary.json")
    for m in METRICS:
        f = tmac(np.array([per[s]["fixed"][m] for s in order]))
        assert abs(f - old_sum["mean"]["fixed"][m]) < 1e-12, m
    # 독립 지표 재계산(_pr_eval_lib.pr_sweep): 모든 사례·시퀀스의 AP·max_F1·R@100P
    worst = 0.0
    n_checked = 0
    spec_t = READ(OUT / "test_eval/protocol.json")
    for s in tests:
        nm = s.replace("/", "_")
        for tag in spec_t["cases"]:
            d = np.load(OUT / "test_eval/test" / nm / f"{tag}.npz")
            if not (d["min_past_gt_dist"] <= 5).any():
                continue
            rr = pr_sweep(d["score"], d["top1_geo_dist"], classify_queries(d["min_past_gt_dist"]))
            for m in ("AP", "max_F1", "R@100P"):
                worst = max(worst, abs(rr[m] - per[s][tag][m]))
            n_checked += 1
    out["independent_metric_recompute"] = dict(tool="_pr_eval_lib.pr_sweep", ledgers=n_checked,
                                               metrics=["AP", "max_F1", "R@100P"], max_abs=worst)
    assert worst < 1e-9, worst

    test = {}
    crossed_check = READ(INDEP / "posthoc_crossed.json")["results"]
    for arm in ("bounded", "diff", "convs_only"):
        res = {}
        for m in METRICS:
            D = np.array([[[100 * (per[s][f"{arm}_s{k}_r{r}"][m] - per[s]["fixed"][m]) for s in order]
                           for r in RUNS] for k in STARTS])
            runs = np.array([[tmac(D[i, j]) for j in range(5)] for i in range(3)])
            mean, iv, by_seed = crossed_interval(D, tmac, cidx)
            if arm != "bounded":  # 기존 사후 구간 재현
                ref = crossed_check[arm][m]
                assert abs(mean - ref["mean"]) < 1e-9 and np.allclose(iv, ref["B_crossed"], atol=1e-9), (arm, m)
            r0 = [tmac(np.array([100 * (per[s][f"{arm}_s{k}_r0"][m] - per[s]["fixed"][m]) for s in order]))
                  for k in STARTS]
            res[m] = dict(mean_pp=mean, crossed_95=iv, positive_runs=int((runs > 0).sum()), per_run=runs.tolist(),
                          per_seed_mean=by_seed, per_start_mean=runs.mean(1).tolist(),
                          sd_runs=float(runs.std(ddof=1)), r0_per_start=r0, r0_mean=float(np.mean(r0)))
        test[arm] = res
    out["test46_gain_pp"] = test
    out["notes"]["test"] = ("45시퀀스→6도시 균등. 이득=조건−fixed(%p). crossed_95=실행 시드 공동 복원추출+도시 안 시퀀스 "
                            "복원추출 10,000회(난수 20261109, posthoc_crossed.py와 같은 흐름). diff·convs_only는 기존 "
                            "posthoc_crossed.json을 1e-9 안에서 재현.")
    # ---------------- 검증 9
    vals = READ(OLD_VAL / "protocol.json")["val"]
    vper = {}
    for s in vals:
        nm = s.replace("/", "_")
        mine = READ(OUT / "val_eval/val" / nm / "summary.json")
        rc = READ(OUT / "val_eval/val" / nm / "reproduction_check.json")
        assert rc["status"] == "passed"
        repro["val:" + s] = rc["checks"]
        prefix += [dict(c, sequence="val:" + s) for c in mine["prefix_checks"]]
        dg = READ(DIAG / "val_eval/val" / nm / "summary.json")["metrics"]
        row = dict(mine["metrics"])
        row["fixed_cpu"] = dg["fixed_cpu"]
        for kind in ("diff", "convs_only"):
            for k in STARTS:
                row[f"{kind}_s{k}_r0"] = dg[f"{kind}_o{k}"]
                for r in RUNS:
                    row[f"{kind}_s{k}_r{r}"] = dg[f"{kind}_s{k}_r{r}"]
        vper[s] = row
    fams = sorted({fam(s) for s in vals if vper[s]["fixed"]["n_pos"]})
    vorder = [s for g in fams for s in vals if fam(s) == g and vper[s]["fixed"]["n_pos"]]
    vidx = [np.array([vorder.index(s) for s in vorder if fam(s) == g]) for g in fams]

    def vmac(v, p=None):
        return float(np.mean([v[ix if p is None else ix[p[c]]].mean() for c, ix in enumerate(vidx)]))

    macro_fam = P.run.base.macro
    for tag in ("fixed", "bounded_s1_r0"):
        ref = macro_fam({s: vper[s][tag] for s in vals})["macro"]
        for m in METRICS:
            assert abs(vmac(np.array([vper[s][tag][m] for s in vorder])) - ref[m]) < 1e-12
    val = {}
    for arm in ("bounded", "diff", "convs_only"):
        base = "fixed" if arm == "bounded" else "fixed_cpu"  # 같은 장치의 균등 감산 대비
        res = {}
        for m in METRICS:
            D = np.array([[[100 * (vper[s][f"{arm}_s{k}_r{r}"][m] - vper[s][base][m]) for s in vorder]
                           for r in RUNS] for k in STARTS])
            runs = np.array([[vmac(D[i, j]) for j in range(5)] for i in range(3)])
            mean, iv, by_seed = crossed_interval(D, vmac, vidx)
            r0 = [vmac(np.array([100 * (vper[s][f"{arm}_s{k}_r0"][m] - vper[s][base][m]) for s in vorder]))
                  for k in STARTS]
            res[m] = dict(mean_pp=mean, crossed_95=iv, positive_runs=int((runs > 0).sum()), per_run=runs.tolist(),
                          per_seed_mean=by_seed, per_start_mean=runs.mean(1).tolist(),
                          sd_runs=float(runs.std(ddof=1)), r0_per_start=r0, r0_mean=float(np.mean(r0)))
        val[arm] = res
    out["val9_gain_pp"] = val
    out["notes"]["val"] = ("양성 있는 시퀀스→계열 균등(기존 검증 매크로와 1e-12 일치). bounded는 이번 GPU 평가, diff·"
                           "convs_only는 2026-11-16 진단의 CPU 재평가(각자 같은 장치의 fixed 대비). 구간은 시험과 같은 "
                           "방식으로 계열 안 시퀀스와 실행 시드를 재표집(난수 20261109).")
    # ---------------- 전달 세기
    eta = {}
    for role, seqs in (("test", tests), ("val", vals)):
        data = [np.load(OUT / f"{role}_eval" / role / s.replace("/", "_") / "strengths.npz") for s in seqs]
        for k in STARTS:
            for r in ALL_RUNS:
                tag = f"bounded_s{k}_r{r}"
                x = np.concatenate([d[tag] for d in data], axis=1)
                assert x.min() >= -0.5
                row = eta.setdefault(tag, {})
                for layer in (0, 1):
                    v = x[layer]
                    row[f"{role}_eta{layer}_mean"] = float(v.mean())
                    for q in (5, 25, 50, 75, 95):
                        row[f"{role}_eta{layer}_p{q:02d}"] = float(np.percentile(v, q))
                    row[f"{role}_eta{layer}_frac_at_bound"] = float((v == -0.5).mean())
                    row[f"{role}_eta{layer}_frac_weaker"] = float((v > -0.5).mean())
                rec = READ(run_folder(k, r) / "training_complete.json")["records"]
                row["train_strength_by_epoch"] = [float(np.mean([z["strength_mean"] for z in rec if z["epoch"] == ep]))
                                                  for ep in range(1, 9)]
                row["train_last_loss"] = float(np.mean([z["loss"] for z in rec if z["epoch"] == 8]))
    out["strengths"] = eta
    diag = READ(DIAG / "result.json")["features"]
    out["strengths_reference_diff"] = {
        t: {key: diag[t][key] for key in ("test_eta0_mean", "test_eta1_mean", "val_eta0_mean", "val_eta1_mean",
                                          "test_eta1_frac_weaker")}
        for t in diag if t.startswith("diff_")}
    # ---------------- 주 결과표 방식 r0 세 시작점 평균과 BEVPlace++
    per_case = old_sum["per_case"]
    seqrows = {s: per[s] for s in tests}
    mac3 = {}
    for k in STARTS:
        mac3[k] = {m: tmac(np.array([seqrows[s][f"bounded_s{k}_r0"][m] for s in order])) for m in METRICS}
    bmean = {m: float(np.mean([mac3[k][m] for k in STARTS])) for m in METRICS}
    bsd = {m: float(np.std([mac3[k][m] for k in STARTS], ddof=1)) for m in METRICS}

    def paired(a_rows, b_rows, seed=20261104, reps=10000):
        """_pdv2_contrast_pipeline_report.report와 같은 절차(시드 평균 차이를 도시 안 시퀀스 재표집)."""
        rng = np.random.default_rng(seed)
        draws = []
        for city, seqs in pilot.CITY.items():
            avail = [q for q in seqs if per_case["fixed"]["sequences"][q]["n_pos"]]
            mat = np.array([[np.mean([100 * (a_rows[s][q][m] - b_rows[s][q][m]) for s in STARTS])
                             for m in METRICS] for q in avail])
            index = rng.integers(0, len(avail), size=(reps, len(avail)))
            draws.append(mat[index].mean(1))
        boot = np.stack(draws).mean(0)
        return {m: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, m in enumerate(METRICS)}

    def rows_of(kind):
        if kind == "bounded":
            return {k: {s: seqrows[s][f"bounded_s{k}_r0"] for s in tests} for k in STARTS}
        if kind == "fixed":
            return {k: per_case["fixed"]["sequences"] for k in STARTS}
        return {k: per_case[f"{kind}_s{k}"]["sequences"] for k in STARTS}

    # 절차 재현: 기존 integrated_report의 diff-bev_native 구간
    ref_iv = READ(OLD_TEST / "integrated_report.json")["comparisons"]["diff-bev_native"]["conditional_95_pp"]
    got = paired(rows_of("diff"), rows_of("bev_native"))
    assert all(np.allclose(got[m], ref_iv[m], atol=1e-9) for m in METRICS), (got, ref_iv)
    table = dict(bounded_r0_mean_percent={m: 100 * v for m, v in bmean.items()},
                 bounded_r0_sd_percent={m: 100 * v for m, v in bsd.items()},
                 bounded_r0_per_start_percent={str(k): {m: 100 * v for m, v in mac3[k].items()} for k in STARTS},
                 nsd_diff_mean_percent={m: 100 * old_sum["mean"]["diff"][m] for m in METRICS},
                 nsd_diff_sd_percent={m: 100 * old_sum["std"]["diff"][m] for m in METRICS},
                 bevplace_mean_percent={m: 100 * old_sum["mean"]["bev_native"][m] for m in METRICS},
                 bevplace_sd_percent={m: 100 * old_sum["std"]["bev_native"][m] for m in METRICS},
                 fixed_percent={m: 100 * old_sum["mean"]["fixed"][m] for m in METRICS},
                 convs_only_mean_percent={m: 100 * old_sum["mean"]["convs_only"][m] for m in METRICS})
    for b in ("bev_native", "diff", "fixed"):
        bm = old_sum["mean"][b]
        table[f"bounded_minus_{b}_pp"] = {m: 100 * (bmean[m] - bm[m]) for m in METRICS}
        table[f"bounded_minus_{b}_conditional_95_pp"] = paired(rows_of("bounded"), rows_of(b))
    table["note"] = ("주 결과표와 같은 집계(45시퀀스→6도시→세 시작점 평균). 구간은 _pdv2_contrast_pipeline_report와 같은 "
                     "절차(난수 20261104); 기존 diff-bev_native 구간을 1e-9 안에서 재현한 뒤 적용.")
    out["main_table_r0"] = table
    # ---------------- 검산 요약
    bounded_r0 = [c for c in prefix if c["condition"].startswith("bounded_") and c["condition"].endswith("_r0")]
    out["prefix_q15"] = dict(all_checks=len(prefix), bounded_r0_checks=len(bounded_r0),
                             max_score_error_all=max(c["score_error"] for c in prefix),
                             max_score_error_bounded_r0=max(c["score_error"] for c in bounded_r0),
                             note="각 시퀀스·사례에서 두 질의를 q+15 접두 입력으로 재계산, 승자 일치(assert)·점수 차 8e-6 안")
    tr = [v for k, v in repro.items() if not k.startswith("val:")]
    vr = [v for k, v in repro.items() if k.startswith("val:")]
    k, r = REPRO
    out["ledger_reproduction"] = dict(
        test_fixed=dict(sequences=len(tr), winners_exact=all(v["fixed"]["winner_agreement"] == 1 for v in tr),
                        max_score_error=max(v["fixed"]["score_max_abs"] for v in tr)),
        test_diff_repro=dict(case=f"diff_s{k}_r{r}", sequences=len(tr),
                             winners_exact=all(v[f"diff_s{k}_r{r}_repro"]["winner_agreement"] == 1 for v in tr),
                             max_score_error=max(v[f"diff_s{k}_r{r}_repro"]["score_max_abs"] for v in tr)),
        val_fixed=dict(sequences=len(vr), winners_exact=all(v["fixed"]["winner_agreement"] == 1 for v in vr),
                       max_score_error=max(v["fixed"]["score_max_abs"] for v in vr)),
        val_diff_repro_vs_cpu=dict(min_winner_agreement=min(v[f"diff_s{k}_r{r}_repro"]["winner_agreement"] for v in vr),
                                   max_score_error=max(v[f"diff_s{k}_r{r}_repro"]["score_max_abs"] for v in vr)),
        train=READ(OUT / "reproduction_train.json"),
        preflight=READ(OUT / "preflight.json"),
    )
    out["test_used_for_selection"] = False
    WRITE(OUT / "result.json", out)
    for name, block in (("TEST", test), ("VAL", val)):
        for arm, res in block.items():
            print(name, arm, {m: (round(v["mean_pp"], 3), np.round(v["crossed_95"], 3).tolist(), v["positive_runs"],
                                  round(v["r0_mean"], 3)) for m, v in res.items()}, flush=True)
    print("TABLE", json.dumps({k: v for k, v in table.items() if k != "note"}, indent=0), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("preflight", "reproduce", "train", "train_all", "evaluate", "eval_worker",
                                      "analyze"))
    p.add_argument("--start", type=int)
    p.add_argument("--run", type=int)
    p.add_argument("--only")
    p.add_argument("--role", choices=("test", "val"), default="test")
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "train":
        train_one(a.start, a.run)
    elif a.action == "train_all":
        train_all(a.only)
    elif a.action == "evaluate":
        evaluate(a.role)
    elif a.action == "eval_worker":
        eval_worker(a.role, a.sequence)
    else:
        globals()[a.action]()
