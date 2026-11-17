#!/usr/bin/env python3
"""현재 NSD의 기하 편향 b_ij를 추론에서만 0으로 두는 개입(재학습 없음).

두 DiffAttnConv 층의 bias_mlp 출력을 0으로 바꾼다. combine_scores(a, b)
= a + b + (a - mean a) * tanh(b - mean b)이므로 b=0이면 점수는 정확히 a다.
나머지(가중치, 게이트, 전달, 후보 합집합, 융합 가중치, 정렬, 시간 판정)는
기존 파이프라인(_pdv2_contrast_pipeline[_test].worker) 그대로다.

사례: diff·convs_only의 세 시작점(원고 체크포인트)과 그 기하 0 변형.
시험에는 독립 반복 convs_only 15개(시작점3×실행1..5)와 그 기하 0 변형을 더한다.
무개입 사례는 기존 원장과 승자·점수를 대조한다. 시험셋은 진단용이며 선택에 쓰지 않는다.

사용법(컨테이너 ring_baseline, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python3 scripts/_pdv2_geometry_bias_ablation_20261116.py run
  python3 scripts/_pdv2_geometry_bias_ablation_20261116.py analyze
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

OUT = ROOT / "results/pdv2_geometry_bias_ablation_20261116"
OLD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
INDEP = Path("/workspace/data/nsd_results/pdv2_diffattn_independent_20261108")
METRICS = ("AP", "max_F1", "R@100P", "R@1")
BOOTSTRAP = 10000
BOOT_SEED = 20261104  # 원고 짝지은 구간(analysis_plan.json)과 같은 난수 시드
STARTS = (1, 2, 3)
RUNS = (1, 2, 3, 4, 5)
SUFFIX = "_nogeo"


def READ(path):
    return json.loads(Path(path).read_text())


def WRITE(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def HASH(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 개입
class ZeroBias:
    """bias_mlp 대신 쓰는 0 출력. 모양과 dtype은 원래 출력과 같다."""

    def __init__(self, mlp):
        self.heads = [m for m in mlp.modules() if hasattr(m, "out_features")][-1].out_features

    def __call__(self, attr):
        import torch

        return torch.zeros(attr.shape[0], self.heads, dtype=attr.dtype, device=attr.device)


def ablate(net):
    """두 층의 기하 편향 출력을 0으로 둔다(가중치는 바꾸지 않음)."""
    from gnn.learned_spectral_difference import ControlledRelationConv

    assert len(net.convs) == 2
    for conv in net.convs:
        assert type(conv) is ControlledRelationConv, type(conv)
        assert conv.geometry_mode == "interaction" and conv.relation_mode == "diff"
        assert conv.bias_mlp is not None
        # nn.Module 속성 등록을 피해 인스턴스 사전에 직접 둔다(state_dict 불변).
        object.__setattr__(conv, "_orig_bias_mlp", conv.bias_mlp)
        del conv._modules["bias_mlp"]
        object.__setattr__(conv, "bias_mlp", ZeroBias(conv._orig_bias_mlp))
    return net


def patch(pipeline):
    """kind가 *_nogeo이면 원래 kind로 불러온 뒤 개입한다."""
    original = pipeline.load_model

    def load_model(case):
        if case["kind"].endswith(SUFFIX):
            net = original(dict(case, kind=case["kind"][: -len(SUFFIX)]))
            return ablate(net)
        return original(case)

    pipeline.load_model = load_model


# ---------------------------------------------------------------- 사례
def checkpoints():
    test = READ(OLD_TEST / "protocol.json")["cases"]
    base = {}
    for kind in ("diff", "convs_only"):
        for k in STARTS:
            c = test[f"{kind}_s{k}"]
            base[f"{kind}_s{k}"] = dict(kind=kind, seed=k, checkpoint=c["checkpoint"],
                                        checkpoint_sha256=c["checkpoint_sha256"])
    indep = READ(INDEP / "test_eval/protocol.json")["cases"]
    runs = {}
    for k in STARTS:
        for r in RUNS:
            c = indep[f"convs_only_s{k}_r{r}"]
            runs[f"convs_only_s{k}_r{r}"] = dict(kind="convs_only", seed=k, checkpoint=c["checkpoint"],
                                                 checkpoint_sha256=c["checkpoint_sha256"])
    return base, runs


def with_ablation(cases):
    out = {}
    for tag, c in cases.items():
        out[tag] = c
        stem, _, tail = tag.partition("_s")
        out[f"{stem}{SUFFIX}_s{tail}"] = dict(c, kind=c["kind"] + SUFFIX)
    return out


def prepare():
    path = OUT / "protocol.json"
    if path.exists():
        return READ(path)
    sources = [
        Path(__file__),
        ROOT / "scripts/_pdv2_contrast_pipeline.py",
        ROOT / "scripts/_pdv2_contrast_pipeline_test.py",
        ROOT / "src/gnn/geometry_interaction.py",
        ROOT / "src/gnn/learned_spectral_difference.py",
        ROOT / "src/gnn/contrast_spectral_difference.py",
        OLD_VAL / "protocol.json",
        OLD_VAL / "summary.json",
        OLD_TEST / "protocol.json",
        OLD_TEST / "summary.json",
        INDEP / "test_eval/protocol.json",
    ]
    base, runs = checkpoints()
    for c in (*base.values(), *runs.values()):
        p = Path(c["checkpoint"])
        p = p if p.is_absolute() else ROOT / p
        assert HASH(p) == c["checkpoint_sha256"], p
    spec = dict(
        created_unix=time.time(),
        intervention="두 DiffAttnConv 층의 bias_mlp 출력을 0으로 바꿈(추론만). b=0이면 combine_scores 결과가 내용 점수 a와 같다. 다른 모든 가중치·후보·융합·정렬·판정 동일. h1이 바뀌므로 둘째 층의 a와 둘째 게이트 세기는 자연 전파로 바뀐다.",
        val_cases=with_ablation(base),
        test_cases=with_ablation({**base, **runs}),
        report="검증9: 원래 선택과 같은 계열 매크로(시작점별) 후 세 시작점 평균. 시험46: 공통 질의 195,726, 양성 45시퀀스→도시6 균등→세 시작점 평균. 짝지은 95% 구간은 원고와 같이 시퀀스별 세 시작점 차이 평균을 도시 안 시퀀스 수 유지 10,000회 복원추출(난수 20261104).",
        exposure="시험셋은 진단용이다. 결과로 설정을 고르지 않는다.",
        sources={str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p): HASH(p) for p in sources},
    )
    WRITE(path, spec)
    return spec


def spec_for(role):
    path = OUT / f"{role}_eval" / "protocol.json"
    if path.exists():
        return READ(path)
    top = prepare()
    parent = OLD_VAL if role == "val" else OLD_TEST
    spec = dict(READ(parent / "protocol.json"))
    spec.update(created_unix=time.time(), cases=top[f"{role}_cases"],
                parent_protocol_sha256=HASH(OUT / "protocol.json"),
                scope="기하 편향 0 추론 개입. 설정 선택 없음.")
    WRITE(path, spec)
    return spec


def link_pool(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        target = dst / ("candidate_pool" + suffix)
        if not target.exists():
            target.symlink_to(os.path.relpath(src / ("candidate_pool" + suffix), dst))


def reference(role, name, tag):
    """무개입 사례의 기존 원장 경로."""
    if role == "val":
        return OLD_VAL / "val" / name / f"{tag}.npz"
    if "_r" in tag:
        return INDEP / "test_eval/test" / name / f"{tag}.npz"
    return OLD_TEST / "test" / name / f"{tag}.npz"


def worker(role, sequence):
    name = sequence.replace("/", "_")
    folder = OUT / f"{role}_eval" / role / name
    if role == "val":
        import _pdv2_contrast_pipeline as P

        P.e.configure()
        P.OUT = OUT / "val_eval"
        P.run.OUT = P.OUT
        P.prepare = lambda: spec_for("val")
        patch(P)
        link_pool(OLD_VAL / "val" / name, folder)
        P.worker("val", sequence)
    else:
        import _pdv2_contrast_pipeline_test as T

        T.OUT = OUT / "test_eval"
        T.prepare = lambda: spec_for("test")
        T.pipeline.e.configure()
        patch(T.pipeline)
        link_pool(OLD_TEST / "test" / name, folder)
        T.worker("test", sequence)
    spec = spec_for(role)
    checks = {}
    for tag in spec["cases"]:
        if SUFFIX in tag:
            continue
        new = np.load(folder / f"{tag}.npz")
        ref = np.load(reference(role, name, tag))
        for field in ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist"):
            np.testing.assert_array_equal(new[field], ref[field])
        error = float(np.abs(new["score"] - ref["score"]).max())
        assert error < 8e-6, (tag, error)
        checks[tag] = dict(winners_exact=True, score_max_abs_error=error,
                           score_bitwise_equal=bool(np.array_equal(new["score"], ref["score"])))
    WRITE(folder / "noop_check.json", dict(status="passed", checks=checks))


# ---------------------------------------------------------------- 주의 변화 통계
def stats_worker(role, sequence):
    """기하 점수 b의 이웃 내 변동과 b=0 개입에 따른 주의 변화."""
    import torch
    from torch_geometric.utils import softmax
    import _pdv2_contrast_pipeline as P
    from gnn.geometry_interaction import combine_scores, incoming_mean

    P.e.configure()
    if role == "test":
        import _pdv2_contrast_pipeline_test as T

        T.OUT = OUT / "test_eval"
        T.prepare = lambda: spec_for("test")
        T.activate()
    name = sequence.replace("/", "_")
    target = OUT / "attention_stats" / role / f"{name}.json"
    if target.exists():
        return
    graph, _, _ = P.e.prior.graph_inputs(role, sequence)
    src, tgt = graph.edge_index
    n = len(graph.x)
    cases = {t: c for t, c in spec_for("val")["cases"].items() if SUFFIX not in t}

    def nstd(v):  # 이웃 내 표준편차(모집단), 중심별 [N, H]
        mean = incoming_mean(v, tgt, n)
        sq = torch.zeros(n, v.shape[1], device=v.device, dtype=v.dtype)
        sq.index_add_(0, tgt, (v - mean) ** 2)
        cnt = torch.bincount(tgt, minlength=n).clamp_min(1)[:, None]
        return (sq / cnt).sqrt()

    def tv(p, q):  # 중심별 총변동거리 [N, H]
        out = torch.zeros(n, p.shape[1], device=p.device, dtype=p.dtype)
        out.index_add_(0, tgt, (p - q).abs())
        return 0.5 * out

    def summary(x):
        x = x.flatten().double().cpu().numpy()
        return dict(mean=float(x.mean()), p50=float(np.median(x)), p95=float(np.percentile(x, 95)),
                    max=float(x.max()), count=int(x.size))

    result = {}
    with torch.no_grad():
        for tag, case in cases.items():
            net = P.load_model(case)
            captured = []
            hooks = [conv.register_forward_pre_hook(
                lambda m, args, kw: captured.append((args[0], kw.get("edge_attr"))), with_kwargs=True)
                for conv in net.convs]
            z0 = net(graph)
            att0 = [conv.attention.clone() for conv in net.convs]
            eta0 = net.last_strengths.clone()
            for h in hooks:
                h.remove()
            row = {}
            for layer, (conv, (x, attr)) in enumerate(zip(net.convs, captured)):
                q = conv.W_q(x[tgt]).view(-1, conv.heads, conv.head_dim)
                k = conv.W_k(x[src] - x[tgt]).tanh().view(-1, conv.heads, conv.head_dim)
                a = (q * k).sum(-1) / conv.scale
                b = conv.bias_mlp(attr)
                full = combine_scores(a, b, tgt, n, conv.geometry_mode)
                p_full = softmax(full, tgt, num_nodes=n)
                p_a = softmax(a, tgt, num_nodes=n)
                recompute_err = float((p_full - att0[layer]).abs().max())
                assert recompute_err < 1e-5, recompute_err
                row[f"layer{layer + 1}"] = dict(
                    recompute_attention_max_err=recompute_err,
                    std_b_within_neighborhood=summary(nstd(b)),
                    std_a_within_neighborhood=summary(nstd(a)),
                    std_logit_within_neighborhood=summary(nstd(full)),
                    mean_abs_b=float(b.abs().mean()),
                    tv_direct=summary(tv(p_full, p_a)),
                )
                if layer == 0:
                    p_a_first = p_a
            ablate(net)
            z1 = net(graph)
            # 첫 층은 입력이 같으므로 개입 후 주의 = softmax(a)여야 한다(b=0이면 점수 = a).
            first_err = float((net.convs[0].attention - p_a_first).abs().max())
            assert first_err < 1e-6, first_err
            row["layer1_ablated_equals_softmax_a_max_err"] = first_err
            for layer, conv in enumerate(net.convs):
                row[f"layer{layer + 1}"]["tv_total"] = summary(tv(att0[layer], conv.attention))
            eta1 = net.last_strengths
            refine = (z0 - graph.spectrum).abs().pow(2).sum().sqrt()
            row["eta_abs_change"] = [summary((eta1[i] - eta0[i]).abs()) for i in range(2)]
            row["eta_original_mean"] = [float(eta0[i].mean()) for i in range(2)]
            row["refined_spectrum_rel_change"] = float(
                (z1 - z0).abs().pow(2).sum().sqrt() / refine)
            result[tag] = row
            del net
    WRITE(target, result)
    print("STATS", role, sequence, flush=True)


# ---------------------------------------------------------------- 실행
def run_all():
    prepare()
    val = spec_for("val")["val"]
    test = spec_for("test")["test"]
    jobs = ([("worker", "val", s) for s in val] + [("stats_worker", "val", s) for s in val]
            + [("worker", "test", s) for s in test] + [("stats_worker", "test", s) for s in test])
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(job):
        action, role, sequence = job
        log = logs / f"{action}_{role}_{sequence.replace('/', '_')}.log"
        with log.open("a") as stream:
            subprocess.run([sys.executable, __file__, action, "--role", role, "--sequence", sequence],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, j) for j in jobs]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            WRITE(OUT / "status.json", dict(done=i, total=len(futures)))
            if i % 10 == 0 or i == len(futures):
                print("DONE", i, len(futures), flush=True)
    analyze()


def analyze():
    import _pdv2_contrast_pipeline as P
    import _pdv2_pilot as pilot

    macro_fam = P.run.base.macro
    # ---- 검증9
    vspec = spec_for("val")
    vrows = {s: READ(OUT / "val_eval/val" / s.replace("/", "_") / "summary.json")["metrics"]
             for s in vspec["val"]}
    for s in vspec["val"]:
        assert READ(OUT / "val_eval/val" / s.replace("/", "_") / "noop_check.json")["status"] == "passed"
    vper = {t: macro_fam({s: r[t] for s, r in vrows.items()}) for t in vspec["cases"]}
    vold = READ(OLD_VAL / "summary.json")
    val = {}
    for kind in ("diff", "convs_only"):
        for arm in (kind, kind + SUFFIX):
            tags = [f"{arm}_s{k}" for k in STARTS]
            val[arm] = dict(
                mean={m: 100 * float(np.mean([vper[t]["macro"][m] for t in tags])) for m in METRICS},
                per_start={t: {m: 100 * vper[t]["macro"][m] for m in METRICS} for t in tags},
                families_mean={f: {m: 100 * float(np.mean([vper[t]["families"][f][m] for t in tags]))
                                   for m in METRICS} for f in vper[tags[0]]["families"]},
            )
        for m in METRICS:  # 무개입 = 기존 보고값
            assert abs(val[kind]["mean"][m] / 100 - vold["mean"][kind][m]) < 1e-12, (kind, m)
        val[f"{kind}{SUFFIX}-{kind}"] = dict(
            mean_pp={m: val[kind + SUFFIX]["mean"][m] - val[kind]["mean"][m] for m in METRICS},
            per_start_pp={str(k): {m: 100 * (vper[f"{kind}{SUFFIX}_s{k}"]["macro"][m]
                                             - vper[f"{kind}_s{k}"]["macro"][m]) for m in METRICS}
                          for k in STARTS},
        )

    # ---- 시험46
    tspec = spec_for("test")
    per = {}
    count = 0
    for s in tspec["test"]:
        folder = OUT / "test_eval/test" / s.replace("/", "_")
        assert READ(folder / "noop_check.json")["status"] == "passed"
        ref = READ(folder / "reference_validation.json")
        assert ref["status"] == "passed"
        count += ref["queries"]
        per[s] = READ(folder / "summary.json")["metrics"]
    assert count == 195726
    cities = {c: [s for s in seqs if per[s]["diff_s1"]["n_pos"]] for c, seqs in pilot.CITY.items()}
    assert sum(len(v) for v in cities.values()) == 45
    order = [s for seqs in cities.values() for s in seqs]
    index = [np.array([order.index(s) for s in seqs]) for seqs in cities.values()]

    def vec(tag, m):
        return np.array([per[s][tag][m] for s in order], float)

    def macro(v, picks=None):
        if picks is None:
            return float(np.mean([v[ix].mean() for ix in index]))
        return float(np.mean([v[ix[p]].mean() for ix, p in zip(index, picks)]))

    old = READ(OLD_TEST / "summary.json")
    rng = np.random.default_rng(BOOT_SEED)
    picks = [[rng.integers(0, len(ix), len(ix)) for ix in index] for _ in range(BOOTSTRAP)]

    def compare(tags_a, tags_b):
        out = {}
        for m in METRICS:
            d = np.mean([vec(a, m) - vec(b, m) for a, b in zip(tags_a, tags_b)], axis=0)
            boot = np.array([macro(d, p) for p in picks])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[m] = dict(mean_pp=100 * macro(d), ci95_pp=[100 * lo, 100 * hi],
                          per_model_pp=[100 * (macro(vec(a, m)) - macro(vec(b, m)))
                                        for a, b in zip(tags_a, tags_b)])
        return out

    test = {}
    for kind in ("diff", "convs_only"):
        for arm in (kind, kind + SUFFIX):
            tags = [f"{arm}_s{k}" for k in STARTS]
            test[arm] = dict(
                mean={m: 100 * float(np.mean([macro(vec(t, m)) for t in tags])) for m in METRICS},
                per_start={t: {m: 100 * macro(vec(t, m)) for m in METRICS} for t in tags},
                city_mean={c: {m: 100 * float(np.mean([vec(t, m)[ix].mean() for t in tags])) for m in METRICS}
                           for c, ix in zip(cities, index)},
            )
        for m in METRICS:
            assert abs(test[kind]["mean"][m] / 100 - old["mean"][kind][m]) < 1e-9, (kind, m)
        test[f"{kind}{SUFFIX}-{kind}"] = compare([f"{kind}{SUFFIX}_s{k}" for k in STARTS],
                                                  [f"{kind}_s{k}" for k in STARTS])
    # 독립 반복 convs_only 15개
    run_tags = [f"convs_only_s{k}_r{r}" for k in STARTS for r in RUNS]
    abl_tags = [f"convs_only{SUFFIX}_s{k}_r{r}" for k in STARTS for r in RUNS]
    test["convs_only_runs15"] = dict(
        mean={m: 100 * float(np.mean([macro(vec(t, m)) for t in run_tags])) for m in METRICS})
    test[f"convs_only_runs15{SUFFIX}"] = dict(
        mean={m: 100 * float(np.mean([macro(vec(t, m)) for t in abl_tags])) for m in METRICS})
    runs_cmp = compare(abl_tags, run_tags)
    for m in METRICS:
        runs_cmp[m]["negative_runs"] = int(sum(v < 0 for v in runs_cmp[m]["per_model_pp"]))
    test[f"convs_only_runs15{SUFFIX}-convs_only_runs15"] = runs_cmp

    # ---- 주의 통계
    attention = {}
    for role, seqs in (("val", vspec["val"]), ("test", tspec["test"])):
        rows = {s: READ(OUT / "attention_stats" / role / (s.replace("/", "_") + ".json")) for s in seqs}
        attention[role] = {}
        for tag in rows[seqs[0]]:
            agg = {}
            for layer in ("layer1", "layer2"):
                def pooled(key, stat="mean"):
                    w = np.array([rows[s][tag][layer][key]["count"] for s in seqs], float)
                    v = np.array([rows[s][tag][layer][key][stat] for s in seqs], float)
                    return float((w * v).sum() / w.sum())
                agg[layer] = dict(
                    tv_total_mean=pooled("tv_total"), tv_total_p95_seqavg=float(np.mean(
                        [rows[s][tag][layer]["tv_total"]["p95"] for s in seqs])),
                    tv_total_max=float(max(rows[s][tag][layer]["tv_total"]["max"] for s in seqs)),
                    tv_direct_mean=pooled("tv_direct"),
                    std_b_mean=pooled("std_b_within_neighborhood"),
                    std_a_mean=pooled("std_a_within_neighborhood"),
                    std_logit_mean=pooled("std_logit_within_neighborhood"),
                    recompute_attention_max_err=float(max(rows[s][tag][layer]["recompute_attention_max_err"]
                                                          for s in seqs)),
                )
            agg["eta2_abs_change_mean"] = float(np.mean([rows[s][tag]["eta_abs_change"][1]["mean"] for s in seqs]))
            agg["eta2_original_mean"] = float(np.mean([rows[s][tag]["eta_original_mean"][1] for s in seqs]))
            agg["refined_spectrum_rel_change_seqavg"] = float(
                np.mean([rows[s][tag]["refined_spectrum_rel_change"] for s in seqs]))
            attention[role][tag] = agg

    report = dict(val_percent=val, test_percent=test, attention=attention,
                  bootstrap=dict(replicates=BOOTSTRAP, seed=BOOT_SEED,
                                 scheme="시퀀스별 모델 차이 평균 → 도시 안 시퀀스 수 유지 짝지은 복원추출, 도시6 균등"),
                  test_used_for_selection=False)
    WRITE(OUT / "result.json", report)
    brief = {k: v.get("mean", v.get("mean_pp")) for k, v in val.items()}
    print(json.dumps(dict(val=brief), ensure_ascii=False, indent=1))
    for k, v in test.items():
        if "-" in k:
            print(k, {m: (round(v[m]["mean_pp"], 3), [round(x, 3) for x in v[m]["ci95_pp"]]) for m in METRICS})
        else:
            print(k, {m: round(x, 3) for m, x in v["mean"].items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "run", "worker", "stats_worker", "analyze"))
    p.add_argument("--role", choices=("val", "test"))
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "worker":
        worker(a.role, a.sequence)
    elif a.action == "stats_worker":
        stats_worker(a.role, a.sequence)
    elif a.action == "run":
        run_all()
    else:
        globals()[a.action]()
