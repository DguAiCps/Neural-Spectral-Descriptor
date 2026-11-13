#!/usr/bin/env python3
"""사전학습부터 과거 전용인 무지연(지연 0) NSD: 시간 전용 사전학습의 시간 간선도 과거 5키프레임만 남긴다.

기존 무지연 모델 Z1(`_pdv2_zero_lag.py`)은 대칭 창으로 사전학습한 시작점에서 과거 정제 + 과거 시간 증거로 추가 학습만
했다. 여기서는 사전학습(`_pdv2_spectral_keys.train`이 부르는 `_pdv2_temporal_derivatives.train`, 20에포크·900갱신,
InfoNCE)의 학습 그래프에서 시간 간선을 src<tgt(과거 이웃 -5..-1)만 남기고, 나머지 사전학습 설정은 그대로 둔다. 그
시작점 세 개에서 Z1과 같은 추가 학습(512갱신)을 하고 시험46에서 무지연 조건으로 평가한다.

- 사전학습: 기존 경로를 그대로 부르고 출력 폴더(root)와, 과거 창이면 `temporal_subgraph` 결과에 과거 간선 필터만 더한다.
  표본(같은 183,249 삼중쌍·순서), 손실, 학습률 스케줄, AMP, 시드, 체크포인트(20에포크)는 같다. 대칭 창도 같은 경로로
  다시 돌려 기존 체크포인트와 비교한다.
- 추가 학습: `_pdv2_window_side_learned.train('past', k, 'diff')`에 `_pdv2_zero_lag`의 무지연 plan을 넣는 Z1 절차를
  그대로 쓰고, 시작점 경로(W.PRETRAIN)만 새 사전학습으로 바꾼다. 같은 경로에 기존 시작점을 넣으면 Z1 체크포인트와
  비트 단위로 같아야 한다.
- 시험: `_pdv2_zero_lag`의 무지연 평가 연결(install_eval, 접두 절단 검사)을 그대로 쓴다. 시험셋으로 고르는 설정은 없다.

산출물은 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python scripts/_pdv2_zero_lag_pretrain.py protocol                         # 실행 전 protocol.json 저장
  python scripts/_pdv2_zero_lag_pretrain.py pretrain --window symmetric --seed 1   # 사전학습 1회(새 프로세스)
  python scripts/_pdv2_zero_lag_pretrain.py pretrain_all                     # 대칭 3 + 과거 3, 대칭 재현 비교
  python scripts/_pdv2_zero_lag_pretrain.py finetune                         # P1 3개 + Z1 재학습 재현(+조건부 대조)
  python scripts/_pdv2_zero_lag_pretrain.py test                             # 시험46 평가·검사·분석
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import _pdv2_zero_lag as Z  # noqa: E402

W = Z.W
READ, WRITE, HASH = W.READ, W.WRITE, W.HASH
OUT = Path(os.environ.get("NSD_ZERO_LAG_PRETRAIN_OUT",
                          "/workspace/data/nsd_results/pdv2_zero_lag_pretrain_20261112"))
PRE = OUT / "pretrain"      # PRE/<tag>/s{k}/octave288/{checkpoints,training_progress.json,...}
FINE = OUT / "finetune"     # FINE/<name>/runs/<window>/s{k}/diff/epoch8.pth
TEST = OUT / "test_eval"
LOGS = OUT / "logs"
ORIGINAL_PRETRAIN = W.PRETRAIN  # results/pdv2_spectral_keys_20261025 (대칭 창 사전학습)
ARM = "octave288"
STARTS, METRICS, GROUPS = W.STARTS, W.METRICS, W.GROUPS
BOOTSTRAP, BOOT_SEED = W.BOOTSTRAP, W.BOOT_SEED

# 추가 학습 작업: 이름 -> (사전학습 폴더, 정제 창, 시간 증거). z1_repro는 기존 시작점으로 Z1 절차 재학습(비트 대조).
FINETUNES = {"p1": (PRE / "past", "past", "past"), "z1_repro": (ORIGINAL_PRETRAIN, "past", "past")}
CONDITIONAL_FINETUNES = {"z1r": (PRE / "symmetric", "past", "past"),
                         "nsdr": (PRE / "symmetric", "symmetric", "both")}


def pretrain_root(tag: str) -> Path:
    return PRE / tag


def pretrain_checkpoint(root: Path, k: int, epoch: int = 20) -> Path:
    return root / f"s{k}" / ARM / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"


def finetune_checkpoint(name: str, k: int) -> Path:
    window = FINETUNES.get(name, CONDITIONAL_FINETUNES.get(name))[1]
    return FINE / name / "runs" / window / f"s{k}" / "diff" / "epoch8.pth"


# ---------------------------------------------------------------- 사전학습 그래프
def past_pretrain_graph(graph):
    """사전학습 그래프에서 시간 간선을 과거 이웃(src<tgt)만 남긴다. `W.past_graph`의 간선 선택과 같다."""
    g = graph.clone()
    keep = g.edge_index[0] < g.edge_index[1]
    g.edge_index, g.edge_attr, g.edge_type = g.edge_index[:, keep], g.edge_attr[keep], g.edge_type[keep]
    offset = g.edge_index[0] - g.edge_index[1]
    assert int(offset.min()) >= -5 and int(offset.max()) <= -1  # 메시지는 과거 1..5키프레임에서만 온다
    return g


def graph_for(window: str):
    """사전학습과 같은 방식으로 학습 그래프를 읽는다(base.train의 GRAPH -> temporal_subgraph)."""
    import torch
    import _pdv2_spectral_keys as SK

    original = torch.load(SK.OUT / f"graph_{ARM}.pt", map_location="cpu", weights_only=False)
    graph = SK.base.temporal_subgraph(original, np.arange(original.num_nodes))
    return past_pretrain_graph(graph) if window == "past" else graph


def pretrain_causal_check(net, graph, sequences: np.ndarray) -> dict:
    """첫 학습 시퀀스 가운데 노드 출력의 입력 기울기가 닿는 범위. 과거 창이면 미래 노드 기울기가 정확히 0이다."""
    import torch
    import _pdv2_spectral_keys as SK

    ids = np.flatnonzero(sequences == 0)
    g = SK.base.temporal_subgraph(graph, ids).cuda()
    g.x.requires_grad_(True)
    i = len(ids) // 2
    (gx,) = torch.autograd.grad(net(g)[i].float().abs().sum(), [g.x])
    size = gx.abs().sum(1).cpu().numpy()
    hit = np.flatnonzero(size > 0)
    return dict(node=i, future_max=float(size[i + 1:].max()), future_nodes=int((size[i + 1:] > 0).sum()),
                past_nodes=int((size[:i] > 0).sum()), reach=[int(hit.min() - i), int(hit.max() - i)])


# ---------------------------------------------------------------- 사전학습
def pretrain_worker(window: str, seed: int, tag: str) -> None:
    """`_pdv2_spectral_keys.main`의 train과 같은 프로세스 설정으로 기존 학습 함수를 부른다. 바꾸는 것은 출력 폴더와 창뿐이다."""
    import logging
    import torch
    import _pdv2_spectral_keys as SK

    assert window in ("symmetric", "past")
    torch.set_num_threads(4)  # SK.main의 train 설정과 같다
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    base = SK.base
    spec = SK.protocol()  # 기존 사전학습 runtime 소스 해시 검사
    mine = READ(OUT / "protocol.json")
    root = pretrain_root(tag)
    folder = root / f"s{seed}" / ARM
    if (folder / "pretrain_audit.json").exists():  # 완료된 실행은 다시 검사하지 않는다(기록 보존)
        return
    base.PREPARED, base.GRAPH = SK.OUT / f"prepared_{ARM}.pt", SK.OUT / f"graph_{ARM}.pt"
    base.root = lambda s, a: root / f"s{s}" / a
    base.EPOCHS = SK.EPOCHS
    base.model = lambda selected: SK.model(selected, seed)
    hashes = spec["graph_hashes"][ARM] if window == "symmetric" else mine["pretraining"]["past_graph_hashes"]
    base.protocol = lambda *args: {**SK.protocol(), "graph_hashes": hashes}
    subgraph = base.temporal_subgraph
    if window == "past":
        base.temporal_subgraph = lambda graph, ids: past_pretrain_graph(subgraph(graph, ids))
    deterministic = torch.are_deterministic_algorithms_enabled()
    tick = time.time()
    SK.ORIGINAL_TRAIN(seed, ARM)
    seconds = time.time() - tick
    base.temporal_subgraph = subgraph
    assert READ(folder / "initial_model.json") == spec["initial"][str(seed)][ARM]
    state = torch.load(base.PREPARED, map_location="cpu", weights_only=False)
    graph = graph_for(window)
    for k, h in hashes.items():
        assert SK.array_hash(getattr(graph, k).numpy()) == h
    net = SK.model(ARM, seed).cuda().eval()
    net.load_state_dict(torch.load(pretrain_checkpoint(root, seed), map_location="cpu",
                                   weights_only=False)["model_state_dict"], strict=True)
    causal = pretrain_causal_check(net, graph, state["sequences"])
    if window == "past":
        assert causal["future_max"] == 0 and causal["future_nodes"] == 0 and causal["past_nodes"] > 0, causal
    else:
        assert causal["future_nodes"] > 0 and causal["past_nodes"] > 0, causal
    WRITE(folder / "pretrain_audit.json", dict(
        window=window, seed=seed, tag=tag, deterministic_algorithms=deterministic, wall_seconds=seconds,
        edges=int(graph.edge_index.shape[1]), causal_check_epoch20=causal,
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()))
    print("PRETRAINED", window, seed, tag, round(seconds, 1), causal, flush=True)


def call_pretrain(window: str, seed: int, tag: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"pretrain_{tag}_s{seed}.log").open("a") as stream:
        subprocess.run([sys.executable, "-u", __file__, "pretrain", "--window", window, "--seed", str(seed),
                        "--tag", tag], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def compare_states(a_path: Path, b_path: Path) -> dict:
    import torch

    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert sa.keys() == sb.keys()
    floats = [n for n in sa if sa[n].is_floating_point()]
    diff = {n: float((sa[n].double() - sb[n].double()).abs().max()) for n in floats}
    rel = {n: diff[n] / max(float(sb[n].double().abs().max()), 1e-30) for n in floats}
    worst = max(diff, key=diff.get)
    return dict(file_sha256_equal=HASH(a_path) == HASH(b_path),
                weights_exact=all(torch.equal(sa[n], sb[n]) for n in sa),
                tensors_different=sum(not torch.equal(sa[n], sb[n]) for n in sa), tensors=len(sa),
                max_abs=max(diff.values()), max_rel=max(rel.values()), worst_tensor=worst,
                train_losses_equal=a.get("train_losses") == b.get("train_losses"))


def epoch_losses(folder: Path) -> list:
    return [r["loss"] for r in READ(folder / "training_progress.json")["records"]]


def pretrain_all() -> None:
    """대칭 3개를 먼저 돌려 기존 체크포인트와 비교하고, 과거 3개를 돌린다. 대칭이 다르면 같은 조건 반복 1회로 실행 간 차이를 잰다."""
    assert (OUT / "protocol.json").exists()
    timing = {}
    for window in ("symmetric", "past"):
        for k in STARTS:
            tick = time.time()
            call_pretrain(window, k, window)
            timing[f"{window}_s{k}"] = time.time() - tick
            print("PRETRAIN DONE", window, k, round(timing[f"{window}_s{k}"], 1), flush=True)
    rows = {}
    for k in STARTS:
        old_folder = ORIGINAL_PRETRAIN / f"s{k}" / ARM
        new_folder = PRE / "symmetric" / f"s{k}" / ARM
        rows[f"s{k}"] = {f"epoch{e}": compare_states(pretrain_checkpoint(PRE / "symmetric", k, e),
                                                     pretrain_checkpoint(ORIGINAL_PRETRAIN, k, e)) for e in (10, 20)}
        old_l, new_l = epoch_losses(old_folder), epoch_losses(new_folder)
        rows[f"s{k}"]["epoch_loss_max_abs"] = float(np.max(np.abs(np.subtract(old_l, new_l))))
        old_o = READ(old_folder / "training_complete.json")["orders"]
        new_o = READ(new_folder / "training_complete.json")["orders"]
        rows[f"s{k}"]["triplet_orders_equal"] = old_o == new_o
    exact = all(r["epoch20"]["weights_exact"] for r in rows.values())
    repeat = None
    if not exact:
        tick = time.time()
        call_pretrain("symmetric", 1, "symmetric_repeat")
        timing["symmetric_repeat_s1"] = time.time() - tick
        repeat = {f"epoch{e}": compare_states(pretrain_checkpoint(PRE / "symmetric_repeat", 1, e),
                                              pretrain_checkpoint(PRE / "symmetric", 1, e)) for e in (10, 20)}
        repeat["epoch_loss_max_abs"] = float(np.max(np.abs(np.subtract(
            epoch_losses(PRE / "symmetric_repeat/s1" / ARM), epoch_losses(PRE / "symmetric/s1" / ARM)))))
    past = {}
    for k in STARTS:
        sym_l = epoch_losses(PRE / "symmetric" / f"s{k}" / ARM)
        past_l = epoch_losses(PRE / "past" / f"s{k}" / ARM)
        past[f"s{k}"] = dict(symmetric_epoch_loss=sym_l, past_epoch_loss=past_l,
                             original_symmetric_epoch_loss=epoch_losses(ORIGINAL_PRETRAIN / f"s{k}" / ARM),
                             finite=bool(np.isfinite(past_l).all()),
                             audit=READ(PRE / "past" / f"s{k}" / ARM / "pretrain_audit.json"),
                             symmetric_audit=READ(PRE / "symmetric" / f"s{k}" / ARM / "pretrain_audit.json"))
    report = dict(
        symmetric_reproduction=dict(status="bit_exact" if exact else "not_bit_exact", per_seed=rows,
                                    repeat_s1_vs_rerun_s1=repeat),
        conditional_arms_required=not exact,
        rule=READ(OUT / "protocol.json")["pretraining"]["reproduction_rule"],
        pretraining_logs=past, wall_seconds=timing)
    WRITE(OUT / "pretrain_report.json", report)
    print("PRETRAIN_REPORT", json.dumps(report["symmetric_reproduction"], ensure_ascii=False)[:3000], flush=True)


# ---------------------------------------------------------------- 추가 학습
def finetune_jobs() -> dict:
    jobs = dict(FINETUNES)
    if READ(OUT / "pretrain_report.json")["conditional_arms_required"]:
        jobs.update(CONDITIONAL_FINETUNES)
    return jobs


def finetune() -> None:
    """Z1 절차(W.train + 무지연 학습 plan)를 시작점만 바꿔 부른다. z1_repro는 기존 Z1 체크포인트와 비트 단위로 같아야 한다."""
    import torch

    report = READ(OUT / "pretrain_report.json")
    for k in STARTS:
        assert pretrain_checkpoint(PRE / "past", k).exists()
    audit = Z.install_train_plans()
    WRITE(FINE / "train_plans.json", audit)
    jobs = finetune_jobs()
    tick = time.time()
    for name, (pretrain, window, evidence) in jobs.items():
        Z.EVIDENCE["mode"] = evidence
        W.PRETRAIN, W.OUT = pretrain, FINE / name  # W.build가 읽는 시작점, W.run_folder가 쓰는 출력 위치
        for k in STARTS:
            W.train(window, k, "diff")
    Z.EVIDENCE["mode"] = None
    W.PRETRAIN, W.OUT = ORIGINAL_PRETRAIN, Z.PAST

    def state(path):
        return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]

    repro = {}
    for k in STARTS:
        old = Z.zero_lag_checkpoint(k)
        assert HASH(old) == READ(old.parent / "training_complete.json")["sha256"]
        a, b = state(finetune_checkpoint("z1_repro", k)), state(old)
        repro[f"zl_past_diff_s{k}"] = dict(exact=all(torch.equal(a[n], b[n]) for n in a),
                                           max_abs=max(float((a[n].double() - b[n].double()).abs().max()) for n in a),
                                           original=str(old))
    training = {}
    for name in jobs:
        for k in STARTS:
            record = READ(finetune_checkpoint(name, k).parent / "training_complete.json")
            losses = np.array([r["loss"] for r in record["records"]]).reshape(8, -1)
            training[f"{name}_s{k}"] = dict(
                finite=bool(np.isfinite(losses).all()), epoch_mean_loss=losses.mean(1).tolist(),
                final_epoch_strength_mean=float(np.mean([r["strength_mean"] for r in record["records"][-losses.shape[1]:]])),
                seconds=record["seconds"], init_error=record["init_error"], causal_check=record["causal_check"],
                start=str(jobs[name][0]), window=jobs[name][1], evidence=jobs[name][2])
    status = "passed" if all(r["exact"] for r in repro.values()) else "failed"
    out = dict(z1_retrain_reproduction=dict(status=status, checkpoints=repro), training=training,
               jobs={k: [str(v[0]), v[1], v[2]] for k, v in jobs.items()}, wall_seconds=time.time() - tick,
               pretrain_symmetric_reproduction=report["symmetric_reproduction"]["status"])
    WRITE(OUT / "finetune_report.json", out)
    print("FINETUNE", json.dumps(dict(status=status, repro=repro), ensure_ascii=False), flush=True)
    assert status == "passed", repro
    assert all(v["finite"] for v in training.values())


# ---------------------------------------------------------------- 시험
def checkpoint_case(kind: str, k: int, path: Path, evidence: str) -> dict:
    digest = HASH(path)
    assert READ(path.parent / "training_complete.json")["sha256"] == digest
    return dict(kind=kind, seed=k, checkpoint=str(path), checkpoint_sha256=digest, evidence=evidence)


def test_spec() -> dict:
    """기존 시험 명세에서 사례만 바꾼다(`Z.test_spec`과 같은 방식). 무지연 사례 이름은 zl_로 시작한다."""
    path = TEST / "protocol.json"
    if path.exists():
        return READ(path)
    old = READ(W.OLD_TEST / "protocol.json")
    jobs = finetune_jobs()
    cases = {}
    for k in STARTS:
        case = dict(old["cases"][f"diff_s{k}"], evidence="both")
        assert HASH(ROOT / case["checkpoint"]) == case["checkpoint_sha256"]
        cases[f"diff_s{k}"] = case
    cases["zl_identity"] = dict(kind="identity", seed=0, checkpoint=None, evidence="past")
    cases["zl_past_fixed"] = dict(kind="past_fixed", seed=0, checkpoint=None, evidence="past")
    for k in STARTS:
        cases[f"zl_past_diff_s{k}"] = checkpoint_case("past_diff", k, Z.zero_lag_checkpoint(k), "past")
    for k in STARTS:
        cases[f"zl_p1_past_diff_s{k}"] = checkpoint_case("past_diff", k, finetune_checkpoint("p1", k), "past")
    if "z1r" in jobs:
        for k in STARTS:
            cases[f"zl_z1r_past_diff_s{k}"] = checkpoint_case("past_diff", k, finetune_checkpoint("z1r", k), "past")
        for k in STARTS:
            cases[f"nsdr_diff_s{k}"] = checkpoint_case("diff", k, finetune_checkpoint("nsdr", k), "both")
    spec = dict(old)
    spec.update(created_unix=time.time(), cases=cases,
                test_policy="고정 설정의 시험 보고. 설정 선택 없음. 조건은 한 번 실행.")
    WRITE(path, spec)
    return spec


def reference_ledger(tag: str, name: str):
    """기존 원장이 있는 참조 사례의 경로. 새 사례는 None."""
    if tag.startswith("diff_s"):
        return W.OLD_TEST / "test" / name / f"{tag}.npz"
    if tag in ("zl_identity", "zl_past_fixed") or tag.startswith("zl_past_diff_s"):
        return Z.TEST / "test" / name / f"{tag}.npz"
    return None


def init_checks(pipeline, graph) -> list:
    """새 사전학습 시작점의 추가 학습 전 모델(W.build 초기화) 출력이 과거 균일 감산 -0.5와 같은지 시험 그래프에서 잰다."""
    import torch

    rows = []
    jobs = finetune_jobs()
    with torch.no_grad():
        fixed = pipeline.transform(None, graph, "past_fixed")
        for name in ("p1", "z1r"):
            if name not in jobs:
                continue
            W.PRETRAIN = jobs[name][0]
            for k in STARTS:
                net = W.build(torch, pipeline.e, k, "diff")
                z = pipeline.transform(net, graph, "past_diff")
                rows.append(dict(start=name, seed=k, max_abs_vs_past_uniform=float((z - fixed).abs().max()),
                                 max_abs_fixed=float(fixed.abs().max())))
                del net, z
    W.PRETRAIN = ORIGINAL_PRETRAIN
    return rows


def test_worker(sequence: str) -> None:
    """한 시퀀스의 사례를 평가하고 참조 원장 재현, 초기화 출력, 접두 절단 검사를 기록한다."""
    import _pdv2_contrast_pipeline_test as T

    T.OUT = TEST
    T.prepare = test_spec
    T.pipeline.e.configure()
    Z.install_eval(T.pipeline)
    Z.test_spec = test_spec  # Z.prefix_checks가 읽는 사례 목록
    Z.NEGATIVE_CONTROLS = ("diff_s1",)  # 앞뒤 증거 대조: 접두 절단에서 점수가 달라져야 한다
    name = sequence.replace("/", "_")
    folder = TEST / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        dst = folder / ("candidate_pool" + suffix)
        if not dst.exists():
            shutil.copy2(W.OLD_TEST / "test" / name / ("candidate_pool" + suffix), dst)
    T.worker("test", sequence)
    target = folder / "zero_lag_pretrain_check.json"
    if target.exists():
        assert READ(target)["status"] == "passed"
        return
    spec = test_spec()
    reproduction = {}
    for tag in spec["cases"]:
        ref = reference_ledger(tag, name)
        if ref is None:
            continue
        new, old = np.load(folder / f"{tag}.npz"), np.load(ref)
        for field in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist"):
            np.testing.assert_array_equal(new[field], old[field])
        error = float(np.abs(new["score"] - old["score"]).max())
        assert error < 8e-6, (tag, error)
        reproduction[tag] = dict(winners_exact=True, score_error=error, reference=str(ref))
    graph, _, _ = T.pipeline.e.prior.graph_inputs("test", sequence)
    init = init_checks(T.pipeline, graph)
    assert all(r["max_abs_vs_past_uniform"] < 4e-6 for r in init), init
    cache = T.inputs("test", sequence)[0]
    n = len(cache["descriptors"])
    q = np.load(folder / "zl_identity.npz")["query_idx"]
    np.testing.assert_array_equal(q, np.arange(30, n - 15))
    prefix = Z.prefix_checks(T.pipeline, sequence, folder)
    WRITE(target, dict(status="passed", reproduction=reproduction, init=init, prefix=prefix,
                       query_range=dict(frames=n, first=int(q[0]), last=int(q[-1]), count=len(q))))
    print("CHECKED", sequence, flush=True)


def test() -> None:
    assert READ(OUT / "finetune_report.json")["z1_retrain_reproduction"]["status"] == "passed"
    spec = test_spec()
    logs = TEST / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def call(sequence):
        with (logs / (sequence.replace("/", "_") + ".log")).open("a") as stream:
            subprocess.run([sys.executable, __file__, "test_worker", "--sequence", sequence],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in spec["test"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            print("TEST_WORKER DONE", i, len(futures), round(time.time() - tick, 1), flush=True)
    WRITE(OUT / "test_timing.json", dict(seconds=time.time() - tick))
    analyze()


# ---------------------------------------------------------------- 분석
def arms_and_contrasts(tags) -> tuple:
    arms = {
        "symmetric_nsd": [f"diff_s{k}" for k in STARTS],
        "P1_past_pretrain_zero_lag": [f"zl_p1_past_diff_s{k}" for k in STARTS],
        "Z1_zero_lag_learned": [f"zl_past_diff_s{k}" for k in STARTS],
        "Z2_zero_lag_no_refine": ["zl_identity"],
        "Z3_zero_lag_past_uniform": ["zl_past_fixed"],
    }
    contrasts = {
        "P1_minus_symmetric": [(f"zl_p1_past_diff_s{k}", f"diff_s{k}") for k in STARTS],
        "P1_minus_Z1": [(f"zl_p1_past_diff_s{k}", f"zl_past_diff_s{k}") for k in STARTS],
        "P1_minus_Z2": [(f"zl_p1_past_diff_s{k}", "zl_identity") for k in STARTS],
        "P1_minus_Z3": [(f"zl_p1_past_diff_s{k}", "zl_past_fixed") for k in STARTS],
        # 기존 무지연 실험의 대조(구간까지 재현 확인)
        "Z1_minus_symmetric": [(f"zl_past_diff_s{k}", f"diff_s{k}") for k in STARTS],
        "Z2_minus_symmetric": [("zl_identity", f"diff_s{k}") for k in STARTS],
        "Z1_minus_Z2": [(f"zl_past_diff_s{k}", "zl_identity") for k in STARTS],
        "Z1_minus_Z3": [(f"zl_past_diff_s{k}", "zl_past_fixed") for k in STARTS],
    }
    if "zl_z1r_past_diff_s1" in tags:
        arms["Z1r_sym_repretrain_zero_lag"] = [f"zl_z1r_past_diff_s{k}" for k in STARTS]
        arms["NSDr_sym_repretrain_symmetric"] = [f"nsdr_diff_s{k}" for k in STARTS]
        contrasts.update({
            "P1_minus_Z1r": [(f"zl_p1_past_diff_s{k}", f"zl_z1r_past_diff_s{k}") for k in STARTS],
            "P1_minus_NSDr": [(f"zl_p1_past_diff_s{k}", f"nsdr_diff_s{k}") for k in STARTS],
            "Z1r_minus_Z1": [(f"zl_z1r_past_diff_s{k}", f"zl_past_diff_s{k}") for k in STARTS],
            "NSDr_minus_symmetric": [(f"nsdr_diff_s{k}", f"diff_s{k}") for k in STARTS],
        })
    return arms, contrasts


def analyze() -> None:
    """방향별 6도시 매크로, 시작점 평균·SD, 도시 안 시퀀스 짝지은 재표집 차이. 집계·재표집은 `Z.analyze`와 같다."""
    import _pdv2_pilot as pilot
    import _review_direction_metrics as direction

    spec = test_spec()
    tags = list(spec["cases"])
    arms, contrasts = arms_and_contrasts(tags)
    rows, rows_new = {}, {}  # rows: 참조 사례는 기존 원장, 나머지는 새 원장. rows_new: 참조 사례를 이 경로로 다시 계산한 원장
    ledger_errors, prefix, init, total = [], [], [], 0
    for s in spec["test"]:
        name = s.replace("/", "_")
        check = READ(TEST / "test" / name / "zero_lag_pretrain_check.json")
        assert check["status"] == "passed"
        ledger_errors += [v["score_error"] for v in check["reproduction"].values()]
        prefix += [dict(r, sequence=s) for r in check["prefix"]]
        init += [dict(r, sequence=s) for r in check["init"]]
        with np.load(W.DIRECTIONS / "directions" / (name + ".npz")) as d:
            label = {k: d[k] for k in ("query_idx", "group")}
        total += len(label["query_idx"])
        masks = dict(all=None, forward_only=label["group"] == 0, reverse_only=label["group"] == 1)
        rows[s], rows_new[s] = {}, {}
        for t in tags:
            ref = reference_ledger(t, name)
            for store, path in ((rows, ref or TEST / "test" / name / f"{t}.npz"),
                                (rows_new, TEST / "test" / name / f"{t}.npz" if ref else None)):
                if path is None:
                    continue
                with np.load(path) as z:
                    np.testing.assert_array_equal(z["query_idx"], label["query_idx"])
                    store[s][t] = {g: direction.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"], m)
                                   for g, m in masks.items()}
    assert total == 195726, total

    valid = {g: {c: [s for s in seqs if rows[s]["diff_s1"][g] is not None] for c, seqs in pilot.CITY.items()}
             for g in GROUPS}
    for s in rows:
        for g in GROUPS:
            assert len({rows[s][t][g] is None for t in tags}) == 1

    def table(source):
        return {g: {t: {m: {c: np.array([source[s][t][g][m] for s in seqs]) for c, seqs in valid[g].items() if seqs}
                        for m in METRICS} for t in source[spec["test"][0]]} for g in GROUPS}

    values, values_new = table(rows), table(rows_new)

    def macro(by_city: dict, picks: dict = None):
        parts = [v.mean() if picks is None else v[picks[c]].mean(axis=1) for c, v in by_city.items()]
        return np.mean(parts, axis=0)

    rng = np.random.default_rng(BOOT_SEED)
    result = dict(arms=arms, contrasts={k: [list(p) for p in v] for k, v in contrasts.items()}, groups={})
    for g in GROUPS:
        sizes = {c: len(seqs) for c, seqs in valid[g].items() if seqs}
        picks = {c: rng.integers(0, n, (BOOTSTRAP, n)) for c, n in sizes.items()}
        block = dict(sequences_per_city=sizes, percent={}, differences_pp={})
        for name, arm_tags in arms.items():
            block["percent"][name] = {}
            for m in METRICS:
                per = [100 * float(macro(values[g][t][m])) for t in arm_tags]
                block["percent"][name][m] = dict(mean=float(np.mean(per)), per_start=per,
                                                 sd=float(np.std(per, ddof=1)) if len(per) > 1 else 0.0)
        for name, pairs in contrasts.items():
            block["differences_pp"][name] = {}
            for m in METRICS:
                diff = {c: np.mean([values[g][a][m][c] - values[g][b][m][c] for a, b in pairs], axis=0)
                        for c in sizes}
                boot = 100 * macro(diff, picks)
                block["differences_pp"][name][m] = dict(
                    mean_pp=100 * float(macro(diff)),
                    per_start_pp=[100 * float(macro(values[g][a][m]) - macro(values[g][b][m])) for a, b in pairs],
                    ci95_pp=[float(v) for v in np.percentile(boot, [2.5, 97.5])])
        result["groups"][g] = block

    # 재현: 다시 계산한 참조 원장의 시작점별 매크로와, 기존 무지연 실험의 보고값(평균·대조 구간)
    old = READ(Z.OUT / "result.json")["groups"]
    gaps = {}
    for g in GROUPS:
        for m in METRICS:
            for arm in ("symmetric_nsd", "Z1_zero_lag_learned", "Z2_zero_lag_no_refine", "Z3_zero_lag_past_uniform"):
                reported = old[g]["percent"][arm][m]["per_start"]
                for i, t in enumerate(arms[arm]):
                    gaps[f"reused_{t}_{g}_{m}"] = abs(100 * macro(values[g][t][m]) - reported[i])
                    gaps[f"recomputed_{t}_{g}_{m}"] = abs(100 * macro(values_new[g][t][m]) - reported[i])
            for name in ("Z1_minus_symmetric", "Z2_minus_symmetric", "Z1_minus_Z2", "Z1_minus_Z3"):
                a, b = result["groups"][g]["differences_pp"][name][m], old[g]["differences_pp"][name][m]
                gaps[f"contrast_{name}_{g}_{m}"] = max(abs(a["mean_pp"] - b["mean_pp"]),
                                                       *[abs(x - y) for x, y in zip(a["ci95_pp"], b["ci95_pp"])])
    max_gap = float(max(gaps.values()))
    worst = max(gaps.items(), key=lambda kv: kv[1])

    zl = [r for r in prefix if r["evidence"] == "past"]
    negative = [r for r in prefix if r["evidence"] == "both"]
    causality = dict(
        rule="질의 q까지의 접두 입력(q+1프레임)으로 그래프·정제·후보 검색·시간 대응·수락 점수를 다시 계산해 전체 입력 원장과 대조(Z.prefix_checks)",
        zero_lag=dict(checks=len(zl), sequences=len({r["sequence"] for r in zl}),
                      conditions=sorted({r["condition"] for r in zl}),
                      winners_equal=all(r["winner_full"] == r["winner_prefix"] for r in zl),
                      plans_equal=all(r["plan_equal"] for r in zl),
                      max_plan_frame_minus_query=max(r["max_plan_frame"] - r["query"] for r in zl),
                      max_coefficient_error=max(r["coefficient_error"] for r in zl),
                      max_score_error=max(r["score_error"] for r in zl),
                      max_original_plan_on_prefix_score_error=max(r["original_plan_on_prefix_score_error"] for r in zl),
                      p1=dict(checks=sum(r["condition"].startswith("zl_p1") for r in zl),
                              max_score_error=max(r["score_error"] for r in zl if r["condition"].startswith("zl_p1")))),
        negative_controls={c: dict(checks=sum(r["condition"] == c for r in negative),
                                   score_changed=sum(r["condition"] == c and r["score_error"] > 8e-6 for r in negative),
                                   winner_changed=sum(r["condition"] == c and r["winner_full"] != r["winner_prefix"]
                                                      for r in negative),
                                   max_score_error=max(r["score_error"] for r in negative if r["condition"] == c))
                           for c in sorted({r["condition"] for r in negative})})
    init_summary = {}
    for r in init:
        key = f"{r['start']}_s{r['seed']}"
        init_summary[key] = max(init_summary.get(key, 0.0), r["max_abs_vs_past_uniform"])
    result.update(
        sanity_checks=dict(ledger_max_score_error=max(ledger_errors), max_reproduction_gap_pp=max_gap,
                           worst_reproduction_gap=list(worst), queries=total),
        causality=causality,
        initialization_on_test=dict(rule="W.build(사전학습 시작점, diff) 직후 출력과 과거 균일 감산 -0.5의 시험46 그래프 최대 절대 차",
                                    max_abs_per_start=init_summary, sequences=len({r["sequence"] for r in init})),
        pretraining=READ(OUT / "pretrain_report.json"),
        finetuning=READ(OUT / "finetune_report.json"),
        script_sha256=HASH(Path(__file__)),
        protocol_script_sha256=READ(OUT / "protocol.json")["sources"]["scripts/_pdv2_zero_lag_pretrain.py"],
        aggregation="시험46 중 해당 방향 양성이 있는 시퀀스 -> 도시 -> 양성이 있는 도시(6) 평균 -> 시작점 3개 평균(SD는 시작점 간, ddof=1).",
        bootstrap=dict(replicates=BOOTSTRAP, seed=BOOT_SEED,
                       scheme="도시 안 시퀀스 수 유지 짝지은 복원추출. 시작점은 고정하고 차이는 시작점 평균. 방향 묶음 순서와 난수 사용 순서는 기존 무지연 분석과 같다."),
        test_used_for_selection=False)
    WRITE(OUT / "result.json", result)
    WRITE(OUT / "reproduction.json", dict(
        pretraining_symmetric=READ(OUT / "pretrain_report.json")["symmetric_reproduction"],
        finetune_z1_retrain=READ(OUT / "finetune_report.json")["z1_retrain_reproduction"],
        evaluation=dict(ledger_max_score_error=max(ledger_errors), max_macro_or_ci_gap_pp=max_gap,
                        worst=list(worst), gaps_over_1e_9=int(sum(v > 1e-9 for v in gaps.values())),
                        rule="대칭 diff·Z1·Z2·Z3를 이 경로로 다시 계산해 기존 원장(승자 일치, 점수 차 8e-6 미만)과 기존 "
                             "보고값(시작점별 매크로, 대조의 평균·구간)에 대조")))
    print(json.dumps({g: {a: round(v["AP"]["mean"], 2) for a, v in b["percent"].items()}
                      for g, b in result["groups"].items()}, ensure_ascii=False), flush=True)
    print("MAX_REPRODUCTION_GAP", max_gap, worst, flush=True)


# ---------------------------------------------------------------- 사전 기록
def protocol() -> None:
    path = OUT / "protocol.json"
    if path.exists():
        print("PROTOCOL EXISTS", path, flush=True)
        return
    import torch
    import _pdv2_spectral_keys as SK

    spec = SK.protocol(True)  # 기존 사전학습의 runtime·inputs 해시가 그대로인지 확인
    sym, past = graph_for("symmetric"), graph_for("past")
    hashes = {k: SK.array_hash(getattr(past, k).numpy()) for k in ("x", "edge_index", "edge_attr", "edge_type")}
    assert {k: SK.array_hash(getattr(sym, k).numpy()) for k in hashes} == spec["graph_hashes"][ARM]
    state = torch.load(SK.OUT / f"prepared_{ARM}.pt", map_location="cpu", weights_only=False)
    no_incoming = int(past.num_nodes - torch.unique(past.edge_index[1]).numel())
    triplets = {}
    for epoch in (1, 6, 11, 16):
        t = SK.base.triplets(epoch)
        triplets[f"epoch{epoch}"] = dict(rows=len(t), positive_after_anchor=float((t[:, 1] > t[:, 0]).mean()),
                                         negative_after_anchor=float((t[:, 2] > t[:, 0]).mean()),
                                         same_sequence=bool((state["sequences"][t[:, 0]] == state["sequences"][t[:, 1]]).all()
                                                            and (state["sequences"][t[:, 0]] == state["sequences"][t[:, 2]]).all()),
                                         min_abs_gap_positive=int(np.abs(t[:, 1] - t[:, 0]).min()),
                                         min_abs_gap_negative=int(np.abs(t[:, 2] - t[:, 0]).min()))
    sources = [Path(__file__), Path(Z.__file__), Path(W.__file__), Path(SK.__file__), Path(SK.base.__file__),
               ROOT / "src/gnn/trainer.py", ROOT / "src/gnn/model.py", ROOT / "src/gnn/geometry_interaction.py",
               ROOT / "src/gnn/learned_spectral_difference.py", ROOT / "src/gnn/contrast_spectral_difference.py",
               ROOT / "scripts/_pdv2_contrast_pipeline.py", ROOT / "scripts/_pdv2_contrast_pipeline_test.py",
               ROOT / "scripts/_pdv2_diffattn_contrast_recalibrated.py", ROOT / "scripts/_pdv2_encoder_place_lib.py",
               SK.OUT / "protocol.json", SK.OUT / f"graph_{ARM}.pt", SK.OUT / f"prepared_{ARM}.pt",
               W.RECAL / "protocol.json", W.OLD_TEST / "protocol.json", Z.OUT / "protocol.json", Z.OUT / "result.json"]
    sources += [pretrain_checkpoint(ORIGINAL_PRETRAIN, k, e) for k in STARTS for e in (10, 20)]
    sources += [Z.zero_lag_checkpoint(k) for k in STARTS]
    WRITE(path, dict(
        created_unix=time.time(),
        purpose="NSD에서 질의 이후 관측만 뺀 구성을 학습의 모든 단계(시간 전용 사전학습, 추가 학습, 평가)에 적용한 무지연 "
                "Past-only NSD(P1)를 만들고, 시험46에서 대칭 NSD·기존 Z1(대칭 사전학습 + 과거 추가 학습)·Z2(정제 없음 + 과거 증거)와 짝지어 비교한다.",
        pretraining=dict(
            path="scripts/_pdv2_spectral_keys.py train(seed, octave288) -> _pdv2_temporal_derivatives.train -> GNNTrainer.train "
                 "(20에포크, 에포크당 45갱신 = 900갱신, InfoNCE 온도 0.1, Adam 5e-4/1e-5, AMP, 3에포크 워밍업 + 코사인, "
                 "삼중쌍 1,024개 x 4배치 누적, 고정 삼중쌍 파일 original_epoch{1,6,11,16}, 시드별 에포크 시드 seed*100000+epoch). "
                 "같은 프로세스 설정(torch 스레드 4, TF32 끔). 결정적 알고리즘 강제 없음(기존과 같음).",
            only_change="과거 창: temporal_subgraph 결과에서 시간 간선을 src<tgt(메시지가 과거 1..5키프레임에서 오는 간선)만 남긴다. "
                        "_pdv2_window_side_learned.past_graph의 간선 선택과 같다(사전학습 그래프에는 transport_mask가 없다). "
                        f"간선 722,820 -> {int(past.edge_index.shape[1])}, 들어오는 간선이 없는 노드(시퀀스 첫 프레임) {no_incoming}개.",
            output_root="출력 폴더만 pretrain/<tag>/s{k}/octave288로 바꾼다. 기존 results/pdv2_spectral_keys_20261025에는 쓰지 않는다.",
            checkpoint="20에포크(checkpoint_epoch_20.pth). 기존과 같다.",
            past_graph_hashes=hashes,
            symmetric_graph_hashes=spec["graph_hashes"][ARM],
            reproduction_rule="대칭 창을 같은 새 경로로 세 시드 모두 다시 사전학습해 기존 20에포크 체크포인트와 가중치를 비교한다. "
                              "세 시드 모두 비트 단위로 같으면 P1은 기존 Z1·대칭 NSD와 직접 짝짓는다. 하나라도 다르면 (a) 대칭 시드1을 "
                              "한 번 더 돌려 실행 간 차이를 재고, (b) 대칭 재사전학습 시작점으로 Z1 절차(Z1r)와 대칭 추가 학습(NSDr)을 "
                              "학습·평가해 P1-Z1r, P1-NSDr, Z1r-Z1, NSDr-대칭 대조를 함께 보고한다. 결정은 시험 결과를 보기 전에 한다.",
            future_dependence_audit=dict(
                temporal_window="정제 그래프의 시간 간선이 유일한 프레임 간 연결이다. 과거 창에서는 src<tgt만 남는다.",
                triplets=dict(note="양성(5 m 안)·음성(10 m 초과, 원 특징 L2 최근접)은 같은 시퀀스의 |차| >= 30키프레임 프레임을 "
                                   "GT 자세로 미리 고른 고정 파일이며 창과 무관하다. 양성·음성의 약 절반은 기준보다 뒤 프레임이다. "
                                   "이것은 오프라인 지도 신호이며 각 노드의 임베딩은 자기 그래프 이웃만 읽으므로 추론 시 미래 관측 "
                                   "의존이 아니다. 지시대로 바꾸지 않는다.",
                              per_refresh=triplets),
                normalization="BN은 학습 중 한 그래프 전체(72,306노드)의 배치 통계를 쓰고 누적 통계를 만든다. 추론과 추가 학습은 "
                              "평가 모드의 고정 누적 통계를 쓰므로 시점 간 의존이 아니다. 학습 중 배치 통계에는 미래 프레임도 섞이며 "
                              "이는 대칭·과거 창 모두 같다. 바꾸지 않는다.",
                other_fixed="원키 재계량(remetric_octave288.npy, 학습 GT 쌍), 상관 보정 W(학습 원 복소 스펙트럼 2차 모멘트), "
                            "손실 척도 loss_fit(대칭 창 균일 감산 -0.5와 앞뒤 증거 plan으로 학습8에서 적합), 융합 가중치·75:25·P·30 상수"
                            "(대칭 사전학습 모델의 검증 선택)는 고정값으로 그대로 쓴다."),
        ),
        finetuning=dict(
            procedure="Z1과 같다: _pdv2_window_side_learned.train('past', k, 'diff') + _pdv2_zero_lag.install_train_plans(EVIDENCE=past). "
                      "사전학습 시드 k의 20에포크 시작점, ContrastSpectralDifference(diff) 균등 감산 -0.5 초기화(Q 투영·기하 MLP 최종층·"
                      "게이트 최종층 0), 과거 그래프, 학습8, 8에포크x8시퀀스x8갱신=512, 배치 64, 실행 시드0 표본 순서, AdamW lr1e-4·감쇠1e-5, "
                      "기울기 절단 1, 재보정 BCE, 결정적 알고리즘. 바꾸는 것은 시작점 경로(W.PRETRAIN)뿐이다.",
            jobs=dict(p1="과거 사전학습 시작점 -> Z1 절차", z1_repro="기존 대칭 사전학습 시작점 -> Z1 절차(기존 Z1 체크포인트와 비트 대조)",
                      conditional=dict(z1r="대칭 재사전학습 시작점 -> Z1 절차", nsdr="대칭 재사전학습 시작점 -> 대칭 추가 학습(diff, 앞뒤 증거)")),
            fixed_values="손실 척도 loss_fit은 대칭 창 균일 감산 -0.5 초기 출력과 앞뒤 증거 plan으로 맞춘 값이다(사전학습 가중치와는 무관: "
                         "초기 출력이 가중치와 무관한 균일 감산). 추론 척도, W, 융합 가중치는 다시 맞추지 않는다. 기존 Z1과 같은 원칙.",
        ),
        evaluation=dict(
            queries="기존 공통 질의(각 시퀀스 q=30..n-16, 시험46 합 195,726개)와 같은 후보 풀·정답·방향 라벨. 무지연 조건(정제 과거 그래프 + 과거 증거).",
            cases="diff_s{k}(대칭 NSD), zl_identity(Z2), zl_past_fixed(Z3 = 추가 학습 전 상태), zl_past_diff_s{k}(Z1), zl_p1_past_diff_s{k}(P1), "
                  "조건부 zl_z1r_past_diff_s{k}, nsdr_diff_s{k}. 참조 사례는 이 경로로 다시 계산해 기존 원장과 대조한다.",
            metrics=list(METRICS), groups=list(GROUPS),
            aggregation="해당 방향 양성이 있는 시퀀스 -> 도시 -> 6도시 평균 -> 시작점 3개 평균(SD ddof=1).",
            bootstrap=f"도시 안 양성 시퀀스 수 유지 짝지은 복원추출 {BOOTSTRAP}회, 시드 {BOOT_SEED}(기존 무지연 분석과 같은 난수 순서). 시작점 고정, 차이는 시작점 평균.",
            contrasts="P1-대칭, P1-Z1, P1-Z2, P1-Z3, (재현) Z1-대칭, Z2-대칭, Z1-Z2, Z1-Z3, (조건부) P1-Z1r, P1-NSDr, Z1r-Z1, NSDr-대칭"),
        checks=dict(
            causality="무지연 사례 전부와 시퀀스 46개 각각 질의 3개에서 q+1프레임 접두 입력으로 다시 계산(Z.prefix_checks): 반환 장소 같음, "
                      "정제 계수 차 <4e-6, plan 같음·q 이후 프레임 미참조, 점수 차 <8e-6. 대조 diff_s1은 점수가 달라져야 한다.",
            initialization="추가 학습 시 학습 그래프에서 init_error<4e-6(W.train), 시험46 그래프에서 새 시작점 초기화 출력과 과거 균일 감산 -0.5의 차 <4e-6.",
            pretraining_causality="사전학습 20에포크 모델(평가 모드)의 첫 학습 시퀀스 가운데 노드 출력 기울기가 과거 창에서는 미래 노드에 0.",
            training_log="사전학습 에포크 손실 유한, 추가 학습 손실·기울기 유한(W.train assert), 에포크 평균 손실·게이트 세기 기록.",
            reproduction="대칭 NSD·Z1·Z2·Z3 원장과 시작점별 매크로, Z1-대칭·Z2-대칭·Z1-Z2·Z1-Z3 구간을 기존 결과와 대조."),
        selection="시험셋으로 고르는 설정 없음. 조건은 한 번 실행한다.",
        sources={(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)): HASH(p) for p in sources}))
    print("PROTOCOL WRITTEN", path, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("protocol", "pretrain", "pretrain_all", "finetune", "test", "test_worker", "analyze"))
    p.add_argument("--window", choices=("symmetric", "past"))
    p.add_argument("--seed", type=int, choices=STARTS)
    p.add_argument("--tag")
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "pretrain":
        pretrain_worker(a.window, a.seed, a.tag or a.window)
    elif a.action == "test_worker":
        test_worker(a.sequence)
    else:
        globals()[a.action]()
