#!/usr/bin/env python3
"""무지연 Past-only NSD의 이웃 수 맞춤 변형 P10: 정제·수락 증거·사전학습·추가 학습 모두 과거 10키프레임을 쓴다.

P1(`_pdv2_zero_lag_pretrain.py`)은 대칭 NSD의 층별 이웃 10개(앞 5 + 뒤 5) 대신 과거 5개만 쓴다. P10은 층별 이웃을
과거 10개(오프셋 -10..-1)로 두어 대칭 NSD와 이웃 수를 맞춘다. 두 층이므로 정제는 i-20까지 읽는다.

- 그래프: 시간 간선을 같은 시퀀스의 (i-d -> i), d=1..10으로 둔다. 간선 속성은 기존 그래프와 같은 함수
  (`_pdv2_input_alignment_lib.pose_attributes`, 같은 자세)로 만든다. 간선 부호기는 거리·회전각만 읽고 오프셋별
  매개변수가 없으므로 매개변수 수와 구조는 그대로다.
- 전달 칸: 기존 11칸(오프셋 -5..+5, 가운데 5는 자기 자신)을 재사용한다. 칸 = (오프셋 + 5) mod 11이므로 -5..-1은 기존
  과거 칸 0..4, -10..-6은 쓰지 않던 미래 칸 6..10에 들어간다. `ContrastSpectralDifference`의
  `dense[tgt, src - tgt + 5]`는 음수 색인이 같은 칸으로 감기므로 코드 수정 없이 같은 대응이 된다(실행 전 검사).
- 수락의 시간 증거: 질의 쪽 오프셋 -10..-1, 후보 쪽 거리 대응 창 [j*-20, j*+20](기존 규칙: 창 반경 = 최대 오프셋의 2배,
  5 -> 10을 10 -> 20으로). β=11/15, 중복 후보 가중치, 정규화, 중심 +4/15, 방향 최댓값은 그대로다. 유효 이웃 10개면
  중심 가중치 1/3(대칭과 같음).
- 사전학습: P1과 같은 경로(`_pdv2_spectral_keys`의 학습 함수, 같은 표본·손실·스케줄·20에포크), 그래프만 P10.
- 추가 학습: Z1/P1과 같은 `_pdv2_window_side_learned.train('past', k, 'diff')`. 학습 그래프(TRAIN_INPUTS)와 학습 plan만
  P10으로 바꾼다. 손실 척도·W·융합 가중치는 다시 맞추지 않는다.
- 시험: 기존 공통 질의, 같은 후보 풀. 시험셋으로 고르는 설정은 없다.

산출물: /workspace/data/nsd_results/pdv2_zero_lag_pretrain_20261112/p10/
사용법(컨테이너 ring_baseline 안, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python scripts/_pdv2_zero_lag_p10.py protocol      # 구조 검사·학습 대응 창 감사·protocol.json 저장
  python scripts/_pdv2_zero_lag_p10.py pretrain_all  # 사전학습 3개(시드마다 새 프로세스)
  python scripts/_pdv2_zero_lag_p10.py finetune      # 학습 그래프 생성 + 추가 학습 3개
  python scripts/_pdv2_zero_lag_p10.py test          # 시험46 평가·검사·분석
"""
import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import _pdv2_zero_lag_pretrain as P  # noqa: E402

Z, W = P.Z, P.W
READ, WRITE, HASH = W.READ, W.WRITE, W.HASH
OUT = P.OUT / "p10"
PRE = OUT / "pretrain"
FINE = OUT / "finetune"
TRAIN_INPUTS = OUT / "train_inputs"
TEST = OUT / "test_eval"
LOGS = OUT / "logs"
ARM, STARTS, METRICS, GROUPS = P.ARM, P.STARTS, P.METRICS, P.GROUPS
WINDOW = 10           # 과거 이웃 수(층별)
RADIUS = 2 * WINDOW   # 후보 쪽 거리 대응 창 반경(기존 10 = 2 x 5)
OFFSETS = np.arange(-WINDOW, 0)
EVIDENCE = {"mode": None}  # "p10" 또는 "both"
ORIGINAL = {}


# ---------------------------------------------------------------- 그래프
def past_edges(sequences: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """같은 시퀀스 안의 (i, i+d), d=1..window 간선. 메시지는 앞선 프레임 i에서 뒤 프레임 i+d로 간다."""
    n = len(sequences)
    offsets = np.arange(1, window + 1)
    source = np.broadcast_to(np.arange(n)[:, None], (n, window))
    target = source + offsets[None]
    valid = target < n
    safe = np.clip(target, 0, n - 1)
    valid &= sequences[source] == sequences[safe]
    return np.stack((source[valid], target[valid]))


def slots(edges: np.ndarray) -> np.ndarray:
    """오프셋(src-tgt, -10..-1)을 기존 11칸에 둔다: (오프셋+5) mod 11. -5..-1 -> 0..4, -10..-6 -> 6..10."""
    offset = edges[0] - edges[1]
    assert offset.min() >= -WINDOW and offset.max() <= -1
    return (offset + 5) % 11


def to_p10(graph, poses: np.ndarray, sequences: np.ndarray = None):
    """기존 ±5 그래프의 노드 입력(x, 복소 계수, 방향)을 그대로 두고 시간 간선·속성·전달 칸만 과거 10개로 바꾼다."""
    import torch
    from _pdv2_input_alignment_lib import pose_attributes

    n = graph.num_nodes
    sequences = np.zeros(n, np.int64) if sequences is None else np.asarray(sequences)
    assert len(poses) == n
    old = graph.edge_index.cpu().numpy()
    # 기존 간선 속성이 같은 함수·같은 자세로 재현되는지 먼저 확인한다.
    assert np.array_equal(pose_attributes(poses, old), graph.edge_attr.cpu().numpy())
    edges = past_edges(sequences)
    device = graph.x.device
    g = graph.clone()
    g.edge_index = torch.as_tensor(edges, device=device)
    g.edge_attr = torch.as_tensor(pose_attributes(poses, edges), device=device)
    g.edge_type = torch.zeros(edges.shape[1], dtype=graph.edge_type.dtype, device=device)
    if hasattr(graph, "transport_mask"):
        src, tgt = edges
        index = np.broadcast_to(np.arange(n)[:, None], (n, 11)).copy()
        mask = np.zeros((n, 11), np.float32)
        index[tgt, slots(edges)], mask[tgt, slots(edges)] = src, 1.0
        assert int(mask.sum()) == len(src) and not mask[:, 5].any()
        g.transport_indices = torch.as_tensor(index, device=device)
        g.transport_mask = torch.as_tensor(mask, device=device)
    return g


def check_p10_graph(graph):
    """W.train의 past_graph 자리에 넣는 검사: 간선이 모두 과거 1..10키프레임이고 전달 칸이 간선과 정확히 같다."""
    import torch

    src, tgt = graph.edge_index
    assert bool((src < tgt).all()) and int((tgt - src).max()) <= WINDOW
    rebuilt = torch.zeros_like(graph.transport_mask)
    rebuilt[tgt, (src - tgt + 5) % 11] = 1
    assert torch.equal(rebuilt, graph.transport_mask)
    return graph


# ---------------------------------------------------------------- 시간 증거
def p10_neighbor_plans(path, query, candidate) -> dict:
    """`retrieval.sequence_evidence.neighbor_plans`의 arc_dense와 같은 규칙에서 질의 오프셋만 -10..-1, 창 반경만 20."""
    from retrieval.sequence_evidence import distance_indices

    q, j = np.asarray(query, np.int64), np.asarray(candidate, np.int64)
    qs, js, masks = [], [], []
    for direction in (1, -1):
        qn = q[:, None] + OFFSETS[None]
        qsafe = np.clip(qn, 0, len(path) - 1)
        jn = distance_indices(path, j[:, None], direction * (path[qsafe] - path[q, None]), radius=RADIUS)
        valid = (qn >= 0) & (qn < len(path)) & (jn >= 0) & (jn < len(path)) & (jn != j[:, None])
        qs.append(qn)
        js.append(jn)
        masks.append(valid)
    dense = np.stack(qs), np.stack(js), np.stack(masks)
    return {"arc_dense": dense, "arc_unique": dense}


def modules():
    _, sg = Z.modules()
    ORIGINAL.setdefault("plan", Z.ORIGINAL["plan"])
    ORIGINAL.setdefault("neighbor_plans", Z.ORIGINAL["neighbor_plans"])
    return sg


def p10_plan(travel, q, j) -> dict:
    """기존 plan을 그대로 부르되 그 안의 neighbor_plans만 P10 규칙으로 바꾼다."""
    sg = modules()
    sg.neighbor_plans = p10_neighbor_plans
    try:
        plan = ORIGINAL["plan"](travel, q, j)
    finally:
        sg.neighbor_plans = ORIGINAL["neighbor_plans"]
    assert (plan["query"] <= np.asarray(q)[:, None, None]).all()
    return plan


def evidence_plan(travel, q, j) -> dict:
    mode = EVIDENCE["mode"]
    assert mode in ("p10", "both"), mode
    return p10_plan(travel, q, j) if mode == "p10" else ORIGINAL["plan"](travel, q, j)


def install_train_plans() -> dict:
    """W.train이 읽는 학습 풀의 plan_*을 P10 plan으로 바꾼다. 저장된 앞뒤 plan을 같은 함수로 정확히 다시 만드는지 먼저 확인."""
    e, _ = Z.modules()
    modules()
    sequences = READ(W.RECAL / "protocol.json")["train"]
    names = {s.replace("/", "_"): s for s in sequences}
    pools = (e.prior.PARENT / "pools/train").resolve()
    original = ORIGINAL.setdefault("ARR", e.run.ARR)
    cache, audit = {}, {}

    def arr(path):
        data = original(path)
        path = Path(path)
        if path.name != "pool.npz" or path.parent.parent.resolve() != pools:
            return data
        name = path.parent.name
        if name not in cache:
            travel = e.run.inputs("train", names[name])[3]
            q, j = data["query_idx"], data["j_star"]
            again = ORIGINAL["plan"](travel, q, j)
            assert all(np.array_equal(again[k], data["plan_" + k]) for k in again), name
            plan = p10_plan(travel, q, j)
            cache[name] = plan
            w = plan["weight"]
            qn, jn, valid = p10_neighbor_plans(travel, q, j)["arc_dense"]
            audit[names[name]] = dict(
                queries=len(q), center_weight_mean=float(w[..., 0].mean()),
                weighted_neighbors_mean=float((w[..., 1:] > 0).sum(-1).mean()),
                max_query_offset=int((plan["query"] - q[:, None, None]).max()),
                candidate_window_edge_fraction=float((np.abs(jn - j[None, :, None])[valid] == RADIUS).mean()))
        data.update({"plan_" + k: v for k, v in cache[name].items()})
        return data

    e.run.ARR = arr
    for s in sequences:
        arr(e.prior.PARENT / "pools/train" / s.replace("/", "_") / "pool.npz")
    return audit


def window_audit() -> dict:
    """학습8 풀에서 질의 오프셋 -10..-1의 후보 쪽 대응이 창 경계에 걸리는 비율(반경 10 대 20)과 대칭 ±5·반경 10 기준."""
    from retrieval.sequence_evidence import distance_indices

    e, _ = Z.modules()
    rows = {}
    for s in READ(W.RECAL / "protocol.json")["train"]:
        data = e.run.ARR(e.prior.PARENT / "pools/train" / s.replace("/", "_") / "pool.npz")
        travel = e.run.inputs("train", s)[3]
        q, j = data["query_idx"], data["j_star"]
        row = {}
        for label, offsets, radius in (("symmetric_pm5_r10", np.r_[-5:0, 1:6], 10),
                                       ("past10_r10", OFFSETS, 10), ("past10_r20", OFFSETS, RADIUS)):
            edge = []
            for direction in (1, -1):
                qn = q[:, None] + offsets[None]
                qsafe = np.clip(qn, 0, len(travel) - 1)
                jn = distance_indices(travel, j[:, None], direction * (travel[qsafe] - travel[q, None]), radius=radius)
                valid = (qn >= 0) & (qn < len(travel)) & (jn >= 0) & (jn < len(travel)) & (jn != j[:, None])
                edge.append((np.abs(jn - j[:, None]) == radius)[valid])
            row[label] = float(np.concatenate(edge).mean())
        rows[s] = row
    return dict(per_sequence=rows, mean={k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))},
                rule="창 경계(|j'-j*| = 반경)에 걸린 유효 대응 비율. 학습8만 사용(시험 미사용).")


# ---------------------------------------------------------------- 사전학습
def pretrain_graph():
    """사전학습 그래프(기존 graph_octave288의 x·같은 GT 자세)를 P10 간선으로 만든다."""
    import torch
    import _pdv2_spectral_keys as SK

    original = torch.load(SK.OUT / f"graph_{ARM}.pt", map_location="cpu", weights_only=False)
    sym = SK.base.temporal_subgraph(original, np.arange(original.num_nodes))
    state = torch.load(SK.OUT / f"prepared_{ARM}.pt", map_location="cpu", weights_only=False)
    return to_p10(sym, state["poses"], state["sequences"]), state


def pretrain_worker(seed: int) -> None:
    """P1의 사전학습 경로와 같은 프로세스 설정. 학습 그래프만 P10으로 바꾼다."""
    import logging
    import torch
    import _pdv2_spectral_keys as SK

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    base = SK.base
    spec = SK.protocol()
    mine = READ(OUT / "protocol.json")
    folder = PRE / f"s{seed}" / ARM
    if (folder / "pretrain_audit.json").exists():
        return
    graph, state = pretrain_graph()
    hashes = mine["pretraining"]["graph_hashes"]
    base.PREPARED, base.GRAPH = SK.OUT / f"prepared_{ARM}.pt", SK.OUT / f"graph_{ARM}.pt"
    base.root = lambda s, a: PRE / f"s{s}" / a
    base.EPOCHS = SK.EPOCHS
    base.model = lambda selected: SK.model(selected, seed)
    base.protocol = lambda *args: {**SK.protocol(), "graph_hashes": hashes}
    subgraph = base.temporal_subgraph
    base.temporal_subgraph = lambda g, ids: graph.clone()  # base.train의 temporal_subgraph(GRAPH, 전체) 자리
    deterministic = torch.are_deterministic_algorithms_enabled()
    tick = time.time()
    SK.ORIGINAL_TRAIN(seed, ARM)
    seconds = time.time() - tick
    base.temporal_subgraph = subgraph
    assert READ(folder / "initial_model.json") == spec["initial"][str(seed)][ARM]
    net = SK.model(ARM, seed).cuda().eval()
    net.load_state_dict(torch.load(P.pretrain_checkpoint(PRE, seed), map_location="cpu",
                                   weights_only=False)["model_state_dict"], strict=True)
    causal = P.pretrain_causal_check(net, graph, state["sequences"])
    assert causal["future_max"] == 0 and causal["future_nodes"] == 0 and causal["reach"][0] == -2 * WINDOW, causal
    WRITE(folder / "pretrain_audit.json", dict(
        window=f"past{WINDOW}", seed=seed, deterministic_algorithms=deterministic, wall_seconds=seconds,
        edges=int(graph.edge_index.shape[1]), causal_check_epoch20=causal,
        parameters=sum(p.numel() for p in net.parameters())))
    print("PRETRAINED p10", seed, round(seconds, 1), causal, flush=True)


def pretrain_all() -> None:
    assert (OUT / "protocol.json").exists()
    LOGS.mkdir(parents=True, exist_ok=True)
    timing = {}
    for k in STARTS:
        tick = time.time()
        with (LOGS / f"pretrain_s{k}.log").open("a") as stream:
            subprocess.run([sys.executable, "-u", __file__, "pretrain", "--seed", str(k)], cwd=ROOT,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        timing[f"s{k}"] = time.time() - tick
        print("PRETRAIN DONE p10", k, round(timing[f"s{k}"], 1), flush=True)
    logs = {}
    for k in STARTS:
        logs[f"s{k}"] = dict(
            p10_epoch_loss=P.epoch_losses(PRE / f"s{k}" / ARM),
            p1_epoch_loss=P.epoch_losses(P.PRE / "past" / f"s{k}" / ARM),
            symmetric_rerun_epoch_loss=P.epoch_losses(P.PRE / "symmetric" / f"s{k}" / ARM),
            audit=READ(PRE / f"s{k}" / ARM / "pretrain_audit.json"))
        logs[f"s{k}"]["finite"] = bool(np.isfinite(logs[f"s{k}"]["p10_epoch_loss"]).all())
    WRITE(OUT / "pretrain_report.json", dict(pretraining_logs=logs, wall_seconds=timing))


# ---------------------------------------------------------------- 추가 학습
def build_train_inputs() -> dict:
    """학습8 그래프(W.TRAIN_INPUTS의 기존 ±5 그래프와 같은 입력)를 P10 간선·전달 칸으로 바꿔 저장한다."""
    import torch

    e, _ = Z.modules()
    TRAIN_INPUTS.mkdir(parents=True, exist_ok=True)
    audit = {}
    for s in READ(W.RECAL / "protocol.json")["train"]:
        name = s.replace("/", "_")
        path = TRAIN_INPUTS / (name + ".pt")
        if path.with_suffix(".json").exists():
            assert HASH(path) == READ(path.with_suffix(".json"))["graph_sha256"]
            audit[s] = READ(path.with_suffix(".json"))
            continue
        graph = torch.load(W.TRAIN_INPUTS / (name + ".pt"), map_location="cpu", weights_only=False)
        poses = e.run.inputs("train", s)[0]["poses"]
        g = check_p10_graph(to_p10(graph, poses))
        torch.save(g, path)
        audit[s] = dict(source=str(W.TRAIN_INPUTS / (name + ".pt")), source_sha256=HASH(W.TRAIN_INPUTS / (name + ".pt")),
                        nodes=g.num_nodes, edges=int(g.edge_index.shape[1]), graph_sha256=HASH(path))
        WRITE(path.with_suffix(".json"), audit[s])
    return audit


def finetune() -> None:
    """P1과 같은 W.train('past', k, 'diff'). 시작점·학습 그래프·학습 plan만 P10으로 바꾼다."""
    for k in STARTS:
        assert P.pretrain_checkpoint(PRE, k).exists()
    inputs = build_train_inputs()
    plans = install_train_plans()
    WRITE(FINE / "train_plans.json", dict(inputs=inputs, plans=plans))
    tick = time.time()
    W.PRETRAIN, W.OUT, W.TRAIN_INPUTS = PRE, FINE, TRAIN_INPUTS
    W.past_graph = check_p10_graph  # W.train은 'past' 창에서 학습 그래프를 이 함수에 통과시킨다
    for k in STARTS:
        W.train("past", k, "diff")
    training = {}
    for k in STARTS:
        record = READ(FINE / "runs/past" / f"s{k}" / "diff" / "training_complete.json")
        losses = np.array([r["loss"] for r in record["records"]]).reshape(8, -1)
        training[f"p10_s{k}"] = dict(
            finite=bool(np.isfinite(losses).all()), epoch_mean_loss=losses.mean(1).tolist(),
            final_epoch_strength_mean=float(np.mean([r["strength_mean"] for r in record["records"][-losses.shape[1]:]])),
            seconds=record["seconds"], init_error=record["init_error"], causal_check=record["causal_check"],
            trainable_parameters=record["trainable_parameters"])
        assert record["causal_check"]["final"]["reach"][0] == -2 * WINDOW, record["causal_check"]
    WRITE(OUT / "finetune_report.json", dict(training=training, wall_seconds=time.time() - tick))
    print("FINETUNE p10", json.dumps(training, ensure_ascii=False)[:1500], flush=True)
    assert all(v["finite"] for v in training.values())


def finetune_checkpoint(k: int) -> Path:
    return FINE / "runs/past" / f"s{k}" / "diff" / "epoch8.pth"


# ---------------------------------------------------------------- 시험
def test_spec() -> dict:
    path = TEST / "protocol.json"
    if path.exists():
        return READ(path)
    old = READ(W.OLD_TEST / "protocol.json")
    cases = {"zl_p10_identity": dict(kind="identity", seed=0, checkpoint=None, evidence="p10"),
             "zl_p10_fixed": dict(kind="p10_fixed", seed=0, checkpoint=None, evidence="p10")}
    for k in STARTS:
        cases[f"zl_p10_diff_s{k}"] = P.checkpoint_case("p10_diff", k, finetune_checkpoint(k), "p10")
    spec = dict(old)
    spec.update(created_unix=time.time(), cases=cases,
                test_policy="고정 설정의 시험 보고. 설정 선택 없음. 조건은 한 번 실행.")
    WRITE(path, spec)
    return spec


def install_eval(pipeline, sequence: str) -> None:
    """kind가 p10_로 시작하면 같은 노드 입력의 P10 그래프에 원래 변환·모델을 쓰고, 사례의 evidence로 plan을 고른다."""
    sg = modules()
    transform, load = pipeline.transform, pipeline.load_model
    poses = pipeline.run.inputs("test", sequence)[0]["poses"]
    cache = {}

    def p10_transform(net, graph, kind: str):
        if not kind.startswith("p10_"):
            return transform(net, graph, kind)
        n = graph.num_nodes
        if n not in cache:
            cache[n] = to_p10(graph, poses[:n])
        return transform(net, cache[n], kind[len("p10_"):])

    def p10_load(case: dict):
        EVIDENCE["mode"] = case["evidence"]  # worker는 사례마다 load_model -> transform -> plan 순서로 부른다
        if case["kind"].startswith("p10_"):
            case = dict(case, kind=case["kind"][len("p10_"):])
        return load(case)

    pipeline.transform, pipeline.load_model = p10_transform, p10_load
    sg.plan = evidence_plan


def prefix_checks(pipeline, sequence: str, folder: Path) -> list:
    """`Z.prefix_checks`와 같은 규칙: q까지만 준 입력(q+1프레임)으로 반환 장소·plan·수락 점수를 다시 계산해 원장과 대조."""
    import torch

    spec = test_spec()
    run, e, sg = pipeline.run, pipeline.e, modules()
    cache, _, _, travel, bank, _, _, _ = run.inputs("test", sequence)
    graph, _, _ = e.prior.graph_inputs("test", sequence)
    weights = run.cal.fusion_weights(spec["confidence"])[run.cal.FUSIONS.index("aux_half")]
    queries = np.load(folder / "zl_p10_identity.npz")["query_idx"]
    positions = sorted({len(queries) // 3, 2 * len(queries) // 3, len(queries) - 1})
    prefixes = {}
    for at in positions:
        end = int(queries[at]) + 1
        small, short_travel, _ = e.prior.graph_inputs("test", sequence, end)
        prefixes[end] = (small, short_travel)

    def cuda(plan):
        return {k: torch.from_numpy(v).cuda() for k, v in plan.items()}

    rows = []
    with torch.no_grad():
        for tag, case in spec["cases"].items():
            kind = case["kind"]
            net = pipeline.load_model(case)
            z = pipeline.transform(net, graph, kind)
            with np.load(folder / f"{tag}.npz") as ledger:
                ledger = {k: ledger[k] for k in ("query_idx", "j_star", "score")}
            for at in positions:
                t = int(ledger["query_idx"][at])
                end = t + 1
                small, short_travel = prefixes[end]
                short = pipeline.transform(net, small.clone(), kind)
                skey = pipeline.raw_key(cache["descriptors"][:end])
                short_bank = (tuple(v[:end] for v in bank) if kind == "identity"
                              else run.base.variant_inputs(short.cpu().numpy(), 8))
                winner = int(run.cal.candidates(skey, (bank[0][:end], *short_bank[1:]), weights[None], t)[0])
                full_plan = sg.plan(travel, np.array([t]), ledger["j_star"][at:at + 1])
                plan = sg.plan(short_travel, np.array([t]), np.array([winner]))
                score = float(e.prior.score(short, cuda(plan)).cpu().numpy()[0])
                rows.append(dict(
                    condition=tag, evidence=case["evidence"], query=t, prefix_frames=end, full_frames=len(z),
                    winner_full=int(ledger["j_star"][at]), winner_prefix=winner,
                    coefficient_error=float((short - z[:end]).abs().max()),
                    plan_equal=all(np.array_equal(full_plan[k], plan[k]) for k in plan),
                    min_plan_query_offset=int((plan["query"] - t).min()),
                    max_plan_frame=int(max(plan["query"].max(), plan["candidate"].max())),
                    score_full=float(ledger["score"][at]),
                    score_error=abs(score - float(ledger["score"][at]))))
                r = rows[-1]
                assert r["winner_full"] == r["winner_prefix"], r
                assert r["coefficient_error"] < 4e-6 and r["plan_equal"] and r["max_plan_frame"] <= t, r
                assert r["score_error"] < 8e-6, r
            del net, z
    return rows


def init_checks(pipeline, graph) -> list:
    """P10 시작점의 추가 학습 전 모델(W.build) 출력이 P10 균일 감산 -0.5와 같은지 시험 그래프에서 잰다."""
    import torch

    rows = []
    W.PRETRAIN = PRE
    with torch.no_grad():
        fixed = pipeline.transform(None, graph, "p10_fixed")
        for k in STARTS:
            net = W.build(torch, pipeline.e, k, "diff")
            z = pipeline.transform(net, graph, "p10_diff")
            rows.append(dict(seed=k, max_abs_vs_p10_uniform=float((z - fixed).abs().max()),
                             max_abs_fixed=float(fixed.abs().max())))
            del net, z
    W.PRETRAIN = P.ORIGINAL_PRETRAIN
    return rows


def test_worker(sequence: str) -> None:
    import _pdv2_contrast_pipeline_test as T

    T.OUT = TEST
    T.prepare = test_spec
    T.pipeline.e.configure()
    T.activate()
    install_eval(T.pipeline, sequence)
    name = sequence.replace("/", "_")
    folder = TEST / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        dst = folder / ("candidate_pool" + suffix)
        if not dst.exists():
            shutil.copy2(W.OLD_TEST / "test" / name / ("candidate_pool" + suffix), dst)
    T.worker("test", sequence)
    target = folder / "p10_check.json"
    if target.exists():
        assert READ(target)["status"] == "passed"
        return
    # 정제 없음 사례는 증거만 바뀌므로 반환 장소가 기존 원장(Z2)과 같아야 한다.
    new, old = np.load(folder / "zl_p10_identity.npz"), np.load(Z.TEST / "test" / name / "zl_identity.npz")
    np.testing.assert_array_equal(new["query_idx"], old["query_idx"])
    np.testing.assert_array_equal(new["j_star"], old["j_star"])
    graph, _, _ = T.pipeline.e.prior.graph_inputs("test", sequence)
    init = init_checks(T.pipeline, graph)
    assert all(r["max_abs_vs_p10_uniform"] < 4e-6 for r in init), init
    cache = T.inputs("test", sequence)[0]
    n = len(cache["descriptors"])
    q = new["query_idx"]
    np.testing.assert_array_equal(q, np.arange(30, n - 15))
    prefix = prefix_checks(T.pipeline, sequence, folder)
    WRITE(target, dict(status="passed", init=init, prefix=prefix,
                       identity_winners_equal_Z2=True,
                       query_range=dict(frames=n, first=int(q[0]), last=int(q[-1]), count=len(q),
                                        past10_offsets_in_range=bool(q.min() - WINDOW >= 0),
                                        refinement_reach_in_range=bool(q.min() - 2 * WINDOW >= 0))))
    print("CHECKED", sequence, flush=True)


def test() -> None:
    assert (OUT / "finetune_report.json").exists()
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
LEDGERS = {  # 사례 -> 원장 폴더(시퀀스 이름 아래)
    "diff": W.OLD_TEST / "test", "zl_identity": Z.TEST / "test", "zl_past_diff": Z.TEST / "test",
    "zl_p1_past_diff": P.TEST / "test", "zl_p10": TEST / "test"}


def ledger_path(tag: str, name: str) -> Path:
    for prefix in ("zl_p10", "zl_p1_past_diff", "zl_past_diff", "zl_identity", "diff"):
        if tag.startswith(prefix):
            return LEDGERS[prefix] / name / f"{tag}.npz"
    raise KeyError(tag)


def analyze() -> None:
    """P1 분석(`P.analyze`)과 같은 집계·재표집(같은 난수 순서). P1 대조는 P1 결과와 같아야 한다."""
    import _pdv2_pilot as pilot
    import _review_direction_metrics as direction

    spec = test_spec()
    arms = {
        "symmetric_nsd": [f"diff_s{k}" for k in STARTS],
        "P10_past10_pretrain_zero_lag": [f"zl_p10_diff_s{k}" for k in STARTS],
        "P1_past_pretrain_zero_lag": [f"zl_p1_past_diff_s{k}" for k in STARTS],
        "Z1_zero_lag_learned": [f"zl_past_diff_s{k}" for k in STARTS],
        "Z2_zero_lag_no_refine": ["zl_identity"],
        "P10_init_past10_uniform": ["zl_p10_fixed"],
        "Z2_10_no_refine_past10_evidence": ["zl_p10_identity"],
    }
    contrasts = {
        "P10_minus_symmetric": [(f"zl_p10_diff_s{k}", f"diff_s{k}") for k in STARTS],
        "P10_minus_P1": [(f"zl_p10_diff_s{k}", f"zl_p1_past_diff_s{k}") for k in STARTS],
        "P10_minus_Z2": [(f"zl_p10_diff_s{k}", "zl_identity") for k in STARTS],
        "P10_minus_P10init": [(f"zl_p10_diff_s{k}", "zl_p10_fixed") for k in STARTS],
        "P10_minus_Z2_10": [(f"zl_p10_diff_s{k}", "zl_p10_identity") for k in STARTS],
        "Z2_10_minus_Z2": [("zl_p10_identity", "zl_identity")],
        "P1_minus_symmetric": [(f"zl_p1_past_diff_s{k}", f"diff_s{k}") for k in STARTS],
        "P1_minus_Z2": [(f"zl_p1_past_diff_s{k}", "zl_identity") for k in STARTS],
    }
    tags = sorted({t for v in arms.values() for t in v})
    rows, prefix, init, total = {}, [], [], 0
    for s in spec["test"]:
        name = s.replace("/", "_")
        check = READ(TEST / "test" / name / "p10_check.json")
        assert check["status"] == "passed"
        prefix += [dict(r, sequence=s) for r in check["prefix"]]
        init += [dict(r, sequence=s) for r in check["init"]]
        with np.load(W.DIRECTIONS / "directions" / (name + ".npz")) as d:
            label = {k: d[k] for k in ("query_idx", "group")}
        total += len(label["query_idx"])
        masks = dict(all=None, forward_only=label["group"] == 0, reverse_only=label["group"] == 1)
        rows[s] = {}
        for t in tags:
            with np.load(ledger_path(t, name)) as z:
                np.testing.assert_array_equal(z["query_idx"], label["query_idx"])
                rows[s][t] = {g: direction.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"], m)
                              for g, m in masks.items()}
    assert total == 195726, total
    valid = {g: {c: [s for s in seqs if rows[s]["diff_s1"][g] is not None] for c, seqs in pilot.CITY.items()}
             for g in GROUPS}
    values = {g: {t: {m: {c: np.array([rows[s][t][g][m] for s in seqs]) for c, seqs in valid[g].items() if seqs}
                      for m in METRICS} for t in tags} for g in GROUPS}

    def macro(by_city: dict, picks: dict = None):
        parts = [v.mean() if picks is None else v[picks[c]].mean(axis=1) for c, v in by_city.items()]
        return np.mean(parts, axis=0)

    rng = np.random.default_rng(P.BOOT_SEED)
    result = dict(arms=arms, contrasts={k: [list(p) for p in v] for k, v in contrasts.items()}, groups={})
    for g in GROUPS:
        sizes = {c: len(seqs) for c, seqs in valid[g].items() if seqs}
        picks = {c: rng.integers(0, n, (P.BOOTSTRAP, n)) for c, n in sizes.items()}
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
                diff = {c: np.mean([values[g][a][m][c] - values[g][b][m][c] for a, b in pairs], axis=0) for c in sizes}
                boot = 100 * macro(diff, picks)
                block["differences_pp"][name][m] = dict(
                    mean_pp=100 * float(macro(diff)),
                    per_start_pp=[100 * float(macro(values[g][a][m]) - macro(values[g][b][m])) for a, b in pairs],
                    ci95_pp=[float(v) for v in np.percentile(boot, [2.5, 97.5])])
        result["groups"][g] = block
    # P1 분석과 같은 값인지(같은 원장·같은 난수 순서)
    p1 = READ(P.OUT / "result.json")["groups"]
    gaps = {}
    for g in GROUPS:
        for m in METRICS:
            for arm in ("symmetric_nsd", "P1_past_pretrain_zero_lag", "Z1_zero_lag_learned", "Z2_zero_lag_no_refine"):
                gaps[f"{arm}_{g}_{m}"] = abs(result["groups"][g]["percent"][arm][m]["mean"] - p1[g]["percent"][arm][m]["mean"])
            for name in ("P1_minus_symmetric", "P1_minus_Z2"):
                a, b = result["groups"][g]["differences_pp"][name][m], p1[g]["differences_pp"][name][m]
                gaps[f"{name}_{g}_{m}"] = max(abs(a["mean_pp"] - b["mean_pp"]),
                                              *[abs(x - y) for x, y in zip(a["ci95_pp"], b["ci95_pp"])])
    result.update(
        consistency_with_p1=dict(max_gap_pp=float(max(gaps.values())), worst=list(max(gaps.items(), key=lambda kv: kv[1]))),
        causality=dict(
            rule="질의 q까지의 접두 입력(q+1프레임)으로 P10 그래프·정제·후보 검색·시간 대응·수락 점수를 다시 계산해 전체 입력 원장과 대조",
            checks=len(prefix), sequences=len({r["sequence"] for r in prefix}),
            conditions=sorted({r["condition"] for r in prefix}),
            winners_equal=all(r["winner_full"] == r["winner_prefix"] for r in prefix),
            plans_equal=all(r["plan_equal"] for r in prefix),
            max_plan_frame_minus_query=max(r["max_plan_frame"] - r["query"] for r in prefix),
            min_plan_query_offset=min(r["min_plan_query_offset"] for r in prefix),
            max_coefficient_error=max(r["coefficient_error"] for r in prefix),
            max_score_error=max(r["score_error"] for r in prefix)),
        initialization_on_test=dict(max_abs=max(r["max_abs_vs_p10_uniform"] for r in init),
                                    sequences=len({r["sequence"] for r in init})),
        pretraining=READ(OUT / "pretrain_report.json"),
        finetuning=READ(OUT / "finetune_report.json"),
        protocol=str(OUT / "protocol.json"),
        script_sha256=HASH(Path(__file__)),
        protocol_script_sha256=READ(OUT / "protocol.json")["sources"]["scripts/_pdv2_zero_lag_p10.py"],
        aggregation="P1 분석과 같다.", bootstrap=dict(replicates=P.BOOTSTRAP, seed=P.BOOT_SEED),
        test_used_for_selection=False)
    WRITE(OUT / "result.json", result)
    print(json.dumps({g: {a: round(v["AP"]["mean"], 2) for a, v in b["percent"].items()}
                      for g, b in result["groups"].items()}, ensure_ascii=False), flush=True)
    print("P1_CONSISTENCY", result["consistency_with_p1"], flush=True)


# ---------------------------------------------------------------- 사전 기록
def protocol() -> None:
    path = OUT / "protocol.json"
    if path.exists():
        print("PROTOCOL EXISTS", path, flush=True)
        return
    import torch
    import _pdv2_spectral_keys as SK

    assert (P.OUT / "result.json").exists()  # P1을 먼저 끝내고 저장한다
    SK.protocol()
    graph, state = pretrain_graph()
    offsets = (graph.edge_index[0] - graph.edge_index[1]).numpy()
    hashes = {k: SK.array_hash(getattr(graph, k).numpy()) for k in ("x", "edge_index", "edge_attr", "edge_type")}
    counts = {"symmetric": sum(p.numel() for p in SK.model(ARM, 1).parameters())}
    # 전달 칸 대응 검사: 기존 코드의 dense[tgt, src-tgt+5]가 음수 색인을 (오프셋+5) mod 11 칸으로 감는지.
    dense = torch.zeros(len(OFFSETS), 11)
    dense[torch.arange(len(OFFSETS)), torch.as_tensor(OFFSETS) + 5] = 1
    assert torch.equal(dense.argmax(1), torch.as_tensor((OFFSETS + 5) % 11))
    if torch.cuda.is_available():
        dense = torch.zeros(len(OFFSETS), 11, device="cuda")
        dense[torch.arange(len(OFFSETS), device="cuda"), torch.as_tensor(OFFSETS, device="cuda") + 5] = 1
        assert torch.equal(dense.argmax(1).cpu(), torch.as_tensor((OFFSETS + 5) % 11))
    # 학습 그래프 한 개로 P10 변환·초기 출력·인과 범위 검사
    e, _ = Z.modules()
    e.configure()
    s0 = READ(W.RECAL / "protocol.json")["train"][0]
    g0 = torch.load(W.TRAIN_INPUTS / (s0.replace("/", "_") + ".pt"), map_location="cuda", weights_only=False)
    p0 = check_p10_graph(to_p10(g0, e.run.inputs("train", s0)[0]["poses"]))
    W.PRETRAIN = P.PRE / "past"  # 구조 검사용 시작점(P1 사전학습). P10 학습에는 쓰지 않는다.
    net = W.build(torch, e, 1, "diff")
    with torch.no_grad():
        init = float((net(p0) - e.prior.filtered(p0, -0.5)).abs().max())
    causal = W.causal_check(net, p0)
    W.PRETRAIN = P.ORIGINAL_PRETRAIN
    assert init < 4e-6 and causal["future_nodes"] == 0 and causal["reach"] == [-2 * WINDOW, 0], (init, causal)
    finetune_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    del net
    audit = window_audit()
    sources = [Path(__file__), Path(P.__file__), Path(Z.__file__), Path(W.__file__), Path(SK.__file__),
               ROOT / "src/retrieval/sequence_evidence.py", ROOT / "src/gnn/spectral_difference.py",
               ROOT / "src/gnn/spectral_transport.py", ROOT / "src/gnn/contrast_spectral_difference.py",
               ROOT / "src/gnn/learned_spectral_difference.py", ROOT / "src/gnn/model.py",
               ROOT / "scripts/_pdv2_input_alignment_lib.py", P.OUT / "protocol.json", P.OUT / "result.json"]
    WRITE(path, dict(
        created_unix=time.time(),
        purpose="이웃 수를 대칭 NSD(층별 10개)와 맞춘 무지연 과거 전용 NSD(P10). 사전학습·추가 학습·평가 모두 과거 10키프레임. "
                "P1(과거 5개)과 같은 절차로 한 번 실행해, 대칭 NSD와의 차이가 미래 관측의 부재인지 이웃 수 절반인지 분리한다.",
        definition=dict(
            refinement=f"각 키프레임의 시간 이웃 = 같은 시퀀스의 앞선 1..{WINDOW}키프레임(층별 최대 {WINDOW}개, 대칭 NSD와 같은 수). "
                       f"두 층이라 정제 계수 z'_i는 i-{2 * WINDOW}..i에만 의존.",
            edges="간선 (i-d -> i), d=1..10. 간선 속성은 기존 그래프와 같은 pose_attributes(거리 log1p/5, 상대 회전각/π, 0, 0, 1), 같은 자세"
                  "(사전학습: GT, 학습8: 기존 학습 그래프 자세, 시험: KISS-ICP). 기존 그래프의 속성이 같은 함수로 비트 단위 재현됨을 매번 검사.",
            slots="전달 칸 11개(자기 칸 5 제외 10칸)를 재사용: 칸 = (오프셋+5) mod 11 -> -5..-1은 0..4(기존 과거 칸), -10..-6은 6..10"
                  "(기존 미래 칸). transport_difference·ContrastSpectralDifference 코드 수정 없음(음수 색인이 같은 칸으로 감김을 CPU·CUDA에서 검사).",
            parameters=dict(pretraining_model=counts["symmetric"], finetune_trainable=finetune_params,
                            note="간선 부호기는 거리·회전각과 간선 종류만 읽고 오프셋별 매개변수가 없어 매개변수 수와 구조가 대칭 NSD와 같다."),
            temporal_evidence=f"질의 쪽 오프셋 -{WINDOW}..-1(대칭은 -5..-1, +1..+5). 후보 쪽 거리 대응 창 반경 {RADIUS}(기존 10). "
                              "기존 규칙 '창 반경 = 최대 오프셋의 2배'(10 = 2x5)를 그대로 적용한 최소 변경이다. β=11/15, 중복 후보 가중치, "
                              "정규화, 중심 +4/15, 정·역방향 최댓값, 융합 가중치, W, 75:25, 점수 척도는 그대로. 유효 이웃 10개면 중심 1/3.",
            candidate_window_audit=audit,
            query_time="q=t. 후보 쪽 프레임 <= j*+20 <= q-10, 그 정제는 과거만 읽으므로 모두 과거.",
            queries="기존 공통 질의 q=30..n-16(195,726). q>=30이라 q-10, q-20이 모두 시퀀스 안.",
            new_hyperparameters="창 크기 10(지시), 후보 창 반경 20(기존 규칙에서 유도). 그 밖에 없음."),
        pretraining=dict(
            path="P1과 같다(_pdv2_spectral_keys 학습 함수, 20에포크·900갱신, 같은 삼중쌍·순서·손실·스케줄·시드, 결정적 알고리즘 강제 없음). "
                 "base.train의 temporal_subgraph 자리에 P10 그래프를 넣는다.",
            graph_hashes=hashes, edges=int(graph.edge_index.shape[1]),
            offsets={int(k): int(v) for k, v in zip(*np.unique(offsets, return_counts=True))},
            checkpoint="20에포크"),
        finetuning=dict(
            procedure="P1과 같다: W.train('past', k, 'diff'), 512갱신, 같은 표본·손실 척도·학습률. 바꾸는 것: 시작점(P10 사전학습), "
                      "학습 그래프(W.TRAIN_INPUTS -> p10/train_inputs, 기존 학습 그래프의 노드 입력에 P10 간선·칸), 학습 plan(P10 증거), "
                      "W.past_graph -> 검사 함수(칸 6~10을 끄지 않음).",
            structure_check=dict(train_sequence=s0, init_error=init, causal=causal)),
        evaluation=dict(
            cases="zl_p10_diff_s{1,2,3}(P10), zl_p10_fixed(P10 균일 감산 -0.5 = 추가 학습 전 상태), zl_p10_identity(정제 없음 + P10 증거, 보조 참조). "
                  "참조 원장: 대칭 NSD(기존 시험), Z1·Z2(기존 무지연), P1(이 실험 P1).",
            contrasts="P10-대칭, P10-P1, P10-Z2, P10-P10초기, P10-Z2_10(보조), Z2_10-Z2(보조), P1-대칭·P1-Z2(P1 결과와 일치 확인)",
            metrics=list(METRICS), groups=list(GROUPS),
            bootstrap="P1과 같음(도시 안 양성 시퀀스 10,000회, 시드 20261111, 같은 난수 순서)."),
        checks=dict(causality="P10 사례 전부 x 46시퀀스 x 질의 3개 접두 절단(q+1프레임): 반환 장소 같음, 계수 차 <4e-6, plan 같음·q 이후 미참조, 점수 차 <8e-6",
                    initialization="학습 그래프·시험46 그래프에서 P10 시작점 초기화 출력 = P10 균일 감산 -0.5 (<4e-6)",
                    identity_winners="zl_p10_identity의 반환 장소 = 기존 Z2 원장"),
        selection="시험셋으로 고르는 설정 없음. P10은 위 정의대로 한 번 실행.",
        sources={(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)): HASH(p) for p in sources}))
    print("PROTOCOL WRITTEN", path, json.dumps(audit["mean"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("protocol", "pretrain", "pretrain_all", "finetune", "test", "test_worker", "analyze"))
    p.add_argument("--seed", type=int, choices=STARTS)
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "pretrain":
        pretrain_worker(a.seed)
    elif a.action == "test_worker":
        test_worker(a.sequence)
    else:
        globals()[a.action]()
