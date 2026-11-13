#!/usr/bin/env python3
"""학습 8의 최종 과거 지도 오답과 기존 배치·손실 신호를 연결한다.

ring_baseline에서 실행한다. 시드 1의 이미 선택된 10에포크를 진단하며
새 모델 학습·검증 재선택·시험 재평가는 하지 않는다.
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_focus as focus
from _pdv2_matching_mine import TRAIN, cache_path, spec_path
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from _pdv2_training_coverage import prefix_distance
from _pdv2_lag15_campaign import bank_for
from _pdv2_sequence_fixed_test import frozen_scores
from _pdv2_train_failure_lib import (
    batch_geometry, batch_membership, cosine_descent_response,
    normalization_gradient, wrong_place_neighbors,
)
from _train_attention_neighbors import array_hash
from _xs_eval_lib import normalize
from _pr_eval_lib import classify_queries, operating_point, pr_sweep

OUT = REPO / "results/pdv2_train_failure_20261018"
PREPARED = REPO / "results/pdv2_pair_sampling_20261013/prepared.pt"
GRAPH = REPO / "results/pdv2_context_mechanism_20261014/train_graph.pt"
FREEZE = REPO / "results/pdv2_sequence_fixed_test_20261018/freeze.json"
METHOD = "arc_unique_b100"


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def arrays(path: Path) -> dict:
    """수치 원장을 읽는다."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k != "meta"}


def prepare() -> dict:
    """학습 전용 진단 범위와 기존 선택 상수·원천을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = read(path)
        for p, h in spec["sources"].items():
            assert sha256(REPO / p) == h, p
        return spec
    original = focus.protocol()
    frozen = read(FREEZE)
    selected = frozen["seeds"]["1"]
    assert selected["epoch"] == 10 and frozen["selected"]["method"] == METHOD
    assert sha256(REPO / selected["checkpoint"]) == selected["checkpoint_sha256"]
    source = {REPO / p for p in original["runtime_sha256"]}
    source.update((Path(__file__), PREPARED, GRAPH, FREEZE,
                   REPO / selected["checkpoint"], focus.base.RVEC,
                   focus.OUT / "s1/standard_original/batch_order.json",
                   focus.OUT / "s1/standard_original/training_graph.json"))
    for name in ("scripts/_pdv2_train_failure_lib.py", "tests/gnn/test_training_exposure.py",
                 "scripts/_pdv2_training_coverage.py", "scripts/_pdv2_matching_mine.py",
                 "scripts/_pdv2_sequence_fixed_test.py", "scripts/_pdv2_no_context_ablation.py",
                 "scripts/_xs_eval_lib.py", "scripts/_causal_pr_dump.py",
                 "src/retrieval/sequence_evidence.py", "src/retrieval/spectral_evidence.py"):
        source.add(REPO / name)
    source.update(focus.SAMPLES / f"original_epoch{e}.npz" for e in (1, 6))
    for sensor, seq in TRAIN:
        name = sensor + "/" + seq
        metadata = read(spec_path(name).with_suffix(".json"))
        assert sha256(cache_path(name)) == metadata["cache_sha256"]
        assert sha256(spec_path(name)) == metadata["sha256"]
        source.update((cache_path(name), spec_path(name), spec_path(name).with_suffix(".json")))
    spec = {
        "created_unix": time.time(), "seed": 1, "epoch": 10,
        "selected": selected, "train_roster": [s+"/"+q for s, q in TRAIN],
        "query_rule": "30<=q<N-15, same session j<=q-30",
        "scope": "학습 GT 키프레임·시간 그래프·고정 재계량/융합/판정. 최종 과거 검색을 재생하나 KISS-ICP 온라인 성능 평가 아님.",
        "cohorts": "비재방문(min_past>25)의 고정 검증 임계값 오수락/거절, 양성 정답수락, 전체 비재방문. 5~25m는 해석에서 제외.",
        "coverage": "선택된 체크포인트까지 1~10에포크 실제 삼중쌍·배치 순열. 명시 오답, 배치 정답열의 암묵 오답, 반대 방향 exact 쌍을 모두 센다. 같은 오답 장소는 승자5m이내·동일세션·과거후보.",
        "gradient": "선택된 모델 FP32 eval 출력에서 기존 모든 배치를 재평가. InfoNCE logit 기울기 질량과 문맥 출력 독립 변수의 하강 방향. 과거 학습 중 기울기나 실제 파라미터 업데이트 재현 아님.",
        "batch_size": 1024, "accumulation": 4, "temperature": .1,
        "validation_data_evaluated": False, "frozen_validation_constants_used": True,
        "test_data_used": False, "training_updates": 0,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(source)},
    }
    write_json(path, spec)
    print("PREPARED", len(source), flush=True)
    return spec


def keys(spec: dict) -> None:
    """실제 학습 그래프에서 손실용·검색용 출력을 만들고 접두 범위를 확인한다."""
    target = OUT / "keys_complete.json"
    if target.exists():
        for name, expected in read(target)["outputs"].items():
            assert sha256(OUT / name) == expected
        return
    state = torch.load(PREPARED, map_location="cpu", weights_only=False)
    graph = torch.load(GRAPH, map_location="cpu", weights_only=False)
    keep = graph.edge_type == 0
    graph.edge_index, graph.edge_attr, graph.edge_type = (
        graph.edge_index[:, keep], graph.edge_attr[keep], graph.edge_type[keep])
    if hasattr(graph, "edge_pose_label"):
        graph.edge_pose_label = graph.edge_pose_label[keep]
    hashes = {k: array_hash(getattr(graph, k).numpy()) for k in ("x", "edge_index", "edge_attr", "edge_type")}
    assert hashes == read(focus.OUT / "s1/standard_original/training_graph.json")["hashes"]
    np.testing.assert_array_equal(graph.x.numpy(), state["descriptors"])
    seqids = np.asarray(state["sequences"])
    edge = graph.edge_index.numpy()
    assert np.all(seqids[edge[0]] == seqids[edge[1]])
    assert np.abs(edge[0]-edge[1]).max() <= 5
    _, model = focus.load_model(1, "standard_original", 10)
    model.gnn.key_scale = None
    model.gnn.use_key_remetrize = False
    with torch.no_grad():
        raw = model(graph.to("cuda")).float().cpu().numpy()
    assert raw.shape == (72306, 416) and np.isfinite(raw).all()
    deployment = raw.copy()
    deployment[:, :288] = normalize(state["descriptors"] * np.load(focus.base.RVEC)[None])
    deployment = normalize(deployment)
    np.save(OUT / "training_output.npy", raw)
    np.save(OUT / "retrieval_keys.npy", deployment)
    np.savez_compressed(OUT / "training_metadata.npz", positions=state["poses"][:, :3, 3],
                        sequences=seqids, coordinate_groups=state["frames"])
    prefix_checks = []
    from torch_geometric.data import Data
    for ordinal, (sensor, seq) in enumerate(TRAIN):
        ids = np.flatnonzero(seqids == ordinal)
        assert np.all(np.diff(ids) == 1)
        with np.load(cache_path(sensor+"/"+seq)) as cache:
            np.testing.assert_array_equal(cache["descriptors"], state["descriptors"][ids])
            np.testing.assert_array_equal(cache["poses"], state["poses"][ids])
        for local in (len(ids)//3, 2*len(ids)//3):
            query = ids[local]
            limit = query + 16
            mask = (graph.edge_index[0] < limit) & (graph.edge_index[1] < limit)
            small = Data(x=graph.x[:limit], edge_index=graph.edge_index[:, mask],
                         edge_attr=graph.edge_attr[mask], edge_type=graph.edge_type[mask])
            with torch.no_grad():
                value = model(small)[query].float().cpu().numpy()
            error = float(np.max(np.abs(value - raw[query])))
            np.testing.assert_allclose(value, raw[query], atol=3e-6, rtol=3e-5)
            prefix_checks.append({"sequence": sensor+"/"+seq, "query": int(local), "max_error": error})
    model.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(focus.base.RVEC)).float().cuda(), requires_grad=False)
    with torch.no_grad():
        direct = normalize(model(graph).float().cpu().numpy())
    np.testing.assert_allclose(direct, deployment, atol=3e-7, rtol=3e-6)
    write_json(target, {"graph_hashes": hashes, "prefix_checks": prefix_checks,
        "remetrization_max_error": float(np.max(np.abs(direct-deployment))),
        "outputs": {p.name: sha256(p) for p in (OUT/"training_output.npy", OUT/"retrieval_keys.npy", OUT/"training_metadata.npz")}})
    print("KEYS_DONE", len(raw), flush=True)


def replay(sequence: str, spec: dict) -> None:
    """학습 한 시퀀스의 기존 최종 검색과 선택 판정을 그대로 재생한다."""
    name = sequence.replace("/", "_")
    target = OUT / f"replay_{name}.json"
    if target.exists():
        assert sha256(OUT / f"ledger_{name}.npz") == read(target)["ledger_sha256"]
        return
    started = time.time()
    meta = arrays(OUT / "training_metadata.npz")
    ordinal = spec["train_roster"].index(sequence)
    ids = np.flatnonzero(meta["sequences"] == ordinal)
    embedding = np.load(OUT / "retrieval_keys.npy", mmap_mode="r")[ids]
    positions = meta["positions"][ids].astype(np.float64)
    min_past = prefix_distance(positions)
    conf = spec["selected"]["confidence"]
    with np.load(spec_path(sequence)) as z, np.load(cache_path(sequence)) as cache:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = bank_for("pbev128_main", z)
    data = evaluate_keys(embedding, bank, positions, min_past,
                         {1: tuple(float(x) for x in conf["tuple"])})[1]
    use = data["query_idx"] < len(ids)-15
    data = {k: v[use] for k, v in data.items()}
    q, j = data["query_idx"], data["j_star"]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    chunks = []
    for start in range(0, len(q), 256):
        sl = slice(start, start+256)
        chunks.append(frozen_scores(bank, travel, q[sl], j[sl], conf["decision"], data[conf["decision"]][sl])[METHOD])
    data[METHOD] = np.concatenate(chunks)
    data["global_query"] = q + ids[0]
    data["global_winner"] = j + ids[0]
    assert np.all(j <= q-30)
    checks = []
    from _causal_pr_dump import dump_op
    # Canonical full evaluation on a short actual prefix checks the copied evaluator.
    limit = min(160, len(ids))
    cut_bank = (bank[0][:limit], bank[1][:limit], bank[2][:limit], tuple(x[:limit] for x in bank[3]))
    exact = dump_op(embedding[:limit], *cut_bank, positions[:limit], min_past[:limit], tuple(conf["tuple"]), 100)
    for key, value in exact.items():
        np.testing.assert_allclose(data[key][:len(value)], value, atol=1e-12, rtol=0)
    for row in (len(q)//3, 2*len(q)//3):
        query = int(q[row])
        distance = np.linalg.norm(positions[:query-29]-positions[query], axis=1).min()
        np.testing.assert_allclose(distance, data["min_past_gt_dist"][row], atol=1e-12)
        end = query+16
        b = (bank[0][:end], bank[1][:end], bank[2][:end], tuple(x[:end] for x in bank[3]))
        small = frozen_scores(b, travel[:end], q[row:row+1], j[row:row+1], conf["decision"], data[conf["decision"]][row:row+1])[METHOD]
        np.testing.assert_array_equal(small, data[METHOD][row:row+1])
        checks.append(query)
    qc = classify_queries(data["min_past_gt_dist"])
    curve = pr_sweep(data[METHOD], data["top1_geo_dist"], qc)
    tau = spec["selected"]["new_tau"]["tau"]
    accepted = data[METHOD] >= tau
    counts = {"queries": len(q), "positive": int(qc["positive"].sum()), "negative": int(qc["negative"].sum()),
              "negative_false_accept": int((qc["negative"] & accepted).sum()),
              "positive_correct_accept": int((qc["positive"] & accepted & (data["top1_geo_dist"]<=5)).sum())}
    path = OUT / f"ledger_{name}.npz"
    np.savez_compressed(path, **data)
    write_json(target, {"sequence": sequence, "runtime_s": time.time()-started, "counts": counts,
        "metrics": {k: float(curve[k]) for k in ("AP", "max_F1", "R@100P")},
        "tau": tau, "prefix_queries": checks, "canonical_prefix_queries": len(exact["query_idx"]),
        "ledger_sha256": sha256(path)})
    print("REPLAY_DONE", sequence, counts, flush=True)


def distribution(values: np.ndarray) -> dict:
    """유한한 표본의 평균과 분위수를 기록한다."""
    v = np.asarray(values)
    assert np.isfinite(v).all()
    return ({"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
             "p10": float(np.quantile(v, .1)), "p90": float(np.quantile(v, .9))} if len(v) else {"n": 0})


def exposure(spec: dict) -> None:
    """실제 배치 전체의 직접·근처 오답 노출과 고정 모델 손실 압력을 센다."""
    if (OUT / "exposure_complete.json").exists():
        for p, h in read(OUT / "exposure_complete.json")["outputs"].items():
            assert sha256(OUT/p) == h
        return
    started = time.time()
    metadata = arrays(OUT / "training_metadata.npz")
    pos, seq = metadata["positions"], metadata["sequences"]
    total = len(pos)
    winner = np.full(total, -1, np.int64)
    distance, confidence = np.full(total, np.nan), np.full(total, np.nan)
    correct = np.zeros(total, bool)
    for sequence in spec["train_roster"]:
        data = arrays(OUT / ("ledger_"+sequence.replace("/", "_")+".npz"))
        q = data["global_query"]
        winner[q], distance[q], confidence[q] = data["global_winner"], data["min_past_gt_dist"], data[METHOD]
        correct[q] = data["top1_geo_dist"] <= 5
    # Exposure neighborhoods for non-revisits only; every selected pair is a genuine negative.
    wrong = np.where(distance > 25, winner, -1)
    nearby = wrong_place_neighbors(pos, seq, wrong)
    names = ("anchors", "hard_exact", "batch_exact", "hard_nearby", "batch_nearby", "reverse_exact")
    counts = {k: np.zeros(total, np.int64) for k in names}
    masses = {k: np.zeros(total, np.float64) for k in ("negative", "exact", "nearby", "hard", "loss")}
    norms = torch.from_numpy(np.load(OUT / "training_output.npy")).cuda().norm(dim=1, keepdim=True)
    raw = torch.from_numpy(np.load(OUT / "training_output.npy")).cuda()
    unit = F.normalize(raw, dim=1)
    grad = torch.zeros_like(unit)
    expected_order = read(focus.OUT / "s1/standard_original/batch_order.json")["epochs"]
    epoch_records = []
    for epoch in range(1, 11):
        refresh = 1 if epoch <= 5 else 6
        with np.load(focus.SAMPLES / f"original_epoch{refresh}.npz") as z:
            samples = z["triplets"]
        order = np.arange(len(samples))
        np.random.RandomState(100000+epoch).shuffle(order)
        samples = samples[order]
        assert array_hash(order) == expected_order[epoch-1]["permutation_sha256"]
        assert array_hash(samples) == expected_order[epoch-1]["sha256"]
        batch_count = (len(samples)+1023)//1024
        for number, start in enumerate(range(0, len(samples), 1024)):
            batch = samples[start:start+1024]
            a, p, n = batch.T
            mask = batch_membership(batch, wrong, nearby)
            # Each loss is a batch mean, then averaged over its accumulation window.
            window = min(4, batch_count-4*(number//4))
            scale = 1./(len(a)*window)
            with torch.no_grad():
                loss, prob, ga, gp, gn = batch_geometry(unit, torch.from_numpy(batch).cuda())
                for ids, g in ((a, ga), (p, gp), (n, gn)):
                    grad.index_add_(0, torch.from_numpy(ids).cuda(), g*scale)
                probs = prob.cpu().numpy()
                losses = loss.cpu().numpy()
            diag = np.arange(len(a))
            neg_mass = 1-probs[diag, diag]
            exact_mass = (probs[:, :-1]*mask["exact"]).sum(1) + probs[:, -1]*mask["hard_exact"]
            nearby_mass = (probs[:, :-1]*mask["nearby"]).sum(1) + probs[:, -1]*mask["hard_nearby"]
            for key, values in (("anchors", np.ones(len(a), np.int64)), ("hard_exact", mask["hard_exact"]),
                                ("batch_exact", mask["exact"].sum(1)), ("hard_nearby", mask["hard_nearby"]),
                                ("batch_nearby", mask["nearby"].sum(1))):
                np.add.at(counts[key], a, values)
            np.add.at(counts["reverse_exact"], p, mask["reverse"].sum(0))
            np.add.at(counts["reverse_exact"], n, mask["hard_reverse"])
            for key, value in (("negative", neg_mass), ("exact", exact_mass), ("nearby", nearby_mass),
                               ("hard", probs[:, -1]), ("loss", losses)):
                np.add.at(masses[key], a, value*scale)
            if number == 0:
                from gnn.trainer import InfoNCELoss
                tb = torch.from_numpy(batch).cuda()
                torch.testing.assert_close(torch.as_tensor(losses.mean(), device="cuda"),
                    InfoNCELoss(.1)(raw[tb[:, 0]], raw[tb[:, 1]], raw[tb[:, 2]]), atol=2e-6, rtol=2e-6)
        record = {"epoch": epoch, "batches": batch_count, "triplets": len(samples),
                  "sample_order_hash": array_hash(samples)}
        epoch_records.append(record)
        print("EXPOSURE_EPOCH", epoch, flush=True)
    gradient = normalization_gradient(unit, norms, grad)
    q = np.flatnonzero(wrong >= 0)
    response = np.full(total, np.nan)
    response[q] = cosine_descent_response(raw, gradient, torch.from_numpy(q).cuda(),
                                          torch.from_numpy(wrong[q]).cuda()).cpu().numpy()
    tau = spec["selected"]["new_tau"]["tau"]
    negative = distance > 25
    cohort = np.full(total, -1, np.int8)
    cohort[negative & (confidence >= tau)] = 1
    cohort[negative & (confidence < tau)] = 0
    cohort[(distance <= 5) & correct & (confidence >= tau)] = 2
    path = OUT / "exposure_arrays.npz"
    np.savez_compressed(path, winner=winner, min_past_gt_dist=distance, confidence=confidence,
        sequence=seq, cohort=cohort, context_descent_response=response,
        **{f"count_{k}": v for k, v in counts.items()}, **{f"mass_{k}": v for k, v in masses.items()})
    np.save(OUT / "training_output_gradient.npy", gradient.cpu().numpy())
    write_json(OUT / "exposure_complete.json", {"status": "passed", "runtime_s": time.time()-started,
        "epochs": epoch_records, "nearby_edges": int(nearby.nnz),
        "outputs": {p.name: sha256(p) for p in (path, OUT/"training_output_gradient.npy")}})


def report(spec: dict) -> None:
    """프레임 포함·오답 쌍 노출·손실 질량·자료별 결과를 분리한다."""
    a = arrays(OUT / "exposure_arrays.npz")
    cohorts = {"false_accept": a["cohort"] == 1, "negative_reject": a["cohort"] == 0,
               "all_negative": (a["cohort"] == 0) | (a["cohort"] == 1)}
    groups = {"all": np.ones(len(a["winner"]), bool), "kitti": a["sequence"] < 4, "helipr": a["sequence"] >= 4}
    groups.update({seq: a["sequence"] == i for i, seq in enumerate(spec["train_roster"])})
    reports = {}
    for name, group in groups.items():
        reports[name] = {}
        for cohort, selection in cohorts.items():
            mask = group & selection
            n = int(mask.sum())
            if not n:
                reports[name][cohort] = {"n": 0}
                continue
            trained = mask & (a["count_anchors"] > 0)
            exact = a["count_hard_exact"]+a["count_batch_exact"]
            near = a["count_hard_nearby"]+a["count_batch_nearby"]
            shares = {k: a[f"mass_{k}"][trained]/a["mass_negative"][trained] for k in ("exact", "nearby", "hard")}
            value = {"n": n, "anchor_seen": int(trained.sum()),
                "explicit_exact_seen": int((mask & (a["count_hard_exact"] > 0)).sum()),
                "batch_exact_seen": int((mask & (a["count_batch_exact"] > 0)).sum()),
                "exact_any_forward_seen": int((mask & (exact > 0)).sum()),
                "exact_either_direction_seen": int((mask & ((exact+a["count_reverse_exact"]) > 0)).sum()),
                "nearby_forward_seen": int((mask & (near > 0)).sum()),
                "logit_gradient_share": {k: distribution(v) for k, v in shares.items()},
                "context_descent_reduces_wrong_similarity": int((mask & (a["context_descent_response"] < 0)).sum()),
                "context_descent_response": distribution(a["context_descent_response"][mask])}
            reports[name][cohort] = value
    replays = {s: read(OUT / ("replay_"+s.replace("/", "_")+".json")) for s in spec["train_roster"]}
    write_json(OUT / "summary.json", {"seed": 1, "epoch": 10, "cohorts": reports, "replay": replays,
        "scope": spec["scope"], "gradient_scope": spec["gradient"], "test_used": False,
        "new_training": False, "protocol_sha256": sha256(OUT / "protocol.json")})
    for name in ("all", "kitti", "helipr"):
        print("SUMMARY", name, reports[name], flush=True)


def main() -> None:
    """入力을 고정하고 키·과거 검색·학습 노출 진단을 순서대로 실행한다."""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    spec = prepare()
    write_json(OUT / "status.json", {"status": "running", "stage": "keys"})
    keys(spec)
    torch.cuda.empty_cache()
    write_json(OUT / "status.json", {"status": "running", "stage": "train_replay"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda seq: replay(seq, spec), spec["train_roster"]))
    write_json(OUT / "status.json", {"status": "running", "stage": "training_exposure"})
    exposure(spec)
    report(spec)
    prepare()
    write_json(OUT / "status.json", {"status": "complete", "runtime_s": time.time()-started})


if __name__ == "__main__":
    main()
