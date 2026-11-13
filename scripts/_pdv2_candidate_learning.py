#!/usr/bin/env python3
"""후보 목록·거절 손실의 표현 고정/공동 학습 대조를 실행한다."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_candidate_data import OUT, load_data, train_path
from _pdv2_pilot import TRAIN, VAL, FAMILY, OPS
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_lag15_campaign import bank_for, model as reference_model, RVEC, classifier_path
from _pdv2_pair_evaluate import direction_groups
from _xs_eval_lib import normalize, align_scores, column_align_scores, minmax
from _causal_pr_dump import prefix_topk
from _pr_eval_lib import classify_queries, pr_sweep, operating_point
from retrieval.candidate_learning import CandidateScorer, candidate_loss, pair_features, FEATURE_NAMES

CONFIG = REPO / "configs/pdv2_candidate_learning.yaml"
BASE = REPO / "results/pdv2_symmetric3_20261013"
PILOT = REPO / "results/pdv2_matching_mining_20261014"
CFG = yaml.safe_load(CONFIG.read_text())
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def setup() -> None:
    """한 프로세스의 수치·난수 설정을 고정한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])


def protocol(create: bool = False) -> dict:
    """학습 시작 전에 코드·자료·예산·선택 규칙을 동결한다."""
    path = OUT / "protocol.json"
    if path.exists():
        result = read(path)
        for name, digest in result["code_sha256"].items():
            assert sha256(REPO / name) == digest, name
        return result
    assert create
    files = [CONFIG, Path(__file__), REPO / "scripts/_pdv2_candidate_data.py",
             REPO / "scripts/_pdv2_candidate_run.py", REPO / "scripts/_pdv2_candidate_verify.py",
             REPO / "src/retrieval/candidate_learning.py", REPO / "tests/test_candidate_learning.py",
             REPO / "src/gnn/model.py", REPO / "src/gnn/graph_free.py",
             REPO / "src/keyframe/graph_manager.py", REPO / "scripts/_lag_branch_fusion.py",
             REPO / "scripts/_causal_edge_features.py", REPO / "scripts/_xs_eval_lib.py",
             REPO / "scripts/_pr_eval_lib.py"]
    inputs = [REPO / CFG["initial_gnn"], REPO / CFG["initial_point"], RVEC,
              classifier_path(1), BASE / "s1/confidence_selection.json"]
    for sensor, seq in TRAIN:
        p = train_path(sensor + "/" + seq)
        assert p.exists() and read(p.with_suffix(".json"))["sha256"] == sha256(p)
        inputs.append(p)
    result = {"created_unix": time.time(), "config": CFG, "scope": "시드1 학습8·검증9만 쓰는 예비 실험. 시험 실행 없음.",
              "train_sequences": [s + "/" + q for s, q in TRAIN], "val_sequences": [s + "/" + q for s, q in VAL],
              "features": list(FEATURE_NAMES),
              "training_inputs": "KISS-ICP의 인과 주행 추정과 같은 키프레임 선택. GT는 정답·질의 표본 선택에만 사용하며 그래프/점수 입력에서 분리.",
              "model": "같은 기존20에포크 실험의10에포크 GNN/이웃없는투영에서 시작. 두 모델 모두 같은 초기 판정기. head는 표현고정, joint는 표현과판정기 공동학습. NSD/NSD-H 판정기는 별도, 표현은 공유.",
              "online": "q+15까지 입력, 후보j<=q-30. GNN 두통과·간선재선택은 현재 추론과 동일. BN통계 고정·dropout비활성(eval모드)으로 전체시퀀스 통계누출 방지. 표현학습은 두번째 통과에 역전파하며 이산 후보/간선 선택과 간선 특징은 미분하지 않음.",
              "refresh": "joint는1/5/9에포크 시작에 전체추론으로 간선과100+100후보를 다시 계산, 다음4에포크 고정. head는처음후보고정. 평가때는선택체크포인트로전체재계산. 학습과추론의교체주기차이는근사임.",
              "loss": "후보 없음 logit0을 포함한 여러정답 확률합의 음의로그 +0.5*질의별 양성/음성 균등 BCE. 정답<=5m, 비재방문질의>25m, 모호질의제외. 후보안에정답없는양성질의는후보없음으로학습. 정답을후보에강제삽입하지않음.",
              "budget": "12에포크·학습시퀀스당양성/음성각최대1024질의고정·네조건공통순서. AdamW·head1e-3/encoder5e-5·노름1클리핑. 네조건각12에포크. 파라미터수대조아님.",
              "selection": "4/8/12에포크의최종검증AP를시퀀스→3계열→NSD/NSD-H평균. 최고값·1e-12동률이면이른것. 판정점수는학습한최대후보logit으로고정, 임계값만검증계열평균F1최대로선택. 양성없는KITTI09는AP평균제외·오수락별도.",
              "interpretation": "새head효과와표현추가학습효과를분리. 기존동일초기모델의고정융합결과도병기. 이번결과로원고표교체·시험튜닝하지않음.",
              "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in files},
              "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in inputs}}
    write_json(path, result)
    return result


def make_encoder(kind: str) -> tuple:
    """현재 인코더·게이트·재계량과 고정된 시작 가중치를 적용한다."""
    from gnn.graph_free import create_graph_free_model
    cfg, net, clf, blob, rvec, _ = reference_model(1)
    if kind == "point":
        net = create_graph_free_model(input_dim=288, hidden_dim=128, context_dim=128, n_layers=2,
                    n_heads=4, dropout=.1, norm_type="batch_norm", use_residual_gate=True,
                    gate_hidden_dim=64, gate_initial_alpha=.0625,
                    key_remetrize_config={"enabled": True, "learnable": False, "init_path": str(RVEC)}).cuda()
    state = torch.load(REPO / CFG["initial_" + kind], map_location="cuda", weights_only=False)["model_state_dict"]
    missing, extra = net.load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra
    assert cfg["gnn"]["input_dim"] == 288 and cfg["gnn"]["use_residual_gate"]
    net.eval()
    return cfg, net, clf, blob, rvec


def graph_and_embedding(kind: str, cache, layout, bundle: tuple) -> tuple:
    """추론과 같은 두 통과를 수행하고 공동 학습용 두번째 그래프를 보존한다."""
    from torch_geometric.data import Data
    from _lag_branch_fusion import nsd_embedding_lag
    cfg, net, clf, blob, rvec = bundle
    if kind == "point":
        graph = Data(x=torch.from_numpy(cache["descriptors"]).float().cuda(),
                     edge_index=torch.empty((2, 0), dtype=torch.long, device="cuda"))
        with torch.no_grad():
            emb = F.normalize(net(graph), dim=1).cpu().numpy()
        return graph, emb
    captured = []

    class Capture(torch.nn.Module):
        def forward(self, graph):
            captured.append(graph)
            return net(graph)

    emb = nsd_embedding_lag(cache, cache["sensor_name"], cache["descriptors"], layout,
                            cfg, Capture(), clf, blob, rvec, "cuda")
    assert len(captured) == 2
    return captured[1], emb


def nearest_past(positions: np.ndarray) -> np.ndarray:
    """GT의 과거 지도 최근접 거리를 정답 작성에만 사용한다."""
    from scipy.spatial import cKDTree
    result = np.full(len(positions), np.nan)
    for start in range(30, len(positions), 256):
        end = min(start + 256, len(positions))
        prefix = start - 29
        distance, _ = cKDTree(positions[:prefix]).query(positions[start:end])
        for row, q in enumerate(range(start, end)):
            extra = positions[prefix:q - 29]
            result[q] = min(distance[row], np.linalg.norm(extra - positions[q], axis=1).min()) if len(extra) else distance[row]
    return result


def query_roster(role: str, sequence: str, gt, min_past) -> tuple:
    """네 조건이 공유하는 질의와 평가용 방향 라벨을 만든다."""
    positions = gt[:, :3, 3]
    if min_past is None:
        min_past = nearest_past(positions)
    q = np.arange(30, len(positions) - 15, dtype=np.int64)
    if role == "train":
        index = [s + "/" + seq for s, seq in TRAIN].index(sequence)
        rng = np.random.default_rng(1000 + index)
        selected = []
        for mask in (min_past[q] <= 5, min_past[q] > 25):
            subset = q[mask]
            selected.extend(rng.choice(subset, min(len(subset), CFG["queries_per_class_per_sequence"]), replace=False))
        q = np.sort(np.asarray(selected, dtype=np.int64))
    return q, min_past, direction_groups(positions, min_past)


def build_pool(embedding, bank, queries, positions) -> dict:
    """모델이 실제 검색한100+100후보만 보존하고 별도 정답을 붙인다."""
    hkey, hfft, hnorm, column = bank
    candidates = np.full((len(queries), 200), -1, np.int64)
    static = np.zeros((len(queries), 200, 3), np.float64)
    distances = np.full((len(queries), 200), np.inf, np.float64)
    for i, q in enumerate(queries):
        a = prefix_topk(embedding, int(q), int(q) - 29, 100)
        b = prefix_topk(hkey, int(q), int(q) - 29, 100)
        union = np.union1d(a, b)
        assert len(union) <= 200 and union.max() <= q - 30
        candidates[i, :len(union)] = union
        static[i, :len(union)] = np.stack([hkey[union] @ hkey[q],
                 align_scores(hfft, hnorm, int(q), union), column_align_scores(column, int(q), union)], axis=1)
        distances[i, :len(union)] = np.linalg.norm(positions[union] - positions[q], axis=1)
        if i and i % 4000 == 0:
            print("POOL_QUERIES", i, len(queries), flush=True)
    return {"queries": queries, "candidates": candidates, "static": static,
            "correct": distances <= 5., "distances": distances}


def snapshot(kind: str, tag: str, role: str, sequence: str, bundle: tuple) -> Path:
    """현재 표현의 그래프·키·실제 후보 목록을 학습 또는 평가용으로 저장한다."""
    dest = OUT / "pools" / tag / role / sequence.replace("/", "_")
    marker = dest / "complete.json"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print("POOL_START", tag, role, sequence, flush=True)
    cache, layout, spec, gt, min_past, paths = load_data(role, sequence)
    cache["sensor_name"] = sequence.split("/")[0]
    # 정답 위치를 그래프·인코더에 넘기지 않는다.
    assert "poses_gt" not in cache
    if role == "val" and tag.endswith("_initial"):
        source = PILOT / f"s1/{kind}_original/epoch10/keys_val_{sequence.replace('/', '_')}.npz"
        with np.load(source) as z:
            embedding = z["embedding"]
            np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        paths["initial_keys"] = source
    else:
        graph, embedding = graph_and_embedding(kind, cache, layout, bundle)
        if role == "train":
            torch.save(graph.cpu(), dest / "graph.pt")
        del graph
    queries, min_past, groups = query_roster(role, sequence, gt, min_past)
    np.savez_compressed(dest / "keys.npz", embedding=embedding, scan_ids=cache["scan_ids"])
    for op in OPS:
        pool = build_pool(embedding, bank_for(op, spec), queries, gt[:, :3, 3])
        np.savez_compressed(dest / (op + ".npz"), **pool,
                            min_past_gt_dist=min_past[queries], group=groups[queries])
    write_json(marker, {"role": role, "sequence": sequence, "n_queries": len(queries),
               "n_positive": int((min_past[queries] <= 5).sum()),
               "n_negative": int((min_past[queries] > 25).sum()),
               "sources": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()},
               "protocol_sha256": sha256(OUT / "protocol.json"),
               "files": {p.name: sha256(p) for p in dest.iterdir() if p.suffix in (".npz", ".pt")}})
    print("POOL_DONE", tag, role, sequence, len(queries), flush=True)
    return dest


def load_pool(path: Path, op: str, device: str = "cuda") -> dict:
    """작은 학습 후보 묶음을 텐서로 읽는다."""
    with np.load(path / (op + ".npz")) as z:
        result = {k: torch.from_numpy(z[k]).to(device) for k in ("queries", "candidates", "static", "correct")}
        result["static"] = result["static"].float()
        return result


def prepare_pools(kind: str) -> None:
    """처음 표현의 학습·검증 후보를 공통으로 준비한다."""
    protocol()
    bundle = make_encoder(kind)
    for role, roster in (("train", TRAIN), ("val", VAL)):
        for sensor, seq in roster:
            snapshot(kind, kind + "_initial", role, sensor + "/" + seq, bundle)


def train(arm: str) -> None:
    """같은 예산으로 판정기만 또는 인코더까지 최종 과제로 학습한다."""
    protocol()
    assert arm in CFG["arms"]
    dest = OUT / arm
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "training_complete.json").exists():
        return
    assert not list(dest.glob("epoch*.pth")), "부분 학습 체크포인트가 있으면 자동 덮어쓰지 않는다"
    kind, mode = arm.split("_")
    bundle = make_encoder(kind)
    net = bundle[1]
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    for name, p in net.named_parameters():
        p.requires_grad_(mode == "joint" and not name.endswith("key_scale"))
    torch.manual_seed(1001)
    heads = torch.nn.ModuleDict({op: CandidateScorer(tuple(CFG["head_hidden"])) for op in OPS}).cuda()
    torch.save(heads.state_dict(), dest / "initial_heads.pth")
    parameters = [{"params": heads.parameters(), "lr": CFG["head_learning_rate"]}]
    if mode == "joint":
        parameters.append({"params": net.parameters(), "lr": CFG["encoder_learning_rate"]})
    optimizer = torch.optim.AdamW(parameters, weight_decay=CFG["weight_decay"])
    updates, history = 0, []
    started = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        losses = []
        refresh = max(e for e in CFG["candidate_refresh_epochs"] if e <= epoch)
        tag = kind + "_initial" if mode == "head" or refresh == 1 else f"{arm}_refresh{refresh}"
        for sequence_index, (sensor, seq) in enumerate(TRAIN):
            sequence = sensor + "/" + seq
            path = snapshot(kind, tag, "train", sequence, bundle)
            pools = {op: load_pool(path, op) for op in OPS}
            np.testing.assert_array_equal(pools[OPS[0]]["queries"].cpu(), pools[OPS[1]]["queries"].cpu())
            n = len(pools[OPS[0]]["queries"])
            order = np.random.default_rng(epoch * 100 + sequence_index).permutation(n)
            if mode == "head":
                with np.load(path / "keys.npz") as z:
                    frozen_embedding = torch.from_numpy(z["embedding"]).cuda()
            else:
                graph = torch.load(path / "graph.pt", map_location="cuda", weights_only=False)
            for start in range(0, n, CFG["queries_per_forward"]):
                rows = order[start:start + CFG["queries_per_forward"]]
                optimizer.zero_grad(set_to_none=True)
                embedding = frozen_embedding if mode == "head" else F.normalize(net(graph), dim=1)
                total = torch.zeros((), device="cuda")
                for op, pool in pools.items():
                    for offset in range(0, len(rows), CFG["batch_queries"]):
                        ix = torch.as_tensor(rows[offset:offset + CFG["batch_queries"]], device="cuda")
                        candidates = pool["candidates"][ix]
                        valid = candidates >= 0
                        features = pair_features(embedding, pool["queries"][ix], candidates, pool["static"][ix], valid)
                        value = candidate_loss(heads[op](features), pool["correct"][ix], valid, CFG["pair_bce_weight"])
                        total = total + value * len(ix) / len(rows) / len(OPS)
                assert torch.isfinite(total)
                total.backward()
                gradnorm = torch.nn.utils.clip_grad_norm_(list(heads.parameters()) + list(net.parameters()), CFG["gradient_clip"])
                assert torch.isfinite(gradnorm)
                optimizer.step()
                updates += 1
                losses.append(float(total.detach()))
            del pools
            if mode == "joint":
                del graph
            else:
                del frozen_embedding
        row = {"epoch": epoch, "updates": updates, "loss": float(np.mean(losses)), "runtime_s": time.time() - started}
        history.append(row)
        write_json(dest / "training_progress.json", row)
        print("EPOCH", arm, row, flush=True)
        if epoch in CFG["checkpoint_epochs"]:
            torch.save({"encoder": net.state_dict(), "heads": heads.state_dict(), "epoch": epoch,
                        "updates": updates, "protocol_sha256": sha256(OUT / "protocol.json")}, dest / f"epoch{epoch}.pth")
    delta = {k: float((v.cpu().float() - initial[k].float()).abs().max()) for k, v in net.state_dict().items()}
    assert (max(delta.values()) == 0) if mode == "head" else (max(delta.values()) > 0)
    for key in initial:
        if "running_" in key or "num_batches_tracked" in key or key.endswith("key_scale"):
            assert delta[key] == 0, key
    write_json(dest / "training_complete.json", {"history": history, "encoder_max_change": max(delta.values()),
               "batch_norm_frozen": True, "updates": updates,
               "checkpoints": {p.name: sha256(p) for p in dest.glob("epoch*.pth")}})


def evaluate(arm: str, epoch: int, sequence: str) -> None:
    """모든 검증 질의에서 최종 선택과 학습한 판정 점수를 기록한다."""
    protocol()
    kind, mode = arm.split("_")
    checkpoint = OUT / arm / f"epoch{epoch}.pth"
    completed = read(OUT / arm / "training_complete.json")
    assert completed["checkpoints"][checkpoint.name] == sha256(checkpoint)
    dest = OUT / arm / f"val_epoch{epoch}"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / (sequence.replace("/", "_") + ".json")
    if marker.exists():
        return
    bundle = make_encoder(kind)
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    assert state["protocol_sha256"] == sha256(OUT / "protocol.json")
    bundle[1].load_state_dict(state["encoder"], strict=True)
    heads = torch.nn.ModuleDict({op: CandidateScorer(tuple(CFG["head_hidden"])) for op in OPS}).cuda().eval()
    heads.load_state_dict(state["heads"], strict=True)
    tag = kind + "_initial" if mode == "head" else f"{arm}_epoch{epoch}"
    path = snapshot(kind, tag, "val", sequence, bundle)
    with np.load(path / "keys.npz") as z:
        embedding = torch.from_numpy(z["embedding"]).cuda()
    result = {"sequence": sequence, "arm": arm, "epoch": epoch, "ops": {}}
    for op in OPS:
        with np.load(path / (op + ".npz")) as z:
            arrays = {k: z[k] for k in z.files}
        n = len(arrays["queries"])
        selected, confidence = [], []
        with torch.no_grad():
            for start in range(0, n, 128):
                sl = slice(start, start + 128)
                q, c, static = [torch.from_numpy(arrays[k][sl]).cuda() for k in ("queries", "candidates", "static")]
                static = static.float()
                valid = c >= 0
                features = pair_features(embedding, q, c, static, valid)
                logits = heads[op](features).masked_fill(~valid, -float("inf"))
                score, index = logits.max(dim=1)
                selected.extend(index.cpu().tolist())
                confidence.extend(score.cpu().tolist())
        chosen = np.asarray(selected)
        rows = np.arange(n)
        ledger = {"query_idx": arrays["queries"], "min_past_gt_dist": arrays["min_past_gt_dist"],
                  "group": arrays["group"], "j_star": arrays["candidates"][rows, chosen],
                  "top1_geo_dist": arrays["distances"][rows, chosen],
                  "score_learned": np.asarray(confidence), "n_candidates": (arrays["candidates"] >= 0).sum(1)}
        assert (ledger["j_star"] <= ledger["query_idx"] - 30).all()
        assert all(np.isfinite(v).all() for v in ledger.values())
        qc = classify_queries(ledger["min_past_gt_dist"])
        metrics = None
        if qc["n_pos"]:
            sweep = pr_sweep(ledger["score_learned"], ledger["top1_geo_dist"], qc)
            metrics = {k: float(sweep[k]) for k in METRICS[:3]}
            metrics["R@1"] = float((ledger["top1_geo_dist"][qc["positive"]] <= 5).mean())
        target = dest / (sequence.replace("/", "_") + "_" + op + ".npz")
        np.savez_compressed(target, **ledger)
        result["ops"][op] = {"metrics": metrics, "ledger": str(target.relative_to(REPO)), "sha256": sha256(target)}
    result.update(checkpoint_sha256=sha256(checkpoint), protocol_sha256=sha256(OUT / "protocol.json"))
    write_json(marker, result)
    print("EVAL_DONE", arm, epoch, sequence, flush=True)


def report() -> None:
    """최종 검증 AP로 선택하고 고정 임계값과 방향별 결과를 함께 기록한다."""
    from _pdv2_c3 import _family_macro_tau_star
    protocol()
    result = {"scope": "시드1 검증9의 최종 판정 지표. 시험 실행·원고 교체 없음.", "arms": {},
              "protocol_sha256": sha256(OUT / "protocol.json")}
    for arm in CFG["arms"]:
        candidates = []
        for epoch in CFG["checkpoint_epochs"]:
            summaries = [read(OUT / arm / f"val_epoch{epoch}" / f"{s}_{q}.json") for s, q in VAL]
            ops = {}
            for op in OPS:
                groups = {}
                for row in summaries:
                    if row["ops"][op]["metrics"] is not None:
                        sensor = row["sequence"].split("/")[0]
                        groups.setdefault(FAMILY.get(sensor, sensor), []).append(row["ops"][op]["metrics"])
                ops[op] = {k: float(np.mean([np.mean([r[k] for r in g]) for g in groups.values()])) for k in METRICS}
            candidates.append({"epoch": epoch, "ops": ops, "joint_AP": float(np.mean([r["AP"] for r in ops.values()]))})
        best = max(r["joint_AP"] for r in candidates)
        selected = min((r for r in candidates if best - r["joint_AP"] <= 1e-12), key=lambda r: r["epoch"])
        selected["candidates"] = copy.deepcopy(candidates)
        selected["decision"] = {}
        for op in OPS:
            sweeps, groups, ledgers = {}, {}, {}
            for sensor, seq in VAL:
                sequence = sensor + "/" + seq
                p = OUT / arm / f"val_epoch{selected['epoch']}" / f"{sensor}_{seq}_{op}.npz"
                with np.load(p) as z:
                    arrays = {k: z[k] for k in z.files}
                ledgers[sequence] = arrays
                qc = classify_queries(arrays["min_past_gt_dist"])
                if qc["n_pos"]:
                    sweeps[sequence] = pr_sweep(arrays["score_learned"], arrays["top1_geo_dist"], qc)
                    groups.setdefault(FAMILY.get(sensor, sensor), []).append(sequence)
            tau, macro_f1, _ = _family_macro_tau_star(sweeps, groups)
            per_sequence, directions = {}, {str(i): {"n": 0, "correct": 0} for i in range(3)}
            for sequence, a in ledgers.items():
                qc = classify_queries(a["min_past_gt_dist"])
                row = operating_point(a["score_learned"], a["top1_geo_dist"], qc, tau)
                row["negative_false_accepts"] = int((qc["negative"] & (a["score_learned"] >= tau)).sum())
                per_sequence[sequence] = row
                for i in range(3):
                    mask = qc["positive"] & (a["group"] == i)
                    directions[str(i)]["n"] += int(mask.sum())
                    directions[str(i)]["correct"] += int((mask & (a["top1_geo_dist"] <= 5)).sum())
            selected["decision"][op] = {"tau": tau, "macro_F1_at_tau": macro_f1,
                                        "sequences": per_sequence, "directions": directions}
        result["arms"][arm] = selected
    old = read(PILOT / "s1/validation_summary.json")["arms"]
    result["initial_fixed_pipeline"] = {kind: old[kind + "_original"] for kind in ("gnn", "point")}
    result["effects"] = {op: {
        "gnn_joint_minus_head": result["arms"]["gnn_joint"]["ops"][op]["AP"] - result["arms"]["gnn_head"]["ops"][op]["AP"],
        "point_joint_minus_head": result["arms"]["point_joint"]["ops"][op]["AP"] - result["arms"]["point_head"]["ops"][op]["AP"],
        "gnn_minus_point_joint": result["arms"]["gnn_joint"]["ops"][op]["AP"] - result["arms"]["point_joint"]["ops"][op]["AP"]} for op in OPS}
    write_json(OUT / "summary.json", result)
    print("SUMMARY", {a: (v["epoch"], v["ops"]) for a, v in result["arms"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("protocol", "pools", "train", "evaluate", "report"))
    parser.add_argument("--kind", choices=("gnn", "point"))
    parser.add_argument("--arm", choices=CFG["arms"])
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    setup()
    if args.action == "protocol":
        protocol(True)
    elif args.action == "pools":
        prepare_pools(args.kind)
    elif args.action == "train":
        train(args.arm)
    elif args.action == "evaluate":
        evaluate(args.arm, args.epoch, args.sequence)
    else:
        report()
