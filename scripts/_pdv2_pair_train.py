#!/usr/bin/env python3
"""기존 학습 데이터의 쌍 정책만 바꾸는 고정 예산 추가 학습 실험이다."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model
from gnn.pair_sampling import (
    masked_infonce, past_heading, positive_pools, same_place_mask, sample_triplets,
)
from gnn.trainer import InfoNCELoss
from keyframe.graph_manager import build_graph_from_keyframes_batch
from train_multi_dataset import attach_sensor_metadata, load_keyframes_cache
from utils.similarity_stats import SimilarityDistribution

EXPERIMENT = REPO / "configs/pdv2_pair_sampling_pilot.yaml"


def sha(path: Path) -> str:
    """파일 내용의 해시를 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """실험 기록을 원자적으로 쓴다."""
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def settings() -> tuple:
    """고정한 실험과 현재 인코더 설정을 읽는다."""
    exp = yaml.safe_load(EXPERIMENT.read_text())
    cfg = _apply_encoder_preset(yaml.safe_load((REPO / exp["base_config"]).read_text()),
                                exp["encoder_preset"])
    cfg["gnn"]["use_residual_gate"] = exp["use_gated_context"]
    cfg["gnn"]["gate_initial_alpha"] = exp["gate_initial_alpha"]
    assert cfg["gnn"]["input_dim"] == 288
    assert not cfg["encoding"]["spectral_policy"]["enabled"]
    return exp, cfg, REPO / exp["out"]


def hard_negatives(descriptors: np.ndarray, positions: np.ndarray,
                   sequences: np.ndarray) -> tuple:
    """동일 시퀀스의 원래 L2 최난도 오답을 한 번 구해 재사용한다.

    모든 후보 거리를 GPU float64로 계산한 뒤 상위 4개를 원래 NumPy
    float32 식으로 다시 정렬한다. 경계가 가깝다면 전체 후보를 다시 잰다.
    원래 학습은 고정된 d로 같은 오답을 5에포크마다 다시 구한다.
    """
    result = np.full(len(positions), -1, dtype=np.int64)
    checks = []
    for seq in np.unique(sequences):
        ix = np.flatnonzero(sequences == seq)
        raw = descriptors[ix]
        x = torch.as_tensor(raw, dtype=torch.float64, device="cuda")
        p = torch.as_tensor(positions[ix], dtype=torch.float64, device="cuda")
        ids = torch.arange(len(ix), device="cuda")
        fallbacks = 0
        for start in range(0, len(ix), 256):
            end = min(start + 256, len(ix))
            distance = torch.cdist(x[start:end], x).square()
            geo = torch.cdist(p[start:end], p, compute_mode="donot_use_mm_for_euclid_dist")
            invalid = (geo <= 10) | ((ids[start:end, None] - ids[None, :]).abs() < 30)
            distance.masked_fill_(invalid, float("inf"))
            values, candidates = torch.topk(distance, 4, dim=1, largest=False, sorted=True)
            values, candidates = values.cpu().numpy(), candidates.cpu().numpy()
            for row, anchor in enumerate(range(start, end)):
                if not np.isfinite(values[row, 0]):
                    continue
                cand = candidates[row]
                if values[row, -1] - values[row, 0] <= max(values[row, 0] * 1e-4, 1e-8):
                    geo_np = np.linalg.norm(positions[ix] - positions[ix[anchor]], axis=1)
                    cand = np.flatnonzero((geo_np > 10) & (np.abs(np.arange(len(ix)) - anchor) >= 30))
                    fallbacks += 1
                cand = np.sort(cand)
                exact = np.sum((raw[cand] - raw[anchor]) ** 2, axis=1)
                result[ix[anchor]] = ix[cand[np.argmin(exact)]]
        for anchor in np.linspace(0, len(ix) - 1, min(64, len(ix))).astype(int):
            dgeo = np.linalg.norm(positions[ix] - positions[ix[anchor]], axis=1)
            cand = np.flatnonzero((dgeo > 10) & (np.abs(np.arange(len(ix)) - anchor) >= 30))
            expected = ix[cand[np.argmin(np.sum((raw[cand] - raw[anchor]) ** 2, axis=1))]]
            assert expected == result[ix[anchor]], (seq, anchor)
        checks.append({"sequence_id": int(seq), "reference_checks": 64,
                       "full_numpy_fallbacks": fallbacks})
        print("NEGATIVE_DONE", int(seq), len(ix), checks[-1], flush=True)
        del x, p
    return result, checks


def prepare() -> None:
    """학습 8개만 읽어 같은 그래프와 정답·오답 후보를 고정한다."""
    exp, cfg, out = settings()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepared.pt").exists():
        print("PREPARED_EXISTS", flush=True)
        return
    np.random.seed(exp["seed"])
    torch.manual_seed(exp["seed"])
    keyframes, seqs, frames, sensors, beams, roster, paths = [], [], [], [], [], [], []
    for dataset in cfg["data"]["datasets"]["train"]:
        sensor = dataset["type"]
        for seq in dataset["sequences"]:
            name = sensor + "/" + seq
            candidates = list((REPO / "data/preprocessed_v2").glob(f"cache_*_{sensor}_{seq}.npz"))
            assert len(candidates) == 1
            path = candidates[0]
            kfs = load_keyframes_cache(path)
            seq_id = len(roster)
            frame = seq_id
            for group_index, group in enumerate(exp["shared_coordinate_groups"]):
                if name in group:
                    frame = 100 + group_index
            keyframes.extend(kfs)
            seqs.extend([seq_id] * len(kfs))
            frames.extend([frame] * len(kfs))
            sensors.extend([0 if sensor == "kitti" else 2] * len(kfs))
            beams.extend([64.0 if sensor == "kitti" else 16.0] * len(kfs))
            roster.append(name)
            paths.append(path)
    descriptors = np.asarray([kf.descriptor for kf in keyframes], dtype=np.float32)
    poses = np.asarray([kf.pose for kf in keyframes])
    positions = poses[:, :3, 3]
    seqs, frames = np.asarray(seqs), np.asarray(frames)
    assert len(poses) == 72306 and descriptors.shape[1] == 288
    intra, opposite = positive_pools(positions, seqs, frames)
    heading, heading_valid = past_heading(positions, seqs)
    pair_counts = {"intra": sum(map(len, intra)), "opposite": sum(map(len, opposite)),
                   "opposite_anchors": sum(bool(len(p)) for p in opposite),
                   "opposite_cross_sequence": sum(int((seqs[p] != seqs[i]).sum()) for i, p in enumerate(opposite))}
    print("PAIR_COUNTS", pair_counts, flush=True)
    negative, checks = hard_negatives(descriptors, positions, seqs)
    g = cfg["keyframe"]["graph"]
    distribution_path = (REPO / exp["initial_checkpoint"]).parent / "similarity_dist.npz"
    dist = SimilarityDistribution(metric=g["similarity_metric"]).load(str(distribution_path))
    assert dist.fitted
    graph = build_graph_from_keyframes_batch(
        keyframes, temporal_neighbors=cfg["keyframe"]["temporal_neighbors"], device="cuda",
        poses=poses, descriptors=descriptors, similarity_threshold=g["similarity_threshold"],
        similarity_max_k=g["similarity_max_k"], similarity_min_k=0,
        similarity_exclude_temporal=g["similarity_exclude_temporal"], similarity_dist=dist,
        prior_signal=g["prior_signal"], similarity_metric=g["similarity_metric"],
        sequence_ids=seqs, temporal_edge_mode=g["temporal_edge_mode"],
        similarity_causal=g["similarity_causal"], confidence_level=g["confidence_level"],
        base_prior=g["base_prior"], density_k=g["density_k"], density_beta=g["density_beta"],
    )
    graph = attach_sensor_metadata(graph, sensors, beams, "cuda").cpu()
    edge = graph.edge_index.numpy()
    assert np.all(seqs[edge[0]] == seqs[edge[1]]) and np.all(edge[0] < edge[1])
    torch.save(dict(graph=graph, poses=poses, descriptors=descriptors, sequences=seqs,
                    frames=frames, intra=intra, opposite=opposite, negative=negative,
                    heading=heading, heading_valid=heading_valid), out / "prepared.pt")
    paths += [EXPERIMENT, Path(__file__), REPO / exp["base_config"],
              REPO / exp["initial_checkpoint"], distribution_path,
              REPO / "src/gnn/pair_sampling.py", REPO / "src/gnn/trainer.py",
              REPO / "src/gnn/model.py", REPO / "src/keyframe/graph_manager.py"]
    write_json(out / "protocol.json", {
        "experiment": exp, "train_roster": roster, "nodes": len(poses),
        "edges": int(edge.shape[1]), "pair_counts": pair_counts, "negative_checks": checks,
        "scope": "시드 1 동일 체크포인트부터 모델 가중치만 초기화하는 12에포크 추가 학습. 마지막 에포크 고정, 체크포인트 선택 없음.",
        "positive_policy": "기존 3쌍/앵커 예산의 25%를 반대 방향 앵커 균등 추출로 교체. 해당 앵커에서 가능한 양성 균등 추출. 나머지 75%는 기존 쌍.",
        "mask_policy": "같은 좌표계의 3차원 거리 5m 이내 배치 양성은 다른 앵커의 오답 항에서 제외. 대각선 정답과 채굴 오답은 유지.",
        "held_fixed": "구조, 학습 그래프, 초기 가중치, 학습 예산, 최난도 오답 규칙, 옵티마이저, 온도. 추론에서는 기존 재계량·간선 분류기·융합·판정 임계값 고정.",
        "evaluation": "검증 9개만 사용. 시험셋 미사용. 기존 간선 분류기를 고정하므로 전체 파이프라인 재최적화 실험이 아님.",
        "sha256": {str(p.relative_to(REPO)): sha(p) for p in paths},
        "prepared_sha256": sha(out / "prepared.pt"),
    })


def train(arm: str) -> None:
    """동일 초기화와 업데이트 수로 한 정책의 마지막 체크포인트를 만든다."""
    exp, cfg, out = settings()
    assert arm in exp["arms"]
    protocol = json.loads((out / "protocol.json").read_text())
    for filename, digest in protocol["sha256"].items():
        assert sha(REPO / filename) == digest, filename
    random.seed(exp["seed"])
    np.random.seed(exp["seed"])
    torch.manual_seed(exp["seed"])
    torch.cuda.manual_seed_all(exp["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    state = torch.load(out / "prepared.pt", map_location="cpu", weights_only=False)
    graph = state["graph"].to("cuda")
    model = _make_model(cfg, REPO / exp["initial_checkpoint"], "cuda")
    expected = torch.load(REPO / exp["initial_checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"]
    assert set(model.state_dict()) == set(expected)
    assert all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in expected.items())
    optimizer = torch.optim.Adam(model.parameters(), lr=exp["learning_rate"], weight_decay=exp["weight_decay"])
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1., total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp["epochs"] - 3, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[3])
    scaler = torch.amp.GradScaler("cuda", init_scale=exp["grad_scaler_initial"],
                                 growth_interval=exp["grad_scaler_growth_interval"])
    criterion = InfoNCELoss(exp["temperature"])
    positions = torch.as_tensor(state["poses"][:, :3, 3], device="cuda")
    frames = torch.as_tensor(state["frames"], device="cuda")
    bsize, accum = exp["batch_size"], exp["accumulation_steps"]
    logs, triplet_reports = [], []
    started = time.monotonic()
    updates = 0
    masked_count = 0
    for epoch in range(exp["epochs"]):
        if epoch % exp["resample_every"] == 0:
            trips = sample_triplets(state["intra"], state["opposite"], state["negative"],
                                    seed=1000 + epoch,
                                    opposite_fraction=0.0 if arm == "original" else exp["opposite_replacement_fraction"])
            a, p, n = trips.T
            geo = np.linalg.norm(state["poses"][a, :3, 3] - state["poses"][p, :3, 3], axis=1)
            assert np.all(geo <= 5.) and np.all(state["frames"][a] == state["frames"][p])
            assert np.all(state["sequences"][a] == state["sequences"][n])
            assert np.all(np.linalg.norm(state["poses"][a, :3, 3] - state["poses"][n, :3, 3], axis=1) > 10.)
            dot = np.sum(state["heading"][a] * state["heading"][p], axis=1)
            opposite = state["heading_valid"][a] & state["heading_valid"][p] & (dot <= -.5)
            row = {"epoch": epoch + 1, "n_triplets": len(trips), "opposite": int(opposite.sum()),
                   "cross_sequence": int((state["sequences"][a] != state["sequences"][p]).sum()),
                   "unique_anchors": int(len(np.unique(a))),
                   "unique_opposite_pairs": int(len(np.unique(trips[opposite, :2], axis=0)))}
            triplet_reports.append(row)
            np.savez_compressed(out / f"triplets_{arm}_epoch{epoch+1}.npz", triplets=trips)
            print(arm, "SAMPLED", row, flush=True)
        order = np.random.default_rng(2000 + epoch).permutation(len(trips))
        batches = trips[order]
        n_batches = (len(batches) + bsize - 1) // bsize
        model.train()
        losses = []
        t0 = time.monotonic()
        for start in range(0, n_batches, accum):
            stop = min(start + accum, n_batches)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                emb = model(graph)
                loss = torch.zeros((), device="cuda")
                for batch in range(start, stop):
                    a, p, n = batches[batch * bsize:(batch + 1) * bsize].T
                    if arm == "opposite_masked":
                        mask = same_place_mask(positions[a], positions[p], frames[a], frames[p])
                        masked_count += int(mask.sum())
                        term = masked_infonce(emb[a], emb[p], emb[n], mask, exp["temperature"])
                    else:
                        term = criterion(emb[a], emb[p], emb[n])
                    loss = loss + term / (stop - start)
                    losses.append(float(term.detach()))
            assert torch.isfinite(loss), (arm, epoch, start)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - t0,
               "updates_total": updates}
        logs.append(row)
        print(arm, "EPOCH", row, flush=True)
        scheduler.step()
        write_json(out / f"train_{arm}.json", {"epochs": logs, "sampling": triplet_reports,
                   "masked_terms_total": masked_count, "complete": False})
    model.eval()
    target = out / f"{arm}.pth"
    torch.save({"epoch": exp["epochs"] - 1, "model_state_dict": model.state_dict(),
                "experiment": exp, "arm": arm, "optimizer_steps": updates,
                "initial_checkpoint_sha256": sha(REPO / exp["initial_checkpoint"]),
                "protocol_sha256": sha(out / "protocol.json")}, target)
    write_json(out / f"train_{arm}.json", {"epochs": logs, "sampling": triplet_reports,
               "masked_terms_total": masked_count, "complete": True,
               "checkpoint_sha256": sha(target), "runtime_s": time.monotonic() - started})
    print("TRAIN_DONE", arm, updates, flush=True)
    del model, graph, state
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    """컨테이너에서 준비와 학습을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=("original", "opposite", "opposite_masked", "all"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.prepare:
        prepare()
    if args.arm:
        for arm in settings()[0]["arms"] if args.arm == "all" else [args.arm]:
            train(arm)


if __name__ == "__main__":
    main()
