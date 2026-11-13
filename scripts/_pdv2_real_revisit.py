#!/usr/bin/env python3
"""기존 최종 순위 학습에 실제 반대 방향 재방문의 구분 제약을 더한다."""
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_fusion_guard as prior
import _pdv2_temporal_augmentation as evaluator
from _pdv2_fusion_guard_gradient import vector_gradient
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from retrieval.revisit_learning import revisit_terms

base, factor = prior.base, prior.factor
OUT = REPO / "results/pdv2_real_revisit_20261025"
CONFIG = REPO / "configs/pdv2_real_revisit.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
SUPPORT = REPO / "results/pdv2_temporal_augmentation_20261024/training_support"
METHOD = CFG["final_score"]
model, initial_model, terms_for = prior.model, prior.initial_model, prior.terms_for
pool_path, load_pool = prior.pool_path, prior.load_pool


def protocol(inputs: bool = False) -> dict:
    """학습 전에 고정한 코드와 입력의 해시를 검증한다."""
    spec = base.read(OUT / "protocol.json")
    for field in (("runtime", "inputs") if inputs else ("runtime",)):
        for path, digest in spec[field].items():
            assert sha256(REPO / path) == digest, path
    return spec


def auxiliary_batches(epoch: int, count: int, steps: int) -> list:
    """각 에포크의 모든 실제 재방문 질의를 중복 없이 배치에 나눈다."""
    order = np.random.default_rng(91000 + epoch).permutation(count)
    return list(np.array_split(order, steps))


def local_auxiliary(graph, groups: np.ndarray, pool: dict, rows: np.ndarray) -> Tuple:
    """필요한 출력의 시간 두 홉을 보존한 부분 그래프와 지역 번호를 만든다."""
    selected = {k: pool[k][rows] for k in ("anchors", "positives", "negatives")}
    seeds = np.unique(np.concatenate([v.reshape(-1) for v in selected.values()]))
    seeds = seeds[seeds >= 0]
    expanded = seeds[:, None] + np.arange(-10, 11)[None]
    safe = np.clip(expanded, 0, graph.num_nodes - 1)
    valid = (
        (expanded >= 0)
        & (expanded < graph.num_nodes)
        & (groups[safe] == groups[seeds, None])
    )
    nodes = np.unique(expanded[valid])
    mapping = np.full(graph.num_nodes, -1, np.int64)
    mapping[nodes] = np.arange(len(nodes))
    local = base.temporal_subgraph(graph, nodes)
    indices = {
        k: torch.from_numpy(np.where(v >= 0, mapping[v.clip(0)], -1)).cuda()
        for k, v in selected.items()
    }
    assert all(np.all(mapping[v[v >= 0]] >= 0) for v in selected.values())
    return local.cuda(), indices, nodes, seeds


def pair_terms(embedding: torch.Tensor, indices: dict, mode: str) -> dict:
    """동일한 실제 쌍을 전체 키 또는 문맥 부분의 유사도로 학습한다."""
    return revisit_terms(
        embedding,
        **indices,
        context_only=mode == "pair_context",
        temperature=CFG["temperature"],
    )


def prepare() -> None:
    """학습 전용 쌍·오답·초기 기울기 계수·승격 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    assert not (OUT / "plan.json").exists(), "부분 준비 결과를 덮어쓰지 않는다"
    parent = prior.protocol(True)
    support = base.read(SUPPORT / "independent_validation.json")
    assert support["passed"] and not support["test_used"]
    for path, digest in support["sources"].items():
        assert sha256(REPO / path) == digest, path
    runtime, inputs = dict(parent["runtime"]), dict(parent["inputs"])
    for name in (
        "scripts/_pdv2_real_revisit.py",
        "scripts/_pdv2_real_revisit_run.py",
        "scripts/_pdv2_real_revisit_finish.py",
        "scripts/_pdv2_temporal_augmentation.py",
        "scripts/_pdv2_fusion_guard_gradient.py",
        "src/retrieval/revisit_learning.py",
        "tests/retrieval/test_revisit_learning.py",
        "configs/pdv2_real_revisit.yaml",
    ):
        runtime[name] = sha256(REPO / name)
    for path in (
        SUPPORT / "available_opposite_pairs.npz",
        SUPPORT / "independent_validation.json",
        prior.OUT / "rank_all/training_complete.json",
    ):
        inputs[str(path.relative_to(REPO))] = sha256(path)
    write_json(
        OUT / "plan.json",
        {
            "runtime": runtime,
            "inputs": inputs,
            "config": CFG,
            "coefficient_rule": "초기63배치 파라미터 평균 기울기: 순위/각 보조 손실의 노름비×.25, [.01,100] 제한.",
            "pair_rule": "기존1477질의·8590실제 반대 방향 정답쌍 모두. 각 에포크 질의1회. "
            "공통좌표의25m초과 오답 중 초기 전체키상위32∪문맥상위32, 재채굴 없음. "
            "각 정답을 같은 오답집합과 개별 비교하고 정답평균 후 질의평균. 온라인 규약 변경 없음.",
            "selection": "최종 검증AP로2/4에포크 중 선택. 시작/새 정상 대조 각각 대비 기존 승격 기준. 시험 없음.",
            "test_used": False,
        },
    )
    roster = base.exp.arrays(SUPPORT / "available_opposite_pairs.npz")
    assert roster["sequence_names"].tolist() == parent["train"]
    graphs, positions, keys, groups = [], [], [], []
    for i, sequence in enumerate(parent["train"]):
        graph = torch.load(
            pool_path(sequence) / "graph.pt", map_location="cpu", weights_only=False
        )
        cache = base.exp.arrays(factor.paths_for("train", sequence)["cache"])
        key = base.exp.arrays(factor.key_path("train", "interaction", sequence))[
            "embedding"
        ]
        assert graph.num_nodes == len(key) == roster["sequence_lengths"][i]
        np.testing.assert_array_equal(graph.x.numpy(), cache["descriptors"])
        graphs.append(graph)
        positions.append(cache["poses"][:, :3, 3].astype(np.float64))
        keys.append(key)
        groups.append(np.full(len(key), i, np.int64))
    graph = Batch.from_data_list(graphs)
    groups, positions, keys = (
        np.concatenate(groups),
        np.concatenate(positions),
        np.concatenate(keys),
    )
    assert graph.num_nodes == 72306
    coordinate = np.array(
        [100 if s.startswith("helipr/") else i for i, s in enumerate(parent["train"])]
    )[groups]
    anchors = np.unique(roster["anchor_global"])
    positives = [
        np.unique(roster["positive_global"][roster["anchor_global"] == a])
        for a in anchors
    ]
    key_gpu = torch.from_numpy(keys).cuda()
    context_gpu = F.normalize(key_gpu[:, 288:], dim=1)
    negatives = []
    for start in range(0, len(anchors), 32):
        aa = anchors[start : start + 32]
        admissible = (coordinate[aa, None] == coordinate[None]) & (
            np.linalg.norm(positions[aa, None] - positions[None], axis=2)
            > CFG["negative_min_distance"]
        )
        valid = torch.from_numpy(admissible).cuda()
        assert valid.sum(1).min() >= CFG["negative_topk_each"]
        choices = []
        for value in (key_gpu, context_gpu):
            score = value[torch.as_tensor(aa, device="cuda")] @ value.T
            choices.append(
                score.masked_fill(~valid, -torch.inf)
                .topk(CFG["negative_topk_each"], dim=1)
                .indices.cpu()
                .numpy()
            )
        negatives.extend(np.unique(np.r_[a, b]) for a, b in zip(*choices))

    def padded(rows):
        a = np.full((len(rows), max(map(len, rows))), -1, np.int64)
        for i, row in enumerate(rows):
            a[i, : len(row)] = row
        return a

    pool = dict(
        anchors=anchors, positives=padded(positives), negatives=padded(negatives)
    )
    assert len(anchors) == 1477 and sum(map(len, positives)) == 8590
    for a, pp, nn in zip(anchors, positives, negatives):
        assert np.all(coordinate[pp] == coordinate[a]) and np.all(
            coordinate[nn] == coordinate[a]
        )
        assert np.all(
            np.linalg.norm(positions[pp] - positions[a], axis=1) <= 5.0 + 1e-10
        )
        assert np.all(
            np.linalg.norm(positions[nn] - positions[a], axis=1)
            > CFG["negative_min_distance"]
        )
    np.savez_compressed(
        OUT / "auxiliary_pool.npz",
        **pool,
        groups=groups,
        positions=positions,
        coordinate=coordinate,
        initial_keys=keys,
    )
    torch.save(graph, OUT / "auxiliary_graph.pt")
    del key_gpu, context_gpu, score, valid, graphs
    net = initial_model()
    params = [p for p in net.parameters() if p.requires_grad]
    initial = {
        k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()
    }
    assert initial == parent["initial"]
    totals = np.zeros((3, sum(p.numel() for p in params)), np.float64)
    batches = auxiliary_batches(1, len(anchors), parent["updates_per_epoch"])
    records, step = [], 0
    torch.cuda.reset_peak_memory_stats()
    for ordinal, sequence in enumerate(parent["train"]):
        n = parent["pool_counts"][sequence]["queries"]
        if not n:
            continue
        primary_graph = torch.load(
            pool_path(sequence) / "graph.pt", map_location="cuda", weights_only=False
        )
        primary_pool = load_pool(sequence)
        order = np.random.default_rng(1000 + ordinal).permutation(n)
        for start in range(0, n, CFG["batch_queries"]):
            rows = torch.as_tensor(
                order[start : start + CFG["batch_queries"]], device="cuda"
            )
            embedding = F.normalize(net(primary_graph), dim=1)
            rank = terms_for(
                embedding, primary_pool, rows, tuple(parent["confidence"]["1"]["tuple"])
            )["rank"].mean()
            gr = vector_gradient(rank, params, False).detach().double().cpu().numpy()
            rank_value = float(rank.detach())
            del embedding, rank
            local, indices, nodes, seeds = local_auxiliary(
                graph, groups, pool, batches[step]
            )
            emb = F.normalize(net(local), dim=1)
            selected = np.searchsorted(nodes, seeds)
            error = float(abs(emb.detach().cpu().numpy()[selected] - keys[seeds]).max())
            np.testing.assert_allclose(
                emb.detach().cpu().numpy()[selected], keys[seeds], atol=4e-6, rtol=4e-5
            )
            scale = len(batches[step]) * len(batches) / len(anchors)
            lk = pair_terms(emb, indices, "pair_key")["loss"].mean() * scale
            lc = pair_terms(emb, indices, "pair_context")["loss"].mean() * scale
            gk = vector_gradient(lk, params, True).detach().double().cpu().numpy()
            gc = vector_gradient(lc, params, False).detach().double().cpu().numpy()
            totals += np.stack((gr, gk, gc))
            records.append(
                dict(
                    step=step,
                    sequence=sequence,
                    primary_rows=len(rows),
                    auxiliary_rows=len(batches[step]),
                    local_nodes=len(nodes),
                    local_edges=local.num_edges,
                    subgraph_key_error=error,
                    rank_loss=rank_value,
                    key_loss=float(lk.detach()),
                    context_loss=float(lc.detach()),
                )
            )
            step += 1
            del emb, lk, lc, local, indices
        del primary_graph, primary_pool
        print("CALIBRATION_DONE", sequence, step, flush=True)
    assert step == 63
    totals /= step
    norms = np.linalg.norm(totals, axis=1)
    coefficients = {
        arm: float(
            np.clip(
                CFG["auxiliary_gradient_ratio"] * norms[0] / norms[i],
                *CFG["coefficient_clip"],
            )
        )
        for i, arm in enumerate(("pair_key", "pair_context"), 1)
    }
    np.savez_compressed(
        OUT / "calibration_gradients.npz",
        rank=totals[0],
        pair_key=totals[1],
        pair_context=totals[2],
    )
    write_json(
        OUT / "calibration.json",
        {
            "norms": norms.tolist(),
            "coefficients": coefficients,
            "cosine_to_rank": {
                arm: float(totals[0] @ totals[i] / norms[0] / norms[i])
                for i, arm in enumerate(("pair_key", "pair_context"), 1)
            },
            "batches": records,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "test_used": False,
        },
    )
    write_json(
        OUT / "pool_audit.json",
        {
            "anchors": len(anchors),
            "positive_pairs": sum(map(len, positives)),
            "negative_pairs": sum(map(len, negatives)),
            "cross_session_positive_pairs": int(
                sum(
                    (groups[pp] != groups[a]).sum() for a, pp in zip(anchors, positives)
                )
            ),
            "anchor_by_sequence": {
                s: int((groups[anchors] == i).sum())
                for i, s in enumerate(parent["train"])
            },
            "initial_rank_pool_overlap": {
                s: int(
                    np.isin(
                        anchors[groups[anchors] == i]
                        - sum(roster["sequence_lengths"][:i]),
                        base.exp.arrays(pool_path(s) / "pool.npz")["queries"],
                    ).sum()
                )
                for i, s in enumerate(parent["train"])
            },
            "source_roster_sha256": sha256(SUPPORT / "available_opposite_pairs.npz"),
            "test_used": False,
        },
    )
    for name in (
        "plan.json",
        "auxiliary_pool.npz",
        "auxiliary_graph.pt",
        "pool_audit.json",
        "calibration.json",
        "calibration_gradients.npz",
    ):
        inputs[str((OUT / name).relative_to(REPO))] = sha256(OUT / name)
    write_json(
        OUT / "protocol.json",
        {
            **{
                k: parent[k]
                for k in (
                    "train",
                    "val",
                    "confidence",
                    "initial",
                    "pool_counts",
                    "updates_per_epoch",
                    "trainable_parameters",
                )
            },
            "config": CFG,
            "runtime": runtime,
            "inputs": inputs,
            "coefficients": coefficients,
            "initial_checkpoint": str(prior.INITIAL.relative_to(REPO)),
            "initial_sha256": sha256(prior.INITIAL),
            "test_used": False,
            "budget": "같은 주학습30726질의·순서·252업데이트, 보조1477질의는 에포크당1회 모두 사용. "
            "정상 대조도 같은 부분그래프를 순전파하되 미분하지 않음. 전체키/문맥 두 조건은 같은 학습쌍과 계산경로. "
            "추가 역전파 때문에 정상 대비 FLOP 동일 주장은 하지 않음.",
            "scope": "사전 고정 시드1 검증 예비. 추론 구조·후보·지연은 그대로. 반대 방향 쌍 반복25% 교체와 다름.",
        },
    )
    protocol(True)
    print("REAL_REVISIT_PREPARED", coefficients, norms.tolist(), flush=True)


def train(arm: str) -> None:
    """기존 순위 손실을 유지하며 실제 재방문 제약을 같은 갱신에 합한다."""
    spec = protocol(True)
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    assert not list(folder.glob("epoch*.pth"))
    torch.manual_seed(1)
    net = initial_model()
    initial = {
        k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()
    }
    assert initial == spec["initial"]
    frozen = {
        k: v.detach().cpu().clone()
        for k, v in net.state_dict().items()
        if "running_" in k or "num_batches_tracked" in k or k.endswith("key_scale")
    }
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )
    graph = torch.load(
        OUT / "auxiliary_graph.pt", map_location="cpu", weights_only=False
    )
    pool = base.exp.arrays(OUT / "auxiliary_pool.npz")
    updates, orders, auxiliary_orders, history = 0, [], [], []
    started = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        batches = auxiliary_batches(
            epoch, len(pool["anchors"]), spec["updates_per_epoch"]
        )
        step, used = 0, 0
        values = np.zeros(3)
        max_grad = 0.0
        tick = time.time()
        for ordinal, sequence in enumerate(spec["train"]):
            n = spec["pool_counts"][sequence]["queries"]
            if not n:
                continue
            primary_graph = torch.load(
                pool_path(sequence) / "graph.pt",
                map_location="cuda",
                weights_only=False,
            )
            primary_pool = load_pool(sequence)
            order = np.random.default_rng(1000 * epoch + ordinal).permutation(n)
            orders.append(
                {"epoch": epoch, "sequence": sequence, "sha256": array_hash(order)}
            )
            for start in range(0, n, CFG["batch_queries"]):
                rows = torch.as_tensor(
                    order[start : start + CFG["batch_queries"]], device="cuda"
                )
                optimizer.zero_grad(set_to_none=True)
                embedding = F.normalize(net(primary_graph), dim=1)
                rank = terms_for(
                    embedding,
                    primary_pool,
                    rows,
                    tuple(spec["confidence"]["1"]["tuple"]),
                )["rank"].mean()
                assert torch.isfinite(rank)
                rank_value = float(rank.detach())
                rank.backward()
                del embedding, rank
                local, indices, nodes, seeds = local_auxiliary(
                    graph, pool["groups"], pool, batches[step]
                )
                auxiliary_orders.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "rows": array_hash(batches[step]),
                        "nodes": array_hash(nodes),
                        "edges": array_hash(local.edge_index.cpu().numpy()),
                    }
                )
                with torch.set_grad_enabled(arm != "clean_rank"):
                    emb = F.normalize(net(local), dim=1)
                    key_loss = pair_terms(emb, indices, "pair_key")["loss"].mean()
                    context_loss = pair_terms(emb, indices, "pair_context")[
                        "loss"
                    ].mean()
                    if arm != "clean_rank":
                        scale = len(batches[step]) * len(batches) / len(pool["anchors"])
                        loss = (
                            spec["coefficients"][arm]
                            * scale
                            * (key_loss if arm == "pair_key" else context_loss)
                        )
                        assert torch.isfinite(loss)
                        loss.backward()
                        del loss
                norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), CFG["gradient_clip"]
                )
                assert torch.isfinite(norm)
                optimizer.step()
                values += [
                    rank_value,
                    float(key_loss.detach()),
                    float(context_loss.detach()),
                ]
                max_grad = max(max_grad, float(norm))
                used += len(rows)
                updates += 1
                step += 1
                del emb, key_loss, context_loss, local, indices
            del primary_graph, primary_pool
        assert step == spec["updates_per_epoch"] and used == 30726
        record = {
            "epoch": epoch,
            "updates": updates,
            "queries": used,
            "auxiliary_queries": len(pool["anchors"]),
            "rank_loss": values[0] / step,
            "pair_key_loss": values[1] / step,
            "pair_context_loss": values[2] / step,
            "gradient_norm_max": max_grad,
            "seconds": time.time() - tick,
        }
        history.append(record)
        write_json(folder / "training_progress.json", record)
        print("EPOCH", arm, record, flush=True)
        if epoch in CFG["checkpoint_epochs"]:
            state = {
                k: v.detach().cpu()
                for k, v in net.state_dict().items()
                if not k.endswith("key_scale")
            }
            assert all(torch.isfinite(v).all() for v in state.values())
            torch.save(
                {
                    "model_state_dict": state,
                    "epoch": epoch,
                    "updates": updates,
                    "protocol_sha256": sha256(OUT / "protocol.json"),
                },
                folder / f"epoch{epoch}.pth",
            )
    for k, v in frozen.items():
        torch.testing.assert_close(net.state_dict()[k].cpu(), v, atol=0, rtol=0)
    write_json(
        folder / "training_complete.json",
        {
            "initial": initial,
            "orders": orders,
            "auxiliary_orders": auxiliary_orders,
            "history": history,
            "updates": updates,
            "seconds": time.time() - started,
            "bn_and_remetric_frozen": True,
            "checkpoints": {p.name: sha256(p) for p in folder.glob("epoch*.pth")},
            "test_used": False,
        },
    )


def bind_evaluator() -> None:
    """동결된 정상 입력 평가기를 새 실험의 출력·프로토콜에 연결한다."""
    evaluator.OUT, evaluator.CFG, evaluator.protocol = OUT, CFG, protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "select"))
    parser.add_argument("--arm", choices=CFG["arms"])
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--role", choices=("train", "val"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action in ("prepare", "train") else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train(args.arm)
    else:
        bind_evaluator()
        if args.action == "select":
            evaluator.select()
        else:
            evaluator.evaluate(args.arm, args.epoch, args.role, args.sequence)
