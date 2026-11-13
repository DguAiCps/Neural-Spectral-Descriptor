#!/usr/bin/env python3
"""고정 시간 모델에서 방향 표본 보강의 실제 학습 신호를 측정한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal3 as old
from _train_attention_neighbors import array_hash
from _pdv2_no_context_ablation import sha256, write_json
from gnn.pair_sampling import same_place_mask
from gnn.trainer import InfoNCELoss

OUT = REPO / "results/pdv2_temporal_focus_20261018"


def main() -> None:
    """FP32·평가 모드에서 손실과 오답 logit 기울기의 분포를 구한다."""
    torch.set_num_threads(2)
    source = REPO / "results/pdv2_context_mechanism_20261014/train_graph.pt"
    graph = torch.load(source, map_location="cpu", weights_only=False)
    mask = graph.edge_type == 0
    graph.edge_index, graph.edge_attr, graph.edge_type = graph.edge_index[:, mask], graph.edge_attr[mask], graph.edge_type[mask]
    hashes = {k: array_hash(getattr(graph, k).numpy()) for k in ("x", "edge_index", "edge_attr", "edge_type")}
    assert hashes == old.read(old.OUT / "s1/temporal/training_graph.json")["hashes"]
    prepared = REPO / "results/pdv2_pair_sampling_20261013/prepared.pt"
    state = torch.load(prepared, map_location="cpu", weights_only=False)
    pos = torch.as_tensor(state["poses"][:, :3, 3], dtype=torch.float64)
    frames = torch.as_tensor(state["frames"])
    seq = torch.as_tensor(state["sequences"])
    order = np.random.RandomState(100006).permutation(183249)[:16 * 1024]
    trips = {p: np.load(OUT / f"samples/{p}_epoch6.npz")["triplets"][order] for p in ("original", "opposite")}
    summary = old.read(old.OUT / "summary.json")
    sources = {str(p.relative_to(REPO)): sha256(p) for p in (Path(__file__), source, prepared,
        OUT / "samples/original_epoch6.npz", OUT / "samples/opposite_epoch6.npz", old.OUT / "summary.json")}
    results = {}
    for seed in old.SEEDS:
        for arm in old.ARMS:
            epoch = summary["per_seed"][str(seed)][arm]["epoch"]
            _, model = old.load_model(seed, arm, epoch, "cpu")
            # 실제 학습의 키 블록에는 추론용 재계량을 쓰지 않는다.
            model.gnn.key_scale = None
            model.gnn.use_key_remetrize = False
            with torch.no_grad():
                emb = F.normalize(model(graph).float(), dim=1)
            assert emb.shape == (72306, 416) and torch.isfinite(emb).all()
            policies = {}
            for policy, samples in trips.items():
                metrics = {}
                for start in range(0, len(samples), 1024):
                    a, p, n = torch.as_tensor(samples[start:start+1024]).T
                    x, y, z = emb[a], emb[p], emb[n]
                    cross, hard = x @ y.T / .1, (x * z).sum(1, keepdim=True) / .1
                    logits = torch.cat((cross, hard), dim=1)
                    prob = logits.softmax(1)
                    target = prob[:, :len(a)].diag()
                    false = same_place_mask(pos[a], pos[p], frames[a], frames[p])
                    same_sequence = seq[a, None] == seq[p][None]
                    loss = torch.logsumexp(logits, dim=1) - cross.diag()
                    torch.testing.assert_close(loss.mean(), InfoNCELoss(.1)(x, y, z), atol=2e-6, rtol=2e-6)
                    values = {"loss": loss, "target_probability": target,
                        "explicit_negative_gradient_fraction": prob[:, -1] / (1-target),
                        "same_place_negative_gradient_fraction": (prob[:, :len(a)] * false).sum(1) / (1-target),
                        "same_sequence_false_negative_count": (false & same_sequence).sum(1),
                        "cross_bridge_false_negative_count": (false & ~same_sequence).sum(1)}
                    for k, v in values.items():
                        metrics.setdefault(k, []).append(v.numpy())
                policies[policy] = {k: float(np.concatenate(v).mean()) for k, v in metrics.items()}
            results[f"s{seed}_{arm}"] = policies
            ckpt = old.checkpoint_dir(seed, arm) / f"checkpoint_epoch_{epoch}.pth"
            sources[str(ckpt.relative_to(REPO))] = sha256(ckpt)
            print("LEARNING", seed, arm, policies, flush=True)
    write_json(OUT / "diagnosis/learning_signal.json", {"results": results, "sources": sources,
        "graph_sha256": hashes, "batch_size": 1024, "batches": 16, "rows": len(order), "test_used": False,
        "scope": "학습 GT 시간 그래프·기존 선택모델·FP32 eval 모드. 6에포크 시드1의 순열 앞16배치에 두표본 정책 적용. 실제 학습 역사나 모델 파라미터 기울기 비중 아님. 같은장소 배치 오답의 손실 logit 기울기 비중."})


if __name__ == "__main__":
    main()
