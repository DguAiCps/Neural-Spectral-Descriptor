#!/usr/bin/env python3
"""고정 학습 배치의 계열별 손실과 실제 모델 파라미터 기울기를 잰다."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_family_balance as run
from gnn.family_weighting import infonce_rows
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash


def geometry(g0: np.ndarray, g1: np.ndarray, fraction: float) -> dict:
    """기울기 크기·방향과 두 손실 하강 방향의 관계를 계산한다."""
    g0, g1 = g0.astype(np.float64), g1.astype(np.float64)
    norm0, norm1 = np.linalg.norm(g0), np.linalg.norm(g1)
    natural0, natural1 = fraction*g0, (1-fraction)*g1
    natural, balanced = natural0+natural1, .5*(g0+g1)
    return {
        "mean_gradient_norm_kitti": float(norm0), "mean_gradient_norm_bridge": float(norm1),
        "gradient_cosine": float(np.dot(g0, g1)/max(norm0*norm1, 1e-30)),
        "natural_bridge_to_kitti_norm_ratio": float((1-fraction)*norm1/max(fraction*norm0, 1e-30)),
        "natural_kitti_dot": float(np.dot(g0, natural)), "natural_bridge_dot": float(np.dot(g1, natural)),
        "balanced_kitti_dot": float(np.dot(g0, balanced)), "balanced_bridge_dot": float(np.dot(g1, balanced)),
    }


def diagnose(stage: str) -> None:
    """Adam 업데이트 없이 계열별 기울기를 같은 24개 배치에서 비교한다."""
    spec = run.protocol()
    root = run.OUT / "diagnostics"
    root.mkdir(exist_ok=True)
    target = root / (stage + ".npz")
    if target.exists():
        assert sha256(target) == run.read(target.with_suffix(".json"))["sha256"]
        return
    torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    net = run.align.old.model()
    if stage == "initial":
        checkpoint = None
    elif stage == "reference":
        checkpoint = run.SAMPLES / "checkpoints/final_model.pth"
    else:
        checkpoint = run.OUT / stage / "checkpoints/final_model.pth"
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    initial = {k: v.clone() for k, v in net.state_dict().items()}
    before = {k: array_hash(v.numpy()) for k, v in initial.items()}
    net = net.cuda().train()
    graph = torch.load(REPO / spec["graph"], map_location="cpu", weights_only=False).cuda()
    data = run.training_data()
    names, parameters = zip(*[(n, p) for n, p in net.named_parameters() if p.requires_grad])
    batches = run.align.previous.load(run.OUT / "probe_batches.npz")
    records, grad0, grad1, additive = [], [], [], []

    def derivative(loss, retain=True):
        values = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                          for p, g in zip(parameters, values)]).cpu().numpy()

    for ordinal, (batch_id, triplets) in enumerate(batches.items()):
        net.load_state_dict(initial, strict=True)
        seed = 202610200 + int(batch_id)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        labels = (data["sequences"][triplets[:, 0]] >= 4).astype(np.int64)
        family = torch.as_tensor(labels, device="cuda")
        ids = torch.as_tensor(triplets, device="cuda")
        embedding = net(graph)
        rows = infonce_rows(embedding[ids[:, 0]], embedding[ids[:, 1]], embedding[ids[:, 2]], .1)
        losses = [rows[family == f].mean() for f in (0, 1)]
        p0 = float(np.mean(labels == 0))
        g0 = derivative(losses[0])
        g1 = derivative(losses[1], retain=ordinal == 0)
        assert np.isfinite(g0).all() and np.isfinite(g1).all()
        if ordinal == 0:
            for name, loss, expected, keep in (
                ("natural", rows.mean(), p0*g0+(1-p0)*g1, True),
                ("balanced", .5*(losses[0]+losses[1]), .5*(g0+g1), False),
            ):
                direct = derivative(loss, retain=keep)
                relative = float(np.linalg.norm(direct.astype(float)-expected)/max(np.linalg.norm(direct.astype(float)), 1e-30))
                assert relative < .002, (stage, name, relative)
                additive.append({"method": name, "relative_l2_error": relative})
        l0, l1 = (float(x.detach()) for x in losses)
        records.append({
            "batch_id": int(batch_id), "rows": len(labels), "n_kitti": int((labels == 0).sum()),
            "kitti_fraction": p0, "loss_mean_kitti": l0, "loss_mean_bridge": l1,
            "natural_loss_bridge_fraction": (1-p0)*l1/(p0*l0+(1-p0)*l1),
            **geometry(g0, g1, p0),
        })
        grad0.append(g0); grad1.append(g1)
        del embedding, rows, losses, ids, family
    net.load_state_dict(initial)
    after = {k: array_hash(v.cpu().numpy()) for k, v in net.state_dict().items()}
    assert before == after
    np.savez_compressed(target, gradient_kitti=np.stack(grad0), gradient_bridge=np.stack(grad1),
                        kitti_fraction=np.array([r["kitti_fraction"] for r in records]),
                        batch_id=np.array([r["batch_id"] for r in records]))
    summary = {
        "median_natural_loss_bridge_fraction": float(np.median([r["natural_loss_bridge_fraction"] for r in records])),
        "median_natural_bridge_to_kitti_gradient_norm_ratio": float(np.median([r["natural_bridge_to_kitti_norm_ratio"] for r in records])),
        "median_gradient_cosine": float(np.median([r["gradient_cosine"] for r in records])),
        "opposed_batches": sum(r["gradient_cosine"] < 0 for r in records),
        "natural_direction_harms_kitti_batches": sum(r["natural_kitti_dot"] < 0 for r in records),
        "natural_direction_harms_bridge_batches": sum(r["natural_bridge_dot"] < 0 for r in records),
        "balanced_direction_harms_kitti_batches": sum(r["balanced_kitti_dot"] < 0 for r in records),
        "balanced_direction_harms_bridge_batches": sum(r["balanced_bridge_dot"] < 0 for r in records),
    }
    write_json(target.with_suffix(".json"), {
        "stage": stage, "sha256": sha256(target), "batches": records, "summary": summary,
        "checkpoint": str(checkpoint.relative_to(REPO)) if checkpoint else None,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "parameters": [{"name": n, "size": p.numel()} for n, p in zip(names, parameters)],
        "state_restored_exactly": True, "additivity_checks": additive,
        "protocol_sha256": sha256(run.OUT / "protocol.json"), "test_used": False})
    print("DIAGNOSED", stage, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initial", "reference", *run.EVAL_ARMS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    diagnose(args.stage)
