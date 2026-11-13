#!/usr/bin/env python3
"""선택 모델을 고정하고 차이 값과 정규화의 크기를 진단한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_grouped_relation as run
from _pdv2_no_context_ablation import sha256, write_json


def quantiles(value: torch.Tensor) -> list:
    """유한한 표본의 최소·10·50·90·최대 분위수를 구한다."""
    a = value.detach().float().cpu().numpy()
    assert np.isfinite(a).all()
    return np.quantile(a, [0, .1, .5, .9, 1]).tolist()


def measure(net: torch.nn.Module, graph) -> tuple:
    """출력을 바꾸지 않는 hook으로 두 층의 메시지와 BN을 관측한다."""
    samples, handles = {}, []
    indices = torch.linspace(0, graph.num_nodes - 1,
                             min(graph.num_nodes, 2048), device="cuda").long()
    for index, (conv, bn) in enumerate(zip(net.gnn.convs, net.gnn.batch_norms)):
        def before(layer, args, number=index):
            x, edges = args[:2]
            samples[number] = {"center": x[indices].detach().float()}
            # 전체 그래프를 바꾸지 않고 결정적으로 고른 간선만 진단한다.
            chosen = torch.linspace(0, edges.shape[1] - 1,
                                     min(edges.shape[1], 16384), device="cuda").long()
            src, tgt = edges[:, chosen]
            relation = x[src] - x[tgt] if layer.value_source == "diff" else x[src]
            z = layer.W_v(relation).reshape(-1, layer.heads, layer.head_dim)
            value = layer.transform_value(z)
            samples[number]["value"] = {
                "edges": len(chosen), "abs_projected_quantiles": quantiles(z.abs()),
                "abs_projected_below_point1": float((z.abs() < .1).float().mean()),
                "abs_projected_above2": float((z.abs() > 2).float().mean()),
                "transformed_rms": float(value.float().square().mean().sqrt()),
                "projected_rms": float(z.float().square().mean().sqrt()),
            }

        def after(layer, args, output, number=index):
            message = args[0][indices].detach().float()
            normalized = output[indices].detach().float()
            center = samples[number].pop("center")
            deviation = (args[0].float().mean(0) - layer.running_mean) / torch.sqrt(layer.running_var + layer.eps)
            zero_response = layer.bias - layer.weight * layer.running_mean / torch.sqrt(layer.running_var + layer.eps)
            center_norm = center.norm(dim=1).clamp_min(1e-12)
            message_norm = message.norm(dim=1).clamp_min(1e-12)
            samples[number]["normalization"] = {
                "nodes_sampled": len(indices),
                "message_norm": quantiles(message_norm),
                "normalized_message_norm": quantiles(normalized.norm(dim=1)),
                "center_norm": quantiles(center_norm),
                "message_over_center": quantiles(message_norm / center_norm),
                "normalized_over_center": quantiles(normalized.norm(dim=1) / center_norm),
                "norm_gain": quantiles(normalized.norm(dim=1) / message_norm),
                "batch_mean_running_mean_z_rms": float(deviation.square().mean().sqrt()),
                "zero_input_bn_response_norm": float(zero_response.norm()),
            }
        handles.append(conv.register_forward_pre_hook(before))
        handles.append(bn.register_forward_hook(after))
    try:
        with torch.no_grad():
            output = net(graph).float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    return output, samples


def main() -> None:
    """학습 전체와 검증 시퀀스별 통계를 보존하며 선택 결과는 바꾸지 않는다."""
    spec, summary = run.protocol(True), run.read(run.OUT / "summary.json")
    assert run.read(run.OUT / "status.json")["audit_passed"]
    output_path = run.OUT / "operator_diagnosis.json"
    if output_path.exists():
        existing = run.read(output_path)
        assert existing["summary_sha256"] == sha256(run.OUT / "summary.json")
        for path, expected in existing["sources"].items():
            assert sha256(REPO / path) == expected
        print("DIAGNOSIS_ALREADY_VERIFIED", run.OUT.name, flush=True)
        return
    original = torch.load(run.base.GRAPH, map_location="cpu", weights_only=False)
    train_graph = run.base.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    records, sources = {}, {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__))}
    for arm in run.ARMS:
        epoch = summary["per_seed"]["1"][arm]["selected"]["epoch"]
        checkpoint = run.base.root(1, arm) / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        sources[str(checkpoint.relative_to(REPO))] = sha256(checkpoint)
        net = run.model(arm).cuda().eval()
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(run.base.exp.old.base.RVEC)).float().cuda(), requires_grad=False)
        _, train_stats = measure(net, train_graph)
        records[arm] = {"epoch": epoch, "train_all": train_stats, "validation": {}}
        for seq in spec["val"]:
            _, _, paths, _ = run.base.exp.old.base.inputs("val", seq)
            cache = run.base.exp.arrays(paths["cache"])
            graph = run.base.make_graph(cache["descriptors"].astype(np.float32),
                cache["poses"], np.zeros(len(cache["poses"]), np.int64)).cuda()
            key, stats = measure(net, graph)
            p = run.base.root(1, arm) / f"epoch{epoch}" / ("keys_" + seq.replace("/", "_") + ".npz")
            reference = run.base.exp.arrays(p)["embedding"]
            key = run.base.normalize(key)
            np.testing.assert_allclose(key, reference, atol=4e-6, rtol=4e-5)
            records[arm]["validation"][seq] = {"layers": stats, "keys_exact": bool(np.array_equal(key, reference)),
                "max_key_error": float(np.max(abs(key - reference)))}
            del graph
        print("DIAGNOSED", arm, epoch, flush=True)
        del net
    write_json(run.OUT / "operator_diagnosis.json", {
        "models": records, "sources": sources, "test_used": False,
        "summary_sha256": sha256(run.OUT / "summary.json"),
        "scope": "시드1·선택 체크포인트 고정·추론 모드. 학습 전체와 검증9를 같은 저장 BN통계로 관측. 층별2048노드/16384간선 결정 표본, BN입력평균은전체. 기존 접두 허용오차로 키 재현, 완전 일치 여부와 최대 오차도 기록. 개입·재학습·원인 확증 아님.",
    })


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
