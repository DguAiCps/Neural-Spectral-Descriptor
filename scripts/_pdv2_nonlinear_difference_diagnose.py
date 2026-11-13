#!/usr/bin/env python3
"""선택된 비선형 차이 모델의 실제 은닉 특징에서 상쇄와 변화를 측정한다."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.utils import softmax

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_nonlinear_difference as experiment
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """동일 입력·가중치의 키 공식만 바꿔 표본 이웃 가중치를 비교한다."""
    run = experiment.base
    spec = run.protocol(True)
    summary = run.read(run.OUT / "summary.json")
    epoch = summary["per_seed"]["1"]["diff_tanh"]["selected"]["epoch"]
    path = run.root(1, "diff_tanh") / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
    net = experiment.model("diff_tanh").cuda().eval()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    records = []
    sequence = None

    def hook(layer_index):
        def measure(conv, args, kwargs):
            h, edge = args[:2]
            centers = torch.linspace(5, len(h)-6, min(512, len(h)-10), device=h.device).long().unique()
            mask = torch.isin(edge[1], centers)
            src, tgt = edge[:, mask]
            center, neighbor = h[tgt].double(), h[src].double()
            q = F.linear(center, conv.W_q.weight.double()).view(-1, conv.heads, conv.head_dim)
            attr = kwargs.get("edge_attr")
            bias = conv.bias_mlp(attr[mask]).double() if attr is not None else 0.
            logit = kwargs.get("edge_logit")
            if logit is not None:
                bias = bias + logit[mask].double()
            weights = {}
            projected = None
            for mode, relation in (("diff", neighbor-center), ("standard", neighbor)):
                key = F.linear(relation, conv.W_k.weight.double()).view(-1, conv.heads, conv.head_dim)
                if mode == "diff":
                    projected = key
                for nonlinear in (False, True):
                    score = (q*(key.tanh() if nonlinear else key)).sum(-1)/conv.scale+bias
                    weights[mode+"_tanh" if nonlinear else mode] = softmax(score, tgt, num_nodes=len(h))
            def difference(a, b):
                error = (weights[a]-weights[b]).abs()
                return {"mean": float(error.mean()), "max": float(error.max())}
            identity = difference("diff", "standard")
            assert identity["max"] < 1e-12
            records.append({"sequence": sequence, "layer": layer_index+1,
                "centers": len(centers), "edges": len(src),
                "linear_diff_vs_standard": identity,
                "tanh_diff_vs_standard": difference("diff_tanh", "standard_tanh"),
                "diff_tanh_vs_linear": difference("diff_tanh", "diff"),
                "absolute_projected_difference_quantiles": torch.quantile(projected.abs().flatten(),
                    torch.tensor([0., .1, .5, .9, 1.], dtype=torch.float64, device=h.device)).cpu().tolist(),
                "projected_abs_below_0_1": float((projected.abs() < .1).double().mean()),
                "projected_abs_above_2": float((projected.abs() > 2).double().mean())})
        return measure

    handles = [conv.register_forward_pre_hook(hook(i), with_kwargs=True)
               for i, conv in enumerate(net.gnn.convs)]
    with torch.no_grad():
        for sequence in spec["val"]:
            _, _, paths, _ = run.exp.old.base.inputs("val", sequence)
            cache = run.exp.arrays(paths["cache"])
            graph = run.make_graph(cache["descriptors"].astype(np.float32), cache["poses"],
                                   np.zeros(len(cache["descriptors"]), dtype=np.int64)).cuda()
            output = net(graph)
            assert torch.isfinite(output).all()
            del graph, output
    for handle in handles:
        handle.remove()
    write_json(run.OUT / "operator_diagnosis.json", {
        "epoch": epoch, "checkpoint_sha256": sha256(path), "records": records,
        "scope": "실제 검증 그래프·선택 diff_tanh의 층 입력에서 키 공식만 대조. 재학습 모델 전체나 최종 성능의 인과 대조가 아님.",
        "test_used": False})
    print("OPERATOR_CHECKED", len(records), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
