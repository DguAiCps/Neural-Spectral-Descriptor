#!/usr/bin/env python3
"""문맥 단독 정규화가 기존 게이트의 최종 판정 기울기를 제거하는지 확인한다."""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_diffattn_comparable as experiment


def main() -> None:
    """학습 자료의 고정 쌍에서만 점수 미분과 크기 불변성을 검사한다."""
    run = experiment.run
    run.configure()
    experiment.activate()
    records = {}
    for sequence in ("kitti/05", "helipr/Bridge01"):
        folder = experiment.PRIOR / "pools" / sequence.replace("/", "_")
        data = run.ARR(folder / "pool.npz")
        graph = torch.load(folder / "graph.pt", map_location="cuda", weights_only=False)
        net = run.model()
        key = F.normalize(net(graph), dim=1)
        plan = {k:torch.from_numpy(data["plan_"+k][:128]).cuda() for k in ("query", "candidate", "weight")}
        gate = [p for n,p in net.named_parameters() if n.startswith("gnn.gate.")]
        conv = [p for n,p in net.named_parameters() if n.startswith("gnn.convs.")]
        normalized = run.context_score(key, plan)
        context = key[:, 288:]
        gated = ((context[plan["query"]]*context[plan["candidate"]]).sum(-1)*plan["weight"]).sum(-1).max(-1).values
        def gradient_norm(score: torch.Tensor, parameters: list) -> float:
            grads = torch.autograd.grad(score.mean(), parameters, retain_graph=True, allow_unused=True)
            return float(torch.sqrt(sum((g.square().sum() for g in grads if g is not None))))
        gradients = {name:{"gate":gradient_norm(score,gate), "convs":gradient_norm(score,conv)}
                     for name,score in (("normalized_context",normalized), ("gated_inner_product",gated))}
        changes = {}
        with torch.no_grad():
            for factor in (.25,2.):
                scaled = key.detach().clone()
                scaled[:,288:] *= factor
                scaled = F.normalize(scaled,dim=1)
                value = run.context_score(scaled,plan)
                changes[str(factor)] = float((value-normalized).abs().max())
                torch.testing.assert_close(value,normalized,atol=5e-7,rtol=0)
        assert gradients["normalized_context"]["convs"] > 0
        assert gradients["gated_inner_product"]["gate"] > 0
        records[sequence] = dict(score_gradient_norm=gradients,
            context_rescaling_max_error=changes,
            alpha_min=float(net.gnn._last_alpha.min()),
            alpha_max=float(net.gnn._last_alpha.max()),
            sources={str(p.relative_to(ROOT)):run.HASH(p) for p in (folder/"graph.pt",folder/"pool.npz")})
        del net,key,context,normalized,gated,graph
        torch.cuda.empty_cache()
    paths = [Path(__file__), ROOT/"src/gnn/model.py", ROOT/"src/retrieval/nonrevisit_learning.py"]
    run.WRITE(experiment.OUT/"gate_gradient_audit.json",dict(status="passed",records=records,
        interpretation="현재 최종 문맥 점수의 독립 L2 정규화는 양수 스칼라 게이트 강도를 제거한다. 게이트는 검색 키에는 남아 있으며 신경망 전체의 기울기가 없는 것은 아니다. 게이트를 유지한 내적은 학습 기울기 경로만 확인한 대안이며 성능 개선을 측정한 것이 아니다.",
        validation_used=False,test_used=False,training_or_model_changes=False,
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths}))
    print("GATE_AUDIT",records,flush=True)


if __name__ == "__main__":
    main()
