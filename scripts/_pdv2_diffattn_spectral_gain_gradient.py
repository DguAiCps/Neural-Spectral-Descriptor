#!/usr/bin/env python3
"""새 이득 경로와 기존 가산 문맥 경로의 학습 미분을 분리한다."""
import numpy as np
import torch
import torch.nn.functional as F

import _pdv2_diffattn_spectral_gain as e


def main() -> None:
    """동일한 초기 균등 배치에서 두 경로의 크기와 방향을 측정한다."""
    e.run.configure()
    spec = e.prepare()
    net = e.model()
    params = [v for v in net.parameters() if v.requires_grad]
    fit = e.run.READ(e.PARENT / "calibration_recenter.json")
    slope = fit["spectral"]["coefficients"][0] / fit["spectral"]["scale"][0]
    result = {}
    for ordinal, seq in enumerate(spec["train"]):
        name = seq.replace("/", "_")
        data = e.run.ARR(e.previous.OUT / "pools" / name / "pool.npz")
        sets = [
            np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)
        ]
        sets = [s for s in sets if len(s)]
        rng = np.random.default_rng(42000 + ordinal)
        chosen = np.concatenate(
            [
                rng.choice(s, 64 // len(sets), replace=len(s) < 64 // len(sets))
                for s in sets
            ]
        )
        graph = torch.load(
            e.PARENT / "pools" / name / "graph.pt",
            map_location="cuda",
            weights_only=False,
        )
        z = e.coefficients(e.run.spec_path(seq))
        plan = {
            k: torch.from_numpy(data["plan_" + k][chosen]).cuda()
            for k in ("query", "candidate", "weight")
        }
        plan["weight"] = plan["weight"].float()
        key = F.normalize(net(graph), dim=-1)
        total, spectral = e.total_score(key, z, plan, 0.5)
        context = total - slope * spectral
        multiplier = (
            total.sigmoid() - torch.from_numpy(data["correct"][chosen]).cuda().float()
        ).detach() / len(chosen)
        grads = []
        for part in (slope * spectral, context):
            g = torch.autograd.grad(
                (part * multiplier).sum(), params, retain_graph=True
            )
            grads.append(torch.cat([v.flatten() for v in g]))
        result[seq] = dict(
            spectral_norm=float(grads[0].norm()),
            context_norm=float(grads[1].norm()),
            cosine=float(F.cosine_similarity(grads[0][None], grads[1][None])),
            loss=float(
                F.binary_cross_entropy_with_logits(
                    total, torch.from_numpy(data["correct"][chosen]).cuda().float()
                ).detach()
            ),
        )
        print(seq, result[seq], flush=True)
        del graph, z, key, total, spectral, context, grads, g
    e.run.WRITE(
        e.OUT / "gradient_diagnostic.json",
        dict(
            sequences=result,
            scope="학습8의초기균등64질의에서최종BCE미분분해. 전체학습기울기분포또는검증효과가아님.",
            test_used=False,
        ),
    )


if __name__ == "__main__":
    main()
