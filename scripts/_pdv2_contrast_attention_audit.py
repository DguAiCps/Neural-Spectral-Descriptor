#!/usr/bin/env python3
"""선택된 모델이 균등 시간 주의와 고정 세기에서 얼마나 달라졌는지 잰다."""
from pathlib import Path

import numpy as np
import torch

import _pdv2_contrast_pipeline as experiment


def main() -> None:
    """검증 자료의 주의·감산 세기를 진단하며 설정을 선택하지 않는다."""
    run = experiment.run
    spec = experiment.prepare()
    results = {}
    with torch.no_grad():
        for sequence in spec["val"]:
            graph, _, _ = experiment.e.prior.graph_inputs("val", sequence)
            src, tgt = graph.edge_index
            degree = torch.bincount(tgt, minlength=len(graph.x)).clamp_min(1)
            result = {}
            for seed in (1, 2, 3):
                model = experiment.load_model(spec["cases"][f"diff_s{seed}"])
                output = model(graph)
                layers = []
                for layer, conv in enumerate(model.convs):
                    a = conv.attention
                    uniform = 1.0 / degree[tgt, None]
                    l1 = torch.zeros(len(graph.x), 4, device=a.device)
                    l1.index_add_(0, tgt, (a - uniform).abs())
                    entropy = torch.zeros_like(l1)
                    entropy.index_add_(0, tgt, -a * a.clamp_min(1e-20).log())
                    scaled_entropy = entropy / degree[:, None].float().log().clamp_min(
                        1.0
                    )
                    strength = model.last_strengths[layer].flatten().cpu().numpy()
                    layers.append(
                        dict(
                            attention_l1_from_uniform=float(l1.mean()),
                            normalized_entropy=float(scaled_entropy.mean()),
                            strength_quantiles=np.quantile(
                                strength, [0.05, 0.25, 0.5, 0.75, 0.95]
                            ).tolist(),
                            strength_mean=float(strength.mean()),
                        )
                    )
                result[str(seed)] = layers
                del model, output
            results[sequence] = result
            del graph
    summary = {
        f"layer{layer+1}": {
            k: float(
                np.mean(
                    [v[str(s)][layer][k] for v in results.values() for s in (1, 2, 3)]
                )
            )
            for k in (
                "attention_l1_from_uniform",
                "normalized_entropy",
                "strength_mean",
            )
        }
        for layer in (0, 1)
    }
    path = experiment.OUT / "attention_audit.json"
    run.WRITE(
        path,
        dict(
            results=results,
            sequence_equal_mean=summary,
            source_sha256=run.HASH(Path(__file__)),
            interpretation="선택모델의검증진단. 균등주의와다르게작동하는지를확인하며각변화의인과적기여나신규성증명은아님. 시험설정선택없음.",
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    experiment.e.configure()
    main()
