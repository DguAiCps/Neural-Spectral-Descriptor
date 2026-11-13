#!/usr/bin/env python3
"""정규화 전 출력의 접두부 오차를 실제 미래 제거와 분리해 진단한다."""
import copy

import numpy as np
import torch

import _pdv2_diffattn_gain_readout as e
from retrieval.context_spectral_gain import spectral_gain


def main() -> None:
    """같은 크기 미래 제거와 실제 절단을 같은 체크포인트에서 비교한다."""
    e.run.configure()
    spec = e.prepare()
    seq = "helipr/Roundabout01"
    name = seq.replace("/", "_")
    cache, _, _, travel, _, _, _, paths = e.run.inputs("val", seq)
    data = e.run.ARR(e.OUT / "pools/val" / name / "pool.npz")
    at = len(data["query_idx"]) // 2
    t = int(data["query_idx"][at])
    end = t + 16
    graph = e.run.make_graph(
        cache["descriptors"], cache["poses"], np.zeros(len(cache["poses"]), np.int64)
    ).cuda()
    short = e.run.make_graph(
        cache["descriptors"][:end], cache["poses"][:end], np.zeros(end, np.int64)
    ).cuda()
    padded = graph.clone()
    padded.x[end:] = 0
    invalid = (padded.edge_index[0] >= end) | (padded.edge_index[1] >= end)
    padded.edge_index[:, invalid] = len(padded.x) - 1
    padded.edge_attr[invalid] = 0
    z = e.prior.coefficients(paths[1])
    p = {
        k: torch.from_numpy(v).cuda()
        for k, v in e.prior.plan(
            travel, data["query_idx"][at : at + 1], data["j_star"][at : at + 1]
        ).items()
    }
    records = {}
    for arm in e.ARMS:
        for epoch in e.EPOCHS:
            tag = f"{arm}_{epoch}"
            net = e.model(e.ARMS[arm][0], e.OUT / arm / f"epoch{epoch}.pth")
            with torch.no_grad():
                full = net(graph)
                repeat = net(graph)
                cut = net(short)
                masked = net(padded)

                def error(a, b):
                    return float((a - b).abs().max())

                gs = spectral_gain(full[: t + 6], 0.5)
                rec = dict(
                    logit_prefix=error(cut[: t + 6, 288:], full[: t + 6, 288:]),
                    logit_padded=error(masked[: t + 6, 288:], full[: t + 6, 288:]),
                    logit_repeat=error(repeat[: t + 6, 288:], full[: t + 6, 288:]),
                    gain_prefix=error(spectral_gain(cut[: t + 6], 0.5), gs),
                    gain_padded=error(spectral_gain(masked[: t + 6], 0.5), gs),
                    score_prefix=error(e.score(cut, z[:end], p), e.score(full, z, p)),
                    score_padded=error(e.score(masked, z, p), e.score(full, z, p)),
                )
            records[tag] = rec
            print(tag, rec, flush=True)
            del net, full, repeat, cut, masked
    e.run.WRITE(
        e.OUT / "prefix_diagnostic.json",
        dict(sequence=seq, query=t, records=records, test_used=False),
    )


if __name__ == "__main__":
    main()
