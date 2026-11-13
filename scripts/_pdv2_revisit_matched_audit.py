#!/usr/bin/env python3
"""실제 재방문 학습 쌍·오답·일회 표본 순서·부분 그래프를 독립 검산한다."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_revisit_matched as run
from _pdv2_fusion_guard_gradient import vector_gradient
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash


def main() -> None:
    """모든 라벨·63배치·기울기 교정을 검사하고 부분 그래프 미분을 대조한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = run.protocol(True)
    pool = run.base.exp.arrays(run.OUT / "auxiliary_pool.npz")
    roster = run.base.exp.arrays(run.SUPPORT / "available_opposite_pairs.npz")
    sources = {**spec["runtime"], **spec["inputs"]}
    anchors = pool["anchors"]
    positions, groups, headings, valid = [], [], [], []
    for i, sequence in enumerate(spec["train"]):
        cache = run.base.exp.arrays(run.factor.paths_for("train", sequence)["cache"])
        p = cache["poses"][:, :3, 3].astype(np.float64)
        ids = np.arange(len(p))
        delta = p - p[np.maximum(0, ids - 5)]
        norm = np.linalg.norm(delta, axis=1)
        positions.append(p)
        groups.append(np.full(len(p), i, np.int64))
        headings.append(delta / np.maximum(norm[:, None], 1e-12))
        valid.append((ids >= 5) & (norm >= 1.0))
    positions, groups, headings, valid = map(
        np.concatenate, (positions, groups, headings, valid)
    )
    np.testing.assert_array_equal(pool["positions"], positions)
    np.testing.assert_array_equal(pool["groups"], groups)
    paircount = negcount = 0
    for index, a in enumerate(anchors):
        pp = pool["positives"][index]
        pp = pp[pp >= 0]
        nn = pool["negatives"][index]
        nn = nn[nn >= 0]
        np.testing.assert_array_equal(
            pp, np.unique(roster["positive_global"][roster["anchor_global"] == a])
        )
        assert np.all(
            np.linalg.norm(positions[pp] - positions[a], axis=1) <= 5.0 + 1e-10
        )
        assert (
            valid[a]
            and valid[pp].all()
            and np.all(headings[pp] @ headings[a] <= -0.5 + 1e-12)
        )
        assert np.all(np.linalg.norm(positions[nn] - positions[a], axis=1) > 25.0)
        for ss in np.unique(groups[np.r_[pp, nn]]):
            assert ss == groups[a] or (
                spec["train"][ss].startswith("helipr/Bridge")
                and spec["train"][groups[a]].startswith("helipr/Bridge")
            )
        paircount += len(pp)
        negcount += len(nn)
    for epoch in range(1, 5):
        batches = run.auxiliary_batches(epoch, len(anchors), 63)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(batches)), np.arange(len(anchors))
        )
        for arm in run.CFG["arms"]:
            complete = run.base.read(run.OUT / arm / "training_complete.json")
            logged = [
                row for row in complete["auxiliary_orders"] if row["epoch"] == epoch
            ]
            assert len(logged) == 63
            for b, row in zip(batches, logged):
                assert row["rows"] == array_hash(b)
    calibration = run.base.read(run.OUT / "calibration.json")
    vectors = run.base.exp.arrays(run.OUT / "calibration_gradients.npz")
    norms = {k: np.linalg.norm(v) for k, v in vectors.items()}
    coefficients = {}
    for arm in ("pair_key", "pair_context"):
        raw = 0.25 * norms["rank"] / norms[arm]
        actual = float(np.clip(raw, *run.CFG["coefficient_clip"]))
        assert abs(actual - spec["coefficients"][arm]) < 1e-14
        coefficients[arm] = {
            "raw": float(raw),
            "used": actual,
            "clipped": bool(abs(raw - actual) > 1e-14),
            "initial_gradient_norm_ratio": float(actual * norms[arm] / norms["rank"]),
        }
    # 독립 시퀀스에서 두 홉 부분 그래프와 전체 그래프의 파라미터 미분을 비교한다.
    sequence = "kitti/02"
    ordinal = spec["train"].index(sequence)
    graph = torch.load(
        run.pool_path(sequence) / "graph.pt", map_location="cpu", weights_only=False
    )
    offset = int((groups < ordinal).sum())
    subset = np.flatnonzero(groups[anchors] == ordinal)[:3]
    selected = {
        k: np.where(pool[k][subset] >= 0, pool[k][subset] - offset, -1)
        for k in ("anchors", "positives", "negatives")
    }
    local, indices, nodes, seeds = run.local_auxiliary(
        graph, np.zeros(graph.num_nodes, np.int64), selected, np.arange(len(subset))
    )
    net = run.initial_model()
    params = [p for p in net.parameters() if p.requires_grad]
    full_indices = {k: torch.from_numpy(v).cuda() for k, v in selected.items()}
    full = F.normalize(net(graph.cuda()), dim=1)
    terms = run.pair_terms(full, full_indices, "pair_context")["loss"].mean()
    direct = vector_gradient(terms, params, False).detach().cpu().numpy()
    full_values = full.detach().cpu().numpy()
    del full, terms
    small = F.normalize(net(local), dim=1)
    local_terms = run.pair_terms(small, indices, "pair_context")["loss"].mean()
    sliced = vector_gradient(local_terms, params, False).detach().cpu().numpy()
    error = float(np.linalg.norm(direct - sliced) / max(np.linalg.norm(direct), 1e-12))
    keyerror = float(
        abs(
            full_values[seeds]
            - small.detach().cpu().numpy()[np.searchsorted(nodes, seeds)]
        ).max()
    )
    assert error < 3e-5 and keyerror < 4e-6
    result = {
        "passed": True,
        "positive_pairs": paircount,
        "negative_pairs": negcount,
        "anchors": len(anchors),
        "epochs_with_each_anchor_once": 4,
        "coefficient_checks": coefficients,
        "subgraph_gradient_relative_error": error,
        "subgraph_key_max_error": keyerror,
        "test_used": False,
    }
    sources[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    for arm in run.CFG["arms"]:
        p = run.OUT / arm / "training_complete.json"
        sources[str(p.relative_to(REPO))] = sha256(p)
    for p, digest in sources.items():
        assert sha256(REPO / p) == digest, p
    write_json(run.OUT / "auxiliary_validation.json", {**result, "sources": sources})
    print("REVISIT_AUXILIARY_VALIDATED", result, flush=True)


if __name__ == "__main__":
    main()
