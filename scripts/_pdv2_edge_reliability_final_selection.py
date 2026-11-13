#!/usr/bin/env python3
"""전체 입력을 그대로 재실행해 저장 후보와 GPU 재생성 후보의 차이를 설명한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_edge_reliability_final as exp
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """미래 입력을 자르지 않은 같은 전체 입력에서 후보·특징·선택의 차이를 센다."""
    import torch
    exp.protocol(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sequence, s, q = "mulran/Sejong01", "mulran", "Sejong01"
    cache, layout, spec, _, _, _ = exp.rel.load_data("val", sequence)
    n = len(cache["scan_ids"])
    with np.load(exp.base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
        old = {k: z[k] for k in z.files}
    with np.load(exp.OUT / "edges" / f"{s}_{q}.npz") as z:
        saved = {k: z[k] for k in z.files}
    bundle = exp.base.load_model("main")
    _, _, new = exp.base.capture(cache, s, layout, bundle, "cuda")
    extra = exp.rel.extra_features(spec, new["query"], new["neighbor"])
    scores = {**exp.probabilities(new["features"], extra), **extra}
    np.testing.assert_array_equal(old["query"], new["query"])
    pk0, pk1 = old["query"] * n + old["neighbor"], new["query"] * n + new["neighbor"]
    shared, i0, i1 = np.intersect1d(pk0, pk1, assume_unique=True, return_indices=True)
    errors = np.abs(old["features"][i0] - new["features"][i1])
    candidate_changes, order_changes, examples = 0, 0, []
    for query in range(30, n):
        lo, hi = np.searchsorted(old["query"], [query, query + 1])
        a, b = old["neighbor"][lo:hi], new["neighbor"][lo:hi]
        order_changes += int(not np.array_equal(a, b))
        candidate_changes += int(not np.array_equal(np.sort(a), np.sort(b)))
    policies = {}
    for op in exp.OPS:
        for arm in exp.ARMS:
            _, ids = exp.choose(new, scores, op, arm)
            oldids = saved[f"selected__{op}__{arm}"]
            key0, key1 = pk0[oldids], pk1[ids]
            removed, added = np.setdiff1d(key0, key1), np.setdiff1d(key1, key0)
            queries = np.unique(np.r_[removed // n, added // n])
            policies[op + "/" + arm] = {"changed_queries": len(queries), "removed_edges": len(removed),
                "added_edges": len(added), "first_queries": queries[:10].tolist()}
            for query in queries[:3]:
                a = oldids[old["query"][oldids] == query]
                b = ids[new["query"][ids] == query]
                examples.append({"op": op, "arm": arm, "query": int(query),
                    "old_neighbors": old["neighbor"][a].tolist(), "new_neighbors": new["neighbor"][b].tolist()})
    result = {"sequence": sequence, "nodes": n, "input_truncated": False,
        "candidate_queries": n - 30, "candidate_set_changed_queries": candidate_changes,
        "candidate_order_changed_queries": order_changes, "shared_pairs": len(shared),
        "features_max_error_by_column": errors.max(0).tolist(),
        "features_changed_pairs_by_column": (errors > 0).sum(0).tolist(),
        "policies": policies, "examples": examples,
        "interpretation": "전체 입력·모델·코드를 유지해도 저장한 GPU 첫 통과와 재생성한 특징/후보가 달라지는지 검사한다. 차이가 있으면 입력 절단만의 효과가 아니다. 이 기록은 새 성능 선택이나 원장 교체에 쓰지 않는다.",
        "script_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(exp.OUT / "protocol.json")}
    write_json(exp.OUT / "full_input_regeneration.json", result)
    print("REGENERATION", candidate_changes, order_changes, policies, flush=True)


if __name__ == "__main__":
    main()
