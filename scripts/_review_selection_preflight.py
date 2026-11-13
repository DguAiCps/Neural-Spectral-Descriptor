#!/usr/bin/env python3
"""묶음 접두 검색을 실제 전 질의·합성 동률에서 검산한다."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from _review_selection_prefix_batch import ledger_exact
from _bevplace_official_dump import ledger_l2


def main() -> None:
    """승자·점수·정답 거리 등 모든 필드의 정확 일치를 확인한다."""
    root = Path("results/pdv2_review_selection_20261105/preflight")
    records = {}
    for name in ("kitti_09", "kitti360_0004"):
        desc = np.load(root / f"descriptors_{name}.npy")
        sensor, seq = name.split("_")
        cp = next(Path("data/preprocessed_v2_odom").glob(f"cache_????????_{sensor}_val_{seq}.npz"))
        with np.load(cp) as z:
            pos = z["poses_gt"][:, :3, 3].astype(np.float64)
        tick = time.time()
        arrays, audit = ledger_exact(desc, pos)
        audit["runtime_s"] = time.time() - tick
        if name == "kitti_09":
            reference = ledger_l2(desc, pos)
        else:
            with np.load(root / f"search_full_{name}.npz") as z:
                reference = {k: z[k] for k in z.files}
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], reference[key])
        audit.update(all_fields_exact=True, n_queries=len(arrays["query_idx"]))
        records[name] = audit
        print(name, audit, flush=True)
    rng = np.random.default_rng(1)
    for name, epsilon in (("synthetic_ties", 0.0), ("synthetic_near_ties", 1e-5)):
        desc = rng.standard_normal((385, 8192)).astype(np.float32)
        desc /= np.linalg.norm(desc, axis=1)[:, None]
        desc[5:140] = desc[0] + epsilon * rng.standard_normal((135, 8192)).astype(np.float32)
        desc /= np.linalg.norm(desc, axis=1)[:, None]
        pos = rng.standard_normal((385, 3))
        arrays, audit = ledger_exact(desc, pos)
        reference = ledger_l2(desc, pos)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], reference[key])
        records[name] = dict(n_queries=len(arrays["query_idx"]), all_fields_exact=True, epsilon=epsilon)
    (root / "batch_preflight.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
