#!/usr/bin/env python3
"""3시드 학습 대조·최종 원장·선택과 CPU 입력 범위를 검증한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_temporal3 import OUT, ARMS, SEEDS, EPOCHS, base, read, protocol, load_model, embedding
from _pdv2_no_context_ablation import sha256, write_json


def decision(e: np.ndarray, bank: tuple, q: int, sel: dict) -> tuple:
    """질의 하나의 고정 검색·융합·수락을 별도로 계산한다."""
    from _causal_pr_dump import prefix_topk
    from _xs_eval_lib import align_scores, column_align_scores, minmax
    key, fft, norm, column = bank
    c = np.unique(np.r_[prefix_topk(e, q, q - 29, 100), prefix_topk(key, q, q - 29, 100)])
    sa, sc = align_scores(fft, norm, q, c), column_align_scores(column, q, c)
    w = sel["tuple"]
    scores = w[0] * minmax(e[c] @ e[q]) + w[1] * minmax(key[c] @ key[q]) + w[2] * minmax(sa) + w[3] * minmax(sc)
    b = int(np.argsort(-scores, kind="stable")[0])
    assert sel["decision"] in ("c_spec", "c_col")
    confidence = float(sc[b] if sel["decision"] == "c_col" else sa[b])
    return {"j_star": int(c[b]), "confidence": confidence, "accepted": confidence >= sel["tau_star"]}, c


def main() -> None:
    """동일 표본·배치·공유층 초기화와 모든 최종 지표를 검사한다."""
    import torch
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_pair_evaluate import ArrayCache
    protocol(True)
    torch.set_num_threads(1)
    summary = read(OUT / "summary.json")
    ledgers, rows, prefix, shared, negatives = 0, 0, [], {}, {}
    for seed in SEEDS:
        left, right = [OUT / f"s{seed}/{a}" for a in ARMS]
        for name in ("training_graph.json", "mining_calls.json", "batch_order.json"):
            assert read(left / name) == read(right / name), (seed, name)
        init = [read(p / "initial_model.json") for p in (left, right)]
        common = set(init[0]["tensor_sha256"]) & set(init[1]["tensor_sha256"])
        assert common and all(init[0]["tensor_sha256"][k] == init[1]["tensor_sha256"][k] for k in common)
        shared[str(seed)] = {"common_tensor_count": len(common), "parameters": {a: i["n_parameters"] for a, i in zip(ARMS, init)}}
        selection = read(base.BASE / f"s{seed}/confidence_selection.json")["ops"]
        negatives[str(seed)] = {}
        for arm in ARMS:
            p = read(OUT / f"s{seed}/{arm}/training_progress.json")
            assert p["epoch"] == 20 and p["optimizer_updates"] == 900
            chosen = summary["per_seed"][str(seed)][arm]
            assert chosen["joint_AP"] == max(r["joint_AP"] for r in chosen["candidates"])
            negatives[str(seed)][arm] = {}
            for epoch in EPOCHS:
                stored = next(r for r in chosen["candidates"] if r["epoch"] == epoch)
                assert stored["ops"] == base.aggregate(seed, arm, epoch, "val")
                count, positive = 0, 0
                for sensor, seq in base.VAL:
                    folder = base.folder(seed, arm, epoch)
                    record = read(folder / f"val_{sensor}_{seq}.json")
                    with np.load(folder / f"keys_val_{sensor}_{seq}.npz") as z:
                        meta = json.loads(str(z["meta"]))
                        assert np.isfinite(z["embedding"]).all()
                        assert meta["graph"]["similarity"] == 0 and meta["graph"]["passes"] == 1
                        assert (meta["graph"]["edges"] == 0) == (arm == "point")
                        assert meta["protocol_sha256"] == sha256(OUT / "protocol.json")
                    for op in base.OPS:
                        r = record["ops"][op]
                        path = REPO / r["ledger"]
                        assert sha256(path) == r["sha256"]
                        a, _ = base.read_ledger(path)
                        ref, _ = base.read_ledger(base.BASE / f"s{seed}/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
                        for key in ("query_idx", "min_past_gt_dist", "group"):
                            np.testing.assert_array_equal(a[key], ref[key])
                        assert np.all(a["j_star"] <= a["query_idx"] - 30) and np.all(a["j_star"] >= 0)
                        sel = selection[op]
                        assert base.sequence_metrics(a, sel["decision"], sel["tau_star"]) == r["metrics"]
                        if epoch == chosen["epoch"]:
                            mask = a["min_past_gt_dist"] > 25.
                            negatives[str(seed)][arm].setdefault(op, {})[sensor + "/" + seq] = {
                                "negative_queries": int(mask.sum()),
                                "false_accepts": int((mask & (a[sel["decision"]] >= sel["tau_star"])).sum())}
                        rows += len(a["query_idx"])
                        ledgers += 1
                    count += record["ops"][base.OPS[0]]["metrics"]["n_queries"]
                    positive += record["ops"][base.OPS[0]]["metrics"]["n_positive"]
                assert (count, positive) == (99236, 10036)
            cfg, net = load_model(seed, arm, chosen["epoch"], "cpu")
            for sequence in ("kitti/09", "mulran/Sejong01"):
                sensor, seq, paths, _ = base.inputs("val", sequence)
                with np.load(paths["cache"]) as z:
                    cache = ArrayCache({k: z[k] for k in z.files})
                raw = cache["descriptors"].astype(np.float32)
                full, _ = embedding(TruncatedCache(cache, 1200), sensor, raw[:1200], cfg, net, arm, "cpu")
                with np.load(paths["spec"]) as z:
                    banks = {op: base.bank_for(op, z) for op in base.OPS}
                for q in (300, 900):
                    cut, _ = embedding(TruncatedCache(cache, q + 16), sensor, raw[:q + 16], cfg, net, arm, "cpu")
                    error = float(np.abs(full[:q + 1] - cut[:q + 1]).max())
                    assert error < 2e-6, (seed, arm, sequence, q, error)
                    for op in base.OPS:
                        da, ca = decision(full, banks[op], q, selection[op])
                        db, cb = decision(cut, banks[op], q, selection[op])
                        assert da == db
                        np.testing.assert_array_equal(ca, cb)
                        prefix.append({"seed": seed, "arm": arm, "sequence": sequence,
                                       "q": q, "op": op, "cpu_max_error": error, "same_final_decision_and_pool": True})
            print("VERIFIED", seed, arm, flush=True)
    assert not list(OUT.glob("s*/*/epoch*/test_*"))
    for op in base.OPS:
        for metric in base.PRIMARY:
            for arm in ARMS:
                vals = [summary["per_seed"][str(s)][arm]["ops"][op]["macro"][metric] for s in SEEDS]
                expected = summary["mean_std"][op][arm][metric]
                assert float(np.mean(vals)) == expected["mean"]
                assert float(np.std(vals, ddof=1)) == expected["std"]
    write_json(OUT / "negative_false_accepts.json", negatives)
    write_json(OUT / "completion_validation.json", {
        "status": "passed", "checked_ledgers": ledgers, "total_rows": rows,
        "common_initialization": shared, "same_triplets_and_batch_order": True,
        "same_training_graph": True, "optimizer_updates_per_run": 900,
        "final_metrics_exactly_reproduced": True, "prefix_checks": prefix, "test_used": False,
        "scope": "첫1200과q+16의CPU입력범위표본검사. 전체GPU스트리밍재현·실시간속도 검증 아님.",
        "protocol_sha256": sha256(OUT / "protocol.json"), "summary_sha256": sha256(OUT / "summary.json")})
    print("PASSED", ledgers, rows, len(prefix), flush=True)


if __name__ == "__main__":
    main()
