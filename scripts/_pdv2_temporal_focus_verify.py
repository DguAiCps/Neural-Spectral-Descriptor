#!/usr/bin/env python3
"""시간 전용 비교의 입력·학습 예산·원장·온라인 입력 범위를 검사한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_temporal_focus as exp
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """선택된 결과와 고정 에포크의 실제 최종 지표를 재계산한다."""
    import torch
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_pair_evaluate import ArrayCache
    from _pdv2_temporal3_verify import decision
    exp.protocol(True)
    torch.set_num_threads(1)
    summary = exp.read(exp.OUT / "summary.json")
    checked, total, prefixes, graphs, parameters = 0, 0, [], [], {}
    for seedstr, arms in summary["per_seed"].items():
        seed = int(seedstr)
        conf = exp.read(exp.base.BASE / f"s{seed}/confidence_selection.json")["ops"]
        initial = {a: exp.read(exp.arm_dir(seed, a) / "initial_model.json") for a in arms}
        common = set.intersection(*(set(v["tensor_sha256"]) for v in initial.values()))
        for key in common:
            assert len({v["tensor_sha256"][key] for v in initial.values()}) == 1
        orders = {a: exp.read(exp.arm_dir(seed, a) / "batch_order.json")["epochs"] for a in arms}
        neworders = [v for a, v in orders.items() if a not in exp.REUSED]
        if neworders:
            assert all([r["permutation_sha256"] for r in x] == [r["permutation_sha256"] for r in neworders[0]] for x in neworders)
        for arm, chosen in arms.items():
            parameters[f"s{seed}_{arm}"] = initial[arm]["n_parameters"]
            d = exp.arm_dir(seed, arm)
            graphs.append(exp.read(d / "training_graph.json"))
            progress = exp.read(d / "training_progress.json")
            assert (progress["epoch"], progress["optimizer_updates"]) == (20, 900)
            # 원래 표본 조건은 재사용 모델과 배치 배열도 같다.
            if arm == "standard_original":
                ref = exp.read(exp.old.OUT / f"s{seed}/temporal/batch_order.json")["epochs"]
                assert [r["sha256"] for r in orders[arm]] == [r["sha256"] for r in ref]
            assert chosen["joint_AP"] == max(c["joint_AP"] for c in chosen["candidates"])
            for epoch in exp.EPOCHS:
                c = next(c for c in chosen["candidates"] if c["epoch"] == epoch)
                assert c["ops"] == exp.base.aggregate(seed, arm, epoch, "val")
                for sensor, seq in exp.base.VAL:
                    record = exp.read(exp.folder(seed, arm, epoch) / f"val_{sensor}_{seq}.json")
                    for op in exp.base.OPS:
                        r = record["ops"][op]
                        path = REPO / r["ledger"]
                        assert sha256(path) == r["sha256"]
                        a, _ = exp.base.read_ledger(path)
                        ref, _ = exp.base.read_ledger(exp.base.BASE / f"s{seed}/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
                        for key in ("query_idx", "min_past_gt_dist", "group"):
                            np.testing.assert_array_equal(a[key], ref[key])
                        assert np.all(a["j_star"] <= a["query_idx"] - 30)
                        assert exp.base.sequence_metrics(a, conf[op]["decision"], conf[op]["tau_star"]) == r["metrics"]
                        checked += 1
                        total += len(a["query_idx"])
            cfg, net = exp.load_model(seed, arm, chosen["epoch"], "cpu")
            for sequence in ("kitti/09", "mulran/Sejong01"):
                sensor, seq, paths, _ = exp.base.inputs("val", sequence)
                with np.load(paths["cache"]) as z:
                    cache = ArrayCache({k: z[k] for k in z.files})
                raw = cache["descriptors"].astype(np.float32)
                kind = "point" if arm.startswith("point") else "temporal"
                full, _ = exp.old.embedding(TruncatedCache(cache, 1200), sensor, raw[:1200], cfg, net, kind, "cpu")
                with np.load(paths["spec"]) as z:
                    banks = {op: exp.base.bank_for(op, z) for op in exp.base.OPS}
                for q in (300, 900):
                    cut, _ = exp.old.embedding(TruncatedCache(cache, q+16), sensor, raw[:q+16], cfg, net, kind, "cpu")
                    err = float(np.abs(full[:q+1] - cut[:q+1]).max())
                    assert err < 2e-6
                    for op in exp.base.OPS:
                        da, ca = decision(full, banks[op], q, conf[op])
                        db, cb = decision(cut, banks[op], q, conf[op])
                        assert da == db
                        np.testing.assert_array_equal(ca, cb)
                        prefixes.append({"seed": seed, "arm": arm, "sequence": sequence, "q": q, "op": op, "max_error": err})
            print("VERIFIED", seed, arm, flush=True)
    assert all(g == graphs[0] for g in graphs)
    assert not list(exp.OUT.glob("s*/*/epoch*/test_*"))
    write_json(exp.OUT / "completion_validation.json", {"status": "passed", "ledgers": checked, "rows": total,
        "same_graph": True, "same_common_initial_tensors": True, "same_permutations": True,
        "updates_per_run": 900, "parameters": parameters, "prefix_checks": prefixes, "test_used": False,
        "scope": "CPU 첫1200/q+16의 표본 입력 범위 검사. 전체GPU실시간 검증 아님."})


if __name__ == "__main__":
    main()
