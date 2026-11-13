#!/usr/bin/env python3
"""간선 선택·품질 라벨·최종 원장과 선택 하한의 입력 절단을 감사한다."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_quality as exp
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """저장된 값에서 독립적으로 지표·라벨·제거 개수를 확인한다."""
    import torch
    from _causal_branch_fusion import select_edges_per_node
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_attention_neighbors_verify_reference import decision
    from _xs_eval_lib import normalize
    exp.protocol(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = exp.read(exp.OUT / "summary.json")
    selection = exp.read(exp.BASE / "s1/confidence_selection.json")["ops"]
    ledger_count = row_count = edge_count = 0
    drift, prefixes = [], []
    for name in exp.MODELS:
        for role, roster in (("val", exp.VAL), ("train", exp.TRAIN)):
            for sensor, seq in roster:
                quality = exp.read(exp.folder(name) / f"quality_{role}_{sensor}_{seq}.json")
                ep = REPO / quality["edge_file"]
                assert sha256(ep) == quality["edge_sha256"]
                with np.load(ep) as z:
                    a = {k: z[k] for k in z.files}
                q, j, p = a["query"], a["neighbor"], a["probability"]
                n = quality["nodes"]
                assert np.all(j <= q - 30) and np.all(j >= 0)
                np.testing.assert_array_equal(a["selected_order"], select_edges_per_node(q, p, 10))
                np.testing.assert_array_equal(np.bincount(q, minlength=n), np.minimum(30, np.maximum(0, np.arange(n) - 29)))
                if role == "val":
                    _, _, paths, _ = exp.alignment.base.inputs("val", sensor + "/" + seq)
                    with np.load(paths["cache"]) as z:
                        positions = z["poses_gt"][:, :3, 3]
                else:
                    with np.load(exp.alignment.cache_path(sensor + "/" + seq)) as z:
                        positions = z["poses"][:, :3, 3]
                d3 = np.linalg.norm(positions[q] - positions[j], axis=1)
                np.testing.assert_array_equal(a["distance_3d"], d3)
                ids = a["selected_order"]
                policy = exp.masks(q[ids], j[ids], p[ids], d3[ids] <= 5.)
                for policy_name, take in policy.items():
                    valid = a["valid_node"][q[ids[take]]]
                    report = quality["policy"][policy_name]["all"]
                    assert report["edges"] == int(valid.sum())
                    assert report["positive_3d"] == int((d3[ids[take]][valid] <= 5).sum())
                    assert report["far_3d"] == int((d3[ids[take]][valid] > 25).sum())
                    if role == "val":
                        out = exp.folder(name, policy_name)
                        with np.load(out / f"keys_{sensor}_{seq}.npz") as z:
                            meta = json.loads(str(z["meta"]))
                            assert meta["selected_edges"] == int(take.sum())
                            assert meta["protocol_sha256"] == sha256(exp.OUT / "protocol.json")
                            assert np.isfinite(z["embedding"]).all()
                            np.testing.assert_allclose(np.linalg.norm(z["embedding"], axis=1), 1., atol=3e-7)
                        record = exp.read(out / f"val_{sensor}_{seq}.json")
                        for op in exp.OPS:
                            rec = record["ops"][op]
                            path = REPO / rec["ledger"]
                            assert sha256(path) == rec["sha256"]
                            arrays, _ = exp.read_ledger(path)
                            ref, _ = exp.read_ledger(exp.BASE / f"s1/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
                            for key in ("query_idx", "min_past_gt_dist", "group"):
                                np.testing.assert_array_equal(arrays[key], ref[key])
                            assert np.all((arrays["j_star"] >= 0) & (arrays["j_star"] <= arrays["query_idx"] - 30))
                            s = selection[op]
                            assert exp.sequence_metrics(arrays, s["decision"], s["tau_star"]) == rec["metrics"]
                            ledger_count += 1
                            row_count += len(arrays["query_idx"])
                            if policy_name == "p000":
                                arm = exp.CFG["models"][name]
                                epoch = exp.protocol()["models"][name]["epoch"]
                                oldrow = exp.read(exp.alignment.folder(1, arm, epoch) / f"val_{sensor}_{seq}.json")
                                oldledger, _ = exp.read_ledger(REPO / oldrow["ops"][op]["ledger"])
                                drift.append({"model": name, "sequence": sensor + "/" + seq, "op": op,
                                              "key_max_error": quality["previous_key_max_error"],
                                              "changed_final_candidate": int((arrays["j_star"] != oldledger["j_star"]).sum()),
                                              "metric_delta": {k: None if rec["metrics"][k] is None else rec["metrics"][k] - oldrow["ops"][op]["metrics"][k] for k in exp.PRIMARY}})
                np.testing.assert_array_equal(np.bincount(q[ids[policy["p050"]]], minlength=n),
                                              np.bincount(q[ids[policy["random050"]]], minlength=n))
                edge_count += len(q)
        chosen = summary["chosen_threshold"][name]
        assert chosen in exp.CFG["selection_policies"]
        assert summary["scores"][name][chosen]["joint_AP"] == max(summary["scores"][name][p]["joint_AP"] for p in exp.CFG["selection_policies"])
        bundle = exp.load_model(name)
        cpu = list(copy.deepcopy(bundle))
        cpu[1].cpu().eval()
        cpu[2].cpu().eval()

        def compute(cache, sensor, layout, loaded, device):
            baseline, graphs, candidate = exp.capture(cache, sensor, layout, loaded, device)
            if chosen == "p000":
                return baseline
            ids = candidate["selected_order"]
            q, j, p = (candidate[k][ids] for k in ("query", "neighbor", "probability"))
            keep = exp.masks(q, j, p, np.zeros(len(q), bool))[chosen]
            with torch.no_grad():
                return normalize(loaded[1](exp.filter_graph(graphs[1], keep)).float().cpu().numpy())

        for sequence in ("kitti/09", "mulran/Sejong01"):
            sensor, seq, paths, _ = exp.alignment.base.inputs("val", sequence)
            with np.load(paths["cache"]) as z:
                cache = exp.ArrayCache({k: z[k] for k in z.files if k != "poses_gt"})
            with np.load(paths["layouts"]) as z:
                layout = z["nsd_layouts"]
            with np.load(exp.folder(name, chosen) / f"keys_{sensor}_{seq}.npz") as z:
                full = z["embedding"]
            cpu_full = compute(cache, sensor, layout, cpu, "cpu")
            with np.load(paths["spec"]) as z:
                banks = {op: exp.alignment.base.bank_for(op, z) for op in exp.OPS}
            for q in (300, 900):
                cut_cache = TruncatedCache(cache, q + 16)
                cpu_cut = compute(cut_cache, sensor, layout[:q + 16], cpu, "cpu")
                error = float(np.abs(cpu_full[:q + 1] - cpu_cut[:q + 1]).max())
                assert error < 2e-6, (name, sequence, q, error)
                gpu_cut = compute(cut_cache, sensor, layout[:q + 16], bundle, "cuda")
                checks = {}
                for op, bank in banks.items():
                    ca, ca_pool = decision(cpu_full, bank, q, selection[op])
                    cb, cb_pool = decision(cpu_cut, bank, q, selection[op])
                    assert ca == cb and np.array_equal(ca_pool, cb_pool)
                    ga, ga_pool = decision(full, bank, q, selection[op])
                    gb, gb_pool = decision(gpu_cut, bank, q, selection[op])
                    checks[op] = {"cpu_equal": True, "gpu_decision_equal": ga == gb,
                                  "gpu_candidate_set_equal": bool(np.array_equal(ga_pool, gb_pool))}
                prefixes.append({"model": name, "policy": chosen, "sequence": sequence, "q": q,
                                 "cpu_max_error": error, "gpu_max_error": float(np.abs(full[:q + 1] - gpu_cut[:q + 1]).max()), "checks": checks})
    assert ledger_count == 288 and not list(exp.OUT.glob("*/*/test_*"))
    limited = any(p["gpu_max_error"] >= 2e-6 or any(not c["gpu_decision_equal"] or not c["gpu_candidate_set_equal"] for c in p["checks"].values()) for p in prefixes)
    write_json(exp.OUT / "completion_validation.json", {
        "status": "passed_with_gpu_numerical_limit" if limited else "passed", "test_used": False,
        "checked_candidate_edges": edge_count, "checked_ledgers": ledger_count, "ledger_rows": row_count,
        "equal_random050_node_counts": True, "same_queries": True, "final_metrics_recomputed": True,
        "oracle_excluded_from_selection": True, "baseline_reproduction": drift, "prefix_checks": prefixes,
        "protocol_sha256": sha256(exp.OUT / "protocol.json"), "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("VERIFIED", ledger_count, row_count, edge_count, flush=True)


if __name__ == "__main__":
    main()
