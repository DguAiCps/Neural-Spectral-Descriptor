#!/usr/bin/env python3
"""3D 라벨·학습 통계·실제 선택 간선·최종 원장·입력 절단을 검증한다."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_labels3d as exp
from _pdv2_no_context_ablation import sha256, write_json, no_context_key


def main() -> None:
    """원천 좌표·저장 특징·최종 판정에서 핵심 통계를 재계산한다."""
    import torch
    from _causal_branch_fusion import select_edges_per_node
    from _lag_branch_fusion import TruncatedCache, nsd_embedding_lag
    from _pdv2_attention_neighbors_verify_reference import decision
    from _xs_eval_lib import normalize

    spec = exp.protocol(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = exp.read(exp.OUT / "summary.json")
    training = {arm: exp.read(exp.OUT / "classifiers" / (arm + ".json")) for arm in ("xy", "xyz")}
    assert training["xy"]["initial_state_sha256"] == training["xyz"]["initial_state_sha256"]
    assert training["xy"]["features_sha256"] == training["xyz"]["features_sha256"]
    fit = {arm: [] for arm in training}
    counts = {arm: {role: {"positive": 0, "negative": 0} for role in ("fit", "holdout")} for arm in training}
    feature_edges, transitions, boundary = 0, {}, 0
    for sensor, seq in exp.TRAIN:
        path = exp.OUT / "features" / f"{sensor}_{seq}.npz"
        assert sha256(path) == training["xy"]["features_sha256"][str(path.relative_to(REPO))]
        with np.load(path) as z:
            q, j, x = z["query"], z["neighbor"], z["features"]
            with np.load(exp.old.alignment.cache_path(sensor + "/" + seq)) as cache:
                delta = cache["poses"][q, :3, 3] - cache["poses"][j, :3, 3]
                n = len(cache["poses"])
            assert x.shape == (len(q), 14) and x.dtype == np.float32 and np.isfinite(x).all()
            assert np.all(j <= q - 30)
            np.testing.assert_array_equal(np.bincount(q, minlength=n), np.minimum(30, np.maximum(0, np.arange(n) - 29)))
            role = "holdout" if sensor + "/" + seq in exp.CFG["holdout"] else "fit"
            for arm, dims in (("xy", 2), ("xyz", 3)):
                distance = np.sqrt(np.sum(delta[:, :dims] ** 2, axis=1))
                np.testing.assert_allclose(distance, z["distance_xy" if arm == "xy" else "distance_3d"], atol=1e-12, rtol=0)
                y = np.full(len(q), -1, np.int8)
                y[distance < 5.] = 1
                y[distance >= 25.] = 0
                np.testing.assert_array_equal(y, z["labels_" + arm])
                for value, label in ((1, "positive"), (0, "negative")):
                    counts[arm][role][label] += int((y == value).sum())
                if role == "fit":
                    fit[arm].append(x[y >= 0])
                boundary += int(((distance == 5) | (distance == 25)).sum())
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    key = f"{a}_to_{b}"
                    transitions[key] = transitions.get(key, 0) + int(((z["labels_xy"] == a) & (z["labels_xyz"] == b)).sum())
            feature_edges += len(q)
    for arm in training:
        record = training[arm]
        assert counts[arm] == record["rows"]
        x = np.concatenate(fit[arm])
        np.testing.assert_array_equal(np.asarray(record["mu"], np.float32), x.mean(0))
        np.testing.assert_array_equal(np.asarray(record["sd"], np.float32), x.std(0) + 1e-6)
        assert record["positive_weight"] == counts[arm]["fit"]["negative"] / counts[arm]["fit"]["positive"]
        assert record["epochs"] == 8 and record["updates"] == 8 * ((len(x) + 65535) // 65536)
        assert record["holdout_used_for_selection"] is False
        del x
    del fit
    selection = exp.read(exp.old.BASE / "s1/confidence_selection.json")["ops"]
    checked_ledgers = ledger_rows = validation_edges = 0
    drift, prefix, probability_errors = [], [], []
    for name in exp.MODELS:
        bundle = exp.load_model(name)
        classifiers = {"original": bundle[2:4], **{a: exp.load_classifier(a) for a in ("xy", "xyz")}}
        for sensor, seq in exp.VAL:
            quality = exp.read(exp.folder(name) / f"quality_{sensor}_{seq}.json")
            edge_path = exp.folder(name) / f"edges_{sensor}_{seq}.npz"
            assert sha256(edge_path) == quality["edge_sha256"]
            _, _, paths, _ = exp.old.alignment.base.inputs("val", sensor + "/" + seq)
            with np.load(paths["cache"]) as cache:
                positions, raw = cache["poses_gt"][:, :3, 3], cache["descriptors"]
                expected_key = no_context_key(raw, bundle[4])[:, :288]
            with np.load(edge_path) as z:
                q, j, d, valid = z["query"], z["neighbor"], z["distance_3d"], z["valid"]
                np.testing.assert_array_equal(d, np.linalg.norm(positions[q] - positions[j], axis=1))
                assert np.all(j <= q - 30)
                for arm in exp.ARMS:
                    p, ids = z["prob_" + arm], z["selected_" + arm]
                    if arm == "temporal":
                        assert len(ids) == 0
                    else:
                        np.testing.assert_array_equal(ids, select_edges_per_node(q, p, 10))
                        np.testing.assert_array_equal(np.bincount(q[ids], minlength=len(raw)), np.minimum(10, np.maximum(0, np.arange(len(raw)) - 29)))
                        clf, blob = classifiers[arm]
                        with torch.no_grad():
                            predicted = torch.sigmoid(clf(torch.from_numpy((z["features"] - blob["mu"]) / blob["sd"]).float().cuda())).cpu().numpy()
                        error = float(np.abs(predicted - p).max())
                        assert error < 3e-6, (name, sensor, seq, arm, error)
                        probability_errors.append(error)
                    for g, code in (("all", None), ("same_available", 0), ("opposite_only", 1), ("other_positive", 2), ("negative", 3), ("ambiguous", 4)):
                        mask = valid if code is None else valid & (z["group"] == code)
                        eligible = mask[q[ids]]
                        report = quality["arms"][arm][g]
                        assert report["edges"] == int(eligible.sum())
                        assert report["positive_3d"] == int((d[ids][eligible] <= 5).sum())
                        assert report["far_3d"] == int((d[ids][eligible] > 25).sum())
                        assert report["nodes_with_true_edge"] == len(np.unique(q[ids][eligible & (d[ids] <= 5)]))
                    key_path = exp.folder(name, arm) / f"keys_{sensor}_{seq}.npz"
                    assert sha256(key_path) == quality["keys_sha256"][arm]
                    with np.load(key_path) as keys:
                        embedding = keys["embedding"]
                        assert embedding.dtype == np.float32 and np.isfinite(embedding).all()
                        np.testing.assert_allclose(np.linalg.norm(embedding, axis=1), 1., atol=3e-7)
                        np.testing.assert_allclose(normalize(embedding[:, :288]), expected_key, atol=3e-7)
                        assert json.loads(str(keys["meta"]))["protocol_sha256"] == sha256(exp.OUT / "protocol.json")
                    record = exp.read(exp.folder(name, arm) / f"val_{sensor}_{seq}.json")
                    for op in exp.OPS:
                        rec = record["ops"][op]
                        assert sha256(REPO / rec["ledger"]) == rec["sha256"]
                        a, _ = exp.old.read_ledger(REPO / rec["ledger"])
                        ref, _ = exp.old.read_ledger(exp.old.BASE / f"s1/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
                        for key in ("query_idx", "min_past_gt_dist", "group"):
                            np.testing.assert_array_equal(a[key], ref[key])
                        assert np.all((a["j_star"] >= 0) & (a["j_star"] <= a["query_idx"] - 30))
                        sel = selection[op]
                        assert exp.old.sequence_metrics(a, sel["decision"], sel["tau_star"]) == rec["metrics"]
                        if arm == "original":
                            if name != "main":
                                prior = exp.old.alignment.folder(1, exp.old.CFG["models"][name], spec["models"][name]["epoch"])
                                row = exp.read(prior / f"val_{sensor}_{seq}.json")
                                ref, _ = exp.old.read_ledger(REPO / row["ops"][op]["ledger"])
                            old_metrics = exp.old.sequence_metrics(ref, sel["decision"], sel["tau_star"])
                            drift.append({"model": name, "sequence": sensor + "/" + seq, "op": op,
                                          "changed_final_candidate": int((a["j_star"] != ref["j_star"]).sum()),
                                          "metric_delta": {m: None if rec["metrics"][m] is None else rec["metrics"][m] - old_metrics[m] for m in exp.PRIMARY}})
                        checked_ledgers += 1
                        ledger_rows += len(a["query_idx"])
                validation_edges += len(q)
        for arm in exp.ARMS:
            rows = {s + "/" + q: exp.read(exp.folder(name, arm) / f"val_{s}_{q}.json") for s, q in exp.VAL}
            for op in exp.OPS:
                assert exp.old.aggregate_rows(rows, "val", op)["macro"] == summary["scores"][name][arm]["ops"][op]["macro"]
        # 저장 키 생성과 별도로, 원래 운영 함수에 새 분류기를 직접 넣어 검사한다.
        bundle = list(bundle)
        bundle[2], bundle[3] = exp.load_classifier("xyz")
        cpu = copy.deepcopy(bundle)
        cpu[1].cpu().eval()
        cpu[2].cpu().eval()

        def compute(cache, sensor, layout, loaded, device):
            cfg, net, clf, blob, rvec, _ = loaded
            return nsd_embedding_lag(cache, sensor, cache["descriptors"], layout,
                                      cfg, net, clf, blob, rvec, device)

        for sequence in ("kitti/09", "mulran/Sejong01"):
            sensor, seq, paths, _ = exp.old.alignment.base.inputs("val", sequence)
            with np.load(paths["cache"]) as z:
                cache = exp.old.ArrayCache({k: z[k] for k in z.files if k != "poses_gt"})
            with np.load(paths["layouts"]) as z:
                layout = z["nsd_layouts"].astype(np.float32)
            with np.load(exp.folder(name, "xyz") / f"keys_{sensor}_{seq}.npz") as z:
                gpu_full = z["embedding"]
            cpu_full = compute(cache, sensor, layout, cpu, "cpu")
            with np.load(paths["spec"]) as z:
                banks = {op: exp.old.alignment.base.bank_for(op, z) for op in exp.OPS}
            for q in (300, 900):
                cut = TruncatedCache(cache, q + 16)
                cpu_cut = compute(cut, sensor, layout[:q + 16], cpu, "cpu")
                error = float(np.abs(cpu_full[:q + 1] - cpu_cut[:q + 1]).max())
                assert error < 2e-6, (name, sequence, q, error)
                gpu_cut = compute(cut, sensor, layout[:q + 16], bundle, "cuda")
                checks = {}
                for op, bank in banks.items():
                    ca, ca_pool = decision(cpu_full, bank, q, selection[op])
                    cb, cb_pool = decision(cpu_cut, bank, q, selection[op])
                    assert ca == cb and np.array_equal(ca_pool, cb_pool)
                    ga, ga_pool = decision(gpu_full, bank, q, selection[op])
                    gb, gb_pool = decision(gpu_cut, bank, q, selection[op])
                    checks[op] = {"cpu_equal": True, "gpu_decision_equal": ga == gb,
                                  "gpu_candidate_set_equal": bool(np.array_equal(ga_pool, gb_pool))}
                prefix.append({"model": name, "sequence": sequence, "q": q, "cpu_max_error": error,
                               "gpu_max_error": float(np.abs(gpu_full[:q + 1] - gpu_cut[:q + 1]).max()), "checks": checks})
    assert checked_ledgers == 216 and not list(exp.OUT.glob("*/*/test_*"))
    limited = any(p["gpu_max_error"] >= 2e-6 or any(not c["gpu_decision_equal"] or not c["gpu_candidate_set_equal"] for c in p["checks"].values()) for p in prefix)
    write_json(exp.OUT / "completion_validation.json", {
        "status": "passed_with_gpu_numerical_limit" if limited else "passed", "test_used": False,
        "training_candidate_edges": feature_edges, "validation_candidate_edges": validation_edges,
        "label_transitions": transitions, "exact_distance_boundary_cases": boundary,
        "identical_initialization_and_features": True, "train_only_standardization_verified": True,
        "classifier_probability_max_error": max(probability_errors), "checked_ledgers": checked_ledgers,
        "ledger_rows": ledger_rows, "same_queries": True, "final_metrics_recomputed": True,
        "baseline_reproduction": drift, "prefix_checks": prefix,
        "protocol_sha256": sha256(exp.OUT / "protocol.json"), "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("VERIFIED", checked_ledgers, ledger_rows, flush=True)


if __name__ == "__main__":
    main()
