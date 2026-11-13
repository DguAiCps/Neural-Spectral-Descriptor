#!/usr/bin/env python3
"""학습 대조·최종 지표·그래프와 q+15 입력 한계를 검사한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_attention_neighbors import (
    OUT, ARMS, EPOCHS, base, read, protocol, load_bundle, embedding,
)
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """저장된 원장 전체 재집계와 선택 모델의 잘린 입력 평가를 수행한다."""
    import torch
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_pair_evaluate import ArrayCache
    protocol(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = read(OUT / "summary.json")
    initial = read(OUT / "s1/diff_full/initial_model.json")
    samples = read(OUT / "s1/diff_full/mining_calls.json")
    common_graph = read(OUT / "s1/diff_full/graph_counts.json")
    assert initial["n_parameters"] == 167561
    ledgers, total_rows, graph_audits, prefixes = 0, 0, [], []
    for arm in ARMS:
        folder = OUT / "s1" / arm
        assert read(folder / "initial_model.json") == initial
        assert read(folder / "mining_calls.json") == samples
        train = read(folder / "training_progress.json")
        assert train["epoch"] == 20 and train["optimizer_updates"] == 900
        graph = read(folder / "graph_counts.json")
        for key in ("nodes", "temporal", "temporal_index_sha256", "temporal_attr_sha256"):
            assert graph[key] == common_graph[key]
        if arm.endswith("temporal"):
            assert graph["similarity"] == 0
        else:
            assert graph == common_graph and graph["similarity"] > 0
        chosen = summary["arms"][arm]
        assert chosen["joint_AP"] == max(r["joint_AP"] for r in chosen["candidates"])
        for epoch in EPOCHS:
            recorded = next(r for r in chosen["candidates"] if r["epoch"] == epoch)
            assert recorded["ops"] == base.aggregate(1, arm, epoch, "val")
            count, positives = 0, 0
            for sensor, seq in base.VAL:
                folder_e = base.folder(1, arm, epoch)
                row = read(folder_e / f"val_{sensor}_{seq}.json")
                with np.load(folder_e / f"keys_val_{sensor}_{seq}.npz") as z:
                    keymeta = json.loads(str(z["meta"]))
                    assert np.isfinite(z["embedding"]).all()
                assert keymeta["value_source"] == arm.split("_")[0]
                assert keymeta["protocol_sha256"] == sha256(OUT / "protocol.json")
                if arm.endswith("temporal"):
                    assert len(keymeta["graphs"]) == 1 and keymeta["graphs"][0]["similarity"] == 0
                else:
                    assert len(keymeta["graphs"]) == 2 and keymeta["graphs"][1]["similarity"] > 0
                graph_audits.append({"arm": arm, "epoch": epoch, "sequence": sensor + "/" + seq,
                                     "graphs": keymeta["graphs"]})
                selection = read(base.BASE / "s1/confidence_selection.json")["ops"]
                for op in base.OPS:
                    record = row["ops"][op]
                    path = REPO / record["ledger"]
                    assert sha256(path) == record["sha256"]
                    arrays, _ = base.read_ledger(path)
                    ref, _ = base.read_ledger(base.BASE / f"s1/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
                    for key in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(arrays[key], ref[key])
                    assert np.all(arrays["j_star"] <= arrays["query_idx"] - 30)
                    assert np.all(arrays["j_star"] >= 0)
                    sel = selection[op]
                    assert base.sequence_metrics(arrays, sel["decision"], sel["tau_star"]) == record["metrics"]
                    ledgers += 1
                    total_rows += len(arrays["query_idx"])
                count += row["ops"][base.OPS[0]]["metrics"]["n_queries"]
                positives += row["ops"][base.OPS[0]]["metrics"]["n_positive"]
            assert (count, positives) == (99236, 10036)
        bundle = load_bundle(arm, chosen["epoch"])
        for sequence in ("kitti/09", "mulran/Sejong01"):
            sensor, seq, paths, _ = base.inputs("val", sequence)
            with np.load(paths["cache"]) as z:
                cache = ArrayCache({k: z[k] for k in z.files})
            with np.load(paths["layouts"]) as z:
                layout = z["nsd_layouts"].astype(np.float32)
            raw = cache["descriptors"].astype(np.float32)
            with np.load(base.folder(1, arm, chosen["epoch"]) / f"keys_val_{sensor}_{seq}.npz") as z:
                full = z["embedding"]
            for q in (300, 900):
                n = q + 16
                cut, _ = embedding(TruncatedCache(cache, n), sensor, raw[:n], layout[:n], bundle, arm)
                error = float(np.abs(cut[:q + 1] - full[:q + 1]).max())
                assert error < 2e-6, (arm, sequence, q, error)
                prefixes.append({"arm": arm, "sequence": sequence, "q": q, "embedding_max_error": error})
        del bundle
        print("VERIFIED", arm, flush=True)
    assert not list(OUT.glob("s*/*/epoch*/test_*"))
    write_json(OUT / "completion_validation.json", {
        "status": "passed", "checked_ledgers": ledgers, "total_rows": total_rows,
        "same_initial_weights": True, "same_training_triplets": True,
        "same_temporal_edges_and_attributes": True, "optimizer_updates_per_arm": 900,
        "parameters_per_arm": initial["n_parameters"], "graphs": graph_audits,
        "prefix_checks": prefixes, "final_metrics_reproduced": True, "test_used": False,
        "protocol_sha256": sha256(OUT / "protocol.json"), "summary_sha256": sha256(OUT / "summary.json"),
    })
    print("PASSED", ledgers, total_rows, flush=True)


if __name__ == "__main__":
    main()
