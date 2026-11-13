#!/usr/bin/env python3
"""고정 추론 실험의 최종 원장·기준 재현·온라인 입력 제약을 검사한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_fixed_fusion as experiment
from _pdv2_fixed_fusion import OUT, CFG, SELECTION, PREVIOUS, base, fixed_decisions
from _pdv2_no_context_ablation import sha256, write_json
from _xs_eval_lib import minmax
from _pr_eval_lib import classify_queries, pr_sweep

VAL, OPS, BASE, PILOT, METRICS = base.VAL, base.OPS, base.BASE, base.PILOT, base.METRICS
read, load_data = base.read, base.load_data


def main() -> None:
    """선택과 무관하게 고정한 기준으로 산출물 전체를 감사한다."""
    base.setup()
    protocol = experiment.protocol()
    for name, digest in protocol["input_sha256"].items():
        assert sha256(REPO / name) == digest, name
    result = {"status": "running", "ledgers": 0, "rows": 0, "initial_pipeline_checks": [],
              "prefix_checks": [], "score_replay_checks": [], "alignment_checks": [],
              "protocol_sha256": sha256(OUT / "protocol.json")}
    counts = [read(OUT / arm / "training_complete.json")["updates"] for arm in CFG["arms"]]
    assert len(set(counts)) == 1 and counts[0] == 252
    result["matched_updates"] = counts[0]
    selection = SELECTION
    for sensor, seq in VAL:
        sequence = sensor + "/" + seq
        cache, layout, spec, gt, md, _ = load_data("val", sequence)
        expected_q = np.arange(30, len(gt) - 15)
        for kind in ("gnn", "point"):
            path = OUT / "pools" / f"{kind}_initial/val/{sensor}_{seq}"
            with np.load(path / "keys.npz") as z:
                embedding = z["embedding"]
            for op in OPS:
                with np.load(path / (op + ".npz")) as z:
                    pools = {k: z[k] for k in z.files}
                refpath = PILOT / f"s1/{kind}_original/epoch10/val_{op}_{sensor}_{seq}.npz"
                with np.load(refpath) as z:
                    ref = {k: z[k] for k in z.files if k != "meta"}
                np.testing.assert_array_equal(pools["queries"], ref["query_idx"])
                weights = selection[op]["tuple"]
                chosen, confidence = [], []
                for i, q in enumerate(pools["queries"]):
                    valid = pools["candidates"][i] >= 0
                    c = pools["candidates"][i, valid]
                    static = pools["static"][i, valid].astype(np.float32)
                    scores = (weights[0] * minmax(embedding[c] @ embedding[q])
                              + weights[1] * minmax(static[:, 0].astype(np.float32))
                              + weights[2] * minmax(static[:, 1]) + weights[3] * minmax(static[:, 2]))
                    best = int(np.argmax(scores))
                    chosen.append(int(c[best]))
                    confidence.append(static[best, 2 if selection[op]["decision"] == "c_col" else 1])
                np.testing.assert_array_equal(chosen, ref["j_star"])
                ix, replay_score, _ = fixed_decisions(embedding, pools, op)
                np.testing.assert_array_equal(pools["candidates"][np.arange(len(ix)), ix], ref["j_star"])
                np.testing.assert_allclose(replay_score, ref[selection[op]["decision"]], atol=2e-6, rtol=0)
                np.testing.assert_allclose(confidence, ref[selection[op]["decision"]], atol=2e-6, rtol=0)
                result["initial_pipeline_checks"].append({"kind": kind, "sequence": sequence, "op": op,
                                                        "same_final_candidates": True})
        for arm in CFG["arms"]:
            for epoch in CFG["checkpoint_epochs"]:
                folder = OUT / arm / f"val_epoch{epoch}"
                meta = read(folder / f"{sensor}_{seq}.json")
                assert meta["protocol_sha256"] == sha256(OUT / "protocol.json")
                for op in OPS:
                    p = REPO / meta["ops"][op]["ledger"]
                    assert sha256(p) == meta["ops"][op]["sha256"]
                    with np.load(p) as z:
                        a = {k: z[k] for k in z.files}
                    np.testing.assert_array_equal(a["query_idx"], expected_q)
                    np.testing.assert_array_equal(a["min_past_gt_dist"], md[expected_q])
                    assert (a["j_star"] <= expected_q - 30).all() and (a["j_star"] >= 0).all()
                    distance = np.linalg.norm(gt[a["j_star"], :3, 3] - gt[expected_q, :3, 3], axis=1)
                    np.testing.assert_array_equal(a["top1_geo_dist"], distance)
                    assert np.isfinite(a["score_fixed"]).all()
                    qc = classify_queries(a["min_past_gt_dist"])
                    if qc["n_pos"]:
                        sweep = pr_sweep(a["score_fixed"], distance, qc)
                        for metric in METRICS[:3]:
                            assert abs(sweep[metric] - meta["ops"][op]["metrics"][metric]) < 1e-12
                    result["ledgers"] += 1
                    result["rows"] += len(expected_q)
    summary = read(OUT / "summary.json")
    for arm, record in summary["arms"].items():
        epoch = record["epoch"]
        for sensor, seq in VAL:
            sequence = sensor + "/" + seq
            folder = OUT / "pools" / f"{arm}_epoch{epoch}/val/{sensor}_{seq}"
            with np.load(folder / "keys.npz") as z:
                embedding = z["embedding"]
            _, _, spec, gt, _, _ = load_data("val", sequence)
            for op in OPS:
                with np.load(folder / (op + ".npz")) as z:
                    data = {k: z[k] for k in z.files}
                ix, scores, ranking = fixed_decisions(embedding, data, op)
                with np.load(OUT / arm / f"val_epoch{epoch}/{sensor}_{seq}_{op}.npz") as z:
                    np.testing.assert_array_equal(data["candidates"][np.arange(len(ix)), ix], z["j_star"])
                    np.testing.assert_array_equal(scores, z["score_fixed"])
                    np.testing.assert_array_equal(ranking, z["rank_score"])
                result["score_replay_checks"].append({"arm": arm, "sequence": sequence, "op": op, "queries": len(ix)})
                sample = np.linspace(0, len(ix) - 1, 32, dtype=int)
                fresh = experiment.reuse.ORIGINAL_BUILD_POOL(embedding, base.bank_for(op, spec), data["queries"][sample], gt[:, :3, 3])
                np.testing.assert_array_equal(fresh["candidates"], data["candidates"][sample])
                np.testing.assert_allclose(fresh["static"], data["static"][sample], atol=1e-6, rtol=1e-6)
                result["alignment_checks"].append({"arm": arm, "sequence": sequence, "op": op,
                    "queries": len(sample), "max_error": float(np.abs(fresh["static"] - data["static"][sample]).max())})
    write_json(OUT / "completion_validation.json", result)
    from _lag_branch_fusion import nsd_embedding_lag, TruncatedCache
    for arm in ("gnn_key", "gnn_fusion"):
        selected = summary["arms"][arm]["epoch"]
        cfg, net, clf, blob, rvec = base.make_encoder("gnn")
        state = torch.load(OUT / arm / f"epoch{selected}.pth", map_location="cpu", weights_only=False)
        net.load_state_dict(state["encoder"], strict=True)
        net.cpu().eval()
        clf.cpu().eval()
        for sequence in ("kitti360/0006", "helipr/Roundabout01"):
            cache, layout, spec, gt, _, _ = load_data("val", sequence)
            sensor = sequence.split("/")[0]
            raw = cache["descriptors"]
            full = nsd_embedding_lag(cache, sensor, raw, layout, cfg, net, clf, blob, rvec, "cpu")
            for query in (128, len(raw) // 2):
                limit = query + 16
                cut = nsd_embedding_lag(TruncatedCache(cache, limit), sensor, raw[:limit], layout[:limit],
                                        cfg, net, clf, blob, rvec, "cpu")
                error = float(np.abs(full[:query + 1] - cut[:query + 1]).max())
                assert error < 2e-6
                score_errors = {}
                for op in OPS:
                    packs = [experiment.reuse.ORIGINAL_BUILD_POOL(e, base.bank_for(op, s), np.array([query]), gt[:len(e), :3, 3])
                             for e, s in ((full, spec), (cut, {k: v[:limit] for k, v in spec.items()}))]
                    np.testing.assert_array_equal(packs[0]["candidates"], packs[1]["candidates"])
                    decisions = [fixed_decisions(e, p, op) for e, p in zip((full, cut), packs)]
                    for left, right in zip(decisions[0], decisions[1]):
                        np.testing.assert_allclose(left, right, atol=2e-6, rtol=0)
                    score_errors[op] = float(np.abs(decisions[0][2] - decisions[1][2]).max())
                result["prefix_checks"].append({"arm": arm, "sequence": sequence, "query": query,
                    "embedding_max_error": error, "rank_score_max_errors": score_errors})
    sources = {}
    for kind in ("gnn", "point"):
        for marker in sorted((PREVIOUS / "pools" / f"{kind}_initial").glob("*/*/complete.json")):
            sources.update(read(marker)["sources"])
    for name, digest in sources.items():
        assert sha256(REPO / name) == digest, name
    result.update(status="passed", source_files_checked=len(sources),
                  summary_sha256=sha256(OUT / "summary.json"), script_sha256=sha256(Path(__file__)))
    write_json(OUT / "completion_validation.json", result)
    print("FIXED_FUSION_VERIFIED", result["ledgers"], result["rows"], flush=True)


if __name__ == "__main__":
    main()
