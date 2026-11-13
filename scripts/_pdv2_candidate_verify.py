#!/usr/bin/env python3
"""후보 학습 실험의 원장·초기 기준·온라인 의존성과 추가 판정 비용을 감사한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_candidate_learning as experiment
from _pdv2_candidate_learning import (OUT, CFG, VAL, OPS, BASE, PILOT, METRICS, read,
                                     make_encoder, load_data, bank_for, build_pool)
from _pdv2_no_context_ablation import sha256, write_json
from _xs_eval_lib import minmax
from _pr_eval_lib import classify_queries, pr_sweep
from retrieval.candidate_learning import CandidateScorer, pair_features


def main() -> None:
    """실제 산출물에서 독립적으로 결과와 인과 제약을 확인한다."""
    experiment.setup()
    protocol = experiment.protocol()
    for name, digest in protocol["input_sha256"].items():
        assert sha256(REPO / name) == digest, name
    result = {"status": "running", "ledgers": 0, "rows": 0, "initial_pipeline_checks": [],
              "prefix_checks": [], "timing_scope": "추가 후보 특징 조립+판정기만. 검색·GNN·정렬·관측대기 제외.",
              "protocol_sha256": sha256(OUT / "protocol.json")}
    states = [torch.load(OUT / arm / "initial_heads.pth", map_location="cpu", weights_only=True) for arm in CFG["arms"]]
    assert all(all(torch.equal(states[0][k], state[k]) for k in states[0]) for state in states)
    counts = [read(OUT / arm / "training_complete.json")["updates"] for arm in CFG["arms"]]
    assert len(set(counts)) == 1
    result["matched_updates"] = counts[0]
    selection = read(BASE / "s1/confidence_selection.json")["ops"]
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
                    static = pools["static"][i, valid]
                    scores = (weights[0] * minmax(embedding[c] @ embedding[q])
                              + weights[1] * minmax(static[:, 0].astype(np.float32))
                              + weights[2] * minmax(static[:, 1]) + weights[3] * minmax(static[:, 2]))
                    best = int(np.argmax(scores))
                    chosen.append(int(c[best]))
                    confidence.append(static[best, 2 if selection[op]["decision"] == "c_col" else 1])
                np.testing.assert_array_equal(chosen, ref["j_star"])
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
                    assert np.isfinite(a["score_learned"]).all()
                    qc = classify_queries(a["min_past_gt_dist"])
                    if qc["n_pos"]:
                        sweep = pr_sweep(a["score_learned"], distance, qc)
                        for metric in METRICS[:3]:
                            assert abs(sweep[metric] - meta["ops"][op]["metrics"][metric]) < 1e-12
                    result["ledgers"] += 1
                    result["rows"] += len(expected_q)
    # CPU 참조로 전체 입력과 q+15 절단 입력을 비교한다. GT는 모델 입력에서 분리한다.
    from _lag_branch_fusion import nsd_embedding_lag, TruncatedCache
    summary = read(OUT / "summary.json")
    selected = summary["arms"]["gnn_joint"]["epoch"]
    cfg, net, clf, blob, rvec = make_encoder("gnn")
    state = torch.load(OUT / "gnn_joint" / f"epoch{selected}.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(state["encoder"], strict=True)
    net.cpu().eval()
    clf.cpu().eval()
    heads = torch.nn.ModuleDict({op: CandidateScorer(tuple(CFG["head_hidden"])) for op in OPS}).eval()
    heads.load_state_dict(state["heads"], strict=True)
    for sequence in ("kitti360/0006", "helipr/Roundabout01"):
        cache, layout, spec, gt, _, _ = load_data("val", sequence)
        sensor = sequence.split("/")[0]
        raw = cache["descriptors"]
        full = nsd_embedding_lag(cache, sensor, raw, layout, cfg, net, clf, blob, rvec, "cpu")
        for q in (128, len(raw) // 2):
            limit = q + 16
            cut = nsd_embedding_lag(TruncatedCache(cache, limit), sensor, raw[:limit], layout[:limit],
                                    cfg, net, clf, blob, rvec, "cpu")
            error = float(np.abs(full[:q + 1] - cut[:q + 1]).max())
            assert error < 2e-6, (sequence, q, error)
            score_errors = {}
            for op in OPS:
                packs = [build_pool(e, bank_for(op, s), np.array([q]), gt[:len(e), :3, 3])
                         for e, s in ((full, spec), (cut, {k: v[:limit] for k, v in spec.items()}))]
                np.testing.assert_array_equal(packs[0]["candidates"], packs[1]["candidates"])
                logits = []
                with torch.no_grad():
                    for emb, pack in zip((full, cut), packs):
                        valid = torch.from_numpy(pack["candidates"] >= 0)
                        feat = pair_features(torch.from_numpy(emb), torch.tensor([q]),
                                             torch.from_numpy(pack["candidates"]), torch.from_numpy(pack["static"]).float(), valid)
                        logits.append(heads[op](feat).numpy()[valid.numpy()])
                score_error = float(np.abs(logits[0] - logits[1]).max())
                assert score_error < 1e-4
                assert int(np.argmax(logits[0])) == int(np.argmax(logits[1]))
                score_errors[op] = score_error
            result["prefix_checks"].append({"sequence": sequence, "query": q, "embedding_max_error": error,
                                            "candidate_score_max_error": score_errors})
    # 같은 추가 판정 경로를 한 질의씩 실제 GPU에서 잰다.
    heads.cuda()
    path = OUT / "pools" / f"gnn_joint_epoch{selected}/val/kitti360_0006"
    with np.load(path / "keys.npz") as z:
        emb = torch.from_numpy(z["embedding"]).cuda()
    timing = {}
    for op in OPS:
        with np.load(path / (op + ".npz")) as z:
            i = len(z["queries"]) // 2
            q, c, static = [torch.from_numpy(z[k][i:i + 1]).cuda() for k in ("queries", "candidates", "static")]
        valid = c >= 0
        static = static.float()
        durations = []
        with torch.no_grad():
            for index in range(120):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                heads[op](pair_features(emb, q, c, static, valid)).masked_fill(~valid, -float("inf")).max(1)
                end.record()
                end.synchronize()
                if index >= 20:
                    durations.append(start.elapsed_time(end))
        timing[op] = {"median_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95)),
                      "n_candidates": int(valid.sum()), "repetitions": 100}
    result.update(status="passed", scorer_timing=timing, summary_sha256=sha256(OUT / "summary.json"),
                  script_sha256=sha256(Path(__file__)))
    write_json(OUT / "completion_validation.json", result)
    print("VERIFIED", result["ledgers"], result["rows"], timing, flush=True)


if __name__ == "__main__":
    main()
