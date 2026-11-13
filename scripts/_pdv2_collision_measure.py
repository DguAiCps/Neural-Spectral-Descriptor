#!/usr/bin/env python3
"""고정된 현재 NSD의 충돌·퍼짐과 검색 단계별 회복을 측정한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  python3 scripts/_pdv2_collision_measure.py --run

원고의 시드 1·46시퀀스 진단이다. 모델·융합·판정 기준을 새로 고르지
않는다. 같은 실행의 키와 매칭을 함께 저장하며 기존 결과를 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]

from _collision_diagnostics_lib import miss_categories, rank_prefix  # noqa: E402
from _pdv2_pilot import TEST  # noqa: E402

DEFAULT_OUT = REPO / "results/pdv2_collision_20261013"
OPS = ("pbev128_main", "hspec12")
STAGES = ("raw", "weighted", "pass1", "context")
EPSILONS = (.05, .1, .2)


def write_json(path: Path, value: dict) -> None:
    """완성된 JSON을 원자적으로 저장한다."""
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def setup_protocol(out: Path) -> dict:
    """시험 분석 전에 고정한 정의와 원천 모델 해시를 보존한다."""
    from _audit_current_nsd import sha256

    files = [Path(__file__), REPO / "scripts/_collision_diagnostics_lib.py",
             REPO / "scripts/_causal_branch_fusion.py", REPO / "scripts/_causal_edge_features.py",
             REPO / "scripts/_causal_pr_dump.py", REPO / "scripts/_xs_eval_lib.py",
             REPO / "src/gnn/model.py", REPO / "src/keyframe/graph_manager.py",
             REPO / "checkpoints/pdv2_causal_s1/best_model.pth",
             REPO / "artifacts/key_remetrize_r_pdv2.npy",
             REPO / "artifacts/edge_classifier_pdv2_causal.pt",
             REPO / "configs/training_place_disjoint_v2_causal.yaml",
             REPO / "results/pdv2_c3_bevclip_s1/odom/causal_tuple_selection.json",
             REPO / "results/pdv2_c3_bevclip_s1/odom/confidence_selection_align.json"]
    protocol = {
        "schema": "nsd-collision-diagnostics-v1", "date": "2026-10-13", "seed": 1,
        "roster": [f"{s}/{q}" for s, q in TEST], "condition": "odom, bevclip, corrected NCLT d",
        "min_index_gap": 30, "match_radius_m": 5., "far_radius_m": 25.,
        "primary_raw_epsilon": .1, "sensitivity_raw_epsilons": list(EPSILONS),
        "categories": {"0": "correct", "1": "spread", "2": "collision",
                       "3": "near_field", "4": "no_revisit"},
        "definition": "같은 과거 gallery에서 원래 정규화 d의 top1 오답만 분해한다. "
                      "best same-place descriptor L2 > epsilon이면 spread, 나머지 오답은 "
                      "top1 지리 거리 >=25m일 때 collision, 5m 초과 25m 미만이면 near_field다. "
                      "다른 표현의 거리 척도에는 이 epsilon을 적용하지 않는다.",
        "stages": list(STAGES),
        "stage_interpretation": "각 키의 검색 대조이며 순차 입력 변환이 아니다. "
                                "두 GNN 통과의 입력은 원래 d이며, weighted는 재계량만 한 대조다.",
        "final_matching": "원래 검증 선택의 100+100 합집합·네 점수·가중치·c_spec·tau를 유지한다.",
        "no_alignment_ranking_diagnostic": "같은 후보에서 원래 w_f,w_m만 남긴 순위 진단. "
                                            "판정 점수를 새로 고른 제거 실험이 아니며 AP 기여로 해석하지 않는다.",
        "aggregation": "시퀀스별 비율→도시 안 평균→도시 6개 평균. 건수·질의 통합 비율도 별도 보고한다.",
        "scope": "이미 본 시험 자료의 시드 1 사후 진단. 모델 선택이나 새 확증 시험이 아니다. "
                 "실패 종류는 원래 d에서 고정해 전체 재방문 분모와 종류별 분모를 함께 보고한다.",
        "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in files},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "고정 프로토콜 또는 원천이 달라졌다"
    else:
        write_json(path, protocol)
    return protocol


def worker(seq: str, out: Path) -> None:
    """시퀀스 하나를 실제 과거 전용 경로로 추론하고 질의 기록을 만든다."""
    import torch
    from _audit_current_nsd import local_path, sha256
    from _causal_branch_fusion import VARIANTS, make_model_causal, nsd_embedding_causal
    from _causal_pr_dump import prefix_topk
    from _xs_eval_lib import (align_scores, column_align_scores, hspec_inputs,
                             minmax, normalize, variant_inputs)

    started = time.monotonic()
    torch.set_num_threads(2)
    sensor, sequence = seq.split("/")
    assert (sensor, sequence) in TEST
    root = REPO / "results/pdv2_c3_bevclip_s1/odom"
    paths = {op: root / f"dump_pdv2_causal_{op}_s1_{sensor}_{sequence}.npz" for op in OPS}
    ledgers = {op: np.load(p) for op, p in paths.items()}
    meta = json.loads(str(ledgers[OPS[0]]["meta"]))
    cache_paths = {k: local_path(v) for k, v in meta["caches"].items() if k != "marker"}
    cache, layout, spectrum = [np.load(cache_paths[k]) for k in ("cache", "layouts", "spec")]
    for z in (layout, spectrum):
        np.testing.assert_array_equal(cache["scan_ids"], z["scan_ids"])
    artifacts = {k: local_path(v) for k, v in meta["artifacts"].items()}
    VARIANTS["collision_current"] = {"ckpt_dir": "pdv2_causal_s1", "rvec": artifacts["rvec"],
                                      "cls": artifacts["classifier"], "config": artifacts["config"]}
    cfg, model, clf, blob, weights, checkpoint = make_model_causal("collision_current", 1, "cuda")
    assert checkpoint == artifacts["checkpoint"]
    raw = cache["descriptors"].astype(np.float32)
    passes = []

    def capture(module, inputs, output):
        np.testing.assert_array_equal(inputs[0].x.cpu().numpy(), raw)
        passes.append(normalize(output.detach().cpu().numpy()))

    handle = model.register_forward_hook(capture)
    embedding = nsd_embedding_causal(cache, sensor, raw, layout["nsd_layouts"].astype(np.float32),
                                     cfg, model, clf, blob, weights, "cuda")
    handle.remove()
    assert len(passes) == 2
    np.testing.assert_array_equal(embedding, passes[1])
    banks = {"raw": normalize(raw), "weighted": normalize(raw * weights),
             "pass1": passes[0], "context": embedding}
    positions = cache["poses_gt"][:, :3, 3]
    queries = np.arange(30, len(raw), dtype=np.int64)
    nq = len(queries)
    arrays = {"query_idx": queries, "query_scan_id": cache["scan_ids"][queries],
              "min_past_gt_dist": np.array([np.linalg.norm(positions[:int(q) - 29] - positions[q], axis=1).min()
                                             for q in queries])}
    positive = arrays["min_past_gt_dist"] <= 5
    for old in ledgers.values():
        np.testing.assert_array_equal(queries, old["query_idx"])
        np.testing.assert_allclose(arrays["min_past_gt_dist"], old["min_past_gt_dist"], atol=1e-9)
    for stage in STAGES:
        for key in ("top1_idx", "best_pos_idx", "best_pos_rank"):
            arrays[f"{stage}_{key}"] = np.full(nq, -1, dtype=np.int64)
        for key in ("top1_geo", "best_pos_dist"):
            arrays[f"{stage}_{key}"] = np.full(nq, np.nan)
    for i in np.flatnonzero(positive):
        query = int(queries[i])
        for stage, bank in banks.items():
            result = rank_prefix(bank, query, positions)
            for key in ("top1_idx", "best_pos_idx", "best_pos_rank", "top1_geo", "best_pos_dist"):
                arrays[f"{stage}_{key}"][i] = result[key]
    for epsilon in EPSILONS:
        arrays[f"raw_category_eps_{epsilon:g}"] = miss_categories(
            arrays["min_past_gt_dist"], arrays["raw_top1_geo"], arrays["raw_best_pos_dist"], epsilon)
    confidence = json.loads((root / "confidence_selection_align.json").read_text())["ops"]
    differences = {}
    for op in OPS:
        op_meta = json.loads(str(ledgers[op]["meta"]))
        # NumPy 2의 float64 스칼라는 float32 점수를 승격하므로 원래 평가처럼
        # Python float를 유지한다. 근소한 점수 차이와 동률의 선택을 보존한다.
        w = tuple(float(value) for value in op_meta["tuple"])
        assert op_meta["tuple"] == confidence[op]["tuple"] and confidence[op]["decision"] == "c_spec"
        hkey, hfft, hnorm, column = (variant_inputs(spectrum["sk_re"].astype(np.float32)
                                                  + 1j * spectrum["sk_im"].astype(np.float32), 8)
                                    if op == "pbev128_main" else hspec_inputs(spectrum["raw_height"], 12))
        for key in ("j_star", "n_candidates", "noalign_j_star", "best_pos_candidate_rank"):
            arrays[f"{op}_{key}"] = np.full(nq, -1, dtype=np.int64)
        for key in ("top1_geo", "noalign_top1_geo", "c_spec", "c_col"):
            arrays[f"{op}_{key}"] = np.full(nq, np.nan)
        for key in ("key_hit", "aux_hit", "union_hit", "accepted"):
            arrays[f"{op}_{key}"] = np.zeros(nq, dtype=bool)
        for i, query in enumerate(queries):
            query = int(query)
            end = query - 29
            a = prefix_topk(embedding, query, end, 100)
            b = prefix_topk(hkey, query, end, 100)
            candidates = np.unique(np.r_[a, b])
            assert candidates.max() <= query - 30
            scores_a = align_scores(hfft, hnorm, query, candidates)
            scores_c = column_align_scores(column, query, candidates)
            noalign = w[0] * minmax(embedding[candidates] @ embedding[query])
            noalign += w[1] * minmax(hkey[candidates] @ hkey[query])
            scores = noalign + w[2] * minmax(scores_a) + w[3] * minmax(scores_c)
            assert scores.dtype == np.float32
            order = np.argsort(-scores, kind="stable")
            best = int(order[0])
            selected = int(candidates[best])
            noalign_j = int(candidates[np.argmax(noalign)])
            distances = np.linalg.norm(positions[candidates] - positions[query], axis=1)
            hit = distances <= 5
            rank = np.flatnonzero(hit[order])
            values = {"j_star": selected, "n_candidates": len(candidates),
                      "noalign_j_star": noalign_j, "best_pos_candidate_rank": int(rank[0] + 1) if len(rank) else -1,
                      "top1_geo": float(distances[best]),
                      "noalign_top1_geo": float(np.linalg.norm(positions[noalign_j] - positions[query])),
                      "c_spec": float(scores_a[best]), "c_col": float(scores_c[best]),
                      "key_hit": bool(np.any(np.linalg.norm(positions[a] - positions[query], axis=1) <= 5)),
                      "aux_hit": bool(np.any(np.linalg.norm(positions[b] - positions[query], axis=1) <= 5)),
                      "union_hit": bool(hit.any()), "accepted": bool(scores_a[best] >= confidence[op]["tau_star"])}
            for key, value in values.items():
                arrays[f"{op}_{key}"][i] = value
            if i % 2000 == 0:
                print(f"{seq} {op}: {i}/{nq}", flush=True)
        old = ledgers[op]
        differences[op] = {"top1_changed": int(np.count_nonzero(arrays[f"{op}_j_star"] != old["j_star"])),
                           "correctness_changed": int(np.count_nonzero((arrays[f"{op}_top1_geo"] <= 5)
                                                                       != (old["top1_geo_dist"] <= 5)))}
    output = out / f"s1_{sensor}_{sequence}.npz"
    meta_out = {"sequence": seq, "seed": 1, "n_frames": len(raw), "n_queries": nq,
                "n_positive": int(positive.sum()), "runtime_s": time.monotonic() - started,
                "caches": {k: str(p.relative_to(REPO)) for k, p in cache_paths.items()},
                "source_ledger_sha256": {op: sha256(path) for op, path in paths.items()},
                "protocol_sha256": sha256(out / "protocol.json"), "confidence": confidence,
                "existing_ledger_differences": differences}
    np.savez_compressed(out / f"keys_s1_{sensor}_{sequence}.npz",
                        scan_ids=cache["scan_ids"], pass1=passes[0], context=embedding)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, meta=json.dumps(meta_out, ensure_ascii=False), **arrays)
    temporary.replace(output)
    write_json(output.with_suffix(".json"), meta_out)
    print(f"DONE {seq}: {meta_out['runtime_s']:.1f}s, {positive.sum()} revisit queries", flush=True)


def main() -> None:
    """고정 프로토콜을 쓰고 시퀀스별 독립 프로세스로 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seq")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    setup_protocol(args.out)
    if args.seq:
        worker(args.seq, args.out)
        return
    if not args.run:
        return
    for index, (sensor, sequence) in enumerate(TEST, 1):
        tag = f"{sensor}_{sequence}"
        output = args.out / f"s1_{tag}.npz"
        if output.exists():
            print(f"SKIP {index}/{len(TEST)} {tag}", flush=True)
            continue
        print(f"START {index}/{len(TEST)} {tag}", flush=True)
        log = args.out / f"run_s1_{tag}.log"
        with log.open("w") as handle:
            subprocess.run([sys.executable, str(Path(__file__)), "--seq", f"{sensor}/{sequence}",
                            "--out", str(args.out)], stdout=handle, stderr=subprocess.STDOUT, check=True)
    print("ALL 46 SEQUENCES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
