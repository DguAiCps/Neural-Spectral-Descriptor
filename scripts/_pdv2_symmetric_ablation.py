#!/usr/bin/env python3
"""현재 입력에서 문맥 없음·과거 문맥·15키프레임 지연 문맥을 비교한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/_pdv2_symmetric_ablation.py --run

검증 9개, 기존 시드 1 모델, 현재 BEV 보정, 고정 융합·판정 설정을 쓴다.
공통 질의는 30 <= q < N-15이며 검색 후보는 j <= q-30이다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import (
    OPS, baseline_root, evaluate_keys, no_context_key, sha256, write_json,
)
from _pdv2_pair_evaluate import ArrayCache, direction_groups, inputs
from _pdv2_pilot import VAL
from _xs_eval_lib import hspec_inputs, normalize, variant_inputs

OUT = REPO / "results/pdv2_symmetric_context_20261013"
PAIR = REPO / "results/pdv2_pair_sampling_20261013"
NO_CONTEXT = REPO / "results/pdv2_no_context_20261013"
ARMS = ("no_context", "past", "symmetric")
LAG = 15
PREFIX_SEQS = ("kitti/09", "helipr/Roundabout01", "mulran/Sejong01")
CFG = REPO / "configs/training_place_disjoint_v2_lag15.yaml"
CKPT = REPO / "checkpoints/pdv2_lag15_s1/best_model.pth"
CLS = REPO / "artifacts/edge_classifier_pdv2_lag15.pt"
RVEC = REPO / "artifacts/key_remetrize_r_pdv2.npy"


def source_ledger(arm: str, op: str, sensor: str, seq: str) -> Path:
    """이미 검증한 고정 설정 대조군 기록의 위치를 반환한다."""
    assert arm in ("no_context", "past")
    if arm == "no_context":
        return NO_CONTEXT / f"val_{op}_s1_{sensor}_{seq}.npz"
    return PAIR / f"val_initial_{op}_{sensor}_{seq}.npz"


def protocol() -> dict:
    """결과를 보기 전에 대상·질의·모델·검색 설정과 원천 해시를 고정한다."""
    path = OUT / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = baseline_root(1) / "confidence_selection_align.json"
    selection = json.loads(selection_path.read_text())["ops"]
    assert all(selection[op]["decision"] == "c_spec" for op in OPS)
    causal_cfg = REPO / "configs/training_place_disjoint_v2_causal.yaml"
    a, b = [yaml.safe_load(p.read_text()) for p in (causal_cfg, CFG)]
    assert a["keyframe"]["temporal_neighbors"] == 20
    assert b["keyframe"]["temporal_neighbors"] == 10
    b["keyframe"]["temporal_neighbors"] = 20
    b["keyframe"]["graph"]["temporal_edge_mode"] = "past_to_current"
    assert a == b, "학습 설정의 차이가 시간 이웃 구성 이외에도 있다"
    files = [Path(__file__), CFG, causal_cfg, CKPT, CLS, RVEC, selection_path,
             PAIR / "protocol.json", NO_CONTEXT / "protocol.json",
             REPO / "checkpoints/pdv2_causal_s1/best_model.pth",
             REPO / "artifacts/edge_classifier_pdv2_causal.pt"]
    files.extend(REPO / p for p in (
        "scripts/_lag_branch_fusion.py", "scripts/_causal_branch_fusion.py",
        "scripts/_causal_edge_features.py", "scripts/_pdv2_no_context_ablation.py",
        "scripts/_pdv2_pair_evaluate.py", "scripts/_pdv2_context_diagnosis.py",
        "scripts/_causal_pr_dump.py", "scripts/_xs_eval_lib.py",
        "scripts/_pr_eval_lib.py", "src/keyframe/graph_manager.py", "src/gnn/model.py",
    ))
    for sensor, seq in VAL:
        _, _, paths, _ = inputs(sensor + "/" + seq)
        files.extend(paths.values())
        files.append(PAIR / f"keys_{sensor}_{seq}.npz")
        for op in OPS:
            files.extend(source_ledger(arm, op, sensor, seq) for arm in ARMS[:2])
    value = {
        "schema": "symmetric-context-val-v1", "date": "2026-10-13", "seed": 1,
        "scope": "검증셋 진단. 기존 각 그래프의 학습·검증 선택 모델을 재사용하며 새 학습·선택·시험 평가는 하지 않는다.",
        "numerics": "두 문맥 모델 모두 GPU FP32, CUDA matmul TF32 비활성. 첫 GPU 기본 TF32 실행의 지연 검사 실패를 진단한 뒤 정밀도를 통일했다.",
        "arms": {"no_context": "[normalize(r*d); zeros(128)]",
                 "past": "기존 과거 10개 모델과 해당 간선 분류기",
                 "symmetric": "기존 앞뒤 5개 모델과 해당 간선 분류기, 지연 15키프레임"},
        "frozen": "같은 odom 키프레임, d, r, BEV clipping, 보조 색인, 두 통과 구조, 후보 상한 100+100, 네 점수, 현재 시드 1 융합과 c_spec 임계값",
        "model_note": "시간 간선을 바꿔 각각 학습한 시스템의 비교다. 동일 가중치에서 간선만 교체하는 실험은 아니다. 간선 분류기도 각 모델의 것을 쓴다.",
        "query_rule": "세 조건 모두 30 <= q < N-15; 끝 15개 제외; 대칭 문맥 판정은 q+15 도착 후",
        "candidate_rule": "j <= q-30 (q+15 기준으로 후보를 늘리지 않는다)",
        "lag": LAG, "prefix_sequences": list(PREFIX_SEQS), "prefix_k": [40, 300, 900],
        "prefix_tolerance": 2e-6,
        "aggregation": "시퀀스 평균 뒤 센서 3계열 균등 평균. 방향별 검색률은 양성 질의를 합산한다.",
        "directions": "과거 5키프레임의 3D 이동 >=1m; 같은 방향 <=60도 정답 존재 / 모든 정답이 유효하고 >=120도 / 나머지",
        "roster": [s + "/" + q for s, q in VAL], "selection": selection,
        "sha256": {str(p.relative_to(REPO)): sha256(p) for p in dict.fromkeys(files)},
    }
    write_json(path, value)
    return value


def lag_model() -> tuple:
    """배포 경로와 같은 필수 인코더·게이트 설정으로 지연 모델을 읽는다."""
    from _causal_branch_fusion import VARIANTS, make_model_causal
    VARIANTS["symmetric_ablation"] = dict(
        ckpt_dir="pdv2_lag15_s{seed}", config=CFG, cls=CLS, rvec=RVEC,
    )
    return make_model_causal("symmetric_ablation", 1, "cuda")


def keys(sequence: str) -> None:
    """현재 검증 입력의 지연 키를 만들고 키 블록·방향·대기 시간을 확인한다."""
    import torch
    from _lag_branch_fusion import nsd_embedding_lag
    from _causal_branch_fusion import VARIANTS, make_model_causal, nsd_embedding_causal
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = protocol()
    sensor, seq, paths, min_past = inputs(sequence)
    dest = OUT / f"keys_{sensor}_{seq}.npz"
    if dest.exists():
        return
    started = time.monotonic()
    cfg, model, clf, blob, r, _ = lag_model()
    with np.load(paths["cache"]) as archive, np.load(paths["layouts"]) as layouts:
        cache = ArrayCache({k: archive[k] for k in archive.files})
        np.testing.assert_array_equal(cache["scan_ids"], layouts["scan_ids"])
        raw = cache["descriptors"].astype(np.float32)
        emb = nsd_embedding_lag(cache, sensor, raw, layouts["nsd_layouts"],
                                cfg, model, clf, blob, r, "cuda")
        model.cpu()
        clf.cpu()
        VARIANTS["past_ablation"] = dict(
            ckpt_dir="pdv2_causal_s{seed}",
            config=REPO / "configs/training_place_disjoint_v2_causal.yaml",
            cls=REPO / "artifacts/edge_classifier_pdv2_causal.pt", rvec=RVEC,
        )
        past_cfg, past_model, past_clf, past_blob, past_r, _ = make_model_causal("past_ablation", 1, "cuda")
        past_emb = nsd_embedding_causal(cache, sensor, raw, layouts["nsd_layouts"],
                                        past_cfg, past_model, past_clf, past_blob, past_r, "cuda")
    assert emb.shape == (len(raw), 416) and emb.dtype == np.float32
    assert np.isfinite(emb).all()
    expected_key = no_context_key(raw, r)[:, :288]
    key_error = float(np.abs(normalize(emb[:, :288]) - expected_key).max())
    assert key_error < 3e-7
    assert past_emb.shape == emb.shape and np.isfinite(past_emb).all()
    assert np.abs(normalize(past_emb[:, :288]) - expected_key).max() < 3e-7
    with np.load(PAIR / f"keys_{sensor}_{seq}.npz") as past:
        meta = json.loads(str(past["meta"]))
        for name, expected in meta["inputs_sha256"].items():
            assert spec["sha256"][name] == expected
        name = "checkpoints/pdv2_causal_s1/best_model.pth"
        assert spec["sha256"][name] == meta["checkpoints_sha256"][name]
        np.testing.assert_array_equal(cache["scan_ids"], past["scan_ids"])
        assert np.abs(normalize(past["initial"][:, :288]) - expected_key).max() < 3e-7
        groups = direction_groups(cache["poses_gt"][:, :3, 3], min_past)
        np.testing.assert_array_equal(groups, past["groups"])
    q = np.arange(30, len(raw) - LAG)
    delay = cache["timestamps"][q + LAG] - cache["timestamps"][q]
    distance = np.linalg.norm(np.diff(cache["poses"][:, :3, 3], axis=0), axis=1)
    travel = np.r_[0., np.cumsum(distance)]
    assert np.isfinite(delay).all() and (delay >= 0).all()
    meta = {"sequence": sequence, "protocol_sha256": sha256(OUT / "protocol.json"),
            "runtime_s": time.monotonic() - started, "key_block_max_error": key_error}
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, symmetric=emb, past=past_emb, groups=groups, scan_ids=cache["scan_ids"],
                        query_idx=q, delay_seconds=delay,
                        delay_travel_m=travel[q + LAG] - travel[q], meta=json.dumps(meta))
    tmp.replace(dest)
    print("KEY_DONE", sequence, meta["runtime_s"], flush=True)


def score(sequence: str) -> None:
    """지연 키를 재검색하고 공통 질의의 세 조건 기록을 보존한다."""
    spec = protocol()
    sensor, seq, paths, min_past = inputs(sequence)
    with np.load(paths["cache"]) as cache, np.load(paths["spec"]) as spectrum, \
            np.load(OUT / f"keys_{sensor}_{seq}.npz") as saved:
        np.testing.assert_array_equal(cache["scan_ids"], spectrum["scan_ids"])
        np.testing.assert_array_equal(cache["scan_ids"], saved["scan_ids"])
        positions = cache["poses_gt"][:, :3, 3]
        for op in OPS:
            selection = spec["selection"][op]
            bank = (variant_inputs(spectrum["sk_re"].astype(np.float32)
                                   + 1j * spectrum["sk_im"].astype(np.float32), 8)
                    if op == "pbev128_main" else hspec_inputs(spectrum["raw_height"], 12))
            for arm in ARMS:
                target = OUT / f"val_{arm}_{op}_{sensor}_{seq}.npz"
                if target.exists():
                    continue
                if arm != "no_context":
                    weights = {1: tuple(float(v) for v in selection["tuple"])}
                    # 미래 입력은 유지하고 평가 질의만 아래에서 공통 구간으로 자른다.
                    arrays = evaluate_keys(saved[arm], bank, positions, min_past, weights)[1]
                    provenance = {"keys_sha256": sha256(OUT / f"keys_{sensor}_{seq}.npz")}
                else:
                    source = source_ledger(arm, op, sensor, seq)
                    with np.load(source) as old:
                        arrays = {k: old[k] for k in old.files if k != "meta"}
                        old_meta = json.loads(str(old["meta"]))
                        if arm == "past":
                            previous_selection = old_meta
                        else:
                            previous_protocol = json.loads((NO_CONTEXT / "protocol.json").read_text())
                            assert old_meta["protocol_sha256"] == sha256(NO_CONTEXT / "protocol.json")
                            previous_selection = previous_protocol["selection_by_seed"]["1"][op]
                        assert tuple(previous_selection["tuple"]) == tuple(selection["tuple"])
                    provenance = {"source": str(source.relative_to(REPO)),
                                  "source_sha256": sha256(source)}
                keep = arrays["query_idx"] < len(positions) - LAG
                arrays = {k: v[keep] for k, v in arrays.items()}
                np.testing.assert_array_equal(arrays["query_idx"], saved["query_idx"])
                arrays["group"] = saved["groups"][arrays["query_idx"]]
                assert (arrays["j_star"] <= arrays["query_idx"] - 30).all()
                meta = {"sequence": sequence, "arm": arm, "op": op,
                        "tuple": selection["tuple"], "tau_star": selection["tau_star"],
                        "protocol_sha256": sha256(OUT / "protocol.json"), **provenance}
                tmp = target.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **arrays, meta=json.dumps(meta))
                tmp.replace(target)
                print("SCORE_DONE", sequence, op, arm, flush=True)


def prefix(sequence: str) -> None:
    """실제 지연 체크포인트를 잘린 입력에서 다시 실행해 지연 상한을 검증한다."""
    import torch
    from _lag_branch_fusion import TruncatedCache, nsd_embedding_lag
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = protocol()
    assert sequence in spec["prefix_sequences"]
    sensor, seq, paths, _ = inputs(sequence)
    cfg, model, clf, blob, r, _ = lag_model()
    with np.load(paths["cache"]) as archive, np.load(paths["layouts"]) as layouts, \
            np.load(OUT / f"keys_{sensor}_{seq}.npz") as saved:
        cache = ArrayCache({k: archive[k] for k in archive.files})
        raw, layout, full = cache["descriptors"].astype(np.float32), layouts["nsd_layouts"], saved["symmetric"]
        rows = []
        for k in spec["prefix_k"]:
            for extra in (LAG, LAG - 1):
                n = k + extra + 1
                emb = nsd_embedding_lag(TruncatedCache(cache, n), sensor, raw[:n],
                                        layout[:n], cfg, model, clf, blob, r, "cuda")
                error = float(np.abs(emb[:k + 1] - full[:k + 1]).max())
                rows.append({"k": k, "lookahead": extra, "max_error_rows_le_k": error,
                             "max_error_at_k": float(np.abs(emb[k] - full[k]).max())})
                if extra == LAG:
                    assert error <= spec["prefix_tolerance"], rows[-1]
    write_json(OUT / f"prefix_{sensor}_{seq}.json", {
        "sequence": sequence, "checkpoint_sha256": sha256(CKPT),
        "classifier_sha256": sha256(CLS), "rows": rows,
        "protocol_sha256": sha256(OUT / "protocol.json"),
    })
    print("PREFIX_DONE", sequence, flush=True)


def run() -> None:
    """GPU 키는 순차 생성하고 CPU 검색은 두 프로세스까지 겹쳐 실행한다."""
    protocol()
    def child(action: str, sequence: str) -> None:
        with (OUT / f"{action}_{sequence.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, __file__, "--" + action, "--seq", sequence],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(action.upper(), sequence, flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for sensor, seq in VAL:
            sequence = sensor + "/" + seq
            child("keys", sequence)
            futures.append(pool.submit(child, "score", sequence))
        for sequence in PREFIX_SEQS:
            child("prefix", sequence)
        for future in futures:
            future.result()
    print("VALIDATION_COMPLETE", flush=True)


def main() -> None:
    """시험셋 옵션이 없는 검증 전용 명령을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq")
    for name in ("protocol", "keys", "score", "prefix", "run"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.protocol:
        protocol()
    for action in ("keys", "score", "prefix"):
        if getattr(args, action):
            globals()[action](args.seq)
    if args.run:
        run()


if __name__ == "__main__":
    main()
