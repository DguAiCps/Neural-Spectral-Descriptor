#!/usr/bin/env python3
"""학습 쌍 추가 학습의 검증 9개를 기존 검색 설정으로 비교한다."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import (
    OPS, baseline_root, evaluate_keys, ledger_path, local_path, metrics, sha256, write_json,
)
from _pdv2_pilot import FAMILY, VAL
from _pdv2_context_diagnosis import heading
from _xs_eval_lib import hspec_inputs, variant_inputs

OUT = REPO / "results/pdv2_pair_sampling_20261013"
ARMS = ("initial", "original", "opposite", "opposite_masked")
GROUPS = ("same_available", "opposite_only", "other")


class ArrayCache(dict):
    """배열을 한 번 읽고 NPZ의 files 속성을 유지한다."""

    @property
    def files(self) -> list:
        """기존 그래프 도구에 배열 이름 목록을 제공한다."""
        return list(self)


def inputs(sequence: str) -> tuple:
    """기존 검증 기록이 사용한 캐시와 정답 질의를 읽는다."""
    sensor, seq = sequence.split("/")
    assert (sensor, seq) in VAL
    path = ledger_path("val", 1, OPS[0], sensor, seq)
    with np.load(path) as archive:
        meta = json.loads(str(archive["meta"]))
        queries = archive["query_idx"]
        min_past = np.full(meta["n_frames"], np.nan)
        min_past[queries] = archive["min_past_gt_dist"]
    paths = {k: local_path(meta["caches"][k]) for k in ("cache", "layouts", "spec")}
    return sensor, seq, paths, min_past


def direction_groups(positions: np.ndarray, min_past: np.ndarray) -> np.ndarray:
    """같은 방향 정답 존재·반대 방향 정답만 존재하는 질의를 나눈다."""
    h, valid = heading(positions)
    tree = cKDTree(positions)
    groups = np.full(len(positions), -1, dtype=np.int8)
    for i in np.flatnonzero(min_past <= 5):
        candidates = np.asarray(tree.query_ball_point(positions[i], 5), dtype=np.int64)
        candidates = candidates[candidates <= i - 30]
        assert len(candidates)
        groups[i] = 2
        dot = np.clip(h[candidates] @ h[i], -1, 1)
        if valid[i] and np.any(valid[candidates] & (dot >= .5)):
            groups[i] = 0
        elif valid[i] and np.all(valid[candidates] & (dot <= -.5)):
            groups[i] = 1
    return groups


def keys(sequence: str) -> None:
    """동일한 기존 분류기로 두 통과 문맥 키를 만든다."""
    import torch
    from _pdv2_pair_train import settings
    from evaluate_kitti_checkpoint import _make_model
    from _causal_branch_fusion import nsd_embedding_causal
    from build_and_train_edge_classifier import EdgeMLP, N_FEAT

    torch.set_num_threads(1)
    exp, base_cfg, _ = settings()
    sensor, seq, paths, min_past = inputs(sequence)
    dest = OUT / f"keys_{sensor}_{seq}.npz"
    if dest.exists():
        print("KEYS_EXIST", sequence, flush=True)
        return
    rpath = REPO / "artifacts/key_remetrize_r_pdv2.npy"
    cpath = REPO / "artifacts/edge_classifier_pdv2_causal.pt"
    cfg = copy.deepcopy(base_cfg)
    cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(rpath)}
    blob = torch.load(cpath, map_location="cpu", weights_only=False)
    classifier = EdgeMLP(N_FEAT).eval().to("cuda")
    classifier.load_state_dict(blob["state_dict"])
    r = np.load(rpath).astype(np.float32)
    saved, checkpoints = {}, {}
    with np.load(paths["cache"]) as cache, np.load(paths["layouts"]) as layouts:
        # keyframes_from은 행을 반복 접근한다. NPZ를 직접 넘기면 각 pose의
        # view마다 전체 poses 배열을 다시 읽고 보존해 메모리가 N²로 늘어난다.
        cache = ArrayCache({name: cache[name] for name in cache.files})
        raw = cache["descriptors"].astype(np.float32)
        np.testing.assert_array_equal(cache["scan_ids"], layouts["scan_ids"])
        for arm in ARMS:
            ckpt = REPO / exp["initial_checkpoint"] if arm == "initial" else OUT / f"{arm}.pth"
            model = _make_model(cfg, ckpt, "cuda")
            saved[arm] = nsd_embedding_causal(cache, sensor, raw, layouts["nsd_layouts"],
                                             cfg, model, classifier, blob, r, "cuda")
            assert saved[arm].shape == (len(raw), 416) and np.isfinite(saved[arm]).all()
            checkpoints[str(ckpt.relative_to(REPO))] = sha256(ckpt)
            print("KEY_DONE", sequence, arm, flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        saved["scan_ids"] = cache["scan_ids"]
        saved["groups"] = direction_groups(cache["poses_gt"][:, :3, 3], min_past)
    meta = {"sequence": sequence, "protocol_sha256": sha256(OUT / "protocol.json"),
            "checkpoints_sha256": checkpoints,
            "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in list(paths.values()) + [rpath, cpath]},
            "script_sha256": sha256(Path(__file__))}
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **saved, meta=json.dumps(meta))
    tmp.replace(dest)


def score(sequence: str) -> None:
    """후보를 다시 검색해 고정 융합·판정 조건의 결과를 저장한다."""
    sensor, seq, paths, min_past = inputs(sequence)
    selection = json.loads((baseline_root(1) / "confidence_selection_align.json").read_text())["ops"]
    with np.load(paths["cache"]) as cache, np.load(paths["spec"]) as spec, \
            np.load(OUT / f"keys_{sensor}_{seq}.npz") as saved:
        np.testing.assert_array_equal(cache["scan_ids"], spec["scan_ids"])
        np.testing.assert_array_equal(cache["scan_ids"], saved["scan_ids"])
        positions = cache["poses_gt"][:, :3, 3]
        for op in OPS:
            bank = (variant_inputs(spec["sk_re"].astype(np.float32) + 1j * spec["sk_im"].astype(np.float32), 8)
                    if op == "pbev128_main" else hspec_inputs(spec["raw_height"], 12))
            weights = {1: tuple(float(v) for v in selection[op]["tuple"])}
            for arm in ARMS:
                target = OUT / f"val_{arm}_{op}_{sensor}_{seq}.npz"
                if target.exists():
                    continue
                print("SCORE_START", sequence, op, arm, flush=True)
                arrays = evaluate_keys(saved[arm], bank, positions, min_past, weights)[1]
                arrays["group"] = saved["groups"][arrays["query_idx"]]
                meta = {"sequence": sequence, "arm": arm, "op": op,
                        "tuple": weights[1], "tau_star": selection[op]["tau_star"],
                        "protocol_sha256": sha256(OUT / "protocol.json"),
                        "keys_sha256": sha256(OUT / f"keys_{sensor}_{seq}.npz"),
                        "script_sha256": sha256(Path(__file__))}
                tmp = target.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **arrays, meta=json.dumps(meta))
                tmp.replace(target)
                print("SCORE_DONE", sequence, op, arm, flush=True)


def aggregate() -> None:
    """검증 3센서군 균등 평균과 방향별 표본 합산 결과를 쓴다."""
    selection = json.loads((baseline_root(1) / "confidence_selection_align.json").read_text())["ops"]
    result = {"scope": "기존 시드 1에서 동일 예산 추가 학습, 마지막 체크포인트, 검증 9개, 검색 설정 고정",
              "protocol_sha256": sha256(OUT / "protocol.json"), "ops": {}, "baseline_reproduction": {}}
    for op in OPS:
        per_arm = {}
        for arm in (*ARMS, "no_context"):
            per_seq, direction = {}, {group: {"n": 0, "key_correct": 0, "full_correct": 0}
                                       for group in GROUPS}
            gate_values = []
            for sensor, seq in VAL:
                path = (REPO / f"results/pdv2_no_context_20261013/val_{op}_s1_{sensor}_{seq}.npz"
                        if arm == "no_context" else OUT / f"val_{arm}_{op}_{sensor}_{seq}.npz")
                with np.load(path) as archive:
                    arrays = {k: archive[k] for k in archive.files if k != "meta"}
                with np.load(OUT / f"keys_{sensor}_{seq}.npz") as saved:
                    groups = saved["groups"][arrays["query_idx"]]
                    if arm != "no_context":
                        emb = saved[arm]
                        gate_values.append(np.linalg.norm(emb[:, 288:], axis=1) / np.linalg.norm(emb[:, :288], axis=1))
                row = metrics(arrays, selection[op]["tau_star"])
                pos = arrays["min_past_gt_dist"] <= 5
                row["key_R@1"] = float((arrays["key_top1_geo"][pos] <= 5).mean())
                row["key_R@100"] = float(arrays["key_hit"][pos].mean())
                row["union_R@200"] = float(arrays["union_hit"][pos].mean())
                row["n_positive"] = int(pos.sum())
                row["n_queries"] = len(pos)
                per_seq[sensor + "/" + seq] = row
                for gi, group in enumerate(GROUPS):
                    mask = pos & (groups == gi)
                    direction[group]["n"] += int(mask.sum())
                    direction[group]["key_correct"] += int((arrays["key_top1_geo"][mask] <= 5).sum())
                    direction[group]["full_correct"] += int((arrays["top1_geo_dist"][mask] <= 5).sum())
                if arm == "initial":
                    old_path = ledger_path("val", 1, op, sensor, seq)
                    with np.load(old_path) as old:
                        previous = {k: old[k] for k in old.files if k != "meta"}
                    with np.load(old_path.with_name("align_" + old_path.name)) as align:
                        np.testing.assert_array_equal(previous["j_star"], align["j_star"])
                        previous["c_spec"] = align["c_spec"]
                    np.testing.assert_array_equal(previous["query_idx"], arrays["query_idx"])
                    np.testing.assert_array_equal(previous["min_past_gt_dist"], arrays["min_past_gt_dist"])
                    result["baseline_reproduction"][op + "/" + sensor + "/" + seq] = {
                        "candidate_changed": int((previous["j_star"] != arrays["j_star"]).sum()),
                        "correct_changed": int(((previous["top1_geo_dist"] <= 5) != (arrays["top1_geo_dist"] <= 5)).sum()),
                        "AP_delta": row["AP"] - metrics(previous, selection[op]["tau_star"])["AP"]}
            families = {}
            names = [k for k in next(iter(per_seq.values())) if not k.startswith("n_")]
            for family in ("kitti", "helipr", "mulran"):
                rows = [v for k, v in per_seq.items() if FAMILY.get(k.split("/")[0], k.split("/")[0]) == family]
                families[family] = {k: float(np.mean([r[k] for r in rows])) for k in names}
            macro = {k: float(np.mean([v[k] for v in families.values()])) for k in names}
            for row in direction.values():
                row["key_R@1"] = row["key_correct"] / row["n"] if row["n"] else None
                row["full_R@1"] = row["full_correct"] / row["n"] if row["n"] else None
            per_arm[arm] = {"macro": macro, "families": families, "per_sequence": per_seq,
                            "direction_pooled": direction,
                            "gate_median": float(np.median(np.concatenate(gate_values))) if gate_values else 0.}
        result["ops"][op] = per_arm
    write_json(OUT / "summary.json", result)
    for op, arms in result["ops"].items():
        for arm, value in arms.items():
            print(op, arm, json.dumps(value["macro"]), flush=True)


def run() -> None:
    """GPU 키 생성은 순차 실행하고 CPU 검색은 최대 두 개씩 실행한다."""
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        def child(action: str, sequence: str) -> None:
            logpath = OUT / f"{action}_{sequence.replace('/', '_')}.log"
            with logpath.open("w") as logfile:
                subprocess.run([sys.executable, __file__, "--" + action, "--seq", sequence],
                               stdout=logfile, stderr=subprocess.STDOUT, check=True)
            print(action.upper() + "_DONE", sequence, flush=True)
        for sensor, seq in VAL:
            sequence = sensor + "/" + seq
            child("keys", sequence)
            futures.append(executor.submit(child, "score", sequence))
        for future in futures:
            future.result()
    aggregate()


def main() -> None:
    """검증 전용 명령을 실행한다. 시험셋 인자는 제공하지 않는다."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq")
    p.add_argument("--keys", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    if a.keys:
        keys(a.seq)
    if a.score:
        score(a.seq)
    if a.aggregate:
        aggregate()
    if a.run:
        run()


if __name__ == "__main__":
    main()
