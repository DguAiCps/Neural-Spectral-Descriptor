#!/usr/bin/env python3
"""BEVPlace++ 시드 2·3을 시드 1과 같은 KISS-ICP 검증 질의로 평가한다.

단일 판정 기준값을 검증에서 고르기 위한 원장이다. 체크포인트는 각 시드의 기존 학습 선택을 그대로 쓰고,
입력 BEV 영상·질의·과거 후보 규칙은 `_pdv2_bevplace_val_odom.py`(시드 1)와 같다. 시험셋은 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

import _pdv2_bevplace_val_odom as seed1
from _pdv2_bevplace_val_odom import ROLE, bev, metrics
from _bevplace_official_dump import _official_model, ledger_l2
from _pdv2_pilot import VAL

OUT = REPO / "results/pdv2_bevplace_val_odom_seeds_20261101"
SELECTION = {1: REPO / "results/pdv2_bevplace_official/train_selection.json",
             2: REPO / "results/pdv2_bevplace_official_s2/train_selection.json",
             3: REPO / "results/pdv2_bevplace_official_s3/train_selection.json"}


def ordered_images(sensor: str, seq: str) -> str:
    """시드 1 검증의 영상 해시와 같은 방식으로 순서·내용을 요약한다."""
    directory = bev.BEV / ROLE / bev.seq_name(sensor, seq)
    digest = hashlib.sha256()
    for sid in np.load(directory / "scan_ids.npy"):
        name = f"{sid:06d}.png"
        digest.update(name.encode())
        digest.update(hashlib.sha256((directory / "bev_imgs" / name).read_bytes()).digest())
    return digest.hexdigest()


def evaluate(seed: int, sequences: list) -> None:
    """선택 체크포인트로 공식 추론·과거 접두 검색을 하고 시드 1 원장과 질의를 대조한다."""
    import torch

    best = seed1.read(SELECTION[seed])["best"]
    checkpoint = REPO / best["checkpoint"]
    assert bev.sha256(checkpoint) == best["sha256"], checkpoint
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"] == best["epoch"]
    torch.set_num_threads(1)
    module, dataset_module, _ = _official_model(checkpoint, 4, 32, "cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for sensor, seq in sequences:
        target = OUT / f"s{seed}_{sensor}_{seq}.json"
        if target.exists():
            continue
        reference = seed1.read(seed1.OUT / f"{sensor}_{seq}.json")
        assert ordered_images(sensor, seq) == reference["images"]["ordered_image_sha256"]
        dataset = dataset_module.InferDataset(seq=bev.seq_name(sensor, seq), dataset_path=str(bev.BEV / ROLE) + "/")
        with np.load(bev.cache_path(ROLE, sensor, seq)) as cache:
            ids = cache["scan_ids"]
            positions = cache["poses_gt"][:, :3, 3].astype(np.float64)
        assert [Path(p).name for p in dataset.imgs_path] == [f"{sid:06d}.png" for sid in ids]
        tick = time.perf_counter()
        descriptors = module.infer(dataset).astype(np.float32)
        encode_s = time.perf_counter() - tick
        assert descriptors.shape == (len(ids), 8192) and np.isfinite(descriptors).all()
        arrays = ledger_l2(descriptors, positions)
        keep = arrays["query_idx"] < len(ids) - 15
        arrays = {key: value[keep] for key, value in arrays.items()}
        with np.load(seed1.OUT / f"dump_bevplace_{sensor}_{seq}.npz") as first:
            for key in ("query_idx", "min_past_gt_dist", "n_candidates"):
                np.testing.assert_array_equal(arrays[key], first[key])
            replay = None
            if seed == 1:
                replay = dict(j_star_equal=bool((arrays["j_star"] == first["j_star"]).all()),
                              max_score_error=float(np.abs(arrays["score_negl2"] - first["score_negl2"]).max()))
        assert (arrays["j_star"] <= arrays["query_idx"] - 30).all()
        ledger = OUT / f"dump_bevplace_s{seed}_{sensor}_{seq}.npz"
        meta = dict(schema="pr-dump-v1", method="bevplace", seed=seed, role=ROLE, epoch=best["epoch"],
                    checkpoint_sha256=best["sha256"], primary_score="score_negl2")
        np.savez_compressed(ledger, meta=json.dumps(meta), **arrays)
        seed1.write(target, dict(sequence=f"{sensor}/{seq}", seed=seed, epoch=best["epoch"],
                                 checkpoint=best["checkpoint"], metrics=metrics(arrays), encode_s=encode_s,
                                 seed1_replay=replay, ledger=str(ledger.relative_to(REPO)),
                                 ledger_sha256=bev.sha256(ledger), test_used=False))
        print("EVALUATED", seed, sensor, seq, metrics(arrays), replay, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=sorted(SELECTION))
    parser.add_argument("--sequence", action="append", choices=[s + "/" + q for s, q in VAL],
                        help="기본값은 검증 9시퀀스 전체. 시드 1은 재현 확인용으로 일부만 돌린다.")
    args = parser.parse_args()
    evaluate(args.seed, [tuple(s.split("/")) for s in args.sequence] if args.sequence else list(VAL))
