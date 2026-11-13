#!/usr/bin/env python3
"""원래 float64 L2 점수와 동률 처리를 보존하는 과거 접두 검색이다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]


def ledger_exact(desc: np.ndarray, pos: np.ndarray, base: dict) -> tuple:
    """FP64 GPU로 제안하고 원래 CPU 거리로 결정하며 근접 동률은 원경로로 돌린다."""
    import torch
    from _disco_official_dump import l2_top1_prefix

    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    vectors = np.ascontiguousarray(desc, dtype=np.float64)
    assert np.isfinite(vectors).all() and len(vectors) == len(pos)
    norms = np.einsum("ij,ij->i", vectors, vectors)
    bank = torch.from_numpy(vectors).cuda()
    norm_gpu = torch.from_numpy(norms).cuda()
    dim = vectors.shape[1]
    unit = np.finfo(np.float64).eps / 2
    gamma = (dim + 8) * unit / (1 - (dim + 8) * unit)
    # 양쪽 내적/거리 누적의 오차와 두 후보 차이를 함께 덮는다.
    margin = max(32 * gamma * float(norms.max()), np.finfo(np.float64).tiny)
    qs = base["query_idx"]
    arrays = {k: v.copy() for k, v in base.items()}
    arrays["j_star"] = np.empty(len(qs), np.int64)
    arrays["score_disco"] = np.empty(len(qs), np.float64)
    arrays["top1_geo_dist"] = np.empty(len(qs), np.float64)
    fallbacks = 0
    smallest_gap = float("inf")
    for offset in range(0, len(qs), 128):
        block = qs[offset:offset + 128]
        common_end = int(block[0] - 29)
        # 모든 질의에 허용되는 공통 과거만 곱한다. 각 질의의 나머지 과거는 CPU로 더한다.
        qgpu = bank[block]
        d2 = norm_gpu[:common_end][None, :] - 2 * (qgpu @ bank[:common_end].T)
        d2 += norm_gpu[block, None]
        vals, idx = torch.topk(d2, min(2, common_end), dim=1, largest=False, sorted=True)
        vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
        for slot, q in enumerate(block):
            end = int(q - 29)
            selected = np.concatenate((idx[slot], np.arange(common_end, end, dtype=np.int64)))
            selected.sort()
            exact_squared = ((vectors[selected] - vectors[q]) ** 2).sum(axis=1)
            order = np.lexsort((selected, exact_squared))
            gap = float(exact_squared[order[1]] - exact_squared[order[0]]) if len(order) > 1 else float("inf")
            common_gap = float(vals[slot, 1] - vals[slot, 0]) if common_end > 1 else float("inf")
            # 원래 np.argpartition의 근접/동률 후보집합까지 보존한다.
            if min(gap, common_gap) <= margin:
                j, distance = l2_top1_prefix(vectors, norms, int(q), end)
                fallbacks += 1
            else:
                winner = order[0]
                j, distance = int(selected[winner]), float(np.sqrt(exact_squared[winner]))
                smallest_gap = min(smallest_gap, gap, common_gap)
            i = offset + slot
            arrays["j_star"][i], arrays["score_disco"][i] = j, -distance
            arrays["top1_geo_dist"][i] = float(np.linalg.norm(pos[j] - pos[q]))
    return arrays, dict(dtype="float64", all_candidates_past_only=True,
        final_distance="원래 NumPy float64 ((D[cand]-D[q])**2).sum -> sqrt",
        near_tie_path="원래 l2_top1_prefix 전체 접두·argpartition·lexsort",
        roundoff_margin_squared_l2=margin, minimum_accepted_squared_gap=smallest_gap,
        full_original_fallback_queries=fallbacks, batch=128, n_queries=len(qs))


def verify() -> None:
    """합성 동률과 실제 긴 시퀀스의 모든 행을 원래 경로와 비교한다."""
    import torch
    from _disco_official_dump import ledger
    from _review_selection_bev import read, write, sha
    from _review_selection_full_eval import OUT, candidate, descriptors, folder, positions, query_base

    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(4)
    home = OUT / "l2_acceleration_audit"
    home.mkdir(parents=True, exist_ok=True)
    checks = []

    def compare(label, vectors, pos, base):
        tick = time.time()
        ref = ledger(vectors, pos)
        original_s = time.time() - tick
        tick = time.time()
        actual, audit = ledger_exact(vectors, pos, base)
        accelerated_s = time.time() - tick
        take = np.searchsorted(ref["query_idx"], base["query_idx"])
        mismatches = {}
        for key, values in actual.items():
            expected = ref[key][take]
            if not np.array_equal(values, expected):
                mismatches[key] = int(np.count_nonzero(values != expected))
        record = dict(label=label, n_queries=len(base["query_idx"]), original_s=original_s,
            accelerated_s=accelerated_s, mismatches=mismatches, audit=audit)
        checks.append(record)
        write(home / "progress.json", checks)
        assert not mismatches, record
        print("EXACT_L2_CHECK", label, original_s, accelerated_s, flush=True)

    rng = np.random.default_rng(67321)
    for tag in ("random", "all_equal", "duplicate", "near_tie"):
        vectors = rng.normal(size=(700, 256)).astype(np.float32)
        if tag == "all_equal": vectors[:] = vectors[0]
        if tag == "duplicate": vectors[::2] = vectors[0]
        if tag == "near_tie":
            vectors[:] = vectors[0]
            vectors[:, 0] = np.nextafter(vectors[:, 0], np.float32(np.inf))
            vectors[::3, 0] = np.nextafter(vectors[::3, 0], np.float32(np.inf))
        pos = rng.normal(size=(700, 3))
        q = np.arange(30, len(pos) - 15)
        base = dict(query_idx=q, n_candidates=q-29,
            min_past_gt_dist=np.array([np.linalg.norm(pos[:int(t)-29]-pos[t],axis=1).min() for t in q]))
        compare(tag, vectors, pos, base)

    spec = read(OUT / "retained_protocol.json")
    for seed, at, sequence in [(1, 0, "mulran/Sejong01"), (1, -1, "mulran/Sejong03"),
                               (2, 49, "kitti360/0006")]:
        row = spec["methods"]["seqot"][str(seed)]["candidates"][at]
        args = argparse.Namespace(method="seqot", seed=seed, candidate=row["id"],
            sequence=sequence, role="val", scope="retained")
        sensor, seq = sequence.split("/")
        ids, pos = positions(sensor, seq, "val")
        vectors, _ = descriptors(args, row, ids, folder("seqot",seed,row["id"],"retained","val"))
        compare(f"seqot_s{seed}_{row['id']}_{sequence}", vectors, pos, query_base(sensor,seq,"val"))
    write(home / "validated.json", dict(completed_unix=time.time(), checks=checks,
        n_queries=sum(r["n_queries"] for r in checks), all_array_fields_exact=True,
        original_sha256=sha(ROOT/"scripts/_disco_official_dump.py"), implementation_sha256=sha(Path(__file__)),
        scope="동일서술자·동일과거후보·원래CPU최종거리/근접동률; 새scorer아님"))


if __name__ == "__main__":
    verify()
