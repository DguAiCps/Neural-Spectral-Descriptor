#!/usr/bin/env python3
"""학습셋의 정렬 오답과 네 조건에 공통인 학습 표본을 준비한다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_matching_mine import OUT, BASE, RVEC, TRAIN, cache_path, spec_path
from _pdv2_no_context_ablation import no_context_key, sha256, write_json
from _pdv2_lag15_campaign import bank_for, OPS
from _causal_pr_dump import prefix_topk
from _xs_eval_lib import align_scores, column_align_scores, minmax
from gnn.matching_mining import paired_triplets


def mine(sequence: str) -> None:
    """학습 스캔의 같은 100+100 후보와 정렬 융합으로 오답 상위 5개씩을 모은다."""
    target = OUT / "mining" / (sequence.replace("/", "_") + "_negatives.npz")
    if target.exists():
        return
    path = cache_path(sequence)
    with np.load(path) as z:
        raw, pos, ids = z["descriptors"], z["poses"][:, :3, 3], z["scan_ids"]
    emb = no_context_key(raw, np.load(RVEC))
    positives = cKDTree(pos).query_ball_tree(cKDTree(pos), 5.)
    active = np.array([q for q in range(30, len(pos) - 15)
                       if any(abs(q - p) >= 30 for p in positives[q])], dtype=np.int64)
    pool = np.full((len(pos), 10), -1, dtype=np.int64)
    selections = json.loads((BASE / "s1/confidence_selection.json").read_text())["ops"]
    diagnostics = {}
    started = time.time()
    with np.load(spec_path(sequence)) as z:
        np.testing.assert_array_equal(ids, z["scan_ids"])
        for op_index, op in enumerate(OPS):
            hkey, hfft, hnorm, column = bank_for(op, z)
            w = selections[op]["tuple"]
            stats = dict(anchors=len(active), with_negative=0, with_positive_in_pool=0,
                         negative_outranks_positive=0, top1_wrong=0)
            for qi, q in enumerate(active):
                a = prefix_topk(emb, q, q - 29, 100)
                b = prefix_topk(hkey, q, q - 29, 100)
                candidates = np.unique(np.r_[a, b])
                assert candidates.max() <= q - 30
                ra = align_scores(hfft, hnorm, q, candidates)
                rc = column_align_scores(column, q, candidates)
                score = (w[0] * minmax(emb[candidates] @ emb[q])
                         + w[1] * minmax(hkey[candidates] @ hkey[q])
                         + w[2] * minmax(ra) + w[3] * minmax(rc))
                distance = np.linalg.norm(pos[candidates] - pos[q], axis=1)
                valid = np.flatnonzero(distance > 10.)
                ordered = valid[np.argsort(-score[valid], kind="stable")][:5]
                pool[q, op_index * 5:op_index * 5 + len(ordered)] = candidates[ordered]
                stats["with_negative"] += int(bool(len(ordered)))
                stats["top1_wrong"] += int(distance[np.argmax(score)] > 5.)
                if np.any(distance <= 5.):
                    stats["with_positive_in_pool"] += 1
                    stats["negative_outranks_positive"] += int(
                        bool(len(ordered)) and score[ordered[0]] > score[distance <= 5.].max())
                if qi % 4000 == 0:
                    print("MINE", sequence, op, qi, len(active), flush=True)
            diagnostics[op] = stats
    # 같은 후보가 두 운영점에 들어왔으면 한 번만 추출한다.
    for q in active:
        unique = np.unique(pool[q][pool[q] >= 0])
        pool[q] = -1
        pool[q, :len(unique)] = unique
        assert np.all(unique <= q - 30)
        assert np.all(np.linalg.norm(pos[unique] - pos[q], axis=1) > 10.)
    np.savez_compressed(target, negative_pool=pool, scan_ids=ids)
    write_json(target.with_suffix(".json"), {"sequence": sequence, "diagnostics": diagnostics,
               "eligible_anchors": int(np.any(pool >= 0, axis=1).sum()),
               "sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                          (path, spec_path(sequence), RVEC, BASE / "s1/confidence_selection.json")},
               "output_sha256": sha256(target), "runtime_s": time.time() - started})
    print("MINE_DONE", sequence, flush=True)


def samples(seed: int) -> None:
    """저장된 원래 오답을 입력 대조 후 재사용하고 두 조건의 삼중쌍을 만든다."""
    import torch
    folder = OUT / f"s{seed}/samples"
    folder.mkdir(parents=True, exist_ok=True)
    reference = REPO / "results/pdv2_pair_sampling_20261013/prepared.pt"
    old = torch.load(reference, map_location="cpu", weights_only=False)
    raw, poses, seqids, pools, positives, provenance = [], [], [], [], [], {}
    offset = 0
    for seqid, (sensor, seq) in enumerate(TRAIN):
        name = sensor + "/" + seq
        cache = cache_path(name)
        path = OUT / "mining" / f"{sensor}_{seq}_negatives.npz"
        with np.load(cache) as z, np.load(path) as n:
            np.testing.assert_array_equal(z["scan_ids"], n["scan_ids"])
            d, p = z["descriptors"], z["poses"]
            raw.append(d)
            poses.append(p)
            seqids.extend([seqid] * len(d))
            pool = n["negative_pool"]
            pools.append(np.where(pool >= 0, pool + offset, -1))
            neighbors = cKDTree(p[:, :3, 3]).query_ball_tree(cKDTree(p[:, :3, 3]), 5.)
            positives.extend(np.array([j + offset for j in js if abs(i - j) >= 30], np.int64)
                             for i, js in enumerate(neighbors))
            offset += len(d)
        provenance[str(cache.relative_to(REPO))] = sha256(cache)
        provenance[str(path.relative_to(REPO))] = sha256(path)
    raw, poses, seqids = np.concatenate(raw), np.concatenate(poses), np.asarray(seqids)
    pool = np.concatenate(pools)
    np.testing.assert_array_equal(raw, old["descriptors"])
    np.testing.assert_array_equal(poses, old["poses"])
    np.testing.assert_array_equal(seqids, old["sequences"])
    negatives = old["negative"]
    reports = []
    for epoch in (1, 6, 11, 16):
        original, matching = paired_triplets(positives, negatives, pool, seed * 100000 + epoch)
        np.testing.assert_array_equal(original[:, :2], matching[:, :2])
        assert len(original) == 183249
        for condition, trips in (("original", original), ("matching", matching)):
            a, p, n = trips.T
            assert np.all(seqids[a] == seqids[p]) and np.all(seqids[a] == seqids[n])
            assert np.all(np.abs(a - p) >= 30) and np.all(np.abs(a - n) >= 30)
            assert np.all(np.linalg.norm(poses[a, :3, 3] - poses[p, :3, 3], axis=1) <= 5.)
            assert np.all(np.linalg.norm(poses[a, :3, 3] - poses[n, :3, 3], axis=1) > 10.)
            np.savez_compressed(folder / f"{condition}_epoch{epoch}.npz", triplets=trips)
        reports.append({"epoch": epoch, "n_triplets": len(original),
                        "eligible": int(np.any(pool[original[:, 0]] >= 0, axis=1).sum()),
                        "changed": int((original[:, 2] != matching[:, 2]).sum())})
    np.savez_compressed(folder / "training_inputs.npz", descriptors=raw, poses=poses, sequences=seqids)
    provenance[str(reference.relative_to(REPO))] = sha256(reference)
    write_json(folder / "audit.json", {"seed": seed, "epochs": reports, "sha256": provenance,
               "sample_sha256": {p.name: sha256(p) for p in folder.glob("*.npz")}})
    print("SAMPLES_DONE", seed, reports, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("mine", "samples"))
    parser.add_argument("--sequence")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    mine(args.sequence) if args.action == "mine" else samples(args.seed)
