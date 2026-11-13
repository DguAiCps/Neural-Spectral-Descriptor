#!/usr/bin/env python3
"""원래 FP32 접두 승자·점수를 보존하는 보수적 FP64 후보 제안이다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def ledger_exact(desc: np.ndarray, pos: np.ndarray, base: dict) -> tuple:
    """반올림 경계보다 확실히 벌어진 승자만 채택하고 나머지는 원경로로 계산한다."""
    import torch
    from _causal_pr_dump import prefix_topk

    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    assert desc.dtype == np.float32 and np.isfinite(desc).all()
    vectors = np.ascontiguousarray(desc)
    exact = vectors.astype(np.float64)
    dim = vectors.shape[1]
    assert dim * np.finfo(np.float32).eps / 2 < 0.01
    norms = np.einsum("ij,ij->i", exact, exact)
    gamma32 = (dim + 8) * (np.finfo(np.float32).eps / 2)
    gamma32 /= 1 - gamma32
    gamma64 = (dim + 8) * (np.finfo(np.float64).eps / 2)
    gamma64 /= 1 - gamma64
    # |fl(x.y)-x.y| <= gamma_D sum|x_i*y_i| <= gamma_D ||x||||y||.
    # 두 후보와 FP64 제안의 오차를 함께 덮는 여유다. FP32 값은 변경하지 않는다.
    margin = max(4 * (gamma32 + gamma64) * float(norms.max()), 1e-300)
    bank = torch.from_numpy(exact).cuda()
    qs = base["query_idx"]
    arrays = {k: v.copy() for k, v in base.items()}
    arrays["j_star"] = np.empty(len(qs), np.int64)
    arrays["score_cos"] = np.empty(len(qs), np.float64)
    arrays["top1_geo_dist"] = np.empty(len(qs), np.float64)
    fallbacks = 0
    minimum_gap = float("inf")
    runtime_checks = []
    check_indices = {0, len(qs) // 2, len(qs) - 1}
    for offset in range(0, len(qs), 128):
        block = qs[offset:offset + 128]
        common_end = int(block[0] - 29)
        # 미래·자기 후보를 곱한 뒤 거르는 방식은 쓰지 않는다.
        scores = bank[block] @ bank[:common_end].T
        vals, indices = torch.topk(scores, min(2, common_end), dim=1, largest=True, sorted=True)
        vals, indices = vals.cpu().numpy(), indices.cpu().numpy()
        for slot, q in enumerate(block):
            end = int(q - 29)
            recent = np.arange(common_end, end, dtype=np.int64)
            selected = np.concatenate((indices[slot], recent))
            values = np.concatenate((vals[slot], exact[recent] @ exact[q]))
            order = np.argsort(-values, kind="stable")
            gap = float(values[order[0]] - values[order[1]]) if len(order) > 1 else float("inf")
            if gap <= margin:
                j = int(prefix_topk(vectors, int(q), end, 1)[0])
                fallbacks += 1
            else:
                j = int(selected[order[0]])
                minimum_gap = min(minimum_gap, gap)
            at = offset + slot
            # 최종 점수는 원래 1D NumPy FP32 내적이다.
            arrays["j_star"][at] = j
            arrays["score_cos"][at] = float(vectors[j] @ vectors[q])
            arrays["top1_geo_dist"][at] = float(np.linalg.norm(pos[j] - pos[q]))
            if at in check_indices:
                original_j = int(prefix_topk(vectors, int(q), end, 1)[0])
                original_score = float(vectors[original_j] @ vectors[q])
                assert original_j == j and original_score == arrays["score_cos"][at]
                runtime_checks.append(dict(query=int(q),winner=j,score=original_score))
    return arrays, dict(proposal_dtype="float64",final_score_dtype="original NumPy float32 dot",
        roundoff_margin=margin,minimum_accepted_gap=minimum_gap,
        original_full_prefix_fallback_queries=fallbacks,n_queries=len(qs),
        all_candidates_past_only=True,runtime_original_checks=runtime_checks,
        near_tie_path="original prefix_topk full GEMV, argpartition, stable sort")


def verify(manifest: Path) -> None:
    """합성 동률 및 동결한 실제 서술자의 모든 질의를 원래 원장과 대조한다."""
    from _causal_bevplace_pr import ledger
    from _review_selection_bev import read, write, sha

    plan = read(manifest)
    home = manifest.parent
    checks = []

    def compare(label, vectors, pos):
        begin = time.time()
        reference = ledger(vectors, pos)
        original_s = time.time() - begin
        take = np.arange(len(vectors) - 45)
        base = {k: reference[k][take] for k in ("query_idx", "n_candidates", "min_past_gt_dist")}
        begin = time.time()
        actual, audit = ledger_exact(vectors, pos, base)
        accelerated_s = time.time() - begin
        mismatches = {k: int(np.count_nonzero(v != reference[k][take])) for k, v in actual.items()
                      if not np.array_equal(v, reference[k][take])}
        checks.append(dict(label=label,n_queries=len(take),original_s=original_s,
            accelerated_s=accelerated_s,mismatches=mismatches,audit=audit))
        write(home / "progress.json", checks)
        assert not mismatches, checks[-1]
        print("EXACT_COSINE",checks[-1],flush=True)

    rng = np.random.default_rng(50309)
    for tag in ("random", "all_equal", "duplicate", "near_tie", "zero"):
        vectors = rng.normal(size=(700, 8448)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        if tag == "all_equal": vectors[:] = vectors[0]
        if tag == "duplicate": vectors[::2] = vectors[0]
        if tag == "near_tie":
            vectors[:] = vectors[0]
            vectors[::3, 0] = np.nextafter(vectors[::3, 0], np.float32(np.inf))
        if tag == "zero": vectors[:] = 0
        compare(tag, vectors, rng.normal(size=(len(vectors), 3)))
    for row in plan["cases"]:
        source = ROOT / row["descriptors"]
        assert sha(source) == row["sha256"]
        compare(row["label"],np.load(source),np.load(ROOT / row["positions"]))
    write(home / "validated.json", dict(completed_unix=time.time(),checks=checks,
        n_queries=sum(c["n_queries"] for c in checks),all_array_fields_exact=True,
        original_sha256=sha(ROOT/"scripts/_causal_bevplace_pr.py"),
        original_prefix_sha256=sha(ROOT/"scripts/_causal_pr_dump.py"),
        implementation_sha256=sha(Path(__file__)),manifest_sha256=sha(manifest),
        scope="원래 FP32 모델/코사인/동률/과거후보를 보존; FP64는 확실한 승자의 제안에만 사용"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    args=parser.parse_args()
    verify(args.manifest)
