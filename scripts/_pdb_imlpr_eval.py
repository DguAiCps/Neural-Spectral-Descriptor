#!/usr/bin/env python3
"""pd2026 baseline eval: ImLPR (CoRL'25, official weights, zero-shot) on the
frozen 42-sequence pd test roster — Table 1.

Protocol is identical to scripts/_pdb_bevplace_eval.py: keyframes + poses
come from the committed pd2026 operating caches
(scripts/_pd_test_cache_manifest.json), intra-session revisit queries with
distance_threshold 5 m / skip_frames 30, R@1/R@5/R@10
(baselines/eval_utils.compute_recall_multi_k — the same scorer every Table-1
baseline row used).

Descriptors are NOT re-encoded here: this driver consumes the FROZEN
descriptor interface produced by the ImLPR RIV/descriptor stage —
    _external/imlpr_data_pd2026/desc_{sensor}_{seq}.npz
with arrays desc (n, 8448) float16 (L2-normalized before the fp16 cast) and
scan_ids (int64, == the pd2026 operating-cache scan_ids). Rows are upcast
fp16 -> float32 on load; compute_recall_multi_k re-normalizes internally, so
the fp16 round-trip is score-neutral. This module imports NO torch.

WEIGHT CONVENTION (documented, mirrors the aed302d lower-bound convention of
scripts/_pdb_ringsharp_eval.py): the official CoRL'25 release checkpoint
_external/ImLPR/weights/ImLPR_default.pth is used ZERO-SHOT — no retraining
on the pd2026 train split. ImLPR was trained on HeLiPR DCC04-06, KAIST04-06
and Riverside04-06, which ARE 9 pd TEST sequences; on those, recall is
train-contaminated (optimistic upper bound) and the result JSON carries
"train_overlap": true. The other 33 sequences are strictly zero-shot.

--method helios serves the HeLiOS (ICRA'25) descriptors on the SAME frozen
interface (desc_helios_{sensor}_{seq}.npz from scripts/_helios_desc.py,
also 8448D fp16), writing helios_{sensor}_{seq}.json; default imlpr is
unchanged.

Usage (inside docker ring_baseline, after the descriptor stage):
    python3 scripts/_pdb_imlpr_eval.py --seqs mulran/DCC03
    python3 scripts/_pdb_imlpr_eval.py            # all 42, resumable
    python3 scripts/_pdb_imlpr_eval.py --method helios --seqs kitti/05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from _pdb_bevplace_eval import MANIFEST, sha256_head  # noqa: E402 (torch-free)

DESC_DIR = REPO / "_external/imlpr_data_pd2026"
DEFAULT_WEIGHTS = REPO / "_external/ImLPR/weights/ImLPR_default.pth"
DESCRIPTOR_DIM = 8448

# ImLPR training split (CoRL'25 paper: HeLiPR DCC04-06 / KAIST04-06 /
# Riverside04-06) intersected with the pd TEST roster: contaminated sequences.
TRAIN_OVERLAP = {("helipr", s) for s in (
    "DCC04", "DCC05", "DCC06", "KAIST04", "KAIST05", "KAIST06",
    "Riverside04", "Riverside05", "Riverside06")}

WEIGHT_CONVENTION = (
    "official CoRL'25 ImLPR release checkpoint used ZERO-SHOT: no retraining "
    "on the pd2026 train split. ImLPR was trained on HeLiPR DCC04-06/"
    "KAIST04-06/Riverside04-06, which are 9 pd TEST sequences; on those, "
    "recall is train-contaminated (optimistic upper bound) and the result "
    "carries train_overlap: true. All other sequences are strictly zero-shot "
    "(aed302d lower-bound convention: train/test overlap documented per "
    "sequence, not retrained away).")

ENCODER_META = {
    "model": "ImLPR (official CoRL'25 code, _external/ImLPR): DINOv2 ViT-S/14"
             " + adapters + SALAD head over 3-channel RIV images",
    "model_name": "dinov2_vits14", "num_trainable_blocks": 2,
    "adapter_frequency": 3, "num_channels": 384, "num_clusters": 64,
    "cluster_dim": 128, "token_dim": 256,
    "descriptor_dim": DESCRIPTOR_DIM, "l2_normalized": True,
    "storage": "float16 (L2-normalized before the fp16 cast)",
}

# ---- HeLiOS (--method helios): same frozen 8448D interface, produced by
# scripts/_helios_desc.py as desc_helios_{sensor}_{seq}.npz. ----
HELIOS_WEIGHTS = REPO / "_external/HeLiOS/weights/HeLiOS.pth"

HELIOS_WEIGHT_CONVENTION = (
    "official ICRA'25 HeLiOS release checkpoint used ZERO-SHOT: no "
    "retraining on the pd2026 train split. HeLiOS was trained on HeLiPR "
    "DCC04-06/KAIST04-06/Riverside04-06, which are 9 pd TEST sequences; on "
    "those, recall is train-contaminated (optimistic upper bound) and the "
    "result carries train_overlap: true. All other sequences are strictly "
    "zero-shot (aed302d lower-bound convention: train/test overlap "
    "documented per sequence, not retrained away). Descriptors are computed "
    "on single-keyframe sensor-frame scans, not HeLiPR-toolbox submaps "
    "(disclosed domain note; see scripts/_helios_desc.py).")

HELIOS_ENCODER_META = {
    "model": "HeLiOS (official ICRA'25 code, _external/HeLiOS): sparse-voxel"
             " MinkowskiEngine backbone + sptr attention + SALAD-style head",
    "config": "_external/HeLiOS/config/model_helios.txt",
    "num_clusters": 64, "cluster_dim": 128, "use_token": True,
    "descriptor_dim": DESCRIPTOR_DIM, "l2_normalized": True,
    "storage": "float16 (L2-normalized before the fp16 cast)",
}

# --method registry: frozen-desc filename prefix + per-method weights and
# disclosure blocks. TRAIN_OVERLAP is shared: both methods were trained on
# the same 9 HeLiPR pd-TEST sequences.
METHODS = {
    "imlpr": dict(prefix="desc_", weights=DEFAULT_WEIGHTS,
                  encoder=ENCODER_META, convention=WEIGHT_CONVENTION,
                  display="ImLPR"),
    "helios": dict(prefix="desc_helios_", weights=HELIOS_WEIGHTS,
                   encoder=HELIOS_ENCODER_META,
                   convention=HELIOS_WEIGHT_CONVENTION, display="HeLiOS"),
    "imlpr_pd": dict(
        prefix="desc_imlpr_pd_",
        weights=REPO / "_external/ImLPR/weights/ImLPR_pd2026_100.pth",
        encoder=ENCODER_META,
        convention=(
            "CLEAN pd2026 place-disjoint retrain of ImLPR: official trainer, "
            "from-scratch init (pretrained DINOv2 backbone + fresh adapters/"
            "SALAD; the released ImLPR_default.pth was NOT used — it is "
            "HeLiPR-Daejeon-trained), official recipe (batch 2048, "
            "TruncatedSmoothAP, 100 epochs; batch_split 32 and num_workers 0 "
            "as 16GB memory knobs), train frames = pd 16 sequences at the "
            "official 3.0 m subsample; checkpoint = pre-declared rule "
            "(highest pd-val sequence-macro 5 m R@1 over saved epochs, "
            "earlier on ties) -> epoch 100, sha256 frozen in "
            "results/pd2026_baselines/imlpr_pd_ckpt_selection.json. "
            "No train/test overlap by construction."),
        display="ImLPR (pd retrain)",
        overlap=frozenset()),
}
# zero-shot methods share the authors' HeLiPR-Daejeon training overlap
METHODS["imlpr"]["overlap"] = TRAIN_OVERLAP
METHODS["helios"]["overlap"] = TRAIN_OVERLAP


def rel_to_repo(p: Path) -> str:
    """Repo-relative string when possible (logging), the path as given
    otherwise (e.g. a --desc-dir override outside the repo)."""
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def load_desc(sensor: str, seq: str, desc_dir: Path = DESC_DIR,
              prefix: str = "desc_"):
    """Frozen descriptor interface: (desc float32 (n, 8448), scan_ids
    int64, path). Rows were L2-normalized before the fp16 cast.
    prefix selects the method: "desc_" (ImLPR) or "desc_helios_"."""
    p = desc_dir / f"{prefix}{sensor}_{seq}.npz"
    if not p.is_file():
        raise FileNotFoundError(
            f"missing frozen descriptor cache: {p} "
            "(run the descriptor stage first)")
    with np.load(p, allow_pickle=False) as z:
        desc = z["desc"].astype(np.float32)
        scan_ids = z["scan_ids"].astype(np.int64)
    assert desc.ndim == 2 and desc.shape[1] == DESCRIPTOR_DIM, desc.shape
    assert len(desc) == len(scan_ids), (p.name, "desc/scan_ids length")
    return desc, scan_ids, p


def eval_sequence(sensor: str, seq: str, cache_rel: str,
                  desc_dir: Path, method: str = "imlpr") -> dict:
    from baselines.eval_utils import compute_recall_multi_k

    cache_path = REPO / cache_rel
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        scan_ids = cache["scan_ids"].astype(np.int64)
        poses = cache["poses"].astype(np.float64)

    desc, desc_sids, desc_path = load_desc(
        sensor, seq, desc_dir, prefix=METHODS[method]["prefix"])
    assert np.array_equal(desc_sids, scan_ids), \
        (sensor, seq, "descriptor scan_ids != operating-cache scan_ids")

    print(f"[{sensor}/{seq}] {len(scan_ids)} keyframes from {desc_path.name} "
          f"(cache {cache_rel})", flush=True)
    recalls, n_queries = compute_recall_multi_k(
        desc, poses, k_values=[1, 5, 10],
        distance_threshold=5.0, skip_frames=30)
    return {
        "method": method,
        "sensor": sensor,
        "sequence": seq,
        "protocol": {"distance_threshold_m": 5.0, "skip_frames": 30,
                     "intra_session": True},
        "cache": cache_rel,
        "desc_cache": rel_to_repo(desc_path),
        "n_keyframes": int(len(scan_ids)),
        "n_queries": int(n_queries),
        "recall_at_1": float(recalls[1]),
        "recall_at_5": float(recalls[5]),
        "recall_at_10": float(recalls[10]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), default="imlpr",
                        help="frozen descriptor stack to evaluate")
    parser.add_argument("--weights", type=Path, default=None,
                        help="official checkpoint (sha-logged only; "
                             "descriptors are consumed precomputed; default: "
                             "the method's release checkpoint)")
    parser.add_argument("--desc-dir", type=Path, default=DESC_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "results/pd2026_baselines")
    parser.add_argument("--seqs", nargs="+", default=None,
                        help="subset like mulran/DCC03 mcd/tuhh_night_09 "
                             "(default: all 42)")
    parser.add_argument("--force", action="store_true",
                        help="re-run sequences whose output JSON exists")
    args = parser.parse_args()

    spec = METHODS[args.method]
    if args.weights is None:
        args.weights = spec["weights"]
    if not args.weights.is_file():
        raise SystemExit(f"missing {spec['display']} checkpoint: {args.weights}")
    weights_id = sha256_head(args.weights)

    roster = json.load(open(MANIFEST))["sequences"]
    wanted = set(args.seqs) if args.seqs else None
    todo = [(s["sensor"], s["sequence"], s["expected"]["operating_cache"])
            for s in roster
            if wanted is None or f"{s['sensor']}/{s['sequence']}" in wanted]
    if wanted is not None and len(todo) != len(wanted):
        have = {f"{a}/{b}" for a, b, _ in todo}
        raise SystemExit(f"unknown --seqs entries: {sorted(wanted - have)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sensor, seq, cache_rel in todo:
        out_path = args.output_dir / f"{args.method}_{sensor}_{seq}.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} exists", flush=True)
            continue
        record = eval_sequence(sensor, seq, cache_rel, args.desc_dir,
                               method=args.method)
        record["weights"] = str(args.weights)
        record["weights_sha256_1mb"] = weights_id
        record["_meta"] = {
            "weight_convention": spec["convention"],
            "train_overlap": (sensor, seq) in spec.get("overlap", TRAIN_OVERLAP),
            "zero_shot_wrt_pd_train": True,
            "weights": str(args.weights),
            "weights_sha256_1mb": weights_id,
            "encoder": dict(spec["encoder"]),
        }
        out_path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"  R@1={record['recall_at_1']:.4f} "
              f"R@5={record['recall_at_5']:.4f} q={record['n_queries']} "
              f"train_overlap={record['_meta']['train_overlap']} "
              f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
