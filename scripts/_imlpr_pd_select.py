#!/usr/bin/env python3
"""ImLPR pd2026 checkpoint selection (pre-declared rule, RING# precedent).

Rule (frozen before any test contact): among the checkpoints the official
trainer saved (save_freq=10 -> _external/ImLPR/weights/ImLPR_pd2026_{10,20,
...,100}.pth; _final.pth duplicates the last periodic epoch when save_freq
divides epochs and is ignored), select the one with the highest pd-validation
sequence-macro 5 m Recall@1 (macro over the 4 pd val sequences kitti/09,
nclt/2012-01-08, helipr/Roundabout03, mulran/Sejong03, equal weight); on
ties, the EARLIER epoch. Records the selected weight path + sha256 into
results/pd2026_baselines/imlpr_pd_ckpt_selection.json. Only after this file
is written may the selected checkpoint touch the test roster.

Per (epoch, sequence): descriptors are extracted from the FULL-roster
validation RIV cache (_external/imlpr_data_pd2026/validation/{sensor}_{seq}/,
built by scripts/_imlpr_pd_prepare.py --stage riv --split val) through the
frozen zero-shot input pipeline + model shim of scripts/_imlpr_desc.py
(prep_frame / load_model, weights overridden per epoch), then scored with the
pd protocol every Table-1 baseline used: baselines/eval_utils.
compute_recall_multi_k, intra-session revisit queries, distance_threshold
5 m, skip_frames 30, on the val-cache poses.

Resumable per (epoch, seq): each cell writes
results/pd2026_baselines/_val_imlpr_pd_e{N}_{sensor}_{seq}.json and is
skipped when present (each cell records its weight sha256; a sha mismatch at
aggregation flags stale cells from an overwritten checkpoint).

Usage (inside docker ring_baseline, GPU):
    python3 scripts/_imlpr_pd_select.py [--batch 16] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from _imlpr_desc import DESC_DIM, load_model, prep_frame, sha256_file  # noqa: E402
from _imlpr_pd_prepare import VAL_SEQS, cache_path, split_dir  # noqa: E402

OUT_DIR = REPO / "results" / "pd2026_baselines"
WEIGHTS_DIR = REPO / "_external" / "ImLPR" / "weights"
MODEL_NAME = "ImLPR_pd2026"  # = config_pd.txt model_name (trainer file prefix)

RULE = ("highest pd-validation sequence-macro 5 m Recall@1 over the saved "
        "epochs (save_freq=10: epochs 10,20,...,100); earlier epoch on ties "
        "(pre-declared before any test contact)")
PROTOCOL = {"distance_threshold_m": 5.0, "skip_frames": 30,
            "intra_session": True,
            "scorer": "baselines/eval_utils.compute_recall_multi_k"}


def rel_to_repo(p: Path) -> str:
    """Repo-relative string when possible, the path as given otherwise."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def discover_checkpoints(weights_dir: Path, model_name: str) -> dict[int, Path]:
    pat = re.compile(rf"^{re.escape(model_name)}_(\d+)\.pth$")
    ckpts = {}
    for p in sorted(weights_dir.glob(f"{model_name}_*.pth")):
        m = pat.match(p.name)
        if m:  # skips _final.pth / _best.pth by construction
            ckpts[int(m.group(1))] = p
    if not ckpts:
        raise SystemExit(f"no {model_name}_<epoch>.pth checkpoints under "
                         f"{weights_dir} — has training finished (or reached "
                         f"the first save_freq epoch)?")
    return dict(sorted(ckpts.items()))


def load_val_sequence(sensor: str, seq: str):
    """Returns (riv_dir, poses) after cross-checking RIV meta vs val cache."""
    riv_dir = split_dir(sensor, seq, "val")
    meta_path = riv_dir / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"[{sensor}/{seq}] val RIV cache incomplete: no "
                         f"{meta_path} (run scripts/_imlpr_pd_prepare.py "
                         f"--stage riv --split val first)")
    meta = json.loads(meta_path.read_text())
    with np.load(cache_path(sensor, seq, "val"), allow_pickle=True) as c:
        scan_ids = c["scan_ids"].astype(np.int64)
        poses = c["poses"].astype(np.float64)
    assert meta["n_frames"] == len(scan_ids), (meta["n_frames"], len(scan_ids))
    assert np.array_equal(np.asarray(meta["scan_ids"], dtype=np.int64),
                          scan_ids), \
        f"[{sensor}/{seq}] val RIV meta scan_ids != val-cache scan_ids"
    return riv_dir, poses


def extract_descriptors(model, riv_dir: Path, n: int, batch: int,
                        device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    desc = np.zeros((n, DESC_DIM), dtype=np.float32)
    t0 = time.time()
    for i0 in range(0, n, batch):
        hi = min(i0 + batch, n)
        x = torch.from_numpy(np.stack(
            [prep_frame(riv_dir / f"{i:06d}.npy") for i in range(i0, hi)]))
        with torch.no_grad():
            _, d = model(x.to(device))  # (B, 8448)
        desc[i0:hi] = F.normalize(d, p=2, dim=1).cpu().numpy()
        if (i0 // batch) % 50 == 0:
            print(f"    {hi}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return desc


def eval_cell(epoch: int, weight: Path, weight_sha: str, sensor: str,
              seq: str, batch: int, device: str, model) -> dict:
    from baselines.eval_utils import compute_recall_multi_k

    riv_dir, poses = load_val_sequence(sensor, seq)
    n = len(poses)
    print(f"  [e{epoch} {sensor}/{seq}] extracting {n} descriptors", flush=True)
    desc = extract_descriptors(model, riv_dir, n, batch, device)
    recalls, n_queries = compute_recall_multi_k(
        desc, poses, k_values=[1, 5, 10],
        distance_threshold=PROTOCOL["distance_threshold_m"],
        skip_frames=PROTOCOL["skip_frames"])
    return {
        "epoch": epoch,
        "sensor": sensor,
        "sequence": seq,
        "recall@1": float(recalls[1]),
        "recall@5": float(recalls[5]),
        "recall@10": float(recalls[10]),
        "n_queries": int(n_queries),
        "n_frames": int(n),
        "weight": rel_to_repo(weight),
        "weight_sha256": weight_sha,
        "protocol": PROTOCOL,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def cell_path(epoch: int, sensor: str, seq: str) -> Path:
    return OUT_DIR / f"_val_imlpr_pd_e{epoch}_{sensor}_{seq}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights-dir", default=str(WEIGHTS_DIR))
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute cells even if their JSON exists")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = discover_checkpoints(Path(args.weights_dir), args.model_name)
    print(f"checkpoints: epochs {sorted(ckpts)}")

    # ---- fill missing (epoch, seq) cells ----
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shas: dict[int, str] = {}
    for epoch, weight in ckpts.items():
        todo = [(s, q) for s, q in VAL_SEQS
                if args.overwrite or not cell_path(epoch, s, q).is_file()]
        shas[epoch] = sha256_file(weight)
        if not todo:
            continue
        print(f"[e{epoch}] loading {weight.name} "
              f"(sha256 {shas[epoch][:12]}…)", flush=True)
        model = load_model(weight, device)
        for sensor, seq in todo:
            res = eval_cell(epoch, weight, shas[epoch], sensor, seq,
                            args.batch, device, model)
            out = cell_path(epoch, sensor, seq)
            tmp = out.with_name("." + out.name + ".tmp")
            tmp.write_text(json.dumps(res, indent=2) + "\n")
            os.replace(tmp, out)
            print(f"  [e{epoch} {sensor}/{seq}] R@1@5m {res['recall@1']:.4f} "
                  f"({res['n_queries']} queries) -> {out.name}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- aggregate + select (max macro; ties -> earlier epoch) ----
    table = {}
    for epoch in sorted(ckpts):
        per_seq = {}
        for sensor, seq in VAL_SEQS:
            d = json.loads(cell_path(epoch, sensor, seq).read_text())
            assert d["weight_sha256"] == shas[epoch], (
                f"e{epoch} {sensor}/{seq}: cell sha {d['weight_sha256'][:12]} "
                f"!= current checkpoint sha {shas[epoch][:12]} — checkpoint "
                f"was overwritten after the cell was scored; rerun with "
                f"--overwrite")
            per_seq[f"{sensor}_{seq}"] = {"recall@1": d["recall@1"],
                                          "n_queries": d["n_queries"]}
        macro = sum(v["recall@1"] for v in per_seq.values()) / len(per_seq)
        table[f"e{epoch}"] = {"macro_r1_5m": macro, "per_seq": per_seq}
        print(f"e{epoch}: macro R@1@5m = {macro:.4f}  "
              + "  ".join(f"{k}={v['recall@1']:.4f}"
                          for k, v in sorted(per_seq.items())))

    best = min(table, key=lambda t: (-table[t]["macro_r1_5m"], int(t[1:])))
    best_epoch = int(best[1:])
    sel = {
        "rule": RULE,
        "val_sequences": [f"{s}/{q}" for s, q in VAL_SEQS],
        "protocol": PROTOCOL,
        "epochs_scored": sorted(ckpts),
        "per_epoch": table,
        "selected_epoch": best,
        "selected_weight": rel_to_repo(ckpts[best_epoch]),
        "weight_sha256": shas[best_epoch],
    }
    out = OUT_DIR / "imlpr_pd_ckpt_selection.json"
    out.write_text(json.dumps(sel, indent=2) + "\n")
    print(f"\nselected: {best} -> {sel['selected_weight']}\n"
          f"sha256={sel['weight_sha256']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
