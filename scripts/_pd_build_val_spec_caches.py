#!/usr/bin/env python3
"""Build stored-spectrum branch caches for the pd-protocol NEW VAL sequences
(kitti 09, nclt 2012-01-08, helipr Roundabout03, mulran Sejong03).

Per keyframe scan (scan_ids taken as-is from the pd2026 val caches):
  sk_re/sk_im  -- (16,12) complex BEV phase sketch, exactly as
                  scripts/_pbev_branch_fusion.pbev_cache builds it
                  (raw BP.project 60-sector max BEV -> _pool_rows 16 ->
                  _phase_sketch 12; NO interpolation).
  raw_height   -- (20,120) max-height polar grid float16, exactly as
                  scripts/_height_candidate_fusion.height_cache stores it
                  (make_height_grids raw z, float16).

Writes data/preprocessed_pd2026/valspec_<sensor>_<seq>.npz
(scan_ids + sk_re + sk_im + raw_height). NEW VAL sequences only -- never test.
Loaders/geometry come from the same modules the paper scripts import, so the
transforms are byte-identical to the paper protocol.

--selftest rebuilds the first 50 kitti 07 TRAIN keyframes of the committed
results/height_sota/cache/K07_TR_height120.npz and compares the height grids
(kitti loader unchanged -> must match); the BEV path was already verified
exactly by scripts/_pd_build_val_layouts.py --selftest.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _pbev_branch_fusion import BP                      # noqa: E402  (paper BEV projector)
from _height_candidate_fusion import make_height_grids  # noqa: E402  (paper height grids)
from encoding.phase_features import _pool_rows, _phase_sketch  # noqa: E402
from _pd_build_val_layouts import _make_loader, _pd_cache, SEQS  # noqa: E402

PD_CACHE = REPO / "data/preprocessed_pd2026"


def spec_for_points(points: np.ndarray):
    bev, _ = BP.project(points, keep_intensity=False)
    sk = _phase_sketch(_pool_rows(bev, 16), 12)          # (16,12) complex64
    _, h_raw, _, _ = make_height_grids(points)
    return sk.real.astype(np.float32), sk.imag.astype(np.float32), \
        h_raw.astype(np.float16)


def build(sensor: str, seq: str, scan_ids, limit=None):
    loader = _make_loader(sensor, seq)
    res, ims, raws = [], [], []
    ids = scan_ids if limit is None else scan_ids[:limit]
    for j, sid in enumerate(ids):
        if j % 500 == 0:
            print(f"  [{sensor}/{seq}] spec {j}/{len(ids)}", flush=True)
        re, im, raw = spec_for_points(loader[int(sid)]["points"])
        res.append(re)
        ims.append(im)
        raws.append(raw)
    return (np.asarray(res, np.float32), np.asarray(ims, np.float32),
            np.asarray(raws, np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="subset like kitti/09 nclt/2012-01-08")
    args = ap.parse_args()

    if args.selftest:
        ref = np.load(REPO / "results/height_sota/cache/K07_TR_height120.npz")
        loader = _make_loader("kitti", "07")
        raws, heights, dens = [], [], []
        for sid in ref["scan_ids"][:50]:
            h, h_raw, d, _ = make_height_grids(loader[int(sid)]["points"])
            heights.append(h.astype(np.float16))
            raws.append(h_raw.astype(np.float16))
            dens.append(d.astype(np.float16))
        raws = np.asarray(raws)
        heights = np.asarray(heights)
        dens = np.asarray(dens)
        print(f"SELFTEST kitti/07 first 50 vs committed K07_TR_height120: "
              f"max|raw_height diff|={np.abs(raws.astype(np.float32) - ref['raw_height'][:50].astype(np.float32)).max():.6f} "
              f"max|height diff|={np.abs(heights.astype(np.float32) - ref['height'][:50].astype(np.float32)).max():.6f} "
              f"max|density diff|={np.abs(dens.astype(np.float32) - ref['density'][:50].astype(np.float32)).max():.6f}")
        return

    todo = SEQS if args.seqs is None else [
        tuple(s.split("/", 1)) for s in args.seqs]
    for sensor, seq in todo:
        out = PD_CACHE / f"valspec_{sensor}_{seq}.npz"
        if out.exists():
            print("SKIP", out.name, flush=True)
            continue
        cache = _pd_cache(sensor, seq, val=True)
        sk_re, sk_im, raw_height = build(sensor, seq, cache["scan_ids"])
        np.savez_compressed(out, scan_ids=cache["scan_ids"], sk_re=sk_re,
                            sk_im=sk_im, raw_height=raw_height)
        print(f"OK {out.name}: sk {sk_re.shape} raw_height {raw_height.shape}",
              flush=True)
    print("PD_VAL_SPEC_DONE", flush=True)


if __name__ == "__main__":
    main()
