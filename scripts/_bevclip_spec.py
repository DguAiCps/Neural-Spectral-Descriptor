#!/usr/bin/env python3
"""Clipped-BEV auxiliary spectrum caches (fix A, 2026-10-10).

The NSD auxiliary block is the 16x12 phase sketch of the max-height BEV grid
(scripts/_pd_build_val_spec_caches.spec_for_points). Cells that hold only
returns below the sensor keep negative heights there. Here the grid is
clipped at zero first, as the height-polar grid of NSD-H already does
(docs/RAW/실험일지/2026-10-10_karlsruhe_diagnosis.md):

    sk = phase_sketch(pool_rows(max(0, BEV), 16), 12)

For one val/test sequence it reads the scan ids of the existing spectrum
caches of both keyframe conditions, projects each scan once, and writes

    data/preprocessed_v2_odom_bevclip/{role}spec_<sensor>_<seq>.npz
    data/preprocessed_v2_bevclip/{role}spec_<sensor>_<seq>.npz

with the same scan_ids and raw_height (copied) and the clipped sk_re/sk_im.
Before writing, the unclipped sketch of the first 20 scans of each condition
must reproduce the stored one. Run inside ring_baseline:

    python3 scripts/_bevclip_spec.py --seq kitti/08
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _pbev_branch_fusion import BP  # noqa: E402  (paper BEV projector)
from _pd_build_test_caches import _make_loader  # noqa: E402
from _pdv2_pilot import TEST, VAL  # noqa: E402
from encoding.phase_features import _phase_sketch, _pool_rows  # noqa: E402

SRC = {"odom": REPO / "data/preprocessed_v2_odom", "gt": REPO / "data/preprocessed_v2"}
DST = {c: p.with_name(p.name + "_bevclip") for c, p in SRC.items()}
N_CHECK = 20


def _sketch(bev):
    return _phase_sketch(_pool_rows(bev, 16), 12)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sensor/seq of a val or test sequence")
    a = ap.parse_args()
    sensor, q = a.seq.split("/")
    assert (sensor, q) in VAL or (sensor, q) in TEST, a.seq
    name = f"{'val' if (sensor, q) in VAL else 'test'}spec_{sensor}_{q}.npz"
    outs = {c: DST[c] / name for c in SRC}
    if all(p.exists() for p in outs.values()):
        print(f"SKIP {a.seq}: clipped spectra exist", flush=True)
        return
    specs = {c: np.load(SRC[c] / name) for c in SRC}
    ids = np.union1d(specs["odom"]["scan_ids"], specs["gt"]["scan_ids"])
    check = {int(s) for z in specs.values() for s in z["scan_ids"][:N_CHECK]}
    loader = _make_loader(sensor, q)
    clipped, unclipped = {}, {}
    for k, sid in enumerate(ids):
        if k % 2000 == 0:
            print(f"  [{a.seq}] scan {k}/{len(ids)}", flush=True)
        bev, _ = BP.project(loader[int(sid)]["points"], keep_intensity=False)
        clipped[int(sid)] = _sketch(np.maximum(bev, 0.0))
        if int(sid) in check:
            unclipped[int(sid)] = _sketch(bev)
    for c, z in specs.items():
        stored = z["sk_re"][:N_CHECK] + 1j * z["sk_im"][:N_CHECK]
        again = np.stack([unclipped[int(s)] for s in z["scan_ids"][:N_CHECK]])
        diff = float(np.abs(again - stored).max())
        print(f"  [{a.seq}] {c}: unclipped sketch vs stored, max diff {diff:.2e}",
              flush=True)
        assert diff <= 1e-5, (a.seq, c, diff)
    for c, z in specs.items():
        sk = np.stack([clipped[int(s)] for s in z["scan_ids"]])
        DST[c].mkdir(parents=True, exist_ok=True)
        tmp = outs[c].with_name(outs[c].stem + ".tmp.npz")
        np.savez_compressed(tmp, scan_ids=z["scan_ids"], sk_re=sk.real.astype(np.float32),
                            sk_im=sk.imag.astype(np.float32), raw_height=z["raw_height"])
        tmp.rename(outs[c])
        print(f"OK {c} {outs[c].relative_to(REPO)}: sk {sk.shape}", flush=True)
    print("BEVCLIP_SPEC_DONE", flush=True)


if __name__ == "__main__":
    main()
