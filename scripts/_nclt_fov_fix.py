#!/usr/bin/env python3
"""NCLT invariant-key caches with the mirrored elevation window (2026-10-11).

The NCLT HDL-32E is mounted upside down; the loader flips z to make the
cloud z-up (src/data/nclt_loader.py), which mirrors the sensor's vertical
field of view to [-10.67, +30.67] deg. The configs still give NCLT the
upright window [-30.67, +10.67] (sensor_elevation_ranges.nclt), so the
range image clamped most returns into its top rows
(docs/RAW/실험일지/2026-10-11_baseline_fidelity_audit.md).

For one NCLT test session this recomputes the 288-D descriptors (and
spectral entropies) of the existing keyframes of both keyframe conditions
with the mirrored window and writes

    data/nclt_fov/{odom,gt}/cache_<key>_nclt_test_<seq>.npz

with every other array copied. Keyframe selection uses poses and points
only, so the keyframes are those a full rebuild would select. Before
writing, the first 20 keyframes of each condition recomputed with the old
window must reproduce the stored descriptors. Run inside ring_baseline:

    python3 scripts/_nclt_fov_fix.py --seq 2012-02-04
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _pd_build_test_caches import (  # noqa: E402
    _effective_config, _make_desc_encoder, _make_loader)
from _pdv2_pilot import TEST  # noqa: E402
from train_multi_dataset import compute_cache_key  # noqa: E402

NCLT_WINDOW = [-10.67, 30.67]          # mirrored HDL-32E window after the z flip
SRC = {"odom": REPO / "data/preprocessed_v2_odom", "gt": REPO / "data/preprocessed_v2"}
DST = REPO / "data/nclt_fov"
N_CHECK = 20


def fixed_config():
    cfg = copy.deepcopy(_effective_config())
    cfg["encoding"]["sensor_elevation_ranges"]["nclt"] = list(NCLT_WINDOW)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="NCLT test session, e.g. 2012-02-04")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    assert ("nclt", a.seq) in TEST, a.seq
    old_cfg, new_cfg = _effective_config(), fixed_config()
    old_key = compute_cache_key(old_cfg, dataset_type="nclt")
    new_key = compute_cache_key(new_cfg, dataset_type="nclt")
    assert old_key != new_key
    name_old = f"cache_{old_key}_nclt_test_{a.seq}.npz"
    outs = {c: DST / c / f"cache_{new_key}_nclt_test_{a.seq}.npz" for c in SRC}
    if all(p.exists() for p in outs.values()):
        print(f"SKIP nclt/{a.seq}: fixed caches exist", flush=True)
        return
    caches = {c: dict(np.load(SRC[c] / name_old)) for c in SRC}
    ids = np.union1d(caches["odom"]["scan_ids"], caches["gt"]["scan_ids"])
    check = {int(s) for z in caches.values() for s in z["scan_ids"][:N_CHECK]}
    loader = _make_loader("nclt", a.seq)
    enc_old = _make_desc_encoder(old_cfg, "nclt", a.device)
    enc_new = _make_desc_encoder(new_cfg, "nclt", a.device)
    desc, ent, desc_old = {}, {}, {}
    for k, sid in enumerate(ids):
        if k % 2000 == 0:
            print(f"  [nclt/{a.seq}] scan {k}/{len(ids)}", flush=True)
        points = loader[int(sid)]["points"]
        d, e = enc_new.encode_points(points, return_entropy=True)
        desc[int(sid)], ent[int(sid)] = d.detach().cpu().numpy(), float(e)
        if int(sid) in check:
            desc_old[int(sid)] = enc_old.encode_points(points).detach().cpu().numpy()
    for c, z in caches.items():
        again = np.stack([desc_old[int(s)] for s in z["scan_ids"][:N_CHECK]])
        diff = float(np.abs(again - z["descriptors"][:N_CHECK]).max())
        scale = float(np.abs(z["descriptors"][:N_CHECK]).max())
        print(f"  [nclt/{a.seq}] {c}: old-window descriptors vs stored, max diff {diff:.2e}"
              f" (max |stored| {scale:.0f})", flush=True)
        assert diff <= 1e-6 * scale, (a.seq, c, diff, scale)   # float32 rounding
    for c, z in caches.items():
        out = dict(z)
        out["descriptors"] = np.stack([desc[int(s)] for s in z["scan_ids"]]).astype(
            z["descriptors"].dtype)
        out["spectral_entropies"] = np.array([ent[int(s)] for s in z["scan_ids"]],
                                             dtype=z["spectral_entropies"].dtype)
        outs[c].parent.mkdir(parents=True, exist_ok=True)
        tmp = outs[c].with_name(outs[c].stem + ".tmp.npz")
        np.savez(tmp, **out)
        tmp.rename(outs[c])
        print(f"OK {c} {outs[c].relative_to(REPO)}: {len(z['scan_ids'])} keyframes", flush=True)
    print("NCLT_FOV_DONE", flush=True)


if __name__ == "__main__":
    main()
