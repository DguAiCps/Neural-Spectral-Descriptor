#!/usr/bin/env python3
"""Fill the 544D/672D rows of tab:aliasrate from the retained dumps.

Same AliasRate definition as scripts/_verify_aliasrate_main.py (eps=0.1,
gamma=25 m, 200k sampled pairs, rng(0), L2-normalized descriptors, 2D-xy far
pairs). Inputs: results/_bench544672/{sensor}_{seq}.npz (desc544 + emb672 +
poses). Output: results/_alias_source/aliasrate_544672.json (per-seq + the
KITTI/NCLT/HeLiPR sensor macros used by the table).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DUMPS = REPO / "results/_bench544672"
OUT = REPO / "results/_alias_source/aliasrate_544672.json"
SENSORS = {"kitti": ["00", "05", "08"],
           "nclt": ["2012-01-08", "2013-01-10"],
           "helipr": ["Town01"]}
EPS, GAMMA, NPAIRS = 0.1, 25.0, 200_000


def aliasrate(desc, poses, rng):
    d = desc.astype(np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    xy = poses[:, :2, 3]
    n = d.shape[0]
    i = rng.integers(0, n, NPAIRS)
    j = rng.integers(0, n, NPAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    geo = np.linalg.norm(xy[i] - xy[j], axis=1)
    dd = np.linalg.norm(d[i] - d[j], axis=1)
    far = geo >= GAMMA
    coll = (dd <= EPS) & far
    return 100.0 * coll.sum() / max(int(far.sum()), 1)


def main():
    out = {"per_seq": {}, "macro": {}}
    for sensor, seqs in SENSORS.items():
        for key in ("desc544", "emb672"):
            vals = []
            for seq in seqs:
                z = np.load(DUMPS / f"{sensor}_{seq}.npz")
                ar = aliasrate(z[key], z["poses"], np.random.default_rng(0))
                out["per_seq"][f"{sensor}/{seq}/{key}"] = round(ar, 4)
                vals.append(ar)
            out["macro"][f"{sensor}/{key}"] = round(float(np.mean(vals)), 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["macro"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
