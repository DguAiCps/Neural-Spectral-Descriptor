#!/usr/bin/env python3
"""Offline split-v2 pilot with the clipped-BEV NSD auxiliary block (fix A,
2026-10-10).

Same as scripts/_pdv2_pilot.py (checkpoints/pdv2_s1, omega, edge classifier,
selection rule, seed 1) except the spectrum caches: val/test spectra come
from data/preprocessed_v2_bevclip (scripts/_bevclip_spec.py). `link` adds
relative links to the other val/test caches of data/preprocessed_v2 into
that folder, so the pd scripts read it as their cache folder unchanged.
NCLT test sessions link the invariant-key caches of data/nclt_fov/gt/
(scripts/_nclt_fov_fix.py, mirrored elevation window, 2026-10-11).
NSD-H (hspec12) should reproduce results/pdv2 up to run-to-run GPU noise.

  results  results/pdv2_bevclip/

Subcommands: link, fusion --tag T, select, apply --seq S, aggregate.
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import _pdv2_pilot as pilot  # noqa: E402

SRC = REPO / "data/preprocessed_v2"
NCLT_FOV = REPO / "data/nclt_fov/gt"
pilot.V2 = REPO / "data/preprocessed_v2_bevclip"
pilot.OUT = REPO / "results/pdv2_bevclip"


def cmd_link():
    for s, q in pilot.VAL + pilot.TEST:
        role = "val" if (s, q) in pilot.VAL else "test"
        assert (pilot.V2 / f"{role}spec_{s}_{q}.npz").is_file(), (s, q)
        cache_dir = NCLT_FOV if s == "nclt" else SRC
        names = [Path(p).name for p in glob.glob(str(cache_dir / f"cache_*_{s}_{role}_{q}.npz"))]
        assert len(names) == 1, (s, q, names)
        for stale in pilot.V2.glob(f"cache_*_{s}_{role}_{q}.npz"):
            if stale.name != names[0]:
                stale.unlink()                     # link to the pre-fix NCLT cache
        for name, src in ((names[0], cache_dir), (f"{role}layouts_{s}_{q}.npz", SRC)):
            dst = pilot.V2 / name
            if not dst.is_symlink():
                dst.symlink_to(Path("..") / src.relative_to(REPO / "data") / name)
            assert dst.resolve() == (src / name).resolve(), dst
    print("BEVCLIP_LINK_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("link")
    s = sub.add_parser("fusion")
    s.add_argument("--tag", choices=list(pilot.VAL_TAGS), required=True)
    sub.add_parser("select")
    s = sub.add_parser("apply")
    s.add_argument("--seq", required=True)
    s.add_argument("--device", default="cuda")
    sub.add_parser("aggregate")
    a = p.parse_args()
    pilot.OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "link":
        cmd_link()
    elif a.cmd == "fusion":
        pilot.cmd_fusion(a.tag)
    elif a.cmd == "select":
        pilot.cmd_select()
    elif a.cmd == "apply":
        pilot.cmd_apply(a.seq, a.device)
    elif a.cmd == "aggregate":
        pilot.cmd_aggregate()


if __name__ == "__main__":
    main()
