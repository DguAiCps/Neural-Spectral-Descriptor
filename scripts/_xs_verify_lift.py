#!/usr/bin/env python3
"""One-shot bit-identity check: lifted helpers in _xs_eval_lib vs the docker
originals (_height_candidate_fusion / _pbev_branch_fusion /
_hspec_branch_fusion, which import torch and cannot load on the host).

Run inside the ring_baseline container:
  python3 scripts/_xs_verify_lift.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import _xs_eval_lib as lib                                    # noqa: E402
from _height_candidate_fusion import (                        # noqa: E402
    align_scores as align_orig, column_align_scores as cas_orig,
    column_align_bank as cab_orig, minmax as minmax_orig, l2 as l2_orig)
from _pbev_branch_fusion import variant_inputs as vi_orig     # noqa: E402
from _hspec_branch_fusion import hspec_inputs as hi_orig      # noqa: E402

SPEC = REPO / "data/preprocessed_pd2026/testspec_nclt_2012-02-04.npz"
N = 400


def same(name, a, b):
    ok = np.array_equal(np.asarray(a), np.asarray(b))
    print(f"  {name}: {'BIT-IDENTICAL' if ok else 'MISMATCH'}")
    assert ok, name
    return ok


def main():
    z = np.load(SPEC)
    sk = (z["sk_re"][:N].astype(np.float32)
          + 1j * z["sk_im"][:N].astype(np.float32))
    raw_height = z["raw_height"][:N]          # f16 on disk, kept as loaded

    print("[variant_inputs pbev n_freqs=8]")
    o = vi_orig(sk, 8)
    m = lib.variant_inputs(sk, 8)
    for name, oo, mm in [("hkey", o[0], m[0]), ("hfft", o[1], m[1]),
                         ("hnorm", o[2], m[2]),
                         ("colbank.fft", o[3][0], m[3][0]),
                         ("colbank.mask", o[3][1], m[3][1])]:
        same(name, oo, mm)

    print("[hspec_inputs n_freqs=12] (f16 raw_height path)")
    oh = hi_orig(raw_height, 12)
    mh = lib.hspec_inputs(raw_height, 12)
    for name, oo, mm in [("hkey", oh[0], mh[0]), ("hfft", oh[1], mh[1]),
                         ("hnorm", oh[2], mh[2]),
                         ("colbank.fft", oh[3][0], mh[3][0]),
                         ("colbank.mask", oh[3][1], mh[3][1])]:
        same(name, oo, mm)

    print("[align_scores / column_align_scores / minmax / l2]")
    rng_cand = np.arange(1, 200, 2)
    for q in (0, 57, N - 1):
        same(f"align_scores q={q}",
             align_orig(o[1], o[2], q, rng_cand),
             lib.align_scores(m[1], m[2], q, rng_cand))
        same(f"column_align_scores q={q}",
             cas_orig(o[3], q, rng_cand),
             lib.column_align_scores(m[3], q, rng_cand))
    x = o[2][:97]
    same("minmax", minmax_orig(x), lib.minmax(x))
    same("l2", l2_orig(o[0][:11] * 3.7), lib.l2(m[0][:11] * 3.7))
    same("column_align_bank.fft", cab_orig(raw_height.astype(np.float32))[0],
         lib.column_align_bank(raw_height.astype(np.float32))[0])
    same("column_align_bank.mask", cab_orig(raw_height.astype(np.float32))[1],
         lib.column_align_bank(raw_height.astype(np.float32))[1])

    print("LIFT_VERIFY_PASS")


if __name__ == "__main__":
    main()
