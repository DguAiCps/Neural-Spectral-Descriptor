#!/usr/bin/env python3
"""C3 chain with the clipped-BEV NSD auxiliary block (fix A, 2026-10-10).

Same as scripts/_pdv2_c3.py except where the spectrum caches come from: val
and test spectra are read from data/preprocessed_v2{_odom}_bevclip/
(scripts/_bevclip_spec.py: sk_re/sk_im from the BEV grid clipped at zero,
raw_height unchanged). Every other cache, the refiner, omega and the
selection rules are those of the C3 run, so for seed 1 NSD-H (hspec12)
reproduces results/pdv2_c3 up to run-to-run GPU noise (a rerun of the
original chain on kitti360/0004 also moves single queries in a few of the 89
tuples; the new hspec12 grid equals that rerun exactly). NCLT test sessions
read the invariant-key caches of data/nclt_fov/{odom,gt}/ (scripts/_nclt_fov_fix.py:
the mirrored elevation window of the upside-down HDL-32E, 2026-10-11); NCLT
is in no validation set, so no selection changes. Each seed is an
independent replicate (all selection refit per seed):

  model       checkpoints/pdv2_causal_s{seed}
  classifier  seed 1: artifacts/edge_classifier_pdv2_causal.pt (the C3 one;
              it does not read the BEV block); seeds 2, 3:
              artifacts/edge_classifier_pdv2_causal_s{seed}.pt (`clf`)
  results     results/pdv2_c3_bevclip_s{seed}/{odom,gt}

Subcommands: clf, val-tuples, select-tuple, val-conf, conf-select,
test-dump, aggregate, align-channels, conf-select-align, aggregate-align
(as in scripts/_pdv2_c3.py).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import _pdv2_c3 as c3  # noqa: E402

SPEC_OF = {"odom": REPO / "data/preprocessed_v2_odom_bevclip",
           "gt": REPO / "data/preprocessed_v2_bevclip"}
NCLT_FOV_OF = {"odom": REPO / "data/nclt_fov/odom", "gt": REPO / "data/nclt_fov/gt"}


def _set_seed(seed):
    c3.SEED = seed
    c3.OUT_ROOT = REPO / f"results/pdv2_c3_bevclip_s{seed}"
    if seed != 1:
        c3.CLS = REPO / f"artifacts/edge_classifier_pdv2_causal_s{seed}.pt"


def _redirect_caches(cond):
    import _causal_val_select as vs
    import _causal_pr_dump as pr
    resolve_val, resolve_test = vs.resolve_val_caches, pr.resolve_caches

    def swap(spec):
        p = SPEC_OF[cond] / spec.name
        assert p.is_file(), p
        return p

    def val_caches(sensor, seq):
        cache, lay, spec = resolve_val(sensor, seq)
        return cache, lay, swap(spec)

    def test_caches(sensor, seq):
        cache, lay, spec, marker = resolve_test(sensor, seq)
        if sensor == "nclt":
            hits = sorted(NCLT_FOV_OF[cond].glob(f"cache_*_nclt_test_{seq}.npz"))
            assert len(hits) == 1, (seq, hits)
            cache = hits[0]
        return cache, lay, swap(spec), marker

    vs.resolve_val_caches = val_caches
    pr.resolve_caches = test_caches


def cmd_clf(only, train_only):
    import _pd_build_edge_classifier as pdec
    import _causal_build_edge_classifier as cec
    pdec.PD_CACHE = c3.V2                 # GT train caches (supervision)
    train_seqs = {}
    for s, q in c3.TRAIN:
        train_seqs.setdefault(s, []).append(q)
    cec.PD_TRAIN_SEQS = train_seqs
    cec.PD_HOLDOUT = c3.EDGE_HOLDOUT
    cec.VARIANTS[c3.VARIANT] = dict(
        ckpt_dir="pdv2_causal_s{seed}", rvec=c3.RVEC, config=c3.CONFIG,
        out=c3.CLS, feat_dir=REPO / f"results/_edge_cls_pdv2_causal_s{c3.SEED}")
    sys.argv = [cec.__file__, "--variant", c3.VARIANT, "--seed", str(c3.SEED)] + \
        (["--only", only] if only else []) + (["--train-only"] if train_only else [])
    cec.main()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("clf")
    s.add_argument("--only")
    s.add_argument("--train-only", action="store_true")
    for name in ("val-tuples", "val-conf"):
        s = sub.add_parser(name)
        s.add_argument("--cond", choices=("odom", "gt"), required=True)
        s.add_argument("--tag", choices=list(c3.VAL_TAGS), required=True)
        s.add_argument("--device", default="cuda")
    for name in ("select-tuple", "conf-select", "aggregate", "align-channels",
                 "conf-select-align", "aggregate-align"):
        s = sub.add_parser(name)
        s.add_argument("--cond", choices=("odom", "gt"), required=True)
    s = sub.add_parser("test-dump")
    s.add_argument("--cond", choices=("odom", "gt"), required=True)
    s.add_argument("--seq", required=True)
    s.add_argument("--device", default="cuda")
    a = p.parse_args()
    _set_seed(a.seed)
    c3.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "clf":
        cmd_clf(a.only, a.train_only)
        return
    _redirect_caches(a.cond)
    if a.cmd == "val-tuples":
        c3.cmd_val_tuples(a.cond, a.tag, a.device)
    elif a.cmd == "select-tuple":
        c3.cmd_select_tuple(a.cond)
    elif a.cmd == "val-conf":
        c3.cmd_val_conf(a.cond, a.tag, a.device)
    elif a.cmd == "conf-select":
        c3.cmd_conf_select(a.cond)
    elif a.cmd == "test-dump":
        c3.cmd_test_dump(a.cond, a.seq, a.device)
    elif a.cmd == "aggregate":
        c3.cmd_aggregate(a.cond)
    elif a.cmd == "align-channels":
        c3.cmd_align_channels(a.cond)
    elif a.cmd == "conf-select-align":
        c3.cmd_conf_select_align(a.cond)
    elif a.cmd == "aggregate-align":
        c3.cmd_aggregate(a.cond, align=True)


if __name__ == "__main__":
    main()
