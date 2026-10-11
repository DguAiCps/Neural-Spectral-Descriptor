#!/usr/bin/env python3
"""C3 chain with the refitted key re-metrization (2026-10-10).

Same as scripts/_pdv2_c3.py (past-only graph, checkpoints/pdv2_causal_s1,
odometry keyframes at inference, selection on the nine val sequences) except
the key block: x_key = w * log1p(d) with w from
scripts/_pdv2_fit_key_logbayes.py (artifacts/key_remetrize_r_pdv2_logbayes.npy)
instead of r * d. The refiner is unchanged (it is trained without the key
weight); the edge classifier is retrained because its features use the
re-metrized key and the pass-1 embedding.

  classifier  artifacts/edge_classifier_pdv2_causal_logbayes.pt
  results     results/pdv2_c3_logbayes/{odom,gt}

Subcommands: clf, val-tuples, select-tuple, val-conf, conf-select, test-dump,
aggregate, align-channels, conf-select-align, aggregate-align (as in
scripts/_pdv2_c3.py).
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

VARIANT = "pdv2_causal_logbayes"
CKPT_DIR = "pdv2_causal_s{seed}"
KEY_TRANSFORM = "log1p"
c3.OUT_ROOT = REPO / "results/pdv2_c3_logbayes"
c3.RVEC = REPO / "artifacts/key_remetrize_r_pdv2_logbayes.npy"
c3.CLS = REPO / "artifacts/edge_classifier_pdv2_causal_logbayes.pt"
c3.VARIANT = VARIANT


def _register_variant():
    import _causal_branch_fusion as cbf
    cbf.VARIANTS[VARIANT] = dict(ckpt_dir=CKPT_DIR, rvec=c3.RVEC, cls=c3.CLS,
                                 config=c3.CONFIG, key_transform=KEY_TRANSFORM)
    return cbf


def cmd_clf(only, train_only):
    import _pd_build_edge_classifier as pdec
    import _causal_build_edge_classifier as cec
    pdec.PD_CACHE = c3.V2
    train_seqs = {}
    for s, q in c3.TRAIN:
        train_seqs.setdefault(s, []).append(q)
    cec.PD_TRAIN_SEQS = train_seqs
    cec.PD_HOLDOUT = c3.EDGE_HOLDOUT
    cec.VARIANTS[VARIANT] = dict(ckpt_dir=CKPT_DIR, rvec=c3.RVEC, config=c3.CONFIG,
                                 out=c3.CLS, key_transform=KEY_TRANSFORM,
                                 feat_dir=REPO / "results/_edge_cls_pdv2_causal_logbayes")
    sys.argv = [cec.__file__, "--variant", VARIANT, "--seed", str(c3.SEED)] + \
        (["--only", only] if only else []) + (["--train-only"] if train_only else [])
    cec.main()


c3._register_variant = _register_variant
c3.cmd_clf = cmd_clf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
    c3.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "clf":
        cmd_clf(a.only, a.train_only)
    elif a.cmd == "val-tuples":
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
