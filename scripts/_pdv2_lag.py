#!/usr/bin/env python3
"""Fixed-lag (15 keyframes) NSD / NSD-H on split v2 (2026-10-09).

Same chain as scripts/_pdv2_c3.py (odometry keyframes at inference, GT for
supervision and scoring, selection on the nine val sequences with the
family-equal rule, past-only candidates j <= t - 30), with one change: the
trajectory graph uses +-5 temporal neighbours (bidirectional) instead of
the past 10, so the decision for keyframe k is made at keyframe k + 15
(scripts/_lag_branch_fusion.py). Similarity edges, classifier-selected
edges, omega (artifacts/key_remetrize_r_pdv2.npy), the confidence
candidates and every selection rule are those of the C3 run.

  model       checkpoints/pdv2_lag15_s{seed}  (configs/training_place_disjoint_v2_lag15.yaml)
  classifier  artifacts/edge_classifier_pdv2_lag15.pt
  results     results/pdv2_lag15/{odom,gt}

Subcommands (one heavy sequence per process):
  prefix-check --seq S [--ckpt-dir D] [--ks ...]  truncation test of the lag bound
  clf [--only S | --train-only]   edge classifier on lag pass-1 embeddings (GT train)
  val-tuples / select-tuple / val-conf / conf-select / test-dump / aggregate
  align-channels / conf-select-align / aggregate-align
                                  as in scripts/_pdv2_c3.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import _pdv2_c3 as c3  # noqa: E402
import _lag_branch_fusion as lag  # noqa: E402

VARIANT = "pdv2_lag15"
CKPT_DIR = "pdv2_lag15_s{seed}"
c3.OUT_ROOT = REPO / "results/pdv2_lag15"
c3.CONFIG = REPO / "configs/training_place_disjoint_v2_lag15.yaml"
c3.CLS = REPO / "artifacts/edge_classifier_pdv2_lag15.pt"
c3.VARIANT = VARIANT


def _register_variant():
    import _causal_branch_fusion as cbf
    cbf.VARIANTS[VARIANT] = dict(ckpt_dir=CKPT_DIR, rvec=c3.RVEC, cls=c3.CLS,
                                 config=c3.CONFIG)
    cbf.nsd_embedding_causal = lag.nsd_embedding_lag
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
    cec.causal_pass1_graph = lag.lag_pass1_graph
    cec.VARIANTS[VARIANT] = dict(ckpt_dir=CKPT_DIR, rvec=c3.RVEC, config=c3.CONFIG,
                                 out=c3.CLS,
                                 feat_dir=REPO / "results/_edge_cls_pdv2_lag15")
    sys.argv = [cec.__file__, "--variant", VARIANT, "--seed", str(c3.SEED)] + \
        (["--only", only] if only else []) + (["--train-only"] if train_only else [])
    cec.main()


c3._register_variant = _register_variant
c3.cmd_clf = cmd_clf


def cmd_prefix_check(seq, ckpt_dir, cls, ks, device):
    """Truncation test on one val sequence (GT keyframe cache): node k's
    embedding from frames [0, k+15] must equal the full-sequence one."""
    import _causal_branch_fusion as cbf
    from _pd_apply_frozen import resolve_caches
    import _pd_apply_frozen as af
    sensor, q = c3._split(seq)
    assert (sensor, q) in c3.VAL, "prefix check runs on validation sequences"
    cbf.VARIANTS["lagcheck"] = dict(ckpt_dir=ckpt_dir, rvec=c3.RVEC, cls=Path(cls),
                                    config=c3.CONFIG)
    af.PD_CACHE = c3.V2
    cache_p, lay_p, _, _ = resolve_caches(sensor, q)
    cache, lay = np.load(cache_p), np.load(lay_p)
    assert (lay["scan_ids"] == cache["scan_ids"]).all()
    d288 = cache["descriptors"].astype(np.float32)
    layouts = lay["nsd_layouts"].astype(np.float32)
    cfg, model, clf, blob, r, ckpt = cbf.make_model_causal("lagcheck", c3.SEED, device)
    n = len(d288)
    ks = [k for k in ks if k + lag.LAG + 1 <= n]
    rows = lag.prefix_check(cache, sensor, d288, layouts, cfg, model, clf, blob, r,
                            device, ks)
    for row in rows:
        print(f"k={row['k']:5d}  |full-prefix(k+15)| {row['at_lag']:.3e}  "
              f"rows<=k {row['at_lag_rows_le_k']:.3e}  "
              f"|full-prefix(k+14)| {row['before_lag']:.3e}", flush=True)
    out = REPO / "logs/lag15" / f"prefix_check_{sensor}_{q}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seq": seq, "checkpoint": str(ckpt), "classifier": cls,
                               "device": device, "lag": lag.LAG, "rows": rows},
                              indent=2))
    print(f"wrote {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prefix-check")
    s.add_argument("--seq", required=True)
    s.add_argument("--ckpt-dir", default=CKPT_DIR)
    s.add_argument("--cls", default=str(c3.CLS))
    s.add_argument("--ks", type=int, nargs="+", default=[40, 300, 900])
    s.add_argument("--device", default="cpu")
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
    if a.cmd == "prefix-check":
        cmd_prefix_check(a.seq, a.ckpt_dir, a.cls, a.ks, a.device)
    elif a.cmd == "clf":
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
