#!/usr/bin/env python3
"""Split v2 PILOT for NSD / NSD-H, single seed (2026-10-07).

Runs the pd2026 pipeline on split v2 (configs/training_place_disjoint_v2.yaml).
The pd scripts are imported and left untouched; this driver only swaps the
sequence lists and the paths (module globals / VARIANTS entries):

  caches      data/preprocessed_v2   (pd2026 files hard-linked under v2 role
                                      names; train/val/test caches share one
                                      schema and builder, see
                                      scripts/_pd_build_test_caches.py)
  checkpoint  checkpoints/pdv2_s1
  omega       artifacts/key_remetrize_r_pdv2.npy
  classifier  artifacts/edge_classifier_pdv2.pt
  results     results/pdv2/

Subcommands (one heavy sequence per process, like the pd chains):
  prepare                        audit the split, link reusable pd2026 caches
  caches --role val|test --seq S build missing operating cache (test only),
                                 layouts and spec for one sequence
  remetrize                      scripts/_pd_fit_key_remetrize.py on v2 train
  edgecls [--only S|--train-only] scripts/_pd_build_edge_classifier.py
  fusion --tag T                 scripts/_pd_branch_fusion.py, one val sequence
  select                         fusion tuple: sensor families weighted
                                 equally, queries pooled within a family
  apply --seq S                  scripts/_pd_apply_frozen.py worker, one test seq
  aggregate                      group summary -> results/pdv2/summary.json
  roster --role train|val|test   print the v2 sequence list

PILOT: the v2 test split is evaluated before split v2 is pre-registered.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PD = REPO / "data/preprocessed_pd2026"
OP = REPO / "data/preprocessed_cross_sensor_operating"
V2 = REPO / "data/preprocessed_v2"
OUT = REPO / "results/pdv2"
CONFIG = REPO / "configs/training_place_disjoint_v2.yaml"
RVEC = REPO / "artifacts/key_remetrize_r_pdv2.npy"
CLS = REPO / "artifacts/edge_classifier_pdv2.pt"
SEED = 1
OPS = ("pbev128_main", "hspec12")

TRAIN = [("kitti", s) for s in ("01", "02", "05", "06")] + \
        [("helipr", f"Bridge0{i}") for i in (1, 2, 3, 4)]
VAL_TAGS = {
    "K09": ("kitti", "09"),
    "K3604": ("kitti360", "0004"), "K3606": ("kitti360", "0006"),
    "RA1": ("helipr", "Roundabout01"), "RA2": ("helipr", "Roundabout02"),
    "RA3": ("helipr", "Roundabout03"),
    "SJ1": ("mulran", "Sejong01"), "SJ2": ("mulran", "Sejong02"),
    "SJ3": ("mulran", "Sejong03"),
}
VAL = list(VAL_TAGS.values())
MCD = ["kth_day_06", "kth_day_09", "kth_day_10", "kth_night_01", "kth_night_04",
       "kth_night_05", "tuhh_day_02", "tuhh_day_03", "tuhh_day_04",
       "tuhh_night_07", "tuhh_night_08", "tuhh_night_09"]
DAEJEON = ["KAIST", "DCC", "Riverside"]
TEST = [("kitti", s) for s in ("00", "07", "08")] + [("kitti360", "0000")] + \
    [("helipr", f"Town0{i}") for i in (1, 2, 3)] + \
    [("helipr", f"{p}0{i}") for p in DAEJEON for i in (4, 5, 6)] + \
    [("mulran", f"{p}0{i}") for p in DAEJEON for i in (1, 2, 3)] + \
    [("nclt", s) for s in ("2012-01-08", "2012-02-04", "2012-03-17", "2012-05-11",
                           "2012-08-04", "2012-11-04", "2012-11-16", "2013-01-10",
                           "2013-02-23")] + \
    [("mcd", s) for s in MCD]
EDGE_HOLDOUT = {"kitti/06", "helipr/Bridge01"}
FAMILY = {"kitti360": "kitti"}  # sensor family (HDL-64E)

# Grouping is not decided yet (docs/할 일.md); both are reported.
CITY = {
    "karlsruhe": ["kitti/00", "kitti/07", "kitti/08", "kitti360/0000"],
    "seoul": [f"helipr/Town0{i}" for i in (1, 2, 3)],
    "daejeon": [f"helipr/{p}0{i}" for p in DAEJEON for i in (4, 5, 6)]
               + [f"mulran/{p}0{i}" for p in DAEJEON for i in (1, 2, 3)],
    "ann_arbor": [f"nclt/{q}" for k, q in TEST if k == "nclt"],
    "stockholm": [f"mcd/{s}" for s in MCD if s.startswith("kth")],
    "hamburg": [f"mcd/{s}" for s in MCD if s.startswith("tuhh")],
}
DATASET = {
    "kitti_family": CITY["karlsruhe"],
    "helipr_town": CITY["seoul"],
    "helipr_daejeon": [f"helipr/{p}0{i}" for p in DAEJEON for i in (4, 5, 6)],
    "mulran_daejeon": [f"mulran/{p}0{i}" for p in DAEJEON for i in (1, 2, 3)],
    "nclt": CITY["ann_arbor"],
    "mcd": [f"mcd/{s}" for s in MCD],
}


def _name(sensor, seq):
    return f"{sensor}/{seq}"


# ---------------------------------------------------------------- prepare
def _audit_split():
    """50 m / 1% rule between splits, both directions, same-frame pairs."""
    pairs = json.loads((REPO / "results/split_audit/geo_overlap.json").read_text())[
        "pairs_same_frame"]
    role = {_name(*x): r for r, L in (("train", TRAIN), ("val", VAL), ("test", TEST))
            for x in L}
    worst = []
    for key, m in pairs.items():
        a, b = key.split("|")
        if a in role and b in role and role[a] != role[b]:
            frac = max(m["frac_scan_50m"], m["frac_len_50m"])
            worst.append((frac, m["min_dist_m"], a, b))
    worst.sort(key=lambda w: (-w[0], w[1]))
    for frac, dmin, a, b in worst[:3]:
        print(f"  closest cross-split: {a}({role[a]}) -> {b}({role[b]}): "
              f"50 m {100 * frac:.2f}%  min {dmin:.1f} m", flush=True)
    bad = [w for w in worst if w[0] > 0.01]
    assert not bad, f"split v2 violates the 50 m / 1% rule: {bad}"
    print(f"AUDIT OK: {len(worst)} cross-split same-frame pairs, none > 1% "
          "within 50 m", flush=True)


def _keys():
    """v2 cache keys per sensor; must equal the pd2026 keys (same encoder)."""
    import yaml
    from train_multi_dataset import apply_encoder_preset, compute_cache_key
    from _pd_build_test_caches import _effective_config, _family, MCD_ELEVATION
    cfg = apply_encoder_preset(yaml.safe_load(open(CONFIG)), "no_interdiff")
    cfg["encoding"].setdefault("sensor_elevation_ranges", {}).setdefault(
        "mcd", list(MCD_ELEVATION))
    ref = _effective_config()
    keys = {}
    for s in ("kitti", "kitti360", "nclt", "helipr", "mulran", "mcd"):
        k = compute_cache_key(cfg, dataset_type=_family(s))
        assert k == compute_cache_key(ref, dataset_type=_family(s)), s
        keys[s] = k
    return keys


def _pd_cache(sensor, seq):
    hits = [h for m in ("test_", "val_", "")
            for h in glob.glob(str(PD / f"cache_*_{sensor}_{m}{seq}.npz"))]
    assert len(hits) <= 1, hits
    return Path(hits[0]) if hits else None


def _first(names):
    for n in names:
        if (PD / n).exists():
            return PD / n
    return None


def _link(src: Path, dst: Path):
    if dst.exists():
        assert dst.stat().st_size == src.stat().st_size, (src, dst)
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def cmd_prepare():
    _audit_split()
    keys = _keys()
    V2.mkdir(parents=True, exist_ok=True)
    plan = []
    for s, q in TRAIN:
        plan.append((_pd_cache(s, q), f"cache_{keys[s]}_{s}_{q}.npz"))
        # edge-classifier layouts, only where pd2026 also had no operating
        # layout cache (scripts/_pd_build_edge_classifier.py _layouts())
        if not (OP / f"{s}_operating_{q}_layout60_stride1.npz").exists():
            plan.append((_first([f"layouts_{s}_{q}_layout60.npz",
                                 f"testlayouts_{s}_{q}.npz"]),
                         f"layouts_{s}_{q}_layout60.npz"))
    for role, seqs in (("val", VAL), ("test", TEST)):
        for s, q in seqs:
            src = _pd_cache(s, q)
            if src is not None:
                plan.append((src, f"cache_{keys[s]}_{s}_{role}_{q}.npz"))
            for kind in ("layouts", "spec"):
                src = _first([f"test{kind}_{s}_{q}.npz", f"val{kind}_{s}_{q}.npz"])
                if src is not None:
                    plan.append((src, f"{role}{kind}_{s}_{q}.npz"))
    missing_src = [d for s, d in plan if s is None]
    assert not missing_src, missing_src
    counts = {}
    for src, dst in plan:
        how = _link(src, V2 / dst)
        counts[how] = counts.get(how, 0) + 1
    print(f"linked into {V2}: {counts}", flush=True)
    todo = []
    for role, seqs in (("val", VAL), ("test", TEST)):
        for s, q in seqs:
            need = [f"{role}layouts_{s}_{q}.npz", f"{role}spec_{s}_{q}.npz"]
            if role == "test":
                need.append(f"cache_{keys[s]}_{s}_test_{q}.npz")
            if not all((V2 / n).exists() for n in need):
                todo.append(f"{role}:{s}/{q}")
    print("STILL TO BUILD:", " ".join(todo) if todo else "nothing", flush=True)


# ---------------------------------------------------------------- caches
def cmd_caches(role, seq, device):
    import numpy as np
    from _pd_build_test_caches import (
        _effective_config, _make_loader, _build_operating_cache,
        _build_layouts_and_spec)
    sensor, q = seq.split("/")
    assert (sensor, q) in (VAL if role == "val" else TEST), seq
    keys = _keys()
    paths = {
        "operating_cache": V2 / f"cache_{keys[sensor]}_{sensor}_{role}_{q}.npz",
        "testlayouts": V2 / f"{role}layouts_{sensor}_{q}.npz",
        "testspec": V2 / f"{role}spec_{sensor}_{q}.npz",
    }
    if all(p.exists() for p in paths.values()):
        print(f"SKIP {seq}: all three artifacts exist", flush=True)
        return
    loader = _make_loader(sensor, q)
    if not paths["operating_cache"].exists():
        assert role == "test", "val operating caches come from prepare"
        _build_operating_cache(_effective_config(), sensor, q, loader,
                               paths["operating_cache"], device)
    cache = np.load(paths["operating_cache"])
    _build_layouts_and_spec(sensor, q, loader, cache, paths)
    print("PDV2_CACHES_DONE", flush=True)


# ---------------------------------------------------------------- selection
def cmd_remetrize():
    import _pd_fit_key_remetrize as rm
    rm.CACHE = V2                # only unmarked (train) caches are harvested
    rm.OUT = {"pd": RVEC}        # 'pd' only matters for skipping nclt
    sys.argv = [rm.__file__, "--variant", "pd"]
    rm.main()


def _edge_module():
    import _pd_build_edge_classifier as ec
    ec.PD_CACHE = V2
    train_seqs = {}
    for s, q in TRAIN:
        train_seqs.setdefault(s, []).append(q)
    ec.VARIANTS["pdv2"] = dict(
        ckpt=REPO / f"checkpoints/pdv2_s{SEED}/best_model.pth",
        rvec=RVEC, config=CONFIG, out=CLS,
        feat_dir=REPO / "results/_edge_cls_pdv2",
        train_seqs=train_seqs, holdout=EDGE_HOLDOUT)
    return ec


def cmd_edgecls(only, train_only):
    ec = _edge_module()
    sys.argv = [ec.__file__, "--variant", "pdv2"] + \
        (["--only", only] if only else []) + (["--train-only"] if train_only else [])
    ec.main()


def _fusion_module():
    import _pd_branch_fusion as bf
    bf.PD_CACHE = V2
    bf.OUT = OUT
    bf.JOBS_PD.update(VAL_TAGS)
    bf.VARIANTS["pdv2"] = dict(ckpt_dir="pdv2_s{seed}", rvec=RVEC, cls=CLS,
                               config=CONFIG, tags=list(VAL_TAGS))
    return bf


def cmd_fusion(tag):
    bf = _fusion_module()
    sys.argv = [bf.__file__, "--variant", "pdv2", "--seed", str(SEED), "--tags", tag]
    bf.main()


def cmd_select():
    rep = json.loads((OUT / f"branch_fusion_pdv2_seed{SEED}.json").read_text())
    tags = list(VAL_TAGS)
    assert all(t in rep for t in tags), [t for t in tags if t not in rep]
    fam = {t: FAMILY.get(VAL_TAGS[t][0], VAL_TAGS[t][0]) for t in tags}
    groups = {}
    for t in tags:
        groups.setdefault(fam[t], []).append(t)
    out = {}
    for op in OPS:
        V = {t: rep[t]["variants"][op] for t in tags}

        def pooled(w, g):
            return (sum(V[t]["n_queries"] * V[t][w] for t in g)
                    / sum(V[t]["n_queries"] for t in g))

        def family_macro(w):
            return statistics.mean(pooled(w, g) for g in groups.values())

        ws = [k for k in V[tags[0]] if k.startswith("(")]
        w = max(ws, key=family_macro)        # ties -> first in WEIGHTS order
        w_q = max(ws, key=lambda x: pooled(x, tags))
        out[f"pdv2/{op}"] = {
            "selected_tuple": w,
            "rule": "argmax over WEIGHTS of the mean over sensor families "
                    "(kitti incl. kitti360, helipr, mulran) of the "
                    "query-pooled R@1 within each family; seed 1 only",
            "selection_pool": tags,
            "family_macro": family_macro(w),
            "per_family": {f: pooled(w, g) for f, g in groups.items()},
            "query_weighted": pooled(w, tags),
            "sequence_macro": statistics.mean(V[t][w] for t in tags),
            "per_sequence": {t: V[t][w] for t in tags},
            "per_sequence_no_fusion_nsd": {t: V[t]["nsd"] for t in tags},
            "n_queries": {t: V[t]["n_queries"] for t in tags},
            "reference_query_weighted_choice": {
                "tuple": w_q, "query_weighted": pooled(w_q, tags),
                "family_macro": family_macro(w_q)},
        }
        print(f"{op}: tuple {w} family-macro {family_macro(w):.4f} "
              f"(per family " + ", ".join(f"{f} {pooled(w, g):.4f}"
                                          for f, g in groups.items())
              + f"); query-weighted choice {w_q}", flush=True)
    (OUT / "tuple_selection.json").write_text(json.dumps(out, indent=2))
    print("PDV2_SELECT_DONE", flush=True)


# ---------------------------------------------------------------- test
def cmd_apply(seq, device):
    _fusion_module()                     # registers VARIANTS['pdv2']
    import _pd_apply_frozen as af
    af.PD_CACHE = V2
    af.OUT = OUT
    af.TUPLES = OUT / "tuple_selection.json"
    sensor, q = seq.split("/")
    assert (sensor, q) in TEST, seq
    args = argparse.Namespace(variant="pdv2", seed=SEED, device=device,
                              pool_k=100, overwrite=False)
    af.run_worker(args, sensor, q)


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def cmd_aggregate():
    summary = {"_meta": {
        "pilot": "split v2 test evaluated before pre-registration",
        "seed": SEED, "config": str(CONFIG.relative_to(REPO)),
        "n_test_sequences": len(TEST),
        "grouping": "not decided; city and dataset groups both reported "
                    "(sequence mean within a group, groups weighted equally)"}}
    for op in OPS:
        per, nq = {}, {}
        for s, q in TEST:
            d = json.loads((OUT / f"test_apply_pdv2_{op}_{SEED}_{s}_{q}.json").read_text())
            per[_name(s, q)], nq[_name(s, q)] = d["r_at_1"], d["n_queries"]
        city = {c: _mean(per[k] for k in m) for c, m in CITY.items()}
        dset = {g: _mean(per[k] for k in m) for g, m in DATASET.items()}
        ref = {}
        for k in per:
            s, q = k.split("/")
            p = REPO / f"results/pd2026/test_apply_pd_{op}_1_{s}_{q}.json"
            if p.exists():
                ref[k] = json.loads(p.read_text())["r_at_1"]
        summary[op] = {
            "city_groups": city,
            "city_macro6": _mean(city.values()),
            "city_macro5_without_nclt": _mean(v for c, v in city.items() if c != "ann_arbor"),
            "dataset_groups": dset,
            "dataset_macro6": _mean(dset.values()),
            "sequence_mean": _mean(per.values()),
            "query_weighted": sum(per[k] * nq[k] for k in per) / sum(nq.values()),
            "per_sequence": per, "n_queries": nq,
            "reference_pd2026_s1_same_sequences": {
                "n": len(ref),
                "pdv2_mean": _mean(per[k] for k in ref),
                "pd2026_s1_mean": _mean(ref.values()),
                "per_sequence": ref},
        }
        S = summary[op]
        print(f"== {op}: city macro6 {S['city_macro6']:.4f}  without NCLT "
              f"{S['city_macro5_without_nclt']:.4f}  dataset macro6 "
              f"{S['dataset_macro6']:.4f}  seq mean {S['sequence_mean']:.4f}  "
              f"query-weighted {S['query_weighted']:.4f}", flush=True)
        print("   " + "  ".join(f"{c} {v:.3f}" for c, v in city.items()), flush=True)
        r = S["reference_pd2026_s1_same_sequences"]
        print(f"   same {r['n']} sequences: pdv2 {r['pdv2_mean']:.4f} vs "
              f"pd2026 s1 {r['pd2026_s1_mean']:.4f}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("PDV2_AGGREGATE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("caches")
    p.add_argument("--role", choices=("val", "test"), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--device", default="cuda")
    sub.add_parser("remetrize")
    p = sub.add_parser("edgecls")
    p.add_argument("--only", default=None)
    p.add_argument("--train-only", action="store_true")
    p = sub.add_parser("fusion")
    p.add_argument("--tag", choices=list(VAL_TAGS), required=True)
    sub.add_parser("select")
    p = sub.add_parser("apply")
    p.add_argument("--seq", required=True)
    p.add_argument("--device", default="cuda")
    sub.add_parser("aggregate")
    p = sub.add_parser("roster")
    p.add_argument("--role", choices=("train", "val", "test"), required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        cmd_prepare()
    elif a.cmd == "caches":
        cmd_caches(a.role, a.seq, a.device)
    elif a.cmd == "remetrize":
        cmd_remetrize()
    elif a.cmd == "edgecls":
        cmd_edgecls(a.only, a.train_only)
    elif a.cmd == "fusion":
        cmd_fusion(a.tag)
    elif a.cmd == "select":
        cmd_select()
    elif a.cmd == "apply":
        cmd_apply(a.seq, a.device)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "roster":
        L = {"train": TRAIN, "val": VAL, "test": TEST}[a.role]
        print(" ".join(_name(s, q) for s, q in L))


if __name__ == "__main__":
    main()
