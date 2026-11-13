#!/usr/bin/env python3
"""SeqOT paper-recipe retraining on split v2 (2026-11-07, review revision).

Independent wrapper around scripts/_seqot_official_train.py and
scripts/_seqot_official_prep.py: nothing in those files or in their output
folders is changed.  Every output of this wrapper lives in
  WORKP   = _external/seqot_paper_recipe_20261107/      (labels, checkpoints, caches)
  OUT     = results/pdv2_baseline_revision_20261107/seqot/  (records)

Paper recipe (arXiv 2209.07951 §III-C, §IV-A; learned_sources/seqot.txt):
  * overlap label Eq. 13: count / min(valid(R_i), valid(R'_j))  -- prep
    overlap_values(..., 'paper') on the existing projected pairs (ovc_*:
    count, valid_cur, valid_ref), then the official gen_training_index
    normalisation (same bins, same seed 1, same draw order) -> new index;
  * positive: overlap > 0.3, otherwise negative (paper wording; code: < 0.3);
  * Npos = Nneg = 6 in both phases (code phase 1: 3 / 3);
  * phase 1 transformer: Adam lr 5e-6, 20 epochs, StepLR(5, 0.9);
  * phase 2 GeM: "similar configuration" with lr 5e-5 -> 20 epochs
    (interpretation; the number is not repeated for GeM in the paper);
  * epochs 1..20: the official loop is range(start + 1, bound), so bound 21.
Everything else (model, loss, margin 0.5, windows, label pair set, the
empty-bin guard, selection rule, evaluation) is the main-table pipeline.

Seeds: env SEED (1, 2, 3) seeds torch / numpy / random before model
creation (_seqot_official_train.seed_all); labels are shared (prep seed 1).

Subcommands (inside ring_baseline, CUDA_VISIBLE_DEVICES=0):
  labels                Eq. 13 index for the 8 TRAIN sequences (+ code reproduction check)
  plan-audit            training-sample audit of both phases' anchor plans (no training)
  probe [--steps N]     warm-up probe in WORKP/probe_s{SEED}: loss / grad / lr / time
  train1 | select1 [--follow] | subdesc | train2 | select2
                        the official phases with the paper recipe, per seed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
sys.dont_write_bytecode = True
os.environ.setdefault("SEQOT_EVAL_WINDOW", "centered")
assert os.environ["SEQOT_EVAL_WINDOW"] == "centered", "main table uses the official centred windows"

import _seqot_official_prep as P  # noqa: E402
import _seqot_official_train as S  # noqa: E402

TAG = "seqot_paper_recipe_20261107"
WORKP = REPO / "_external" / TAG
LABELS_P = WORKP / "labels_eq13"
OUT = REPO / "results/pdv2_baseline_revision_20261107/seqot"
SEED = S.TRAIN_SEED
SEED_DIR = WORKP / f"s{SEED}"

RECIPE = {
    "overlap_formula": "paper",          # Eq. 13
    "negative_rule": "le",               # not (ov > 0.3) -> negative
    "use1": (6, 6), "use2": (6, 6),
    "lr1": 5e-6, "lr2": 5e-5,
    "epochs1": 20, "epochs2": 20,
    "loop_bound1": 21, "loop_bound2": 21,  # range(start + 1, bound) -> 1..20
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


# ------------------------------------------------------------ labels
def label_path(sensor, seq):
    return LABELS_P / P.seq_name(sensor, seq) / "normalized_data.npy"


def cmd_labels():
    """Eq. 13 index per TRAIN sequence; before that the same normalisation
    with the code formula must reproduce the existing index byte for byte
    (so only the denominator differs)."""
    rep_all = {}
    for sensor, seq in P.TRAIN:
        key = f"{sensor}/{seq}"
        d = P.label_dir(sensor, seq)
        out = label_path(sensor, seq)
        with np.load(d / "frames.npz") as z:
            M = len(z["raw_ids"])
        old_meta = json.loads((d / "labels_meta.json").read_text())
        assert old_meta["scale"] == "official" and old_meta["overlap_formula"] == "code", old_meta
        if out.exists():
            rep_all[key] = json.loads((out.parent / "labels_meta.json").read_text())
            print(f"SKIP labels {key}: exists", flush=True)
            continue
        t0 = time.time()
        ic, jc, oc = P.load_explicit(sensor, seq, "code")
        arr_c, _ = P.normalize(M, ic, jc, oc, P.SEED, 1.0)
        old = np.load(d / "normalized_data.npy")
        code_repro = bool(arr_c.shape == old.shape and np.array_equal(arr_c, old))
        assert code_repro, f"{key}: code-formula normalisation does not reproduce the existing index"
        i, j, o = P.load_explicit(sensor, seq, "paper")
        assert np.array_equal(i, ic) and np.array_equal(j, jc)
        assert np.all(j >= i) and np.all(np.isfinite(o))
        assert np.all(o >= oc - 1e-12), "Eq. 13 must be >= the code value (min denominator)"
        arr, rep = P.normalize(M, i, j, o, P.SEED, 1.0)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out.with_suffix(".tmp.npy"), arr)
        os.replace(out.with_suffix(".tmp.npy"), out)
        ov = arr[:, 2].astype(float)
        f1 = np.array([int(x) for x in arr[:, 0]])
        f2 = np.array([int(x) for x in arr[:, 1]])
        ovo = old[:, 2].astype(float)
        rep.update({
            "sequence": key, "overlap_formula": "paper", "scale": "official",
            "code_formula_reproduces_existing_index": code_repro,
            "existing_index_sha256": sha256(d / "normalized_data.npy"),
            "ovc_sha256": {f.name: sha256(f) for f in sorted(d.glob("ovc_*of*.npz"))},
            "explicit_pairs": int(len(i)),
            "explicit_pairs_changed_bin_vs_code": int(np.sum(np.digitize(o, [.1, .3, .5, .7]) != np.digitize(oc, [.1, .3, .5, .7]))),
            "explicit_pairs_gt03": {"paper": int(np.sum(o > .3)), "code": int(np.sum(oc > .3))},
            "rows": {"paper": {"positive_gt03": int(np.sum(ov > .3)), "exact_03": int(np.sum(ov == .3)),
                               "self_rows": int(np.sum(f1 == f2)), "n": int(len(ov))},
                     "code": {"positive_gt03": int(np.sum(ovo > .3)), "exact_03": int(np.sum(ovo == .3)),
                              "self_rows": int(np.sum(old[:, 0] == old[:, 1])), "n": int(len(ovo))}},
            "empty_bin_guard": {"paper_skipped": rep["skipped_empty_bins"],
                                "code_skipped": old_meta.get("skipped_empty_bins")},
            "seconds": round(time.time() - t0, 1)})
        write_json(out.parent / "labels_meta.json", rep)
        rep_all[key] = rep
        print(f"OK labels {key}: {rep['n_rows']} rows, skipped {rep['skipped_empty_bins']}, "
              f"pos rows paper {rep['rows']['paper']['positive_gt03']} code {rep['rows']['code']['positive_gt03']}, "
              f"{rep['seconds']} s", flush=True)
    write_json(OUT / "labels_eq13.json", {"labels_dir": str(LABELS_P.relative_to(REPO)),
                                          "sequences": rep_all,
                                          "index_sha256": {k: sha256(label_path(*k.split("/"))) for k in rep_all}})


# ------------------------------------------------------------ patch the official driver
def _load_train_indices():
    return [S.parse_index(np.load(label_path(s, q))) for s, q in S.TRAIN]


def _label_meta():
    out = {}
    for s, q in S.TRAIN:
        m = json.loads((label_path(s, q).parent / "labels_meta.json").read_text())
        out[P.seq_name(s, q)] = {k: m.get(k) for k in ("overlap_formula", "scale", "n_rows")}
    return out


def _recipe_meta(phase):
    return {"recipe": "paper (arXiv 2209.07951 §III-C, §IV-A), review revision 2026-11-07",
            "protocol": str((OUT / "protocol.json").relative_to(REPO)),
            "lr": RECIPE["lr1"] if phase == 1 else RECIPE["lr2"],
            "use_pos_neg": RECIPE["use1"] if phase == 1 else RECIPE["use2"],
            "overlap": "Eq. 13 min(valid) denominator", "negative_rule": "le (not > 0.3)"}


_ORIG_ANCHOR_PLAN = S.anchor_plan


def _anchor_plan(indices, window_ok, use_pos, use_neg, thresh=S.OVERLAP_THRESH, neg_rule=None):
    return _ORIG_ANCHOR_PLAN(indices, window_ok, use_pos, use_neg, thresh, RECIPE["negative_rule"])


def _stage2_home():
    return S.MODE_WORK, "own (paper recipe, never shared)"


def patch():
    S.SEED_WORK = SEED_DIR
    S.MODE_WORK = SEED_DIR / "centered"
    S.MODE_RESULTS = OUT / f"s{SEED}"
    S.RUNS = SEED_DIR / "runs"
    S.SUBDESC = S.MODE_WORK / "subdesc"
    S.SELECTION = S.MODE_RESULTS / "train_selection.json"
    S.LR1, S.LR2 = RECIPE["lr1"], RECIPE["lr2"]
    S.USE1, S.USE2 = RECIPE["use1"], RECIPE["use2"]
    S.NEG_RULE = RECIPE["negative_rule"]
    S.load_train_indices = _load_train_indices
    S._label_meta = _label_meta
    S._recipe_meta = _recipe_meta
    S.anchor_plan = _anchor_plan
    S.stage2_home = _stage2_home
    assert S.EVAL_WINDOW == "centered" and S.METRIC == "l2" and S.TRAIN_WINDOW == "centered"
    for s, q in S.TRAIN:
        assert label_path(s, q).exists(), f"Eq. 13 labels missing: {s}/{q} (run labels)"


# ------------------------------------------------------------ audits
def _plan_stats(plan, idx):
    """Per-step label facts of a plan (d, a, pos, neg)."""
    lut = []
    for f1, f2, ov in idx:
        lut.append({(int(a), int(b)): float(o) for a, b, o in zip(f1, f2, ov)})
    npos = np.array([len(p) for _, _, p, _ in plan])
    nneg = np.array([len(n) for _, _, _, n in plan])
    self_only = np.array([all(x == a for x in p) for _, a, p, _ in plan])
    any_self = np.array([any(x == a for x in p) for _, a, p, _ in plan])
    pos_ov = np.array([lut[d][(a, x)] for d, a, p, _ in plan for x in p])
    neg_ov = np.array([lut[d][(a, x)] for d, a, _, n in plan for x in n])
    q = lambda v: [float(x) for x in np.quantile(v, [0, .1, .5, .9, 1])] if len(v) else []
    return {"steps": int(len(plan)),
            "steps_per_sequence": S._plan_meta(plan, len(S.TRAIN)),
            "npos_hist": {int(k): int(v) for k, v in zip(*np.unique(npos, return_counts=True))},
            "nneg_hist": {int(k): int(v) for k, v in zip(*np.unique(nneg, return_counts=True))},
            "self_only_positive_steps": int(self_only.sum()),
            "self_only_fraction": float(self_only.mean()) if len(plan) else None,
            "steps_with_self_positive": int(any_self.sum()),
            "positive_overlap_quantiles": q(pos_ov), "negative_overlap_quantiles": q(neg_ov),
            "negatives_with_overlap_eq_03": int(np.sum(neg_ov == .3)),
            "max_negative_overlap": float(neg_ov.max()) if len(neg_ov) else None,
            "min_positive_overlap": float(pos_ov.min()) if len(pos_ov) else None}


def cmd_plan_audit():
    """Stage-1 and stage-2 anchor plans of the paper recipe vs the code recipe
    (same windows / existence rules); proves 6 / 6, Eq. 13 and the negative
    rule reach the training samples.  Stage-2 existence uses n_raw + 1 rows
    (gen_sub_descriptors), independent of the phase-1 weights."""
    sources = [S.StoreSource("train", s, q) for s, q in S.TRAIN]
    paper_idx = _load_train_indices()
    code_idx = [S.parse_index(np.load(P.label_dir(s, q) / "normalized_data.npy")) for s, q in S.TRAIN]
    ok1 = lambda d, a: all(sources[d].exists(int(x)) for x in S.window(a, S.SEQL, "centered"))
    nrow = [src.n_raw + 1 for src in sources]
    ok2 = lambda d, a: all(0 <= int(x) < nrow[d] for x in S.window(a, S.GEM_SEQLEN, "centered"))
    rep = {}
    for name, idx, use1, use2, rule in (("paper", paper_idx, RECIPE["use1"], RECIPE["use2"], "le"),
                                         ("code", code_idx, (3, 3), (6, 6), "lt")):
        p1 = _ORIG_ANCHOR_PLAN(idx, ok1, *use1, S.OVERLAP_THRESH, rule)
        p2 = _ORIG_ANCHOR_PLAN(idx, ok2, *use2, S.OVERLAP_THRESH, rule)
        rep[name] = {"stage1": _plan_stats(p1, idx), "stage2": _plan_stats(p2, idx),
                     "use1": use1, "use2": use2, "negative_rule": rule}
    # negative rule effect on the paper labels
    alt = _ORIG_ANCHOR_PLAN(paper_idx, ok1, *RECIPE["use1"], S.OVERLAP_THRESH, "lt")
    rep["paper_labels_with_code_negative_rule_stage1_steps"] = len(alt)
    rep["label_rows_eq_03"] = {"paper": int(sum(np.sum(ov == .3) for _, _, ov in paper_idx)),
                               "code": int(sum(np.sum(ov == .3) for _, _, ov in code_idx))}
    write_json(OUT / "plan_audit.json", rep)
    print(json.dumps({k: {s: {x: v[s][x] for x in ("steps", "npos_hist", "self_only_fraction")}
                          for s in ("stage1", "stage2")} for k, v in rep.items() if isinstance(v, dict)
                      and "stage1" in v}), flush=True)


def cmd_probe(steps):
    """Warm-up on the real stage-1 plan from the seeded initial weights, in a
    separate folder (the run starts again from the same seed)."""
    import torch
    patch()
    S.seed_all()
    dev = S.device()
    O = S.official()
    sources = [S.StoreSource("train", s, q) for s, q in S.TRAIN]
    plan = S.anchor_plan(S.load_train_indices(),
                         lambda d, a: all(sources[d].exists(int(x)) for x in S.window(a, S.SEQL, "centered")),
                         *S.USE1)
    model = O["featureExtracter"](seqL=S.SEQL).to(dev)
    init_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                        for v in model.state_dict().values())).hexdigest()
    opt, sch = S._lr_sched(model, S.LR1)
    recs = []
    t_all = time.time()
    for k, (d, a, pos, neg) in enumerate(plan[:steps]):
        t0 = time.time()
        arr = S.stage1_batch(sources[d], a, pos, neg, "centered")
        x = torch.from_numpy(arr).to(dev)
        x.requires_grad_(True)
        model.train()
        opt.zero_grad()
        g = model(x)
        o1, o2, o3 = torch.split(g, [1, len(pos), len(neg)], dim=0)
        loss = O["loss"].triplet_loss(o1, o2, o3, S.MARGIN_1, lazy=False)
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)))
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        recs.append({"step": k, "loss": float(loss), "grad_norm": gn, "lr": opt.param_groups[0]["lr"],
                     "batch": int(arr.shape[0]), "npos": len(pos), "nneg": len(neg),
                     "self_only": all(p == a for p in pos), "s": round(time.time() - t0, 4)})
    sec = [r["s"] for r in recs[5:]] or [r["s"] for r in recs]
    rep = {"seed": SEED, "steps": len(recs), "plan_steps_per_epoch": len(plan), "initial_weights_sha256": init_hash,
           "loss_finite": bool(all(np.isfinite(r["loss"]) for r in recs)),
           "grad_finite": bool(all(np.isfinite(r["grad_norm"]) for r in recs)),
           "nonzero_grad_steps": int(sum(r["grad_norm"] > 0 for r in recs)),
           "lr_actual": sorted({r["lr"] for r in recs}), "mean_s_per_step_after5": float(np.mean(sec)),
           "est_epoch_s": float(np.mean(sec) * len(plan)), "total_s": round(time.time() - t_all, 1),
           "device": str(dev), "torch": torch.__version__,
           "cuda_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2) if dev.type == "cuda" else None,
           "records": recs}
    write_json(OUT / f"probe_s{SEED}.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "records"}), flush=True)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels")
    sub.add_parser("plan-audit")
    p = sub.add_parser("probe")
    p.add_argument("--steps", type=int, default=200)
    sub.add_parser("train1")
    p = sub.add_parser("select1")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("subdesc")
    p.add_argument("--batch", type=int, default=1)
    sub.add_parser("train2")
    p = sub.add_parser("select2")
    p.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "labels":
        return cmd_labels()
    if a.cmd == "plan-audit":
        return cmd_plan_audit()
    if a.cmd == "probe":
        return cmd_probe(a.steps)
    patch()
    if a.cmd == "train1":
        S.cmd_train1(RECIPE["loop_bound1"], RECIPE["lr1"], 0)
    elif a.cmd == "select1":
        S.cmd_select1(a.follow, a.batch)
    elif a.cmd == "subdesc":
        S.cmd_subdesc(a.batch)
    elif a.cmd == "train2":
        S.cmd_train2(RECIPE["loop_bound2"], RECIPE["lr2"])
    elif a.cmd == "select2":
        S.cmd_select2(a.batch)


if __name__ == "__main__":
    main()
