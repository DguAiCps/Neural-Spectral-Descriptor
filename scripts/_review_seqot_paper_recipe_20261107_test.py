#!/usr/bin/env python3
"""SeqOT paper-recipe revision (2026-11-07) -- test stage on the frozen
selections.  Nothing here chooses anything: the checkpoints are the VAL
choices of results/pdv2_baseline_revision_20261107/seqot/s{SEED}/train_selection.json.

Subcommands (inside ring_baseline, CUDA_VISIBLE_DEVICES=0):
  reproduce-old [--seqs ...]   input check: the main-table seed-1 SeqOT model
                               (results/pdv2_seqot_official_centered/train_selection.json)
                               on the new test store must give the existing
                               main-table ledgers (dump_seqotl2) of those sequences
  dump                         env SEED: 46 ledgers of the paper-recipe model
                               -> results/pdv2_baseline_revision_20261107/seqot/s{SEED}/odom
  aggregate                    env SEED: _pdv2_c3 PR summary of that folder (as the main table)
  final                        3-seed common-query aggregation (_pdv2_main_baselines_official.aggregate),
                               summary.json, verification.json, ledger / checkpoint hashes
Ledger writers, windows and scoring are those of scripts/_seqot_official_dump.py
(official centred windows, past candidates j <= q - 30, official L2 reported).
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

import _review_seqot_paper_recipe_20261107 as R  # noqa: E402  (imports the official driver S)
import _seqot_official_dump as D  # noqa: E402

S, P = R.S, R.P
TEST_STORE = R.WORKP / "test_inputs"
OUT = R.OUT
OLD_SEL = REPO / "results/pdv2_seqot_official_centered/train_selection.json"
OLD_ODOM = REPO / "results/pdv2_seqot_official_centered/odom"
REPRO_SEQS = ["kitti/08", "kitti360/0000", "helipr/Town01", "mulran/DCC01", "nclt/2012-01-08", "mcd/kth_day_06"]
# The server checkout (e153c26) predates aggregate() and lacks the common-query references and the
# main-table seed summary; byte copies of the local HEAD (635594f) files are pinned here (MANIFEST.json)
# instead of touching the shared scripts/ and results/ that other workers import.
FINAL_INPUTS = R.WORKP / "final_inputs"


def sha256(p):
    return R.sha256(Path(p))


class TestSource:
    """StoreSource interface over the new test store (rows = centred need)."""

    def __init__(self, sensor, seq):
        d = TEST_STORE / P.seq_name(sensor, seq)
        assert (d / "done.json").exists(), f"test store incomplete: {d}"
        with np.load(d / "index.npz") as z:
            self.raw_ids, self.n_raw = z["raw_ids"], int(z["n_raw"])
            self.kf, self.kf_poses = z["kf_scan_ids"], z["poses_gt"]
        self.mm = np.load(d / "range.npy", mmap_mode="r")
        assert len(self.mm) == len(self.raw_ids)
        self.pos = np.full(self.n_raw, -1, dtype=np.int64)
        self.pos[self.raw_ids] = np.arange(len(self.raw_ids))
        self.done = json.loads((d / "done.json").read_text())

    def exists(self, r):
        return 0 <= r < self.n_raw

    def rows(self, raws):
        k = self.pos[np.asarray(raws, dtype=np.int64)]
        assert np.all(k >= 0), "a window reads a raw scan outside the stored centred rows"
        return self.mm[k]


def encode(fe, gem, dev, sensor, seq):
    src = TestSource(sensor, seq)
    cp = P.cache_path("test_odom", sensor, seq)
    with np.load(cp) as z:
        assert np.array_equal(z["scan_ids"], src.kf) and np.array_equal(z["poses_gt"], src.kf_poses)
    t0 = time.time()
    final, _, need = S.final_descriptors(fe, gem, src, src.kf, dev, 64, mode="centered")
    return src, cp, final, need, time.time() - t0


def cmd_reproduce_old(seqs):
    """Uses the existing seed-1 choice only to check the inputs (no selection)."""
    import torch
    dev = S.device()
    sel = json.loads(OLD_SEL.read_text())
    paths = []
    for st in ("stage1", "stage2"):
        c = sel[st]["chosen"]
        p = REPO / c["checkpoint"]
        assert sha256(p) == c["sha256"], p
        paths.append(p)
    fe, gem = S._load_fe(paths[0], dev), S._load_gem(paths[1], dev)
    rep = {"selection": str(OLD_SEL.relative_to(REPO)), "checkpoints": [str(p.relative_to(REPO)) for p in paths],
           "sequences": {}}
    for key in seqs:
        sensor, seq = key.split("/")
        src, cp, final, need, t = encode(fe, gem, dev, sensor, seq)
        _, a_l2 = D.ledgers(final, src.kf_poses[:, :3, 3])
        with np.load(OLD_ODOM / f"dump_seqotl2_{sensor}_{seq}.npz") as z:
            r = {k: bool(np.array_equal(a_l2[k], z[k])) for k in ("query_idx", "j_star", "min_past_gt_dist",
                                                                   "n_candidates", "top1_geo_dist")}
            r["j_star_agree_fraction"] = float(np.mean(a_l2["j_star"] == z["j_star"]))
            r["score_l2_max_abs_diff"] = float(np.max(np.abs(a_l2["score_l2"] - z["score_l2"])))
            r["old_meta_raw_scans_read"] = json.loads(str(z["meta"]))["caches"]["raw_scans_read"]
        r["raw_scans_read"] = int(len(need))
        r["encode_s"] = round(t, 1)
        rep["sequences"][key] = r
        print(key, r, flush=True)
    rep["passed"] = all(v["query_idx"] and v["min_past_gt_dist"] and v["n_candidates"]
                        and v["j_star_agree_fraction"] >= 0.999 and v["score_l2_max_abs_diff"] < 1e-3
                        and v["raw_scans_read"] == v["old_meta_raw_scans_read"]
                        for v in rep["sequences"].values())
    rep["device"], rep["torch"] = str(dev), torch.__version__
    R.write_json(OUT / "test_input_reproduction.json", rep)
    print("REPRO_PASSED" if rep["passed"] else "REPRO_FAILED", flush=True)
    if not rep["passed"]:
        sys.exit(1)


def seed_odom():
    return OUT / f"s{R.SEED}" / "odom"


def _patch_dump():
    R.patch()
    D.SELECTION = S.SELECTION
    D.OUT = seed_odom()
    D.DESC = S.MODE_WORK / "desc" / "test_odom"
    orig = D._meta

    def meta(*a, **k):
        m = orig(*a, **k)
        m["variant"] = "SeqOT paper recipe (arXiv 2209.07951) retrained from scratch on split-v2 train"
        m["training"] = ("paper recipe: Eq.13 overlap labels, >0.3 positive otherwise negative, Npos=Nneg=6, "
                         "phase 1 Adam lr 5e-6 x 20 epochs, phase 2 GeM lr 5e-5 x 20 epochs (interpreted), "
                         "StepLR(5,0.9); epochs chosen on v2 VAL (main-table rule), "
                         f"{S.SELECTION.relative_to(REPO)}; protocol {(OUT / 'protocol.json').relative_to(REPO)}")
        return m
    D._meta = meta


def cmd_dump():
    import torch
    _patch_dump()
    dev = S.device()
    p1, c1 = S._chosen("stage1")
    p2, c2 = S._chosen("stage2")
    fe, gem = S._load_fe(p1, dev), S._load_gem(p2, dev)
    D.DESC.mkdir(parents=True, exist_ok=True)
    for sensor, seq in P.TEST:
        if all((D.OUT / f"dump_{m}_{sensor}_{seq}.npz").exists() for m in D.METHODS):
            print(f"[skip] {sensor}/{seq}", flush=True)
            continue
        src, cp, final, need, t = encode(fe, gem, dev, sensor, seq)
        np.savez(D.DESC / f"{P.seq_name(sensor, seq)}.npz", final=final, kf_scan_ids=src.kf,
                 stage1_sha256=c1["sha256"], stage2_sha256=c2["sha256"])
        extra = {"artifacts": {"stage1": c1, "stage2": c2, "selection": str(S.SELECTION.relative_to(REPO))},
                 "caches": {"operating_cache": str(cp.relative_to(REPO)), "operating_cache_sha256": sha256(cp),
                            "positions": "poses_gt (scoring only)", "raw_scans_read": int(len(need)),
                            "n_raw": int(src.n_raw), "range_store": str((TEST_STORE / P.seq_name(sensor, seq)).relative_to(REPO)),
                            "range_sha256": src.done["range_sha256"]},
                 "timing": {"encode_s": round(t, 1), "encode_ms_per_kf": round(1e3 * t / max(len(final), 1), 2)},
                 "device": str(dev), "torch": torch.__version__}
        D.write_dump(sensor, seq, final, src.kf_poses[:, :3, 3], D.OUT, extra)
    print(f"DUMP_DONE seed {R.SEED}", flush=True)


def cmd_aggregate():
    import _pdv2_c3 as c3
    _patch_dump()
    c3.OUT_ROOT = OUT / f"s{R.SEED}"
    c3.BASELINES = dict(D.METHODS)
    c3.cmd_aggregate_baselines("odom")
    p = seed_odom() / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"]["seqot"] = ("SeqOT paper recipe retrained on split-v2 train; epochs chosen on v2 VAL by L2 R@1 "
                           f"(main-table rule); REPORTED = seqotl2; {S.SELECTION.relative_to(REPO)}")
    s["_meta"]["reported"] = D.REPORTED
    s["_meta"]["train_seed"] = R.SEED
    s["_meta"]["eval_window"] = "centered: official windows (main table; decision at keyframe q + 15)"
    p.write_text(json.dumps(s, indent=2))
    print(f"AGG_DONE seed {R.SEED}", flush=True)


def _training_record(seed):
    w = R.WORKP / f"s{seed}"
    rec = {}
    for st, d in (("stage1", w / "runs/stage1"), ("stage2", w / "centered/runs/stage2")):
        log = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()]
        rec[st] = {"epochs": len(log), "finished": (d / "finished").read_text().strip(),
                   "train_s": round(sum(x["seconds"] for x in log), 1),
                   "steps_per_epoch": sorted({x["steps"] for x in log}),
                   "loss_first_last": [log[0]["loss"], log[-1]["loss"]],
                   "loss_finite": bool(all(np.isfinite(x["loss"]) for x in log)),
                   "lr_next_last": log[-1]["lr_next"], "meta": json.loads((d / "meta.json").read_text())}
    sel = json.loads((OUT / f"s{seed}/train_selection.json").read_text())
    for st in ("stage1", "stage2"):
        e = sel[st]["epochs"]
        rec[st]["selected_epoch"] = sel[st]["chosen"]["epoch"]
        rec[st]["checkpoint"] = sel[st]["chosen"]["checkpoint"]
        rec[st]["checkpoint_sha256"] = sel[st]["chosen"]["sha256"]
        rec[st]["val_family_r1_l2"] = {k: e[k]["family_r1_l2"] for k in sorted(e, key=int)}
        rec[st]["n_scored_epochs"] = len(e)
        rec[st]["val_scoring_s"] = round(sum(v.get("val_s", 0) or 0 for v in e.values()), 1)
    rec["selection_sha256"] = sha256(OUT / f"s{seed}/train_selection.json")
    return rec


def _pinned_aggregator():
    import importlib.util
    man = json.loads((FINAL_INPUTS / "MANIFEST.json").read_text())
    for rel, h in man["sha256"].items():
        assert sha256(FINAL_INPUTS / rel) == h, rel
    spec = importlib.util.spec_from_file_location("_pdv2_main_baselines_official",
                                                  FINAL_INPUTS / "scripts/_pdv2_main_baselines_official.py")
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    M.REPO, M.COMMON = REPO, FINAL_INPUTS / "pdv2_symmetric3_20261013"   # module paths are repo-relative
    return M


def cmd_final():
    from _pr_eval_lib import MIN_INDEX_GAP
    M = _pinned_aggregator()
    sources = {}

    def track(p):
        p = Path(p)
        sources[str(p.relative_to(REPO))] = sha256(p)
        return p

    for f in ("scripts/_review_seqot_paper_recipe_20261107.py", "scripts/_review_seqot_paper_recipe_20261107_test.py",
              "scripts/_review_seqot_paper_recipe_20261107_testinputs.py",
              "scripts/_review_seqot_paper_recipe_20261107_chain.sh", "scripts/_seqot_official_train.py",
              "scripts/_seqot_official_prep.py", "scripts/_seqot_official_dump.py",
              "scripts/_pdv2_c3.py", "scripts/_pr_eval_lib.py", "scripts/_pdv2_pilot.py"):
        track(REPO / f)
    track(FINAL_INPUTS / "scripts/_pdv2_main_baselines_official.py")
    track(FINAL_INPUTS / "MANIFEST.json")
    per_seed, ledgers, checks = {}, {}, {}
    for seed in (1, 2, 3):
        folder = OUT / f"s{seed}" / "odom"
        e = M.aggregate(str(folder.relative_to(REPO)), "dump_seqotl2", "score_l2",
                        "pr_summary_baselines.json", "seqotl2", track)
        assert e is not None, f"seed {seed}: ledgers missing"
        per_seed[str(seed)] = {"source": e["source"], "macro": e["macro"], "city_groups": e["city_groups"],
                               "n_queries": e["n_queries"], "n_positive": e["n_positive"]}
        lst, c = {}, {"ledgers": 0, "causal": True, "checkpoint_match": True, "finite": True}
        sel = json.loads((OUT / f"s{seed}/train_selection.json").read_text())
        for sensor, seq in P.TEST:
            for m in D.METHODS:
                p = folder / f"dump_{m}_{sensor}_{seq}.npz"
                lst[p.name] = sha256(p)
                with np.load(p) as z:
                    meta = json.loads(str(z["meta"]))
                    c["ledgers"] += 1
                    c["causal"] &= bool(np.all(z["j_star"] <= z["query_idx"] - MIN_INDEX_GAP))
                    c["finite"] &= bool(np.all(np.isfinite(z[D.METHODS[m]])))
                    for st in ("stage1", "stage2"):
                        c["checkpoint_match"] &= meta["artifacts"][st]["sha256"] == sel[st]["chosen"]["sha256"]
        for st in ("stage1", "stage2"):
            p = REPO / sel[st]["chosen"]["checkpoint"]
            c[f"{st}_checkpoint_sha256_now"] = sha256(p) == sel[st]["chosen"]["sha256"]
        ledgers[str(seed)] = lst
        checks[str(seed)] = c
    vals = np.array([[per_seed[s]["macro"][k] for k in M.METRICS] for s in ("1", "2", "3")])
    stats = {k: {"mean": float(vals[:, i].mean()), "std": float(vals[:, i].std(ddof=1)),
                 "values": vals[:, i].tolist()} for i, k in enumerate(M.METRICS)}
    old = json.loads(track(FINAL_INPUTS / "pdv2_baseline_seeds/summary.json").read_text())["methods"]["seqot"]
    delta = {k: stats[k]["mean"] - old["mean_std"][k]["mean"] for k in M.METRICS}
    training = {str(s): _training_record(s) for s in (1, 2, 3)}
    summary = {"method": "seqot (paper recipe, review revision 2026-11-07)",
               "protocol": str((OUT / "protocol.json").relative_to(REPO)),
               "protocol_sha256": sha256(OUT / "protocol.json"),
               "aggregate": "per seed 45 seq (KITTI07 excluded) -> 6 cities macro; seed mean, sample SD (ddof=1); "
                            "_pdv2_main_baselines_official.aggregate on the 195,726 common queries",
               "reported_ledger": "dump_seqotl2 (official L2)",
               "per_seed": per_seed, "mean_std": stats,
               "current_main_table_seqot": {"mean_std": old["mean_std"], "seeds": old["seeds"]},
               "paper_minus_current_main": delta,
               "selected_epochs": {s: {st: training[s][st]["selected_epoch"] for st in ("stage1", "stage2")}
                                   for s in training},
               "training": training}
    R.write_json(OUT / "summary.json", summary)
    repro = json.loads((OUT / "test_input_reproduction.json").read_text())
    ver = {"passed": bool(all(c["ledgers"] == 92 and c["causal"] and c["finite"] and c["checkpoint_match"]
                              and c["stage1_checkpoint_sha256_now"] and c["stage2_checkpoint_sha256_now"]
                              for c in checks.values()) and repro["passed"]),
           "per_seed_checks": checks, "test_input_reproduction": repro,
           "common_queries": {s: per_seed[s]["n_queries"] for s in per_seed},
           "aggregate_asserts": "M.aggregate asserts 195726 queries, 61682 positives, KITTI07 excluded, "
                                "self-summary reproduction < 1e-12, j* <= q-30",
           "ledger_sha256": ledgers, "input_sha256": sources,
           "test_used_for_selection": False,
           "selection_files_written_before_test": {
               str(s): os.path.getmtime(OUT / f"s{s}/train_selection.json") <
               min(os.path.getmtime(OUT / f"s{s}/odom/{n}") for n in ledgers[str(s)]) for s in (1, 2, 3)}}
    ver["passed"] &= all(ver["selection_files_written_before_test"].values())
    R.write_json(OUT / "verification.json", ver)
    print(json.dumps({"mean": {k: round(v["mean"], 4) for k, v in stats.items()},
                      "std": {k: round(v["std"], 4) for k, v in stats.items()}, "passed": ver["passed"]}), flush=True)
    print("FINAL_PASSED" if ver["passed"] else "FINAL_FAILED", flush=True)
    if not ver["passed"]:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reproduce-old")
    p.add_argument("--seqs", nargs="+", default=REPRO_SEQS)
    for c in ("dump", "aggregate", "final"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"reproduce-old": lambda: cmd_reproduce_old(a.seqs) if a.cmd == "reproduce-old" else None,
     "dump": cmd_dump, "aggregate": cmd_aggregate, "final": cmd_final}[a.cmd]()


if __name__ == "__main__":
    main()
