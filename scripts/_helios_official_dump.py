#!/usr/bin/env python3
"""HeLiOS official pipeline on split v2 -- part 4: checkpoint choice and
past-only test ledgers (2026-10-11).

Inference = the OFFICIAL evaluation code, executed unchanged:
  eval/evaluate.get_latent_vectors -> datasets/pnv_raw.PNVPointCloudLoader on
  the toolbox scans (x, y, z of the .bin), compute_embedding
  (CartesianQuantizer 0.01, ME.utils.batched_coordinates, features = 1, one
  scan per forward, model(batch)['global']); model loading as evaluate.main()
  (model_factory(config/model_helios.txt), torch.load, load_state_dict).
  Only evaluate.DATASET_FOLDER (module constant "data/") is pointed at
  _external/helios_official_pdv2/data.
The descriptor is L2-normalised inside the model (models/helios.py:124).

select   every checkpoint the official loop saved (_best by validation AP,
         _10 ... _80, _final; identical files counted once) is scored on all
         split-v2 VAL keyframes: intra-sequence 5 m R@1 with a 30-keyframe gap
         (compute_recall_multi_k semantics: top 61 by cosine, temporal
         neighbours removed, first remaining), sensor families weighted
         equally with queries pooled inside a family (_pdv2_c3._family_pooled).
         Written to results/pdv2_helios_official/train_selection.json with
           official_best = <run>/HeLiOS_default_trainning_best.pth (the official
                           rule: highest validation-phase AP on Roundabout01),
                           recorded,
           val_best      = highest family-equal R@1 (ties -> earlier epoch),
           chosen        = val_best (default, user decision 2026-10-11: the rule
                           of the other learned baselines; --chosen official_best).
dump     46 split-v2 TEST sequences on the KISS-ICP keyframes of
         data/preprocessed_v2_odom (scans from _helios_official_prep.py role
         test_odom): ledger = scripts/_causal_bevplace_pr.ledger, unchanged
         (pr-dump-v1; queries t >= 30; exhaustive top-1 over the prefix
         j <= t - 30; score_cos = cosine of the unit descriptors).  The official
         retrieval is a Euclidean KDTree with Euclidean-distance thresholds on
         the same unit vectors: same top-1, ||a - b||^2 = 2 - 2 cos.  The L2
         top-1 of 100 queries per sequence is recomputed into meta.l2_check.
         Scoring uses poses_gt only.
aggregate PR summary via _pdv2_c3.cmd_aggregate_baselines (unchanged).
selftest  CPU: batched R@1 == compute_recall_multi_k; the official
         get_latent_vectors + ledger on 80 test keyframes with a CPU stand-in
         network (the HeLiOS forward needs CUDA) -> pr-dump-v1 format == an
         existing ledger of the same sequence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

import _helios_official_prep as P  # noqa: E402
import _helios_official_train as TR  # noqa: E402  (SEED and seed-specific paths)
from _causal_bevplace_pr import ledger  # noqa: E402  (torch-free, unchanged)
from _pdv2_pilot import TEST, VAL  # noqa: E402
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402

RESULTS = TR.RESULTS          # results/pdv2_helios_official[_s<SEED>]
OUT = RESULTS / "odom"
RECORD = TR.RECORD
SELECTION = RESULTS / "train_selection.json"
METHOD = "helios"
SKIP, RADIUS = 30, 5.0


def _official(config_path: Path | None = None):
    TR.official_path()
    import eval.evaluate as E
    from misc.utils import TrainingParams
    E.DATASET_FOLDER = str(P.DATA)
    cfg = TR.write_config(config_path or TR.CONFIG, P.DATA)
    params = TrainingParams(str(cfg), str(TR.MODEL_CONFIG))
    return E, params


def official_model(E, params, checkpoint: Path, device: str):
    """evaluate.main(): model_factory, torch.load, load_state_dict, .to(device)."""
    import torch
    model = E.model_factory(params.model_params)
    prev_model = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(prev_model)
    model.to(device)
    return model


def descriptors(E, params, model, role, sensor, seq, device, limit=0):
    z = np.load(P.seq_dir(role, sensor, seq) / "index.npz")
    sids = z["scan_ids"][z["generate"]]
    if limit:
        sids = sids[:limit]
    dataset = {i: {"query": P.rel_scan_path(role, sensor, seq, s)} for i, s in enumerate(sids)}
    t0 = time.time()
    desc = E.get_latent_vectors(model, dataset, device, params)     # official
    return desc.astype(np.float32), sids, time.time() - t0


# ------------------------------------------------------------ validation R@1
def recall_at_1(desc, poses, per_query_records=None, block=256):
    """compute_recall_multi_k(desc, poses, [1], 5, 30) with the cosine top-61
    computed by float32 matrix products instead of faiss IndexFlatIP."""
    from baselines.eval_utils import _find_revisit_queries, _score_recalls_from_ranked
    positions = poses[:, :3, 3].astype(np.float64)
    queries = _find_revisit_queries(positions, RADIUS, SKIP)
    if not queries:
        return 0.0, 0
    emb = desc.astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    k = min(1 + 2 * SKIP, len(emb))
    qidx = np.array([q for q, _ in queries])
    ranked = []
    for s in range(0, len(qidx), block):
        qb = qidx[s:s + block]
        S = emb[qb] @ emb.T
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        for row, q, sc in zip(top, qb, S):
            row = row[np.lexsort((row, -sc[row]))]        # score desc, then index
            ranked.append(row[np.abs(row - q) > SKIP][:1])
    rec = _score_recalls_from_ranked(queries, ranked, positions, [1], RADIUS, poses=poses,
                                     per_query_records=per_query_records)
    return float(rec[1]), len(queries)


def _candidates(record):
    run = Path(record["_meta"]["run_dir"])
    best_epoch = max(e["epoch"] for e in record["epochs"] if e["saved_best"])
    last = max(e["epoch"] for e in record["epochs"])
    cands, seen = [], {}
    names = sorted(record["checkpoints"])
    for name in names:
        tag = name.rsplit("_", 1)[1][:-4]          # best | final | <epoch>
        ep = best_epoch if tag == "best" else last if tag == "final" else int(tag)
        sha = record["checkpoints"][name]
        if sha in seen:
            seen[sha]["aliases"].append(name)
            continue
        seen[sha] = {"name": name, "path": str(run / name), "epoch": ep, "sha256": sha, "aliases": [name]}
        cands.append(seen[sha])
    return sorted(cands, key=lambda c: c["epoch"])


def cmd_select(chosen: str, device: str):
    from _pdv2_c3 import _family_pooled
    record = json.loads(RECORD.read_text())
    assert record.get("finished"), f"{RECORD}: training not finished"
    E, params = _official()
    sel = json.loads(SELECTION.read_text()) if SELECTION.exists() else {"candidates": {}}
    for c in _candidates(record):
        entry = sel["candidates"].setdefault(c["name"], {**c, "per_seq": {}})
        assert entry["sha256"] == P.sha256(Path(c["path"])), f"{c['path']} changed"
        model = None
        for sensor, seq in VAL:
            key = f"{sensor}/{seq}"
            if key in entry["per_seq"]:
                continue
            if model is None:
                model = official_model(E, params, Path(c["path"]), device)
            desc, sids, dt = descriptors(E, params, model, "val", sensor, seq, device)
            poses = np.load(P.cache_path("val", sensor, seq))["poses"].astype(np.float64)
            r1, nq = recall_at_1(desc, poses)
            entry["per_seq"][key] = {"r1": r1, "n_queries": nq, "n_frames": int(len(sids)),
                                     "encode_s": round(dt, 1)}
            print(f"  {c['name']} {key}: R@1 {r1:.4f} ({nq} q, {dt:.0f} s)", flush=True)
            P.WORK.mkdir(parents=True, exist_ok=True)
            _write(SELECTION, sel)
        entry["family_macro_r1"] = _family_pooled(entry["per_seq"])
        print(f"== {c['name']} (epoch {c['epoch']}): family R@1 {entry['family_macro_r1']:.4f}", flush=True)
    cands = sorted(sel["candidates"].values(), key=lambda e: e["epoch"])
    vb = max(cands, key=lambda e: (e["family_macro_r1"], -e["epoch"]))
    ob = next(e for e in cands if "HeLiOS_default_trainning_best.pth" in e["aliases"])
    sel["_meta"] = {"rule_official_best": "official trainer: highest validation-phase TruncatedSmoothAP "
                                          "statistic on validation.pickle (Roundabout01, 3 m keyframes, "
                                          "overlap labels), trainer.py:333-336",
                    "rule_val_best": "split-v2 VAL (9 sequences, all keyframes) intra-sequence 5 m R@1, "
                                     "30-keyframe gap, families equal (_pdv2_c3._family_pooled), "
                                     "ties -> earlier epoch",
                    "train_record": str(RECORD.relative_to(REPO))}
    sel["official_best"] = {"name": ob["name"], "epoch": ob["epoch"], "path": ob["path"],
                            "sha256": ob["sha256"], "family_macro_r1": ob["family_macro_r1"]}
    sel["val_best"] = {"name": vb["name"], "epoch": vb["epoch"], "path": vb["path"],
                       "sha256": vb["sha256"], "family_macro_r1": vb["family_macro_r1"]}
    sel["chosen"] = chosen
    _write(SELECTION, sel)
    print(f"SELECT official_best={ob['name']} (epoch {ob['epoch']}, R@1 {ob['family_macro_r1']:.4f}); "
          f"val_best={vb['name']} (epoch {vb['epoch']}, R@1 {vb['family_macro_r1']:.4f}); chosen={chosen}",
          flush=True)


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ------------------------------------------------------------ test ledgers
def _l2_check(desc, arrays, n=100):
    q, js = arrays["query_idx"], arrays["j_star"]
    pick = np.unique(np.linspace(0, len(q) - 1, min(n, len(q))).round().astype(int))
    agree = 0
    for k in pick:
        t = int(q[k])
        d = np.sqrt(np.sum((desc[:t - MIN_INDEX_GAP + 1] - desc[t]) ** 2, axis=1))
        agree += int(np.argmin(d) == js[k])
    return {"n_checked": int(len(pick)), "l2_top1_equals_j_star": int(agree)}


def encode_and_dump(E, params, model, sensor, q, out_path, checkpoint, device, limit=0, extra_meta=None):
    cp = P.cache_path("test_odom", sensor, q)
    with np.load(cp) as z:
        sids_c = z["scan_ids"].astype(np.int64)
        poses_gt = z["poses_gt"].astype(np.float64)
    desc, sids, t_enc = descriptors(E, params, model, "test_odom", sensor, q, device, limit)
    assert np.array_equal(sids, sids_c[:len(sids)]), f"{sensor}/{q}: scans != cache keyframes"
    poses_gt = poses_gt[:len(sids)]
    norms = np.linalg.norm(desc, axis=1)
    t0 = time.time()
    arrays = ledger(desc, poses_gt[:, :3, 3])
    t_led = time.time() - t0
    l2 = _l2_check(desc, arrays)
    pm = json.loads((P.seq_dir("test_odom", sensor, q) / "meta.json").read_text())
    meta = {
        "schema": "pr-dump-v1", "method": METHOD,
        "variant": "official HeLiOS recipe retrained on split-v2 train" + ("" if TR.SEED == 1 else f", seed {TR.SEED}"),
        "sensor": sensor, "sequence": q, "min_index_gap": MIN_INDEX_GAP,
        "n_frames": int(len(desc)), "n_queries": int(len(arrays["query_idx"])),
        "primary_score": "score_cos",
        "scores": {"score_cos": "8448D HeLiOS descriptor cosine of the exhaustive prefix top-1; "
                                "higher = accept. Official retrieval: Euclidean KDTree and "
                                "Euclidean-distance thresholds on the same unit vectors (same top-1, "
                                "L2^2 = 2 - 2 cos)"},
        "descriptor_norm": {"min": float(norms.min()), "max": float(norms.max())},
        "l2_check": l2,
        "official": {"repo": "https://github.com/minwoo0611/HeLiOS", "commit": pm["helios_commit"],
                     "code": "eval/evaluate.py get_latent_vectors / compute_embedding / main() "
                             "model loading; models/*; datasets/pnv_raw.PNVPointCloudLoader",
                     "toolbox": f"HeLiPR-Pointcloud-Toolbox@{P.TOOLBOX_COMMIT[:7]} (scans)"},
        "artifacts": {"checkpoint": str(checkpoint) if checkpoint else None,
                      "checkpoint_sha256": P.sha256(checkpoint) if checkpoint else None,
                      "selection": str(SELECTION.relative_to(REPO))},
        "caches": {"operating_cache": str(cp.relative_to(REPO)), "operating_cache_sha256": pm["cache_sha256"],
                   "scan_dir": str(P.seq_dir("test_odom", sensor, q).relative_to(REPO)),
                   "deskew": pm.get("deskew"), "positions": "poses_gt (scoring only)"},
        "encoder": {"points": 8192, "crop_m": 100, "normalise": "1/100", "quantization": 0.01,
                    "descriptor_dim": int(desc.shape[1]), "l2_normalized": "inside the model (official)"},
        "protocol": ("causal all-frames PR ledger: per-frame descriptors computed once per sequence "
                     "(each depends only on its own scan; HeLiPR scans undistorted with KISS-ICP "
                     "odometry when deskew is on), per-query gemv cosine search restricted to prefix "
                     "rows j <= t - 30 (scripts/_causal_bevplace_pr.ledger, unchanged); KISS-ICP "
                     "keyframes, GT poses for scoring only"),
        "timing": {"encode_s": round(t_enc, 1), "encode_ms_per_kf": round(t_enc / max(len(desc), 1) * 1e3, 2),
                   "ledger_s": round(t_led, 1)},
    }
    if TR.SEED != 1:
        meta["train_seed"] = TR.SEED
    meta.update(extra_meta or {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    print(f"[{sensor}/{q}] n_queries={len(arrays['query_idx'])} l2-check "
          f"{l2['l2_top1_equals_j_star']}/{l2['n_checked']} ({t_enc:.0f} s encode, {t_led:.0f} s ledger) "
          f"-> {out_path}", flush=True)
    return out_path


def _chosen_checkpoint(use: str | None) -> Path:
    sel = json.loads(SELECTION.read_text())
    key = use or sel["chosen"]
    c = sel[key]
    ck = Path(c["path"])
    assert P.sha256(ck) == c["sha256"], "checkpoint changed after selection"
    return ck


def cmd_dump(seqs, checkpoint, use, worker, overwrite, device):
    checkpoint = Path(checkpoint) if checkpoint else _chosen_checkpoint(use)
    todo = seqs or [f"{s}/{q}" for s, q in TEST]
    unknown = sorted(set(todo) - {f"{s}/{q}" for s, q in TEST})
    assert not unknown, f"not split-v2 test sequences: {unknown}"
    if worker:
        assert len(todo) == 1
        sensor, q = todo[0].split("/")
        E, params = _official()
        model = official_model(E, params, checkpoint, device)
        encode_and_dump(E, params, model, sensor, q, OUT / f"dump_{METHOD}_{sensor}_{q}.npz",
                        checkpoint, device)
        return
    for key in todo:   # one sequence per subprocess (memory discipline)
        sensor, q = key.split("/")
        if (OUT / f"dump_{METHOD}_{sensor}_{q}.npz").exists() and not overwrite:
            print(f"[skip] {key}", flush=True)
            continue
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "dump", "--worker",
                             "--seqs", key, "--checkpoint", str(checkpoint)]).returncode
        if rc:
            print(f"WORKER_FAILED {key} rc={rc}", flush=True)
            sys.exit(rc)
    print("HELIOS_OFFICIAL_DUMP_DONE", flush=True)


def cmd_aggregate():
    import _pdv2_c3 as c3
    c3.OUT_ROOT = RESULTS
    c3.BASELINES = {METHOD: "score_cos"}
    c3.cmd_aggregate_baselines("odom")
    p = OUT / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"].pop("seqot", None)   # hard-coded note of the reused aggregator
    sel = json.loads(SELECTION.read_text())
    s["_meta"][METHOD] = ("official HeLiOS recipe retrained on split-v2 train "
                          + ("" if TR.SEED == 1 else f"(seed {TR.SEED}) ") +
                          "(scripts/_helios_official_train.py), checkpoint "
                          f"{sel['chosen']} = {sel[sel['chosen']]['name']}; {SELECTION.relative_to(REPO)}")
    p.write_text(json.dumps(s, indent=2))


# ------------------------------------------------------------ CPU selftest
def selftest(out_json: Path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from baselines.eval_utils import compute_recall_multi_k
    rep = {}
    rng = np.random.default_rng(0)
    for s, q in (("helipr", "Roundabout02"), ("kitti", "09")):
        Pz = np.load(P.cache_path("val", s, q))["poses"].astype(np.float64)
        pos = Pz[:, :3, 3]
        W = rng.standard_normal((6, 256))
        f = np.concatenate([np.cos(pos[:, :2] / 7.0), np.sin(pos[:, :2] / 7.0), np.cos(pos[:, :2] / 3.0)], 1)
        emb = (f @ W + 0.8 * rng.standard_normal((len(pos), 256))).astype(np.float32)
        recs, mine = [], []
        ref, nq_ref = compute_recall_multi_k(emb, Pz, k_values=[1], distance_threshold=5.0,
                                             skip_frames=30, per_query_records=recs)
        r1, nq = recall_at_1(emb, Pz, per_query_records=mine)
        same = nq == nq_ref and r1 == ref[1] and [r["top1_idx"] for r in mine] == [r["top1_idx"] for r in recs]
        rep[f"recall_{s}_{q}"] = {"r1": r1, "ref": ref[1], "n_queries": nq, "equal_incl_top1": bool(same)}
        print(json.dumps(rep[f"recall_{s}_{q}"]), flush=True)
    # official get_latent_vectors + ledger on 80 test keyframes, stand-in network
    from _pd_build_test_caches import _make_loader
    sensor, q, n = "nclt", "2013-01-10", 80
    tb = P.Toolbox()
    with tempfile.TemporaryDirectory(dir=str(P.WORK)) as td:
        td = Path(td)
        z = np.load(P.cache_path("test_odom", sensor, q))
        sids = z["scan_ids"][:n]
        loader = _make_loader(sensor, q)
        d = td / "test_odom" / P.seq_name(sensor, q)
        (d / "LiDAR").mkdir(parents=True)
        for sid in sids:
            pts, _ = P.process_scan(tb, loader[int(sid)]["points"], P.fill_rng(sensor, q, int(sid)))
            pts.astype("<f4").tofile(d / "LiDAR" / f"{int(sid):06d}.bin")
        np.savez(d / "index.npz", scan_ids=z["scan_ids"], generate=np.ones(len(z["scan_ids"]), bool))
        (d / "meta.json").write_text(json.dumps({"helios_commit": P.git_head(P.HELIOS),
                                                 "cache_sha256": P.sha256(P.cache_path("test_odom", sensor, q)),
                                                 "deskew": None}))
        old_data = P.DATA
        P.DATA = td
        try:
            E, params = _official(td / "cfg.txt")
            import torch
            torch.manual_seed(0)
            model = TR._StandIn.make().eval()
            path = encode_and_dump(E, params, model, sensor, q, td / f"dump_{METHOD}_{sensor}_{q}.npz",
                                   None, "cpu", limit=n,
                                   extra_meta={"DRY_RUN": "CPU stand-in network, format check only"})
        finally:
            P.DATA = old_data
        zz = np.load(path)
        ref = np.load(REPO / f"results/pdv2_c3/odom/dump_seqot_{sensor}_{q}.npz")
        m = len(zz["query_idx"])
        fmt = {"ours": {k: [str(zz[k].dtype), list(zz[k].shape)] for k in zz.files},
               "ref_seqot": {k: [str(ref[k].dtype), list(ref[k].shape)] for k in ref.files}}
        for k in ("query_idx", "min_past_gt_dist", "n_candidates"):
            fmt[f"{k}_equal_on_shared_rows"] = bool(np.array_equal(zz[k], ref[k][:m]))
        zm, rm = json.loads(str(zz["meta"])), json.loads(str(ref["meta"]))
        fmt["schema"] = [zm["schema"], rm["schema"]]
        fmt["dtypes_match"] = all(str(zz[k].dtype) == str(ref[k].dtype) for k in
                                  ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist", "n_candidates"))
        fmt["l2_check"] = zm["l2_check"]
        rep["ledger_format"] = fmt
        print(json.dumps(fmt), flush=True)
    ok = all(v["equal_incl_top1"] for k, v in rep.items() if k.startswith("recall_")) and \
        fmt["dtypes_match"] and fmt["schema"] == ["pr-dump-v1", "pr-dump-v1"] and \
        all(fmt[f"{k}_equal_on_shared_rows"] for k in ("query_idx", "min_past_gt_dist", "n_candidates"))
    rep["all_ok"] = bool(ok)
    out_json.write_text(json.dumps(rep, indent=1))
    print("DUMP_SELFTEST_OK" if ok else "DUMP_SELFTEST_FAILED", flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--chosen", choices=("official_best", "val_best"), default="val_best")
    p = sub.add_parser("dump")
    p.add_argument("--seqs", nargs="+", default=None)
    p.add_argument("--checkpoint", default=None, help="default: train_selection.json 'chosen'")
    p.add_argument("--use", choices=("official_best", "val_best"), default=None)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    sub.add_parser("aggregate")
    p = sub.add_parser("selftest")
    p.add_argument("--out-json", type=Path, default=P.LOG_DIR / f"helios_selftest_dump{TR.SFX}.json")
    a = ap.parse_args()
    if a.cmd == "select":
        cmd_select(a.chosen, "cuda")
    elif a.cmd == "dump":
        cmd_dump(a.seqs, a.checkpoint, a.use, a.worker, a.overwrite, "cuda")
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "selftest":
        selftest(a.out_json)


if __name__ == "__main__":
    main()
