#!/usr/bin/env python3
"""DiSCO 저자 공개 가중치를 표2 DiSCO와 같은 시험46 파이프라인으로 평가한다(가중치만 바꾼다).

규칙: results/pdv2_disco_author_ckpt_test46_20261118/protocol.json

  check  (컨테이너) 표2 시드1 체크포인트로 한 시퀀스를 임시 폴더에 다시 만들어 기존 원장과 같은지 확인
  dump   (컨테이너) 저자 가중치로 46개 시험 원장을 새 폴더에 만든다(시퀀스마다 하위 프로세스)
  worker (내부)     한 시퀀스 원장
  aggregate         c3 요약 + 공통 집계기, 6도시·NCLT 제외 5도시 매크로, 표2 3시드 비교
  overlap           NCLT 시험 9개와 공식 학습 영역(시험 사각형 밖)의 50 m 겹침
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

OUT = REPO / "results/pdv2_disco_author_ckpt_test46_20261118"
CKPT = OUT / "weights/disco_author_model.ckpt"
CKPT_SHA = "8665e356f0aab2cf03fdc47aa2b3dd2d32be7a2b420bad35ac2c994f45d7808c"
SEED1_CKPT = REPO / "_external/disco_official_pdv2/runs/Oct12_08-15-52/epoch_00.ckpt"
SEED1_SHA = "80cdb08811707462205da226a594daaed9de52b7b1e30ec78ffe9ecfabd66608"
SUMMARY = REPO / ("results/pdv2_deepring_scaleinit_integration_20261110/"
                  "baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json")
# 리포 안 polar 폴더는 색인만 남아 있다. 표2 원장을 만든 서버 polar를 SHA 일치로 복구한 사본을 쓴다
# (results/pdv2_review_selection_full_20261105/disco_test_input_restore_state.json: all_files_SHA_exact).
POLAR = REPO / "results/pdv2_review_selection_full_20261105/inputs/test_disco/polar"
NCLT_RECT = (-50.0, 150.0, -250.0, 150.0)   # 공식 생성기 p: northing, easting 범위(시험 영역)


def worker(key: str, checkpoint: Path, out_dir: Path) -> None:
    import _disco_official_prep as P
    P.POLAR = POLAR
    import _disco_official_dump as D
    from _disco_official_prep import check_store, sha256
    from _disco_official_train import PolarBank
    D.OUT = out_dir
    sensor, q = key.split("/")
    r = check_store("test_odom", sensor, q, n_sample=16)
    assert r["ok"], r
    _, lp, EV, model = D.official_model(checkpoint)
    bank = PolarBank()
    lp.load_pc_file = bank.load
    bank.add("test_odom", sensor, q)
    variant = ("DiSCO authors' released NCLT weights (README Drive link), Table-2 test46 pipeline"
               if sha256(checkpoint) == CKPT_SHA else "Table-2 seed-1 checkpoint, pipeline check")
    D.encode_and_dump(EV, model, bank, sensor, q, out_dir / f"dump_disco_{sensor}_{q}.npz",
                      checkpoint, "explicit --checkpoint", extra_meta={"variant": variant})


def _run(key, checkpoint, out_dir):
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--seq", key,
           "--checkpoint", str(checkpoint), "--out-dir", str(out_dir)]
    rc = subprocess.run(cmd).returncode
    assert rc == 0, (key, rc)


def cmd_check(key: str, scratch: Path) -> None:
    from _disco_official_prep import sha256
    assert sha256(SEED1_CKPT) == SEED1_SHA
    sensor, q = key.split("/")
    _run(key, SEED1_CKPT, scratch)
    new = np.load(scratch / f"dump_disco_{sensor}_{q}.npz")
    old = np.load(REPO / f"results/pdv2_disco_official/odom/dump_disco_{sensor}_{q}.npz")
    rep = {"sequence": key, "checkpoint_sha256": SEED1_SHA}
    for k in ("query_idx", "j_star", "score_disco", "top1_geo_dist", "min_past_gt_dist", "n_candidates"):
        a, b = new[k], old[k]
        rep[k] = {"equal": bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f")),
                  "max_abs_diff": float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))}
    rep["j_star_agree_fraction"] = float((new["j_star"] == old["j_star"]).mean())
    rep["all_equal"] = all(v["equal"] for k, v in rep.items() if isinstance(v, dict))
    (OUT / "pipeline_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_dump() -> None:
    from _disco_official_prep import sha256
    from _pdv2_pilot import TEST
    assert sha256(CKPT) == CKPT_SHA, "author checkpoint hash mismatch"
    odom = OUT / "odom"
    for s, q in TEST:
        if (odom / f"dump_disco_{s}_{q}.npz").exists():
            print(f"[skip] {s}/{q}", flush=True)
            continue
        _run(f"{s}/{q}", CKPT, odom)
    print("AUTHOR_DUMP_DONE", flush=True)


def _five(entry):
    from _pdv2_main_baselines_official import METRICS
    g = {c: v for c, v in entry["city_groups"].items() if c != "ann_arbor"}
    return {k: float(np.mean([v[k] for v in g.values()])) for k in METRICS}


def cmd_aggregate() -> None:
    import _pdv2_c3 as c3
    from _pdv2_main_baselines_official import METRICS, aggregate, sha256
    c3.OUT_ROOT = OUT
    c3.BASELINES = {"disco": "score_disco"}
    if not (OUT / "odom/pr_summary_baselines.json").exists():   # 컨테이너가 만든 odom/은 root 소유
        c3.cmd_aggregate_baselines("odom")
    sources = {}

    def track(p: Path) -> Path:
        sources[str(p.relative_to(REPO))] = sha256(p)
        return p

    track(Path(__file__))
    track(REPO / "scripts/_pr_eval_lib.py")
    track(REPO / "scripts/_pdv2_main_baselines_official.py")
    author = aggregate(str((OUT / "odom").relative_to(REPO)), "dump_disco", "score_disco",
                       "pr_summary_baselines.json", "disco", track)
    summ = json.loads(track(SUMMARY).read_text())["methods"]["disco"]
    seeds = {}
    for seed, rec in summ["per_seed"].items():
        e = aggregate(rec["source"], "dump_disco", "score_disco", "pr_summary_baselines.json", "disco", track)
        for k in METRICS:   # 표2 요약 재현
            assert abs(e["macro"][k] - rec["macro"][k]) < 1e-12, (seed, k)
        seeds[seed] = {"macro6": e["macro"], "macro5_no_nclt": _five(e), "city_groups": e["city_groups"]}

    def ms(key):
        out = {}
        for k in METRICS:
            v = np.array([s[key][k] for s in seeds.values()])
            out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": v.tolist()}
        return out

    city_ms = {c: {k: {"mean": float(np.mean([s["city_groups"][c][k] for s in seeds.values()])),
                       "std": float(np.std([s["city_groups"][c][k] for s in seeds.values()], ddof=1))}
                   for k in METRICS} for c in author["city_groups"]}
    rep = {
        "protocol": "results/pdv2_disco_author_ckpt_test46_20261118/protocol.json",
        "checkpoint_sha256": sha256(CKPT),
        "author": {"macro6": author["macro"], "macro5_no_nclt": _five(author),
                   "city_groups": author["city_groups"], "per_sequence": author["per_sequence"],
                   "n_queries": author["n_queries"], "n_positive": author["n_positive"]},
        "table2_disco_3seeds": {"macro6": ms("macro6"), "macro5_no_nclt": ms("macro5_no_nclt"),
                                "city_groups": city_ms, "summary_reproduced": True},
        "contamination": "ann_arbor (NCLT 9 sequences) marked dagger: author weights trained on NCLT "
                         "(see nclt_overlap.json)",
        "table2_changed": False,
        "input_sha256": sources,
    }
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    for name, m in (("author", rep["author"]), ("table2", rep["table2_disco_3seeds"])):
        for key in ("macro6", "macro5_no_nclt"):
            v = m[key]
            print(name, key, {k: round(v[k]["mean"] if isinstance(v[k], dict) else v[k], 4)
                              for k in ("AP", "max_F1", "R@100P", "R@1")})


def cmd_overlap() -> None:
    from scipy.spatial import cKDTree
    from _pdv2_pilot import TEST
    ds = Path("/mnt/d/NSD_datasets/nclt")
    sess = [q for s, q in TEST if s == "nclt"]
    xy = {}
    for q in sess:
        a = np.loadtxt(ds / f"{q}/groundtruth_{q}.csv", delimiter=",", usecols=(0, 1, 2))
        a = a[np.isfinite(a).all(1)][::10]
        xy[q] = a[:, 1:3]
    n0, n1, e0, e1 = NCLT_RECT

    def inside(p):
        return (p[:, 0] > n0) & (p[:, 0] < n1) & (p[:, 1] > e0) & (p[:, 1] < e1)

    rows = {}
    for q in sess:
        others = np.concatenate([xy[o][~inside(xy[o])] for o in sess if o != q])
        d, _ = cKDTree(others).query(xy[q])
        rows[f"nclt/{q}"] = {"n_positions": int(len(xy[q])),
                             "outside_test_rectangle": float((~inside(xy[q])).mean()),
                             "within_50m_of_other_sessions_training_area": float((d <= 50).mean()),
                             "within_25m": float((d <= 25).mean())}
    rep = {"rule": "positions from groundtruth csv (every 10th row, finite), shared campus frame; "
                   "training area = outside official rectangle northing -50..150, easting -250..150 "
                   "(_external/DiSCO/generating_queries/nclt/generate_training_tuples_baseline_with_pose.py:35); "
                   "author session list unpublished, so the other 8 local sessions stand in",
           "per_sequence": rows,
           "min_within_50m": min(r["within_50m_of_other_sessions_training_area"] for r in rows.values())}
    (OUT / "nclt_overlap.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--seq", default="kitti/07")
    p.add_argument("--scratch", type=Path, required=True)
    sub.add_parser("dump")
    p = sub.add_parser("worker")
    p.add_argument("--seq", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    sub.add_parser("aggregate")
    sub.add_parser("overlap")
    a = ap.parse_args()
    if a.cmd == "check":
        cmd_check(a.seq, a.scratch)
    elif a.cmd == "dump":
        cmd_dump()
    elif a.cmd == "worker":
        worker(a.seq, a.checkpoint, a.out_dir)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    else:
        cmd_overlap()


if __name__ == "__main__":
    main()
