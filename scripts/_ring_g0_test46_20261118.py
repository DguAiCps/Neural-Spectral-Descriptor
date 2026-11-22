#!/usr/bin/env python3
"""RING 각도 격자 G0 시험46 민감도 확인(규칙: results/pdv2_ring_g0_test46_20261118/protocol.md).

한 시퀀스마다 점유 BEV를 공식 경로(_ring_official_pr.cmd_bev의 RING 부분)로 다시 만들고,
같은 BEV로 두 원장을 쓴다.
  g1/odom/dump_ring_*.npz  공식 generate_RING(0과 2π 중복) — 표2 원장과 같은지 검산용
  g0/odom/dump_ring_*.npz  Radon 각도만 linspace(..., endpoint=False)
원장 생성은 _ring_official_pr.cmd_dump를 그대로 부르고, BEV 공급(bev_shards)과 출력 위치(OUT_ROOT)만 바꾼다.

도커 ring_baseline 안, 리포 루트에서 `ulimit -s 81920` 뒤:
  python3 scripts/_ring_g0_test46_20261118.py run --seq kitti/07
  python3 scripts/_ring_g0_test46_20261118.py aggregate
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _ring_official_lib as rol  # noqa: E402  (before anything importing 'utils')
import _ring_official_pr as rpr  # noqa: E402

OUT = REPO / "results/pdv2_ring_g0_test46_20261118"
COND = "odom"


def generate_ring_g0(bev):
    """공식 generate_RING(core.py:152-159)과 같고 각도만 endpoint=False."""
    import torch
    core = rol.import_ring_core()
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, core.cfg.num_ring, endpoint=False).astype(np.float32))
    radon = core.ParallelBeam(core.cfg.num_sector, angles)
    pc_ring = radon.forward(bev)
    pc_tiring, _ = core.forward_row_fft(pc_ring)
    return pc_ring, pc_tiring


def build_occ(sensor, q, scan_ids):
    """cmd_bev의 RING 점유 BEV 부분과 같은 계산(CPU 전사 검사 포함), 메모리에만 둔다."""
    import torch
    core = rol.import_ring_core()
    loader = rol.make_loader(sensor, q)
    n = len(scan_ids)
    occ = np.zeros((n, 1, 120, 120), np.uint8)
    n_pts = np.zeros(n, np.int64)
    t0 = time.time()
    for k in range(n):
        pts = loader[int(scan_ids[k])]["points"]
        _, pc_norm = rol.official_points(sensor, pts)
        n_pts[k] = len(pc_norm)
        with torch.no_grad():
            if len(pc_norm) > 0:
                occ[k] = core.generate_bev_occ(pc_norm).cpu().numpy()
        rol.check_bev(occ[k], pc_norm, None)
        if k and k % 1000 == 0:
            print(f"  [{sensor}/{q}] BEV {k}/{n} ({(time.time() - t0) / k:.3f} s/kf)", flush=True)
    bad = rol.degenerate_rows(occ, n_pts)
    assert len(bad) == 0, f"{len(bad)} degenerate occupancy rows"
    return occ, n_pts, time.time() - t0


def cmd_run(seq):
    sensor, q = rpr._split(seq)
    outs = {g: OUT / g / COND / f"dump_ring_{sensor}_{q}.npz" for g in ("g1", "g0")}
    if all(p.exists() for p in outs.values()):
        print(f"SKIP {seq}", flush=True)
        return
    core = rol.import_ring_core()
    official = core.generate_RING
    cache = rol.split_cache(COND, "test", sensor, q)
    scan_ids, _ = rol.load_keyframes(cache)
    occ, n_pts, bev_s = build_occ(sensor, q, scan_ids)
    info = {"sensor": sensor, "sequence": q, "cond": COND, "cache": str(cache),
            "n_frames": len(scan_ids), "scan_ids_sha256": rpr._sha(scan_ids),
            "table_row": rol.DATASET_TABLE[sensor], "ring_commit": rol.RING_COMMIT,
            "n_frames_no_points": int((n_pts == 0).sum()),
            "points_per_frame_median": float(np.median(n_pts)),
            "bev_rebuilt_in_memory": True, "occ_sha256": rpr._sha(occ),
            "bev_s_per_kf_this_run": round(bev_s / len(scan_ids), 4)}
    rpr.bev_shards = lambda cond, s, qq, v, ids: (info, iter([occ]))
    for g, fn in (("g1", official), ("g0", generate_ring_g0)):
        if outs[g].exists():
            continue
        core.generate_RING = fn
        rol.OUT_ROOT = OUT / g
        rpr.cmd_dump(COND, seq, ("ring",), False, False)
        z = np.load(outs[g], allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        meta["angle_grid"] = {"g1": "official np.linspace(0, 2*pi, 120) (0 and 2*pi both present)",
                              "g0": "np.linspace(0, 2*pi, 120, endpoint=False)"}[g]
        arrays = {k: z[k] for k in z.files if k != "meta"}
        z.close()
        np.savez_compressed(outs[g], meta=json.dumps(meta), **arrays)
    core.generate_RING = official
    print(f"RUN_DONE {seq} bev {bev_s:.0f} s", flush=True)


def cmd_aggregate():
    """공식 요약(pr_summary_ring_official.json) → 표2와 같은 공통 집계 → 표2 원장 비교."""
    from _pdv2_main_baselines_official import aggregate, sha256
    from _pdv2_pilot import TEST
    sources = {}

    def track(p):
        sources[str(Path(p).relative_to(REPO))] = sha256(Path(p))
        return p

    track(Path(__file__))
    track(OUT / "protocol.md")
    ref_dir = REPO / "results/pdv2_ring_official/odom"
    # G1 재생성 원장 == 표2 원장 (배열 단위)
    ident = {}
    for s, q in TEST:
        a = np.load(OUT / "g1" / COND / f"dump_ring_{s}_{q}.npz")
        b = np.load(ref_dir / f"dump_ring_{s}_{q}.npz")
        ident[f"{s}/{q}"] = {k: bool(np.array_equal(a[k], b[k])) for k in b.files if k != "meta"}
    all_equal = all(all(v.values()) for v in ident.values())
    res = {}
    for g in ("g1", "g0"):
        rpr.cmd_aggregate(COND, root=str(OUT / g))
        res[g] = aggregate(f"results/pdv2_ring_g0_test46_20261118/{g}/{COND}", "dump_ring", "score_corr",
                           "pr_summary_ring_official.json", "ring", track)
    table2 = json.loads(track(REPO / "results/pdv2_deepring_scaleinit_integration_20261110/"
                              "baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
                              ).read_text())["methods"]["ring"]["per_seed"]["-"]["macro"]
    keys = ("AP", "max_F1", "R@100P", "R@1")
    report = {
        "protocol": "results/pdv2_ring_g0_test46_20261118/protocol.md",
        "g1_ledgers_equal_table2_ledgers": all_equal,
        "g1_identity_per_sequence": ident,
        "g1_macro_equals_table2": all(abs(res["g1"]["macro"][k] - table2[k]) < 1e-12 for k in keys),
        "table2_ring_macro": {k: table2[k] for k in keys},
        "g0_macro": {k: res["g0"]["macro"][k] for k in keys},
        "delta_g0_minus_table2_pp": {k: 100 * (res["g0"]["macro"][k] - table2[k]) for k in keys},
        "city_groups": {g: res[g]["city_groups"] for g in ("g1", "g0")},
        "per_sequence": {g: res[g]["per_sequence"] for g in ("g1", "g0")},
        "n_queries": res["g0"]["n_queries"], "n_positive": res["g0"]["n_positive"],
        "input_sha256": sources,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("g1_ledgers_equal_table2_ledgers", "g1_macro_equals_table2",
                                             "table2_ring_macro", "g0_macro", "delta_g0_minus_table2_pp")},
                     indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("--seq", required=True)
    sub.add_parser("aggregate")
    a = ap.parse_args()
    cmd_run(a.seq) if a.cmd == "run" else cmd_aggregate()


if __name__ == "__main__":
    main()
