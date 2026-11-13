#!/usr/bin/env python3
"""주요 실험이 모두 완료된 뒤에만 지역 특징 좌표 대조를 실행한다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

import _review_native_ring_feature_metric as experiment
import _review_native_ring_verify as verifier

ROOT = experiment.OUT
REPO = experiment.native.REPO


def verify() -> None:
    """네 상관 행렬의 지표를 재계산하고 RING 대조 불변을 검증한다."""
    manifest = {}
    for p in (ROOT / "descriptor_cache").glob("*.npz"):
        with np.load(p) as z:
            manifest[p.name] = {"raw_sha256": str(z["raw_sha256"]), "finite": bool(np.isfinite(z["ring"]).all() and np.isfinite(z["ringpp"]).all()), "shapes": {k: list(z[k].shape) for k in ["ring", "ringpp"]}}
    experiment.native.save_json(ROOT / "descriptor_manifest.json", manifest)
    verifier.OUT = ROOT
    verifier.main()
    old = json.loads((ROOT.parent / "maxfix/summary.json").read_text())
    new = json.loads((ROOT / "summary.json").read_text())
    rows = [{"condition": key, "radius_m": float(radius), "original_anisotropic": old["variants"][key][radius], "isotropic_feature_metric": row, "delta_r1_pp": 100*(row["official_R@1"]-old["variants"][key][radius]["official_R@1"])} for key, group in new["variants"].items() for radius, row in group.items()]
    for region in ["test", "all"]:
        assert np.array_equal(np.load(ROOT / f"pair_dists_{region}_ring.npy"), np.load(ROOT.parent / "maxfix" / f"pair_dists_{region}_ring.npy"))
    experiment.native.save_json(ROOT / "comparison.json", {"rows": rows, "checks_passed": True, "RING_matrices_unchanged_bitwise": True, "selection": "특징 좌표 단일요인만; 공통46 미적용; 논문 정답 구현 주장 안 함"})


def main() -> None:
    """완료 마커를 기다리며 GPU1의 기존 작업과 중복 실행하지 않는다."""
    experiment.prepare()
    pidfile = ROOT / "runner.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("지역 특징 진단 실행기가 이미 살아 있다")
    if (ROOT / "comparison.json").exists():
        return
    pidfile.write_text(str(os.getpid()))
    status = ROOT / "runner_status.json"
    while True:
        ring = json.loads((REPO / "results/pdv2_ringpp_maxfix_20261105/server_status.json").read_text())
        hp = ROOT.parent / "helios_regenerated_velodyne/runner_status.json"
        helios = json.loads(hp.read_text()) if hp.exists() else {}
        learned = json.loads((ROOT.parent / "learned_queue_status.json").read_text())
        learned_count = len(learned["completed"])
        learned_complete = learned_count == learned["total"] and not learned["errors"] and learned["active"] is None
        if ring.get("finished") and len(ring["completed"]) == 46 and not ring["errors"] and helios.get("complete") and learned_complete:
            break
        experiment.native.save_json(status, {"stage": "waiting_primary", "ring_completed": len(ring["completed"]), "helios_complete": bool(helios.get("complete")), "learned_completed": learned_count, "learned_total": learned["total"], "learned_complete": learned_complete, "pid": os.getpid(), "unix_time": time.time()})
        time.sleep(60)
    base = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=1", "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "LOKY_MAX_CPU_COUNT=2", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "bash", "-lc"]
    try:
        for stage in ["smoke", "run"]:
            if stage == "smoke" and (ROOT / "smoke.json").exists():
                continue
            experiment.native.save_json(status, {"stage": stage, "pid": os.getpid(), "unix_time": time.time()})
            with (ROOT / (stage+".log")).open("a") as stream:
                subprocess.run(base + ["ulimit -s 81920; exec nice -n 15 python3 -u scripts/_review_native_ring_feature_metric.py " + stage], stdout=stream, stderr=stream, check=True)
        verify()
        experiment.native.save_json(status, {"complete": True, "pid": os.getpid(), "unix_time": time.time()})
    except Exception as exc:
        experiment.native.save_json(status, {"stage": "failed", "error": repr(exc), "pid": os.getpid(), "unix_time": time.time()})
        raise


if __name__ == "__main__":
    main()
