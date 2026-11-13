#!/usr/bin/env python3
"""원인 단일요인 대조의 완료 원장도 주기적으로 회수·독립 검산한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from _review_native_collect import sync, REPO, NATIVE


def main() -> None:
    root = REPO / NATIVE
    while True:
        try:
            sync(NATIVE + "/ring_angles_endpoint_false", ["*.json", "pair_dists_*.npy"], optional=True)
            angle = root / "ring_angles_endpoint_false"
            if (angle / "comparison.json").exists():
                code = "import sys;sys.path.insert(0,'scripts');import _review_native_ring_verify as v;v.OUT=v.OUT/'ring_angles_endpoint_false';v.main()"
                subprocess.run(["python3", "-c", code], cwd=REPO, check=True)
            regen = root / "helios_regenerated_velodyne"
            for model in ["author", "v2_s1", "v2_s2", "v2_s3"]:
                sync(NATIVE + "/helios_regenerated_velodyne/" + model, ["native*.json", "native_pairs/*.npz"], optional=True)
            if (regen / "author/native_partial_report.json").exists():
                subprocess.run(["python3", "scripts/_review_native_helios_regen_eval.py", "verify"], cwd=REPO, check=True)
            subprocess.run(["python3", "scripts/_review_native_learned_report.py"], cwd=REPO, check=True)
            status = {"passed": True, "pid": os.getpid(), "unix_time": time.time()}
        except Exception as exc:
            status = {"passed": False, "error": repr(exc), "pid": os.getpid(), "unix_time": time.time()}
        with (root / "extra_collection_history.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        print(json.dumps(status), flush=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
