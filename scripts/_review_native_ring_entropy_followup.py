#!/usr/bin/env python3
"""고정 엔트로피 대조를 GPU1에서 실행한 뒤 원장과 지표를 검산한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def main() -> None:
    """새 진단 실행만 관리하고 다른 프로세스는 변경하지 않는다."""
    root = OUT / "ring_entropy_formula"
    pidfile = root / "runner.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("엔트로피 진단 실행기가 이미 살아 있다")
    if (root / "comparison.json").exists():
        raise SystemExit("엔트로피 진단이 이미 완료됐다")
    pidfile.write_text(str(os.getpid()))
    status = root / "runner_status.json"
    status.write_text(json.dumps({"stage": "running", "pid": os.getpid(), "unix_time": time.time()}))
    command = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=1", "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "LOKY_MAX_CPU_COUNT=2", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "bash", "-lc", "ulimit -s 81920; exec nice -n 15 python3 -u scripts/_review_native_ring_entropy.py run"]
    with (root / "run.log").open("a") as stream:
        result = subprocess.run(command, stdout=stream, stderr=stream)
    if result.returncode:
        status.write_text(json.dumps({"stage": "failed", "returncode": result.returncode, "pid": os.getpid(), "unix_time": time.time()}))
        result.check_returncode()
    code = '''import json,numpy as np
from pathlib import Path
import sys
sys.path.insert(0,"scripts")
import _review_native_ring_verify as verify
r=Path("results/pdv2_review_native_20261105/ring_entropy_formula")
manifest={}
for p in r.joinpath("descriptor_cache").glob("*.npz"):
 z=np.load(p)
 manifest[p.name]={"raw_sha256":str(z["raw_sha256"]),"finite":bool(np.isfinite(z["ring"]).all() and np.isfinite(z["ringpp"]).all()),"shapes":{k:list(z[k].shape) for k in ["ring","ringpp"]}}
(r/"descriptor_manifest.json").write_text(json.dumps(manifest,indent=2))
verify.OUT=r
verify.main()
old=json.loads((r.parent/"maxfix/summary.json").read_text())
new=json.loads((r/"summary.json").read_text())
rows=[]
for key in new["variants"]:
 for radius,row in new["variants"][key].items():
  rows.append({"condition":key,"radius_m":float(radius),"normalized_entropy":old["variants"][key][radius],"literal_paper_entropy":row,"delta_r1_pp":100*(row["official_R@1"]-old["variants"][key][radius]["official_R@1"])})
for region in ["test","all"]:
 assert np.array_equal(np.load(r/f"pair_dists_{region}_ring.npy"),np.load(r.parent/"maxfix"/f"pair_dists_{region}_ring.npy"))
(r/"comparison.json").write_text(json.dumps({"rows":rows,"checks_passed":True,"RING_matrices_unchanged_bitwise":True,"selection":"논문표수식 단일요인;공통46에미적용"},ensure_ascii=False,indent=2))
'''
    subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-c", code], check=True)
    status.write_text(json.dumps({"complete": True, "pid": os.getpid(), "unix_time": time.time()}))


if __name__ == "__main__":
    main()
