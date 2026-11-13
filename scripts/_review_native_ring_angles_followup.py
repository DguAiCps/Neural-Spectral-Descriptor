#!/usr/bin/env python3
"""확보된 학습형 native 증거를 우선 마친 뒤 RING 각도 대조를 실행한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def main() -> None:
    pidfile = OUT / "ring_angles_followup.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("각도 후속 실행기가 이미 살아 있다")
    pidfile.write_text(str(os.getpid()))
    available = [n for n, row in json.loads((OUT / "helios/input_manifest.json").read_text())["sequences"].items() if row["available"]]
    required = []
    for model in ["author", "v2_s1", "v2_s2", "v2_s3"]:
        required += [OUT / "seqot" / model / "native/report.json"]
        required += [OUT / "helios" / model / "desc" / name / "done.json" for name in available]
        required += [OUT / f"imlpr/k{k}" / model / f"native_batch1_{seq}.json" for k in (15, 25) for seq in ("DCC01", "DCC02", "DCC03")]
    while True:
        ready = sum(p.exists() for p in required)
        (OUT / "ring_angles_followup_status.json").write_text(json.dumps({"waiting_learned_completed": ready, "required": len(required), "pid": os.getpid(), "unix_time": time.time()}, indent=2))
        if ready == len(required):
            break
        time.sleep(30)
    command = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=1", "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-e", "LOKY_MAX_CPU_COUNT=2", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "bash", "-lc", "ulimit -s 81920; exec nice -n 15 python3 -u scripts/_review_native_ring_angles.py run"]
    with (OUT / "ring_angles_run.log").open("a") as stream:
        subprocess.run(command, stdout=stream, stderr=stream, check=True)
    code = '''import json,numpy as np
from pathlib import Path
import sys
sys.path.insert(0,"scripts")
import _review_native_ring_verify as verify
r=Path("results/pdv2_review_native_20261105/ring_angles_endpoint_false")
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
  rows.append({"condition":key,"radius_m":float(radius),"original_endpoint_true":old["variants"][key][radius],"endpoint_false":row,"delta_r1_pp":100*(row["official_R@1"]-old["variants"][key][radius]["official_R@1"])})
(r/"comparison.json").write_text(json.dumps({"rows":rows,"checks_passed":True,"selection":"사전고정각도요인대조;46재평가에적용안함"},ensure_ascii=False,indent=2))
'''
    subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-c", code], check=True)
    (OUT / "ring_angles_followup_status.json").write_text(json.dumps({"complete": True, "pid": os.getpid(), "unix_time": time.time()}, indent=2))


if __name__ == "__main__":
    main()
