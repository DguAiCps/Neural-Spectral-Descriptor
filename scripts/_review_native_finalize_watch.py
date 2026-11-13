#!/usr/bin/env python3
"""수집된 최종 원장을 독립 검산하고 제출 제안까지 자동으로 준비한다."""
import json
import os
from pathlib import Path
import subprocess
import time

from _review_native_pdv2_ringpp import OUT as RING, REPO, write_json, sha256

OUT = REPO / "results/pdv2_review_native_20261105"


def main() -> None:
    """GPU 작업을 건드리지 않고 완료 파일만 읽어 재개 가능한 검산을 한다."""
    finished = {}
    while True:
        try:
            if "ring" not in finished and (RING / "summary_verified.json").exists():
                subprocess.run(["python3", "scripts/_review_native_ringpp_geometry.py"], cwd=REPO, check=True)
                geometry = json.loads((RING / "geometry_independent_verification.json").read_text())
                assert geometry["complete"] and geometry["passed"] and geometry["common_queries"] == 195726
                summary = json.loads((RING / "summary_verified.json").read_text())
                row = summary["ringpp_maxfix"]["macro"]
                delta = summary["delta_previous_pp"]
                text = r"\paragraph{Deterministic voxel aggregation on the common benchmark.}" + "\n"
                text += "Replacing only the non-atomic voxel maximum with an exact maximum gives "
                text += ", ".join(f"{100*row[key]:.3f}\\% {label}" for key,label in [("AP","AP"),("max_F1","maximum F1"),("R@100P","R@100P"),("R@1","R@1")]) + ". "
                text += "The changes relative to the retained RING++ row are " + ", ".join(f"{delta[key]:+.3f}" for key in ["AP","max_F1","R@100P","R@1"]) + " percentage points, respectively. "
                text += "The 46 sequences, 195,726 common queries, past-only candidates, labels, and six-city aggregation are unchanged. Independent recomputation checks every candidate count and geometric label, as well as the ranking metrics and city aggregation. The original runtime and memory measurements do not measure this corrected implementation.\n"
                (OUT / "paper_proposal/ring_common46_addendum.tex").write_text(text)
                finished["ring"] = {"summary_sha256": sha256(RING / "summary_verified.json"), "geometry_sha256": sha256(RING / "geometry_independent_verification.json"), "unix_time": time.time()}
            regen = OUT / "helios_regenerated_velodyne"
            if "helios_regen" not in finished and all((regen / m / "native_partial_report.json").exists() for m in ["author","v2_s1","v2_s2","v2_s3"]):
                subprocess.run(["python3", "scripts/_review_native_helios_regen_eval.py", "verify"], cwd=REPO, check=True)
                subprocess.run(["python3", "scripts/_review_native_helios_regen_compare.py"], cwd=REPO, check=True)
                finished["helios_regen"] = {"comparison_sha256": sha256(regen / "comparison_report.json"), "unix_time": time.time()}
            state = {"pid": os.getpid(), "unix_time": time.time(), "finished": finished, "complete": len(finished)==2, "scope": "이 기록은 RING 공통46과 Velodyne 재생성의 완료 범위다. 전체 공개HeLiOS24쌍 완료 검산은 helios/full_release_final_verification.json에 별도로 기록한다."}
        except Exception as exc:
            state = {"pid": os.getpid(), "unix_time": time.time(), "finished": finished, "error": repr(exc), "complete": False}
        write_json(OUT / "finalize_status.json", state)
        with (OUT / "finalize_history.jsonl").open("a") as stream:
            stream.write(json.dumps(state, ensure_ascii=False)+"\n")
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if state["complete"]:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
