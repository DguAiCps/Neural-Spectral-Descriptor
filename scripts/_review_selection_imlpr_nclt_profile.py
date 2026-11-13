#!/usr/bin/env python3
"""동결된 ImLPR 시험 입력의 CPU 시간만 고정 위치에서 관측한다."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import TEST, read, sha, write
from _review_selection_full_inputs import cache

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def main() -> None:
    """공식 원래 입력 함수를 바꾸지 않고 CPU 한 개의 구간 시간을 잰다."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import cv2
    import _imlpr_official_riv as riv

    cv2.setNumThreads(1)
    home = OUT / "imlpr_nclt_time_profile"
    plan = read(home / "plan.json")
    assert sha(OUT / "retained/selection_imlpr.json") == plan["selection_sha256"]
    assert sha(Path(riv.__file__)) == plan["implementation_sha256"]
    results = []
    for sensor, seq in TEST:
        if sensor != "nclt":
            continue
        ids = np.load(cache(sensor, seq, "test"))["scan_ids"]
        loader = riv.make_loader(sensor, seq)
        track = riv.Track("test_odom", sensor, seq, loader)
        actual_submap = track.submap
        current = {}

        def observed_submap(raw_id: int):
            tick = time.perf_counter()
            value = actual_submap(raw_id)
            current.update(submap_s=time.perf_counter() - tick,
                           points=len(value[0]), raw_scans=len(value[1]))
            return value

        track.submap = observed_submap
        for numerator in plan["position_numerators"]:
            pos = (len(ids) - 1) * numerator // plan["position_denominator"]
            path = home / f"{seq}_{pos:06d}.json"
            if path.exists():
                results.append(read(path))
                continue
            current = dict(sequence=f"{sensor}/{seq}", position=pos,
                           sequence_frames=len(ids), raw_id=int(ids[pos]))
            write(home / "state.json", dict(status="CPU_single_worker", pid=os.getpid(),
                  active=current, completed=len(results), total=27, updated_unix=time.time()))
            tick = time.perf_counter()
            image, _, _ = riv.riv_for("test_odom", sensor, seq, pos, int(ids[pos]), loader, track)
            current["total_s"] = time.perf_counter() - tick
            buffer = io.BytesIO()
            np.save(buffer, image)
            current.update(image_npy_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                           image_shape=list(image.shape), completed_unix=time.time())
            write(path, current)
            results.append(current)
            print(current, flush=True)
        del loader, track
    write(home / "report.json", dict(status="complete", rows=results, completed_unix=time.time(),
          plan_sha256=sha(home / "plan.json"), GPU_used=False, model_inference=False,
          note="공유 부하에서 고정 위치27개 관측; 전체 처리율·단독 효율을 뜻하지 않는다."))
    write(home / "state.json", dict(status="complete", completed=len(results), total=27,
          pid=os.getpid(), completed_unix=time.time()))


if __name__ == "__main__":
    main()
