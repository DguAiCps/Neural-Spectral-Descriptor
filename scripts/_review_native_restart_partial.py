#!/usr/bin/env python3
"""재부팅 당시 HeLiOS 미완료 묶음에서 검증된 출력을 보존하고 부족분을 만든다."""
from pathlib import Path
import json
import time
import numpy as np

import _review_native_helios_regenerate as gen
from _helios_official_prep import run_toolbox, official_toolbox_config


def main() -> None:
    """마커 없는 묶음만 복구하며 기존 정상 출력 바이트를 변경하지 않는다."""
    seq = "Roundabout02-Velodyne"
    ids = gen.stamps(seq)
    root = gen.OUT / "prepared" / seq
    cfg = official_toolbox_config()
    assert cfg["Save"]["accumulatedSize"] == 1, cfg
    recovered = []
    for chunk in sorted(root.glob("chunk_*")):
        if (chunk / "done.json").exists():
            continue
        first = int(chunk.name.split("_")[1])
        needed = ids[first:first + 128]
        prior = {}
        invalid = []
        work = chunk / "restart_20261105_2248"
        bindir = work / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        for stamp in needed:
            output = chunk / "out" / (stamp + ".bin")
            valid = output.exists() and output.stat().st_size == 8192 * 16
            if valid:
                arr = np.fromfile(output, np.float32).reshape(8192, 4)
                valid = bool(np.isfinite(arr).all() and np.any(arr[:, :3]))
            if valid:
                prior[stamp] = gen.native.sha(output)
                continue
            if output.exists():
                saved = work / "interrupted_outputs" / output.name
                saved.parent.mkdir(exist_ok=True)
                assert not saved.exists()
                output.rename(saved)
                invalid.append({"timestamp": stamp, "bytes": saved.stat().st_size, "sha256": gen.native.sha(saved)})
            target = bindir / (stamp + ".bin")
            if not target.exists():
                target.symlink_to(gen.RAW / "Roundabout02/LiDAR/Velodyne" / target.name)
        gen.native.write(work / "before.json", {"preserved": prior, "invalid_preserved": invalid, "missing": len(needed)-len(prior), "unix_time": time.time()})
        output_root = run_toolbox(work, bindir, gen.RAW / "Roundabout02/LiDAR_GT/Velodyne_gt.txt", True)
        rows = []
        for stamp in needed:
            output = chunk / "out" / (stamp + ".bin")
            if stamp in prior:
                assert gen.native.sha(output) == prior[stamp]
            else:
                assert not output.exists()
                (output_root / output.name).rename(output)
            arr = np.fromfile(output, np.float32).reshape(8192, 4)
            assert np.isfinite(arr).all() and np.any(arr[:, :3])
            rows.append({"timestamp": stamp, "raw_sha256": gen.native.sha(chunk / "bin" / output.name), "sha256": gen.native.sha(output)})
        gen.native.write(chunk / "done.json", {"files": rows, "restart": str(work.relative_to(gen.OUT)), "preserved_outputs": len(prior)})
        recovered.append({"chunk": chunk.name, "preserved": len(prior), "new": len(needed)-len(prior)})
    gen.native.write(gen.OUT / "restart_partial_recovery.json", {"chunks": recovered, "unix_time": time.time(), "scope": "정상 출력 SHA 보존, 완료마커 없는 묶음의 누락/중단 파일만 공식 바이너리로 복구"})
    gen.generate(seq, False)


if __name__ == "__main__":
    main()
