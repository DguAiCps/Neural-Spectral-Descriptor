#!/usr/bin/env python3
"""원래 RIV 계산을 서로 다른 키프레임에 나누어 재개 가능하게 실행한다."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

_CTX = None


def initialize(sensor: str, seq: str, home: str) -> None:
    """각 CPU 프로세스에서 독립 로더와 읽기 캐시만 준비한다."""
    global _CTX
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    import cv2
    cv2.setNumThreads(1)
    from _pd_build_test_caches import _make_loader
    from _review_selection_full_test_relay import imlpr_module
    from _review_selection_full_riv_cache import ScanCache
    p = imlpr_module()
    loader = ScanCache(_make_loader(sensor, seq))
    track = p.Track("test_odom", sensor, seq, loader) if p.SENSORS[sensor]["accumulate"] else None
    _CTX = p, sensor, seq, Path(home), loader, track


def block(rows: list) -> dict:
    """서로 겹치지 않는 키프레임의 원래 계산과 원자적 저장을 수행한다."""
    p, sensor, seq, home, loader, track = _CTX
    tick = time.time()
    count = 0
    for pos, sid in rows:
        target = home / f"{pos:06d}.npy"
        if target.exists():
            continue
        image, _, _ = p.riv_for("test_odom", sensor, seq, pos, sid, loader, track)
        temporary = home / f".{pos:06d}.{os.getpid()}.tmp.npy"
        np.save(temporary, image)
        temporary.replace(target)
        count += 1
    return dict(completed_positions=[pos for pos, _ in rows],new_frames=count,seconds=time.time()-tick,
                pid=os.getpid(),cache_hits=loader.hits,cache_misses=loader.misses,cache_bytes=loader.bytes)


def generate(sensor: str, seq: str, home: Path, ids: np.ndarray, processes: int = 3) -> None:
    """동일 계산을 최대3개 독립 CPU에 나누며 완성 파일만 재사용한다."""
    from _review_selection_bev import read, write, sha
    from _review_selection_full_inputs import OUT
    proof = read(OUT / "riv_cache_audit.json")
    root = Path(__file__).resolve().parents[1]
    assert proof["all_arrays_exact"]
    assert proof["implementation_sha256"] == sha(root / "scripts/_review_selection_full_riv_cache.py")
    assert proof["original_sha256"] == sha(root / "scripts/_imlpr_official_riv.py")
    missing = [(i, int(sid)) for i, sid in enumerate(ids) if not (home / f"{i:06d}.npy").exists()]
    blocks = [missing[i:i+32] for i in range(0, len(missing), 32)]
    completed = len(ids) - len(missing)
    tick = time.time()
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes, initializer=initialize, initargs=(sensor, seq, str(home))) as pool:
        for result in pool.imap_unordered(block, blocks):
            completed += len(result["completed_positions"])
            write(home / "progress.json", dict(completed=completed,total=len(ids),seconds=time.time()-tick,
                                               processes=processes,last_block=result))
            print("RIV_PARALLEL",sensor,seq,completed,len(ids),round(time.time()-tick,1),flush=True)
