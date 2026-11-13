#!/usr/bin/env python3
"""원래 로더의 점 순서·좌표·강도를 무손실로 묶어 CPU 계산에 전달한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
from _review_selection_bev import read, write, sha
from _review_selection_full_inputs import OUT, INPUT, cache


def home_for(role: str, sensor: str, seq: str) -> Path:
    """이번 실험에 한정된 임시 무손실 입력 위치를 반환한다."""
    return INPUT / "rawpack" / role / f"{sensor}_{seq}"


class PackedLoader:
    """원래 loader 결과의 float32 바이트를 같은 raw ID로 돌려준다."""
    def __init__(self, home: Path):
        info = read(home / "done.json")
        self.n_raw = info["n_raw"]
        self.raw = np.load(home / "raw_ids.npy")
        self.offsets = np.load(home / "offsets.npy")
        self.points = np.memmap(home / "points.bin", mode="r", dtype=np.float32).reshape(-1, 4)
        self.lookup = {int(s): i for i, s in enumerate(self.raw)}

    def __len__(self):
        return self.n_raw

    def __getitem__(self, sid):
        i = self.lookup[int(sid)]
        return {"points": self.points[self.offsets[i]:self.offsets[i+1]]}


def build(role: str, sensor: str, seq: str) -> None:
    """원래 KISS 과거 창에 필요한 스캔만 원래 로더로 읽고 그대로 저장한다."""
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    from _pd_build_test_caches import _make_loader
    import _imlpr_official_riv as p
    home = home_for(role, sensor, seq)
    home.mkdir(parents=True, exist_ok=True)
    if (home / "done.json").exists():
        return
    ids = np.load(cache(sensor, seq, role))["scan_ids"]
    loader = _make_loader(sensor, seq)
    if p.SENSORS[sensor]["accumulate"]:
        track = p.Track("test_odom", sensor, seq, loader)
        raw = np.unique(np.concatenate([track.window(int(s)) for s in ids]))
    else:
        raw = ids
    np.save(home / "raw_ids.npy", raw)
    progress = home / "progress.json"
    state = read(progress) if progress.exists() else dict(completed=0, n_points=0, seconds=0)
    offsets_path = home / "offsets.npy"
    if state["completed"] and not state.get("data_fsync_before_cursor", False):
        # 구버전 중간 기록은 원시 전체 접두와 대조한 뒤에만 재개한다.
        old_offsets = np.load(offsets_path, mmap_mode="r")
        old_points = np.memmap(home / "points.bin", mode="r", dtype=np.float32).reshape(-1, 4)
        assert old_offsets[state["completed"]] == state["n_points"]
        assert len(old_points) >= state["n_points"]
        for i in range(state["completed"]):
            original = np.ascontiguousarray(loader[int(raw[i])]["points"])
            preserved = old_points[old_offsets[i]:old_offsets[i+1]]
            assert original.shape == preserved.shape and original.tobytes() == preserved.tobytes(), i
        write(home / "legacy_prefix_verified.json", dict(completed_raw_rows=state["completed"],
            all_source_bytes_exact=True, checked_unix=time.time()))
        del old_offsets, old_points
    offsets = np.lib.format.open_memmap(offsets_path, mode="r+" if offsets_path.exists() else "w+", dtype=np.int64, shape=(len(raw)+1,))
    offsets[0] = 0
    tick = time.time()
    path = home / "points.bin"
    with path.open("r+b" if path.exists() else "w+b") as handle:
        handle.truncate(state["n_points"] * 16)
        handle.seek(state["n_points"] * 16)
        for i in range(state["completed"], len(raw)):
            points = np.asarray(loader[int(raw[i])]["points"])
            assert points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 4
            np.ascontiguousarray(points).tofile(handle)
            offsets[i+1] = offsets[i] + len(points)
            if (i+1) % 128 == 0 or i+1 == len(raw):
                handle.flush(); offsets.flush()
                os.fsync(handle.fileno())
                with offsets_path.open("rb") as offset_stream:
                    os.fsync(offset_stream.fileno())
                write(progress, dict(completed=i+1,total=len(raw),n_points=int(offsets[i+1]),
                    data_fsync_before_cursor=True,seconds=state["seconds"]+time.time()-tick))
                print("RAWPACK",role,sensor,seq,i+1,len(raw),round(time.time()-tick,1),flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    # 작은 offset 배열은 mmap을 닫은 뒤 원자적으로 다시 저장하여
    # 해시 대상의 쓰기 완료 순서를 명확히 한다.
    final_offsets=np.array(offsets,copy=True)
    offsets.flush()
    del offsets
    temp=offsets_path.with_suffix(".complete.tmp.npy")
    np.save(temp,final_offsets)
    with temp.open("rb") as stream:os.fsync(stream.fileno())
    temp.replace(offsets_path)
    np.testing.assert_array_equal(np.load(offsets_path),final_offsets)
    info = dict(role=role,sensor=sensor,sequence=seq,n_raw=len(loader),packed_raw_rows=len(raw),n_keyframes=len(ids),
                cache_sha256=sha(cache(sensor,seq,role)),raw_ids_sha256=sha(home/"raw_ids.npy"),
                offsets_sha256=sha(offsets_path),points_sha256=sha(path),points_bytes=path.stat().st_size,
                dtype="float32",channels=4,source="original _make_loader points; no filter/normalization/coordinate/intensity change")
    assert sha(offsets_path)==info["offsets_sha256"], "offset 해시가 안정되기 전에는 완료로 표시하지 않는다"
    write(home / "done.json", info)
    packed = PackedLoader(home)
    checked = []
    for row in sorted(set([0,len(raw)//4,len(raw)//2,len(raw)-1])):
        sid = int(raw[row])
        np.testing.assert_array_equal(packed[sid]["points"], loader[sid]["points"])
        checked.append(sid)
    write(home / "verified.json", dict(all_arrays_exact=True,checked_raw_ids=checked,done_sha256=sha(home/"done.json")))


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("val","test"), required=True)
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    sensor, seq = args.sequence.split("/")
    build(args.role,sensor,seq)
