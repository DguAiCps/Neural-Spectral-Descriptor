#!/usr/bin/env python3
"""공통 KISS 검증 입력을 기존 공식 생성기로 별도 경로에 준비한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
from _review_selection_bev import VAL, read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
INPUT = OUT / "inputs"
ODOM = ROOT / "data/preprocessed_v2_odom"


def cache(sensor: str, seq: str, role: str = "val") -> Path:
    """공통 운영 캐시를 찾는다."""
    found = list(ODOM.glob(f"cache_????????_{sensor}_{role}_{seq}.npz"))
    assert len(found) == 1, found
    return found[0]


def configure(method: str):
    """test_odom의 KISS 분기를 검증9 목록·고유 경로에 연결한다."""
    if method == "helios":
        import _helios_official_prep as p
        p.WORK = INPUT / method
        p.DATA = p.WORK / "data"
    elif method == "imlpr":
        import _imlpr_official_riv as p
        p.WORK = INPUT / method
        p.DATA = p.WORK / "data"
    elif method == "disco":
        import _disco_official_prep as p
        p.WORK = INPUT / method
        p.POLAR = p.WORK / "polar"
    else:
        raise ValueError(method)
    p.ROLES = {"test_odom": (VAL, ODOM, "val_", "test_odom" if method == "imlpr" else "poses_gt")}
    return p


def link(source: Path, target: Path) -> None:
    """동일 원시 스캔에서 만든 입력만 재사용한다."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)


def index(method: str) -> None:
    """입력 정책과 원시 ID에 의한 재사용을 기록한다."""
    marker = INPUT / method / "index_done.json"
    if marker.exists():
        return
    p = configure(method)
    if method == "helios":
        p.cmd_index(["test_odom"], 10.0, "helipr")
    else:
        p.cmd_index("test_odom")
    rows = []
    for sensor, seq in VAL:
        ids = np.load(cache(sensor, seq))["scan_ids"]
        reused = 0
        if method == "helios":
            old = ROOT / f"_external/helios_official_pdv2/data/val/{sensor}_{seq}/LiDAR"
            new = p.seq_dir("test_odom", sensor, seq) / "LiDAR"
            if sensor != "helipr":
                for sid in ids:
                    source = old / f"{sid:06d}.bin"
                    if source.exists():
                        link(source, new / source.name)
                        reused += 1
        elif method == "imlpr":
            old = ROOT / f"_external/imlpr_official_pdv2/data/validation/{sensor}_{seq}"
            new = p.role_dir("test_odom", sensor, seq)
            if not p.SENSORS[sensor]["accumulate"]:
                oldids = np.load(old / "index.npz")["scan_ids"]
                lookup = {int(s): i for i, s in enumerate(oldids)}
                for i, sid in enumerate(ids):
                    if int(sid) in lookup:
                        source = old / f"{lookup[int(sid)]:06d}.npy"
                        if source.exists():
                            link(source, new / f"{i:06d}.npy")
                            reused += 1
        else:
            old = ROOT / f"_external/disco_official_pdv2/polar/val/{sensor}_{seq}"
            new = p.store_dir("test_odom", sensor, seq)
            oldids = np.load(old / "index.npz")["scan_ids"]
            lookup = {int(s): i for i, s in enumerate(oldids)}
            bits = np.lib.format.open_memmap(new / "bits.npy", mode="w+", dtype=np.uint8,
                                             shape=(len(ids), p.NBYTES))
            oldbits = np.load(old / "bits.npy", mmap_mode="r")
            oldhits = np.load(old / "nhits.npy")
            done = np.zeros(len(ids), bool)
            hits = np.full(len(ids), -1, np.int64)
            for i, sid in enumerate(ids):
                if int(sid) in lookup:
                    j = lookup[int(sid)]
                    bits[i], hits[i], done[i] = oldbits[j], oldhits[j], True
                    reused += 1
            bits.flush()
            np.save(new / "done.npy", done)
            np.save(new / "nhits.npy", hits)
        rows.append(dict(sequence=f"{sensor}/{seq}", total=len(ids), reused=reused, new=len(ids)-reused))
    write(marker, dict(created_unix=time.time(), rows=rows,
        role_alias="test_odom은 기존 코드의 KISS 분기를 쓰기 위한 내부 이름이며 실제 목록은 검증9이다",
        input_rule="같은 raw scan ID의 자세 독립 입력만 재사용. HeLiPR deskew/submap은 KISS로 새 생성"))


def seqot(sensor: str, seq: str) -> None:
    """공식 가운데 창에 필요한 누락 거리 영상만 생성한다."""
    import _seqot_official_prep as p
    from _pd_build_test_caches import _make_loader
    home = INPUT / "seqot" / f"{sensor}_{seq}"
    if (home / "done.json").exists():
        return
    mm, raw, _, _, nraw, ext, eraw = p.open_store("val", sensor, seq, ext=True)
    ids = np.load(cache(sensor, seq))["scan_ids"]
    need = p.centered_raw_ids(ids, nraw)
    missing = np.setdiff1d(need, np.union1d(raw, eraw))
    home.mkdir(parents=True, exist_ok=True)
    np.save(home / "missing_ids.npy", missing)
    arr = np.lib.format.open_memmap(home / "missing.npy", mode="r+" if (home / "missing.npy").exists() else "w+",
                                   dtype=mm.dtype, shape=(len(missing), *mm.shape[1:]))
    prog = home / "progress.json"
    start = read(prog)["completed"] if prog.exists() else 0
    loader = _make_loader(sensor, seq) if len(missing) else None
    tick = time.time()
    for i in range(start, len(missing)):
        arr[i] = p.range_image(sensor, loader[int(missing[i])]["points"])
        if i % 100 == 0 or i + 1 == len(missing):
            arr.flush()
            write(prog, dict(completed=i+1, total=len(missing), elapsed_s=time.time()-tick))
    write(home / "done.json", dict(base_rows=len(raw), extension_rows=len(eraw), new_rows=len(missing),
        total_needed=len(need), cache_sha256=sha(cache(sensor, seq))))


class SeqSource:
    """검증 KISS키프레임과 원래+추가 거리영상의 읽기 전용 뷰다."""
    def __init__(self, sensor: str, seq: str):
        import _seqot_official_prep as p
        mm, raw, _, _, self.n_raw, ext, eraw = p.open_store("val", sensor, seq, ext=True)
        home = INPUT / "seqot" / f"{sensor}_{seq}"
        assert (home / "done.json").exists(), home
        self.arrays = [mm, ext, np.load(home / "missing.npy", mmap_mode="r")]
        self.lookup = {int(r): (a, j) for a, raws in enumerate([raw, eraw, np.load(home / "missing_ids.npy")])
                       for j, r in enumerate(raws)}
        z = np.load(cache(sensor, seq))
        self.kf, self.kf_poses = z["scan_ids"], z["poses_gt"]

    def exists(self, r: int) -> bool:
        return 0 <= r < self.n_raw

    def rows(self, raws):
        return np.stack([self.arrays[self.lookup[int(r)][0]][self.lookup[int(r)][1]] for r in raws])


def ringsharp(sensor: str, seq: str) -> None:
    """공식 20x160x160 점유 입력을 ID 단위로 재사용·보충한다."""
    import _ringsharp_official as r
    home = INPUT / "ringsharp" / f"{sensor}_{seq}"
    if (home / "done.json").exists():
        return
    home.mkdir(parents=True, exist_ok=True)
    ids = np.load(cache(sensor, seq))["scan_ids"]
    z = np.load(r.VAL_BEV / f"{sensor}_{seq}.npz")
    lookup = {int(s): i for i, s in enumerate(z["scan_ids"])}
    packed = np.lib.format.open_memmap(home / "packed.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(ids), z["packed"].shape[1]))
    old = z["packed"]
    missing = []
    for i, sid in enumerate(ids):
        if int(sid) in lookup:
            packed[i] = old[lookup[int(sid)]]
        else:
            missing.append(i)
    if missing:
        base, loader = r._base(), r.rol.make_loader(sensor, seq)
        batches = r.bev_batches_from_loader(base, loader, ids[missing], r.rol.ringsharp_bounds(sensor))
        pos = 0
        for batch in batches:
            bits = np.packbits(batch.numpy().reshape(len(batch), -1) > 0, axis=1)
            packed[np.asarray(missing[pos:pos+len(batch)])] = bits
            pos += len(batch)
    packed.flush()
    write(home / "done.json", dict(total=len(ids), reused=len(ids)-len(missing), new=len(missing),
                                   cache_sha256=sha(cache(sensor, seq))))


def work(method: str, sequence: str, shard: int = 0, nshards: int = 1) -> None:
    sensor, seq = sequence.split("/")
    if method in ("seqot", "ringsharp"):
        globals()[method](sensor, seq)
    else:
        p = configure(method)
        p.cmd_gen("test_odom", sequence, shard, nshards)


def run(method: str, workers: int) -> None:
    """CPU 전처리 작업과 결과를 재개 가능한 파일로 보존한다."""
    if method in ("helios", "imlpr", "disco"):
        index(method)
    if method == "disco":
        return  # 누락 GPU 생성은 GPU 작업열에서만 수행한다.
    jobs = [(s+"/"+q, shard, workers) for s,q in VAL for shard in range(workers)] if method in ("helios", "imlpr") else [(s+"/"+q,0,1) for s,q in VAL]
    status = INPUT / method / "runner_status.json"
    write(status, dict(status="running", pid=os.getpid(), jobs=len(jobs), workers=workers, started_unix=time.time()))

    def launch(job):
        seq, shard, ns = job
        log = OUT / "logs" / f"input_{method}_{seq.replace('/', '_')}_{shard}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as out:
            code = subprocess.call([sys.executable,"-u",__file__,"worker","--method",method,
                "--sequence",seq,"--shard",str(shard),"--nshards",str(ns)], stdout=out, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        if code:
            raise RuntimeError(f"{job}: rc {code}, {log}")
        return dict(sequence=seq, shard=shard)

    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(launch, jobs):
            completed.append(result)
            write(status, dict(status="running", pid=os.getpid(), jobs=len(jobs), completed=completed, updated_unix=time.time()))
    write(status, dict(status="complete", jobs=len(jobs), completed=completed, finished_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run","index","worker"))
    parser.add_argument("--method", required=True, choices=("helios","imlpr","disco","seqot","ringsharp"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sequence")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "run":
        run(args.method, args.workers)
    elif args.action == "index":
        index(args.method)
    else:
        work(args.method, args.sequence, args.shard, args.nshards)
