#!/usr/bin/env python3
"""서버에서 NCLT 압축 파일을 색인하고 DiSCO의3m 표본만 보존한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 -u scripts/_review_remaining_native_disco_data.py fetch
압축 원본 전체를 디스크에 저장하지 않는다. 원본 파일은 삭제하지 않는다.
블록 읽기는 재시도하고, 색인 중에는 1GiB마다 부분 색인을 저장해 재실행 시 이어서 쓴다.
재개 동작은 `selftest-resume`이 합성 tar.gz로 확인한다.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_remaining_native_20261106/disco"
S3 = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
DATES = ["2012-06-15", "2012-05-26", "2012-08-20", "2012-09-28", "2012-10-28"]
CHECKPOINT_BYTES = 1024**3
BLOCK_ATTEMPTS = 8


def write(path: Path, value: dict) -> None:
    """상태를 원자적으로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sample(names: list[str], gt_path: Path, upper_bound: bool = True) -> list[dict]:
    """공식 생성기의 최근접 자세와3≤거리<3.5m 표본을 재생한다."""
    gt = np.loadtxt(gt_path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 6))
    names = sorted(names, key=lambda name: int(Path(name).stem))
    timestamps = np.array([int(Path(n).stem) for n in names], dtype=np.int64)
    j = np.clip(np.searchsorted(gt[:, 0], timestamps), 1, len(gt)-1)
    j -= (timestamps - gt[j-1, 0]) < (gt[j, 0] - timestamps)
    previous = None
    rows = []
    for name, ts, row in zip(names, timestamps, gt[j]):
        xy = row[1:3]
        if not np.isfinite(xy).all():
            continue
        if previous is None:
            previous = xy
        distance = np.sqrt(np.sum((xy-previous)**2))
        if distance >= 3 and (not upper_bound or distance < 3.5):
            rows.append({"name": name, "timestamp": int(ts), "xy": xy.tolist(),
                         "yaw": float(row[3]),
                         "test_region": bool(-50 < xy[0] < 150 and -250 < xy[1] < 150)})
            previous = xy
    return rows


class HTTPRange(io.RawIOBase):
    """최대64MiB 캐시로 HTTPS 범위를 읽는 seek 가능한 파일이다."""

    def __init__(self, url: str, date: str):
        self.url, self.date = url, date
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
        response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=40)
        response.raise_for_status()
        assert response.status_code == 206
        self.length = int(response.headers["Content-Range"].split("/")[-1])
        self.etag = response.headers.get("ETag")
        self.position = 0
        self.blocksize = 16 * 1024 * 1024
        self.cache_start, self.cache = -1, b""
        self.blocks = OrderedDict()
        self.received = 0
        self.retries = 0
        self.started = time.time()
        self.stage = "indexing"

    def _block(self, start: int, end: int) -> bytes:
        """범위 응답을 검사하고 일시 오류는 대기 후 다시 받는다. 원본 변경은 재시도하지 않는다."""
        for attempt in range(BLOCK_ATTEMPTS):
            try:
                response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
                response.raise_for_status()
                if response.headers.get("ETag") != self.etag:
                    raise RuntimeError(f"원본 ETag 변경: {response.headers.get('ETag')} != {self.etag}")
                if response.status_code != 206 or \
                        response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.length}":
                    raise OSError(f"범위 응답 불일치: {response.status_code} {response.headers.get('Content-Range')}")
                body = response.content
                if len(body) != end - start + 1:
                    raise OSError(f"짧은 응답 {len(body)}/{end - start + 1}")
                return body
            except (requests.RequestException, OSError) as exc:
                self.retries += 1
                write(OUT / f"download_retry_{self.date}.json", {"date": self.date, "start": start,
                      "attempt": attempt + 1, "retries_total": self.retries,
                      "error": f"{type(exc).__name__}: {exc}"[:500], "unix": time.time()})
                if attempt + 1 == BLOCK_ATTEMPTS:
                    raise
                time.sleep(min(300, 15 * 2**attempt))
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.length)
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.length - self.position
        pieces = []
        remaining = min(size, self.length - self.position)
        while remaining > 0:
            start = (self.position // self.blocksize) * self.blocksize
            if start != self.cache_start and start in self.blocks:
                self.cache_start, self.cache = start, self.blocks.pop(start)
                self.blocks[start] = self.cache
            if start != self.cache_start:
                end = min(start + self.blocksize, self.length) - 1
                self.cache = self._block(start, end)
                self.cache_start = start
                self.blocks[start] = self.cache
                if len(self.blocks) > 4:
                    self.blocks.popitem(last=False)
                self.received += len(self.cache)
                write(OUT / f"download_status_{self.date}.json", {"stage": self.stage, "date": self.date,
                      "compressed_position": start, "archive_bytes": self.length,
                      "network_bytes": self.received, "block_retries": self.retries,
                      "elapsed_s": time.time() - self.started})
            offset = self.position - start
            count = min(remaining, len(self.cache) - offset)
            pieces.append(self.cache[offset:offset+count])
            remaining -= count
            self.position += count
        return b"".join(pieces)

    def readinto(self, buffer) -> int:
        chunk = self.read(len(buffer))
        buffer[:len(chunk)] = chunk
        return len(chunk)


class SelftestInterrupt(Exception):
    """자체 시험에서 색인 도중 중단을 흉내 낸다."""


def index_members(compressed, dest: Path, url: str, etag: str, interrupt_after: int | None = None) -> tuple[dict, dict]:
    """tar 머리를 훑어 velodyne_sync 위치를 모은다. 1GiB마다 부분 색인과 다음 머리 위치를 저장한다.

    큰 멤버(velodyne_hits.bin)를 건너뛰는 동안에도 같은 간격으로 색인을 넓혀 저장한다.
    부분 상태가 같은 URL·ETag이면 그 색인을 읽고 저장된 다음 머리부터 이어 읽는다.
    """
    import tarfile
    partial_index, partial_meta = dest / "gzip.partial.index", dest / "members.partial.json"
    members, start, resumed = {}, 0, None
    if partial_index.exists() and partial_meta.exists():
        state = json.loads(partial_meta.read_text())
        if state["url"] == url and state["etag"] == etag:
            compressed.import_index(str(partial_index))
            members, start = state["members"], int(state["next_header"])
            resumed = {"next_header": start, "members": len(members), "checkpoints": state["checkpoints"]}
    checkpoints = resumed["checkpoints"] if resumed else 0

    def save(next_header: int) -> None:
        nonlocal checkpoints
        tmp = dest / "gzip.partial.index.tmp"
        compressed.export_index(str(tmp))
        tmp.replace(partial_index)
        checkpoints += 1
        write(partial_meta, {"url": url, "etag": etag, "next_header": next_header,
                             "checkpoints": checkpoints, "members": members})
        if interrupt_after is not None and checkpoints >= interrupt_after:
            raise SelftestInterrupt(checkpoints)

    compressed.seek(start)
    saved = start
    tar = tarfile.open(fileobj=compressed, mode="r:")
    for item in tar:
        if item.isfile() and "/velodyne_sync/" in item.name and item.name.endswith(".bin"):
            members[item.name] = {"offset": item.offset_data, "size": item.size}
        if item.size >= CHECKPOINT_BYTES:
            for position in range(item.offset_data + CHECKPOINT_BYTES, tar.offset, CHECKPOINT_BYTES):
                compressed.seek(position)
                save(tar.offset)
            saved = tar.offset
        elif tar.offset - saved >= CHECKPOINT_BYTES:
            save(tar.offset)
            saved = tar.offset
    return members, {"resumed": resumed, "checkpoints": checkpoints}


def fetch(date: str) -> None:
    """원본 tar의 순서와 무관하게 정확 표본을 고르고 필요한 스캔만 저장한다."""
    import indexed_gzip
    import shutil
    dest = OUT / "native_data" / date
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / "complete.json"
    if done.exists():
        return
    gt = dest / "groundtruth.csv"
    if not gt.exists():
        response = requests.get(f"{S3}/ground_truth/groundtruth_{date}.csv", timeout=90)
        response.raise_for_status()
        tmp = dest / "groundtruth.csv.tmp"
        tmp.write_bytes(response.content)
        tmp.replace(gt)
    url = f"{S3}/velodyne_data/{date}_vel.tar.gz"
    remote = HTTPRange(url, date)
    compressed = indexed_gzip.IndexedGzipFile(fileobj=remote, spacing=16 * 1024 * 1024)
    index = dest / "gzip.index"
    members_path = dest / "members.json"
    index_state = {"reused_complete_index": False}
    if index.exists() and members_path.exists():
        state = json.loads(members_path.read_text())
        assert state["url"] == url and state["etag"] == remote.etag, "색인 이후 원본 ETag가 바뀜"
        compressed.import_index(str(index))
        members = state["members"]
        index_state["reused_complete_index"] = True
    else:
        members, partial = index_members(compressed, dest, url, remote.etag)
        index_state.update(partial)
        tmp = dest / "gzip.index.tmp"
        compressed.export_index(str(tmp))
        tmp.replace(index)
        write(members_path, {"url": url, "etag": remote.etag, "members": members})
    if not members:
        raise RuntimeError(f"{date}: velodyne_sync 멤버 0개; 빈 입력으로 진행하지 않음")
    rows = sample(list(members), gt)
    alternate = sample(list(members), gt, upper_bound=False)
    write(dest / "selection.json", {"all_scans": len(members), "selected": len(rows), "rows": rows,
          "rule": "official SUBMAP_INTERVAL_TEST=3; displacement in [3,3.5), nearest pose; first CSV row skipped as header=0"})
    write(dest / "selection_no_upper_bound.json", {"all_scans": len(members), "selected": len(alternate), "rows": alternate,
          "rule": "3m 이상이면 다음 표본 선택; 상한으로 경로 선택이 멈추는 현상을 분리한 진단이며 공개 원본 조건과 구별"})
    union = {row["name"]: row for row in rows + alternate}
    required = sum(members[name]["size"] for name in union)
    assert shutil.disk_usage(dest).free > required + 20 * 1024**3, "서버 여유20GiB를 유지할 수 없음"
    remote.stage = "extracting_selected"
    kept = []
    for row in sorted(union.values(), key=lambda item: members[item["name"]]["offset"]):
        member = members[row["name"]]
        path = dest / "velodyne_sync" / Path(row["name"]).name
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            assert path.stat().st_size == member["size"]
        else:
            compressed.seek(member["offset"])
            blob = compressed.read(member["size"])
            assert len(blob) == member["size"]
            path.write_bytes(blob)
        kept.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write(done, {"date": date, "selected": len(rows), "diagnostic_selected": len(alternate), "union_selected": len(union), "retained_bytes": required,
                 "archive_bytes": remote.length, "network_bytes": remote.received, "block_retries": remote.retries,
                 "index_state": index_state, "elapsed_s": time.time()-remote.started, "raw_sha256": kept})
    compressed.close()
    remote.close()


def selftest_resume(root: Path) -> dict:
    """합성 tar.gz에서 중단 후 재개한 색인·추출이 한 번에 읽은 결과와 같은지 확인한다."""
    import gzip
    import indexed_gzip
    import tarfile
    global CHECKPOINT_BYTES
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "synthetic_vel.tar.gz"
    rng = np.random.default_rng(0)
    with gzip.open(archive, "wb", compresslevel=1) as handle, tarfile.open(fileobj=handle, mode="w:") as tar:
        def add(name: str, blob: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
        add("2099-01-01/velodyne_hits.bin", rng.integers(0, 8, 40 * 1024**2, dtype=np.uint8).tobytes())
        expected = {}
        for k in range(120):
            blob = rng.integers(0, 256, 150_000 + 997 * k, dtype=np.uint8).tobytes()
            name = f"2099-01-01/velodyne_sync/{1000000 + k}.bin"
            add(name, blob)
            expected[name] = hashlib.sha256(blob).hexdigest()
    with tarfile.open(archive, "r:gz") as direct:
        truth = {m.name: {"offset": m.offset_data, "size": m.size} for m in direct
                 if "/velodyne_sync/" in m.name}
    old, CHECKPOINT_BYTES = CHECKPOINT_BYTES, 4 * 1024**2
    cases = []
    try:
        # 5회: 큰 멤버를 건너뛰는 도중, 12회: 작은 멤버 구간에서 중단한다.
        for interrupt_after in [5, 12]:
            dest = root / f"dest_{interrupt_after}"
            dest.mkdir(exist_ok=True)
            for path in dest.glob("*"):
                path.unlink()
            interrupted = False
            with open(archive, "rb") as raw:
                first = indexed_gzip.IndexedGzipFile(fileobj=raw, spacing=1024**2)
                try:
                    index_members(first, dest, "synthetic", "etag", interrupt_after=interrupt_after)
                except SelftestInterrupt:
                    interrupted = True
                first.close()
            partial = json.loads((dest / "members.partial.json").read_text())
            mismatched = []
            with open(archive, "rb") as raw:
                second = indexed_gzip.IndexedGzipFile(fileobj=raw, spacing=1024**2)
                members, state = index_members(second, dest, "synthetic", "etag")
                for name in sorted(members, key=lambda key: members[key]["offset"]):
                    second.seek(members[name]["offset"])
                    if hashlib.sha256(second.read(members[name]["size"])).hexdigest() != expected[name]:
                        mismatched.append(name)
                second.close()
            cases.append({"interrupt_after_checkpoints": interrupt_after, "interrupted": interrupted,
                          "partial_members_at_interrupt": len(partial["members"]),
                          "partial_next_header": partial["next_header"], "resumed": state["resumed"],
                          "members": len(members), "members_equal_direct_listing": members == truth,
                          "extracted_sha_mismatches": len(mismatched),
                          "passed": interrupted and state["resumed"] is not None and members == truth
                          and not mismatched})
    finally:
        CHECKPOINT_BYTES = old
    report = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "sync_members": len(truth),
              "cases": cases, "passed": all(case["passed"] for case in cases)}
    write(root / "selftest_resume.json", report)
    return report


def main() -> None:
    """보유 자료의 목록 준비 또는 서버의 부족 자료 확보를 수행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fetch", "local-plan", "selftest-resume"])
    parser.add_argument("--dates", nargs="+", default=DATES)
    parser.add_argument("--selftest-root", type=Path, default=Path("/tmp/disco_resume_selftest"))
    args = parser.parse_args()
    if args.stage == "selftest-resume":
        report = selftest_resume(args.selftest_root)
        print(json.dumps(report, ensure_ascii=False))
        assert report["passed"]
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "local-plan":
        root = Path("/mnt/d/NSD_datasets/nclt/2012-03-17")
        rows = sample([str(p.relative_to(root)) for p in (root/"velodyne_sync").glob("*.bin")], root/"groundtruth_2012-03-17.csv")
        alternate = sample([str(p.relative_to(root)) for p in (root/"velodyne_sync").glob("*.bin")], root/"groundtruth_2012-03-17.csv", upper_bound=False)
        write(OUT/"local_selection.json", {"date": "2012-03-17", "selected": len(rows), "rows": rows})
        write(OUT/"local_selection_no_upper_bound.json", {"date": "2012-03-17", "selected": len(alternate), "rows": alternate})
        names = sorted({row["name"] for row in rows + alternate})
        (OUT/"local_files.txt").write_text("\n".join(names)+"\n")
        print("selected",len(rows),"no_upper_bound",len(alternate),"union",len(names))
    else:
        try:
            for date in args.dates:
                fetch(date)
            for date in args.dates:
                write(OUT/f"download_status_{date}.json", {"stage": "complete", "date": date})
        except Exception as exc:
            write(OUT/f"download_error_{date}.json", {"type": type(exc).__name__, "message": str(exc)})
            raise


if __name__ == "__main__":
    main()
