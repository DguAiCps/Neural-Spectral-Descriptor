#!/usr/bin/env python3
"""ImLPR 저자 data.tar.gz를 표준입력 스트림으로 받아 목록과 선택 파일만 저장한다.

서버 컨테이너에서 실행한다. 압축 파일 전체를 서버에 복제하지 않는다.

    cat data.tar.gz | ssh joms-gpu docker exec -i -w /workspace/Neural-Spectral-Codec \
        ring_baseline python3 -u scripts/_review_imlpr_author_extract.py --attempt 01

- gzip은 zlib(wbits=31)로 직접 풀어 멤버별 CRC32·ISIZE 검사를 받는다.
- tar는 스트림 모드로 읽고 모든 항목의 이름·형식·크기·SHA-256을 목록에 남긴다.
- 절대경로, `..`, 링크·장치 파일은 저장하지 않는다. 선택 규칙은 SELECTION에 고정한다.
- 출력: results/pdv2_reproduction_orchestration_20261106/imlpr/author_data/attempt_<N>/
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import posixpath
import re
import sys
import tarfile
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/imlpr/author_data"
CHUNK = 8 << 20
GIB = 1 << 30
RESERVE_BYTES = 22 * GIB  # 사용자 규칙 20GiB + 여유 2GiB
DATA_BUDGET_BYTES = 24 * GIB
META_MAX_BYTES = 512 << 20
SELECTION = {
    "metadata_suffixes": [".pickle", ".pkl", ".txt", ".csv", ".json", ".yaml", ".yml", ".md", ".py", ".cfg", ".ini"],
    "metadata_max_bytes": META_MAX_BYTES,
    "data_suffixes": [".npy", ".npz", ".png", ".bin", ".pcd"],
    "mulran_component_regex": r"(?i)(^|[^a-z0-9])(dcc[_-]?0?[1-3])($|[^0-9])|mulran",
    "data_budget_bytes": DATA_BUDGET_BYTES,
    "reserve_free_bytes": RESERVE_BYTES,
    "rule": "메타데이터 확장자는 위치와 무관하게 저장, 데이터 확장자는 경로 구성요소가 MulRan DCC01-03 또는 mulran과 맞을 때만 저장",
}
MULRAN = re.compile(SELECTION["mulran_component_regex"])


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


class GzipStream:
    """압축 바이트를 해시하며 gzip 멤버를 차례로 푸는 읽기 전용 파일 객체."""

    def __init__(self, raw) -> None:
        self.raw = raw
        self.sha = hashlib.sha256()
        self.compressed = 0
        self.decompressed = 0
        self.members = 1
        self.d = zlib.decompressobj(31)
        self.pending = b""
        self.cur = b""
        self.pos = 0
        self.done = False

    def _next_block(self) -> bytes:
        while not self.done:
            if not self.pending:
                chunk = self.raw.read(CHUNK)
                if not chunk:
                    if not self.d.eof:
                        raise EOFError("gzip stream truncated before member trailer")
                    self.done = True
                    return b""
                self.sha.update(chunk)
                self.compressed += len(chunk)
                self.pending = chunk
            if self.d.eof:
                self.d = zlib.decompressobj(31)
                self.members += 1
            out = self.d.decompress(self.pending, 4 << 20)
            self.pending = self.d.unused_data if self.d.eof else self.d.unconsumed_tail
            if out:
                self.decompressed += len(out)
                return out
        return b""

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = 1 << 62
        parts = []
        while n > 0:
            if self.pos >= len(self.cur):
                self.cur, self.pos = self._next_block(), 0
                if not self.cur:
                    break
            take = self.cur[self.pos:self.pos + n]
            self.pos += len(take)
            n -= len(take)
            parts.append(take)
        return b"".join(parts)

    def drain(self) -> int:
        """tar 끝 표시 뒤 남은 자료까지 읽어 마지막 CRC를 검사하고 0이 아닌 바이트 수를 센다."""
        nonzero = len(self.cur[self.pos:]) - self.cur[self.pos:].count(0)
        self.cur, self.pos = b"", 0
        while True:
            block = self._next_block()
            if not block:
                return nonzero
            nonzero += len(block) - block.count(0)


def safe_name(name: str) -> tuple[str | None, str]:
    if not name or "\x00" in name or "\\" in name or any(ord(c) < 32 for c in name):
        return None, "invalid_characters"
    if name.startswith("/"):
        return None, "absolute_path"
    parts = name.split("/")
    if ".." in parts:
        return None, "parent_component"
    norm = posixpath.normpath(name)
    if norm in (".", "") or norm.startswith("../") or norm == ".." or norm.startswith("/"):
        return None, "unsafe_normalized"
    return norm, "ok"


def npy_header(head: bytes) -> dict | None:
    if not head.startswith(b"\x93NUMPY") or len(head) < 10:
        return None
    major = head[6]
    if major == 1:
        hlen, start = int.from_bytes(head[8:10], "little"), 10
    else:
        hlen, start = int.from_bytes(head[8:12], "little"), 12
    try:
        d = ast.literal_eval(head[start:start + hlen].decode("latin1"))
        return {"descr": str(d["descr"]), "shape": list(d["shape"]), "fortran_order": bool(d["fortran_order"])}
    except Exception as exc:  # 헤더 파싱 실패도 목록에 남긴다.
        return {"error": repr(exc)}


def classify(norm: str, size: int) -> str:
    suffix = posixpath.splitext(norm)[1].lower()
    if suffix in SELECTION["metadata_suffixes"]:
        return "metadata" if size <= META_MAX_BYTES else "metadata_too_large"
    if suffix in SELECTION["data_suffixes"] and any(MULRAN.search(p) for p in norm.split("/")[:-1] + [posixpath.basename(norm)]):
        return "mulran_data"
    return "manifest_only"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    p.add_argument("--expect-bytes", type=int, default=0)
    a = p.parse_args()
    root = OUT / f"attempt_{a.attempt}"
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"attempt folder exists: {root}")
    files_root = root / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    real_files_root = os.path.realpath(files_root)
    status_path = root / "extract_status.json"
    manifest = (root / "manifest.jsonl").open("w")
    gz = GzipStream(sys.stdin.buffer)
    start = time.time()
    last = 0.0
    counts: dict[str, int] = {}
    selected_bytes = {"metadata": 0, "mulran_data": 0}
    selected_files = {"metadata": 0, "mulran_data": 0}
    skipped: list[dict] = []
    blocked: list[dict] = []
    dirs: dict[str, dict] = {}
    seen: set[str] = set()
    duplicates = 0
    error = None
    trailing_nonzero = None
    n_members = 0

    def status(stage: str, current: str = "") -> None:
        el = time.time() - start
        write_json(status_path, {"stage": stage, "attempt": a.attempt, "pid": os.getpid(),
            "compressed_bytes": gz.compressed, "expect_bytes": a.expect_bytes,
            "decompressed_bytes": gz.decompressed, "members": n_members, "selected_files": selected_files,
            "selected_bytes": selected_bytes, "elapsed_s": el,
            "compressed_MBps": gz.compressed / max(el, 1e-6) / 1e6, "current": current[-200:],
            "updated_unix": time.time()})

    try:
        with tarfile.open(fileobj=gz, mode="r|") as tf:
            for m in tf:
                n_members += 1
                norm, why = safe_name(m.name)
                kind = ("reg" if m.isreg() else "dir" if m.isdir() else "sym" if m.issym() else "lnk" if m.islnk()
                        else "chr" if m.ischr() else "blk" if m.isblk() else "fifo" if m.isfifo() else "other")
                row = {"name": m.name, "type": kind, "size": int(m.size), "mode": int(m.mode), "mtime": int(m.mtime)}
                if m.linkname:
                    row["linkname"] = m.linkname
                if norm is None or kind not in ("reg", "dir"):
                    row["blocked"] = why if norm is None else f"type_{kind}"
                    blocked.append(row)
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["blocked"] = counts.get("blocked", 0) + 1
                    continue
                if kind == "dir":
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["dir"] = counts.get("dir", 0) + 1
                    continue
                choice = classify(norm, m.size)
                parent = posixpath.dirname(norm)
                ds = dirs.setdefault(parent, {"files": 0, "bytes": 0, "suffixes": {}, "npy_headers": {}})
                ds["files"] += 1
                ds["bytes"] += int(m.size)
                suf = posixpath.splitext(norm)[1].lower()
                ds["suffixes"][suf] = ds["suffixes"].get(suf, 0) + 1
                if norm in seen:
                    duplicates += 1
                    row["duplicate_name"] = True
                seen.add(norm)
                write = choice in ("metadata", "mulran_data")
                if write:
                    st = os.statvfs(root)
                    free = st.f_bavail * st.f_frsize
                    over_budget = choice == "mulran_data" and selected_bytes["mulran_data"] + m.size > DATA_BUDGET_BYTES
                    if free - m.size < RESERVE_BYTES or over_budget:
                        write = False
                        row["skip_reason"] = "budget" if over_budget else "free_space_reserve"
                        skipped.append({"name": norm, "size": int(m.size), "reason": row["skip_reason"], "free": free})
                f = tf.extractfile(m)
                h = hashlib.sha256()
                head = b""
                out = None
                if write:
                    dest = files_root / norm
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not os.path.realpath(dest.parent).startswith(real_files_root + os.sep) and \
                            os.path.realpath(dest.parent) != real_files_root:
                        raise RuntimeError(f"path escapes output: {norm}")
                    tmp = dest.with_name(dest.name + ".part")
                    out = open(tmp, "wb")
                got = 0
                while True:
                    b = f.read(CHUNK)
                    if not b:
                        break
                    if len(head) < 4096:
                        head += b[:4096 - len(head)]
                    h.update(b)
                    got += len(b)
                    if out is not None:
                        out.write(b)
                if got != m.size:
                    raise RuntimeError(f"short member {norm}: {got} != {m.size}")
                row["sha256"] = h.hexdigest()
                if suf == ".npy":
                    hdr = npy_header(head)
                    row["npy"] = hdr
                    if hdr and "shape" in hdr:
                        key = f"{hdr['descr']}|{tuple(hdr['shape'])}"
                        ds["npy_headers"][key] = ds["npy_headers"].get(key, 0) + 1
                if out is not None:
                    out.close()
                    tmp.replace(dest)
                    row["saved"] = str(Path("files") / norm)
                    selected_files[choice] += 1
                    selected_bytes[choice] += int(m.size)
                row["class"] = choice
                counts[choice] = counts.get(choice, 0) + 1
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                if time.time() - last > 15:
                    last = time.time()
                    manifest.flush()
                    status("streaming", norm)
        trailing_nonzero = gz.drain()
    except Exception as exc:  # 실패도 기록하고 비정상 종료한다.
        error = repr(exc)
    manifest.close()
    elapsed = time.time() - start
    passed = (error is None and gz.done and (not a.expect_bytes or gz.compressed == a.expect_bytes)
              and not skipped)
    result = {"attempt": a.attempt, "passed": passed, "error": error,
              "compressed_bytes": gz.compressed, "expect_bytes": a.expect_bytes,
              "compressed_sha256": gz.sha.hexdigest(), "gzip_members": gz.members,
              "gzip_crc32_isize_verified_by_zlib": error is None and gz.done,
              "decompressed_bytes": gz.decompressed, "tar_members": n_members,
              "trailing_nonzero_bytes_after_tar_end": trailing_nonzero,
              "counts": counts, "duplicate_names": duplicates, "blocked": blocked[:500], "n_blocked": len(blocked),
              "skipped_selected": skipped[:500], "n_skipped_selected": len(skipped),
              "selected_files": selected_files, "selected_bytes": selected_bytes,
              "selection": SELECTION, "directories": dirs, "elapsed_s": elapsed,
              "compressed_MBps": gz.compressed / max(elapsed, 1e-6) / 1e6,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "finished_unix": time.time()}
    write_json(root / "extract_result.json", result)
    status("finished" if passed else "failed")
    print(json.dumps({k: result[k] for k in ("attempt", "passed", "error", "compressed_bytes", "compressed_sha256",
                                              "gzip_members", "decompressed_bytes", "tar_members", "counts",
                                              "selected_files", "selected_bytes", "n_blocked", "n_skipped_selected")}),
          flush=True)
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
