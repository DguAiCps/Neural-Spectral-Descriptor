#!/usr/bin/env python3
"""Extract selected KITTI files from a partially downloaded odometry ZIP.

The official KITTI velodyne archive is a single large ZIP. For sequence-level
smoke experiments, this script reads local ZIP headers sequentially and extracts
entries that are already present in an incomplete download. It supports stored
and deflated entries with known local-header sizes.
"""

from __future__ import annotations

import argparse
import os
import struct
import zlib
from pathlib import Path


LOCAL_FILE_HEADER = b"PK\x03\x04"


def _read_exact(f, n: int) -> bytes | None:
    data = f.read(n)
    if len(data) != n:
        return None
    return data


def extract_partial(zip_path: Path, output_dir: Path, prefixes: list[str]) -> int:
    extracted = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(zip_path, "rb") as f:
        while True:
            sig = _read_exact(f, 4)
            if sig != LOCAL_FILE_HEADER:
                break

            header = _read_exact(f, 26)
            if header is None:
                break
            _, flag, method, _, _, _, csize, usize, nlen, xlen = struct.unpack(
                "<HHHHHIIIHH", header
            )
            name_raw = _read_exact(f, nlen)
            extra = _read_exact(f, xlen)
            if name_raw is None or extra is None:
                break
            name = name_raw.decode("utf-8", "replace")

            if flag & 0x08:
                raise RuntimeError(
                    f"Unsupported streaming ZIP entry with data descriptor: {name}"
                )

            target = output_dir / name
            want = any(name.startswith(prefix) for prefix in prefixes)
            if name.endswith("/"):
                if want:
                    target.mkdir(parents=True, exist_ok=True)
                continue

            if not want:
                f.seek(csize, os.SEEK_CUR)
                continue

            payload = _read_exact(f, csize)
            if payload is None:
                break

            if target.exists() and target.stat().st_size == usize:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if method == 0:
                data = payload
            elif method == 8:
                data = zlib.decompress(payload, -zlib.MAX_WBITS)
            else:
                raise RuntimeError(f"Unsupported ZIP compression method {method}: {name}")

            with open(target, "wb") as out:
                out.write(data)
            extracted += 1

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("output_dir")
    parser.add_argument("--sequence", default=None, help="Single sequence to extract, e.g. 00")
    parser.add_argument("--sequences", nargs="+", default=None, help="Sequences to extract in one pass")
    args = parser.parse_args()

    if args.sequences is not None:
        sequences = args.sequences
    elif args.sequence is not None:
        sequences = [args.sequence]
    else:
        sequences = ["00"]

    prefixes = []
    for sequence in sequences:
        seq = f"{int(sequence):02d}"
        prefixes.extend([
            f"dataset/sequences/{seq}/velodyne/",
            f"dataset/sequences/{seq}/",
            f"dataset/poses/{seq}.txt",
        ])
    n = extract_partial(Path(args.zip_path), Path(args.output_dir), prefixes)
    print(f"extracted_or_updated={n}")


if __name__ == "__main__":
    main()
