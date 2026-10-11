#!/usr/bin/env python3
"""Fetch only the oxts/ members of KITTI raw *_sync.zip archives via HTTP range
requests (official S3 bucket), independently of results/kitti_odometry_geo_reference.json.
Output: <out>/<drive>_sync/oxts/{data/*.txt,timestamps.txt}
"""
import io, sys, zipfile, urllib.request
from pathlib import Path

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{d}/{d}_sync.zip"
DRIVES = ["2011_10_03_drive_0027", "2011_10_03_drive_0042", "2011_10_03_drive_0034",
          "2011_09_30_drive_0018", "2011_09_30_drive_0020", "2011_09_30_drive_0027",
          "2011_09_30_drive_0028", "2011_09_30_drive_0033"]


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = []  # (start, bytes)

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def prefetch(self, start, end):
        self.cache.append((start, self._get(start, end)))

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        for s, b in self.cache:
            if s <= self.pos and end <= s + len(b):
                out = b[self.pos - s:end - s]
                self.pos = end
                return out
        out = self._get(self.pos, end) if end > self.pos else b""
        self.pos = end
        return out


def main(out_root):
    out_root = Path(out_root)
    for d in DRIVES:
        dst = out_root / f"{d}_sync" / "oxts"
        if (dst / "timestamps.txt").exists():
            print(d, "exists"); continue
        rf = RangeFile(URL.format(d=d))
        # central directory lives at the end: prefetch last 8 MB
        rf.prefetch(max(0, rf.size - 8 * 2**20), rf.size)
        z = zipfile.ZipFile(rf)
        infos = [i for i in z.infolist() if "/oxts/" in i.filename and not i.is_dir()]
        lo = min(i.header_offset for i in infos)
        hi = max(i.header_offset + 30 + len(i.filename) + i.compress_size + 1024 for i in infos)
        print(d, len(infos), "oxts members, fetching", (hi - lo) / 1e6, "MB")
        rf.prefetch(lo, min(hi, rf.size))
        for i in infos:
            rel = i.filename.split("/oxts/", 1)[1]
            p = dst / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(i))
        print(d, "done", len(list((dst / "data").glob("*.txt"))), "data files")


if __name__ == "__main__":
    main(sys.argv[1])
