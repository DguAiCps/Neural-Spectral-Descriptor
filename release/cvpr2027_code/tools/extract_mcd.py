#!/usr/bin/env python3
"""Extract one MCD sequence from its ROS1 bag into the per-scan layout read by nsd/data.py.

    python tools/extract_mcd.py --bag <mcd dir>/<seq>/<seq>_os1_64.bag \
        --gt <mcd dir>/<seq>/gt/<seq>/pose_inW.csv --out <extracted dir>/<seq>

Writes
    <out>/Ouster/<timestamp_ns>.bin   float32 [x, y, z, intensity] per point (64 x 1024, zero returns kept)
    <out>/scan_poses.csv              timestamp_ns,x,y,z  (reference position, linear interpolation of pose_inW)

Scan timestamps are the PointCloud2 header stamps (same clock as the reference poses). Scans outside the
reference time range are dropped. Requires the `rosbags` package.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

POINTS_TOPIC = "/os_cloud_node/points"
# Ouster point layout in the MCD bags: point_step 48, x/y/z float32 at 0/4/8, intensity float32 at 16.
POINT_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity"], "formats": ["<f4"] * 4,
                        "offsets": [0, 4, 8, 16], "itemsize": 48})


def load_reference(path: Path):
    """pose_inW.csv (num, t, x, y, z, qx, qy, qz, qw) -> times [s] and positions, sorted by time."""
    rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
    rows = rows[np.argsort(rows[:, 0])]
    return rows[:, 0], rows[:, 1:4]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path, help="pose_inW.csv of the sequence")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    from rosbags.highlevel import AnyReader  # only needed here

    gt_t, gt_xyz = load_reference(args.gt)
    scan_dir = args.out / "Ouster"
    scan_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with AnyReader([args.bag]) as reader:
        conns = [c for c in reader.connections if c.topic == POINTS_TOPIC]
        if not conns:
            raise SystemExit(f"topic {POINTS_TOPIC} not found in the bag")
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t < gt_t[0] or t > gt_t[-1]:
                continue
            pts = np.frombuffer(msg.data, dtype=POINT_DTYPE)
            cloud = np.empty((len(pts), 4), dtype=np.float32)
            for c, name in enumerate(("x", "y", "z", "intensity")):
                cloud[:, c] = pts[name]
            ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            cloud.tofile(scan_dir / f"{ts_ns}.bin")
            rows.append((ts_ns, *(np.interp(t, gt_t, gt_xyz[:, k]) for k in range(3))))

    with open(args.out / "scan_poses.csv", "w") as f:
        f.write("timestamp_ns,x,y,z\n")
        for ts_ns, x, y, z in rows:
            f.write(f"{ts_ns},{x:.6f},{y:.6f},{z:.6f}\n")
    print(f"{len(rows)} scans written")


if __name__ == "__main__":
    main()
