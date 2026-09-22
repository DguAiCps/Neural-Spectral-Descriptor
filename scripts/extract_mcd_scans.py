#!/usr/bin/env python3
"""Extract MCD (Multi-Campus Dataset) ROS1 bags into MulRan-style per-scan bins.

Input layout (per sequence):
    <data_root>/<seq>/<seq>_os1_64.bag                      ROS1 bag, /os_cloud_node/points
    <data_root>/<seq>/gt/<seq>/pose_inW.csv                 num,t,x,y,z,qx,qy,qz,qw @ ~10 Hz (t = epoch seconds)

Output layout (MulRan convention, cf. src/data/mulran_loader.py):
    <out_root>/<seq>/Ouster/<timestamp_ns>.bin              flat float32 [x,y,z,intensity] x 65536 (64x1024, zeros kept)
    <out_root>/<seq>/scan_poses.csv                         timestamp_ns,x,y,z (world, linear interp of pose_inW)
    <out_root>/<seq>/extraction_summary.json

Scan timestamps are the PointCloud2 header stamps (same epoch base as the GT).
Scans outside the GT time range are dropped (no extrapolation).

Usage:
    python3 scripts/extract_mcd_scans.py --seq tuhh_night_09 \
        --data-root /mnt/d/NSD_datasets/mcd --out-root /mnt/d/NSD_datasets/mcd_extracted
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

POINTS_TOPIC = "/os_cloud_node/points"

# Ouster point layout in the MCD bags: point_step=48,
# x/y/z float32 @ 0/4/8, intensity float32 @ 16.
POINT_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity"],
    "formats": ["<f4", "<f4", "<f4", "<f4"],
    "offsets": [0, 4, 8, 16],
    "itemsize": 48,
})


def load_gt(gt_csv: Path):
    """Load pose_inW.csv -> (t_sec, xyz) sorted by time."""
    rows = np.loadtxt(gt_csv, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
    order = np.argsort(rows[:, 0])
    rows = rows[order]
    return rows[:, 0], rows[:, 1:4]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", required=True, help="sequence name, e.g. tuhh_night_09")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/d/NSD_datasets/mcd"))
    ap.add_argument("--out-root", type=Path, default=Path("/mnt/d/NSD_datasets/mcd_extracted"))
    ap.add_argument("--elev-sample-scans", type=int, default=10,
                    help="number of scans (evenly spaced) used for elevation/z statistics")
    args = ap.parse_args()

    bag = args.data_root / args.seq / f"{args.seq}_os1_64.bag"
    gt_csv = args.data_root / args.seq / "gt" / args.seq / "pose_inW.csv"
    if not bag.is_file():
        raise SystemExit(f"bag not found: {bag}")
    if not gt_csv.is_file():
        raise SystemExit(f"GT not found: {gt_csv}")

    out_dir = args.out_root / args.seq
    scan_dir = out_dir / "Ouster"
    scan_dir.mkdir(parents=True, exist_ok=True)

    gt_t, gt_xyz = load_gt(gt_csv)

    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == POINTS_TOPIC]
        if not conns:
            raise SystemExit(f"topic {POINTS_TOPIC} not in {bag}")
        n_msgs = sum(c.msgcount for c in conns)

        # Pass 1: which scan indices to sample for elevation stats.
        sample_idx = set(np.linspace(0, n_msgs - 1, args.elev_sample_scans).astype(int).tolist())

        written = []          # (ts_ns, x, y, z)
        dropped_no_pose = 0
        valid_counts = []
        elev_min, elev_max = [], []
        elev_p1, elev_p99 = [], []
        z_ground = []

        for i, (conn, _bag_ts, raw) in enumerate(reader.messages(connections=conns)):
            msg = reader.deserialize(raw, conn.msgtype)
            ts_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if ts_sec < gt_t[0] or ts_sec > gt_t[-1]:
                dropped_no_pose += 1
                continue

            pts = np.frombuffer(msg.data, dtype=POINT_DTYPE)
            cloud = np.empty((pts.shape[0], 4), dtype=np.float32)
            cloud[:, 0] = pts["x"]
            cloud[:, 1] = pts["y"]
            cloud[:, 2] = pts["z"]
            cloud[:, 3] = pts["intensity"]

            ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            cloud.tofile(scan_dir / f"{ts_ns}.bin")

            x = np.interp(ts_sec, gt_t, gt_xyz[:, 0])
            y = np.interp(ts_sec, gt_t, gt_xyz[:, 1])
            z = np.interp(ts_sec, gt_t, gt_xyz[:, 2])
            written.append((ts_ns, x, y, z))

            if i in sample_idx:
                r_xy = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
                valid = r_xy > 0.1
                valid_counts.append(int(valid.sum()))
                elev = np.degrees(np.arctan2(cloud[valid, 2], r_xy[valid]))
                elev_min.append(float(elev.min()))
                elev_max.append(float(elev.max()))
                elev_p1.append(float(np.percentile(elev, 1)))
                elev_p99.append(float(np.percentile(elev, 99)))
                z_ground.append(float(np.percentile(cloud[valid, 2], 90)))

    poses = np.array([(x, y, z) for _, x, y, z in written])
    step = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    path_len = float(step.sum())
    duration = (written[-1][0] - written[0][0]) / 1e9 if len(written) > 1 else 0.0

    with open(out_dir / "scan_poses.csv", "w") as f:
        f.write("timestamp_ns,x,y,z\n")
        for ts_ns, x, y, z in written:
            f.write(f"{ts_ns},{x:.6f},{y:.6f},{z:.6f}\n")

    summary = {
        "seq": args.seq,
        "bag": str(bag),
        "topic": POINTS_TOPIC,
        "n_bag_msgs": n_msgs,
        "n_scans_written": len(written),
        "n_dropped_outside_gt": dropped_no_pose,
        "points_per_scan": 65536,
        "valid_points_per_scan_mean": float(np.mean(valid_counts)),
        "valid_points_per_scan_min": int(np.min(valid_counts)),
        "valid_points_per_scan_max": int(np.max(valid_counts)),
        "elevation_deg": {
            "min": float(np.min(elev_min)),
            "max": float(np.max(elev_max)),
            "p1_mean": float(np.mean(elev_p1)),
            "p99_mean": float(np.mean(elev_p99)),
        },
        "z_p90_sensor_frame_mean": float(np.mean(z_ground)),
        "duration_s": duration,
        "path_length_m": path_len,
        "mean_speed_mps": path_len / duration if duration > 0 else 0.0,
        "gt_time_range": [float(gt_t[0]), float(gt_t[-1])],
    }
    with open(out_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
