#!/usr/bin/env python3
"""KITTI-360 vs KITTI-odometry route-overlap audit.

Checks whether the KITTI-360 test drives (0002, 0009) geographically overlap
the KITTI odometry train sequences (01/02/06/07) and eval sequences (00/05/08),
all recorded in Karlsruhe, Germany.

Test routes come from the local KITTI-360 oxts (lat/lon per frame). Odometry
routes come from results/kitti_odometry_geo_reference.json (exact raw-drive
oxts polylines, devkit frame ranges; see the per-sequence 'source' field), or
from --odometry-oxts-dir containing <raw_drive>_sync/oxts/data/*.txt.

For each (KITTI-360 drive, odometry sequence) pair it reports the minimum
pairwise haversine distance between the route polylines and the percentage of
test-route points within 50 m / 200 m of the train route.

Usage:
    python3 scripts/audit_kitti360_route_overlap.py
    python3 scripts/audit_kitti360_route_overlap.py --odometry-oxts-dir /path/to/raw_oxts
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# KITTI odometry devkit readme: sequence -> (raw drive, start frame, end frame)
ODOM_MAP = {
    "00": ("2011_10_03_drive_0027", 0, 4540),
    "01": ("2011_10_03_drive_0042", 0, 1100),
    "02": ("2011_10_03_drive_0034", 0, 4660),
    "05": ("2011_09_30_drive_0018", 0, 2760),
    "06": ("2011_09_30_drive_0020", 0, 1100),
    "07": ("2011_09_30_drive_0027", 0, 1100),
    "08": ("2011_09_30_drive_0028", 1100, 5170),
}
TRAIN_SEQS = ("01", "02", "06", "07")
EVAL_SEQS = ("00", "05", "08")

EARTH_RADIUS_M = 6371008.8


def haversine_m(lat1, lon1, lat2, lon2):
    """Pairwise haversine distance in meters. lat1/lon1 (N,), lat2/lon2 (M,) -> (N, M)."""
    lat1, lon1 = np.radians(lat1)[:, None], np.radians(lon1)[:, None]
    lat2, lon2 = np.radians(lat2)[None, :], np.radians(lon2)[None, :]
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_oxts_polyline(data_dir, start=0, end=None, subsample=1):
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no oxts txt files in {data_dir}")
    stop = len(files) if end is None else end + 1
    latlon = []
    for f in files[start:stop:subsample]:
        lat, lon = f.read_text().split()[:2]
        latlon.append((float(lat), float(lon)))
    return np.array(latlon)


def load_odometry_routes(args):
    """Return {seq: (polyline (N,2), confidence, source)}."""
    routes = {}
    if args.odometry_oxts_dir:
        root = Path(args.odometry_oxts_dir)
        for seq, (drive, start, end) in ODOM_MAP.items():
            data_dir = root / f"{drive}_sync" / "oxts" / "data"
            poly = load_oxts_polyline(data_dir, start, end, args.odometry_subsample)
            routes[seq] = (poly, "exact-oxts", f"local raw oxts {data_dir}")
    else:
        ref = json.loads(Path(args.geo_reference).read_text())
        for seq, entry in ref["sequences"].items():
            poly = np.array(entry["polyline_latlon"])
            routes[seq] = (poly, entry["confidence"], entry["source"])
    return routes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kitti360-root", type=Path,
                        default=Path("/mnt/d/NSD_datasets/kitti360"))
    parser.add_argument("--kitti360-drives", nargs="+", default=["0002", "0009"])
    parser.add_argument("--kitti360-subsample", type=int, default=20)
    parser.add_argument("--geo-reference", type=Path,
                        default=REPO_ROOT / "results/kitti_odometry_geo_reference.json")
    parser.add_argument("--odometry-oxts-dir", type=Path, default=None,
                        help="dir with <raw_drive>_sync/oxts/data/*.txt; overrides --geo-reference")
    parser.add_argument("--odometry-subsample", type=int, default=5,
                        help="frame subsampling when reading --odometry-oxts-dir")
    args = parser.parse_args()

    test_routes = {}
    for drive in args.kitti360_drives:
        data_dir = (args.kitti360_root / "data_poses"
                    / f"2013_05_28_drive_{drive}_sync" / "oxts" / "data")
        poly = load_oxts_polyline(data_dir, subsample=args.kitti360_subsample)
        test_routes[drive] = poly
        print(f"KITTI-360 {drive}: {len(poly)} pts (every {args.kitti360_subsample}th frame), "
              f"lat [{poly[:, 0].min():.5f}, {poly[:, 0].max():.5f}], "
              f"lon [{poly[:, 1].min():.5f}, {poly[:, 1].max():.5f}]")

    odom_routes = load_odometry_routes(args)
    for seq in sorted(odom_routes):
        poly, conf, _ = odom_routes[seq]
        print(f"odometry {seq} ({ODOM_MAP[seq][0]}): {len(poly)} pts, confidence={conf}")

    header = (f"{'360-drive':>9}  {'odom':>4}  {'role':>5}  {'min_dist_m':>10}  "
              f"{'<50m_%':>7}  {'<200m_%':>8}  {'confidence':>12}")
    print("\n" + header)
    print("-" * len(header))
    for drive, test_poly in test_routes.items():
        for seq in sorted(odom_routes):
            odom_poly, conf, _ = odom_routes[seq]
            dist = haversine_m(test_poly[:, 0], test_poly[:, 1],
                               odom_poly[:, 0], odom_poly[:, 1])
            nearest = dist.min(axis=1)  # per test point
            role = "train" if seq in TRAIN_SEQS else "eval"
            print(f"{drive:>9}  {seq:>4}  {role:>5}  {nearest.min():>10.1f}  "
                  f"{100 * (nearest < 50).mean():>7.2f}  "
                  f"{100 * (nearest < 200).mean():>8.2f}  {conf:>12}")

    print("\nNote: distances are point-to-point minima over subsampled polylines; "
          "with the default subsampling the discretization error is < ~15 m.")


if __name__ == "__main__":
    main()
