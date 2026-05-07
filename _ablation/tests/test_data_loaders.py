from pathlib import Path

import numpy as np

from data.helipr_loader import HeLiPRLoader
from data.nclt_loader import NCLTLoader


def _write_nclt_scan(path: Path) -> None:
    dtype = np.dtype(
        [
            ("x", "<u2"),
            ("y", "<u2"),
            ("z", "<u2"),
            ("intensity", "u1"),
            ("padding", "u1"),
            ("extra", "<u4"),
        ]
    )
    raw = np.zeros(3, dtype=dtype)
    raw["x"] = np.asarray([20000, 20010, 20020], dtype=np.uint16)
    raw["y"] = np.asarray([20000, 20010, 20020], dtype=np.uint16)
    raw["z"] = np.asarray([20000, 20010, 20020], dtype=np.uint16)
    raw["intensity"] = np.asarray([10, 20, 30], dtype=np.uint8)
    raw.tofile(path)


def _write_helipr_scan(path: Path) -> None:
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("intensity", np.float32),
            ("ring", np.uint16),
            ("time", np.float32),
        ]
    )
    raw = np.zeros(3, dtype=dtype)
    raw["x"] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    raw["y"] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    raw["z"] = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    raw["intensity"] = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    raw.tofile(path)


def test_nclt_loader_matches_scan_to_nearest_groundtruth(tmp_path: Path):
    root = tmp_path / "nclt"
    scan_dir = root / "2013-01-10" / "velodyne_sync"
    scan_dir.mkdir(parents=True)
    _write_nclt_scan(scan_dir / "96.bin")

    gt = root / "2013-01-10" / "groundtruth_2013-01-10.csv"
    gt.write_text(
        "timestamp,x,y,z,roll,pitch,yaw\n"
        "90,1,2,3,0,0,0\n"
        "110,4,5,6,0,0,0\n",
        encoding="utf-8",
    )

    loader = NCLTLoader(root, "2013-01-10", lazy_load=True)
    item = loader[0]

    assert len(loader) == 1
    assert item["points"].shape == (3, 4)
    assert item["pose"].shape == (4, 4)
    assert np.allclose(item["pose"][:3, 3], [1.0, 2.0, 3.0])
    assert item["timestamp"] == 96.0


def test_helipr_loader_accepts_dataset_root_plus_sequence(tmp_path: Path):
    root = tmp_path / "helipr"
    seq_dir = root / "Town01" / "Town01"
    scan_dir = seq_dir / "LiDAR" / "Velodyne"
    gt_dir = seq_dir / "LiDAR_GT"
    scan_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    _write_helipr_scan(scan_dir / "96.bin")
    (gt_dir / "Velodyne_gt.txt").write_text(
        "90 1 2 3 0 0 0 1\n"
        "110 4 5 6 0 0 0 1\n",
        encoding="utf-8",
    )

    loader = HeLiPRLoader(root, "Town01", lazy_load=True)
    item = loader[0]

    assert len(loader) == 1
    assert item["points"].shape == (3, 4)
    assert item["pose"].shape == (4, 4)
    assert np.allclose(item["pose"][:3, 3], [1.0, 2.0, 3.0])
    assert item["timestamp"] == 96.0
