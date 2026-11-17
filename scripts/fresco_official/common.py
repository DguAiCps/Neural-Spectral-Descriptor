"""공식 FreSCo 평가의 공통 경로·명단·원시 파일 대응 (2026-11-07).

로컬(producer)과 서버(receiver·MATLAB·원장)가 같이 쓴다. torch를 부르지 않는다.
키프레임 = 분할 v2 캐시의 scan_ids(시험: data/preprocessed_v2_odom의 KISS-ICP 키프레임,
검증: data/preprocessed_v2의 val 캐시). 정답 자세 poses_gt는 원장 라벨에만 쓴다.
원시 파일 = 각 데이터셋 로더의 파일 목록(frame_files / scan_files)[scan_id].
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]

from _pdv2_pilot import CITY, TEST, VAL  # noqa: E402

TAG = "pdv2_fresco_official_20261107"
RESULTS = REPO / "results" / TAG / "pipeline"
NAS_ROOT = Path(os.environ.get("FRESCO_NAS_ROOT",
                               "/mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107"))
SERVER = "joms-gpu"
SERVER_REPO = "/home/qkrrlans00/Neural-Spectral-Codec"

FRESCO_COMMIT = "2bf5ceef48513524cedb4ce69dcbd2bd6091fb09"
PREP_COMMIT = "d94040ed626c7275b6f428deddde2ddca74419d7"

# 원시 점 순서·투영을 정하는 공식 선택기(또는 적응) 이름. protocol.json의 G1~G4.
FAMILY = {"kitti": "kitti_hdl64", "kitti360": "kitti_hdl64", "mulran": "mulran_os1",
          "mcd": "mcd_os1_transposed", "nclt": "nclt_oxford_hdl32", "helipr": "helipr_vlp16"}

ODOM = REPO / "data/preprocessed_v2_odom"
V2 = REPO / "data/preprocessed_v2"
ROLES = {"test": (TEST, ODOM, "test_"), "val": (VAL, V2, "val_")}

MIN_ROOT_FREE = 30 * 2 ** 30
MIN_NAS_FREE = 200 * 2 ** 30


def key(sensor: str, seq: str) -> str:
    return f"{sensor}/{seq}"


def name(sensor: str, seq: str) -> str:
    return f"{sensor}_{seq}"


def cache_path(role: str, sensor: str, seq: str) -> Path:
    _, cdir, mid = ROLES[role]
    hits = sorted(glob.glob(str(cdir / f"cache_????????_{sensor}_{mid}{seq}.npz")))
    assert len(hits) == 1, (role, sensor, seq, hits)
    return Path(hits[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def raw_files(sensor: str, seq: str) -> list[Path]:
    """로더의 인덱스 공간 그대로의 원시 파일 목록(scan_id로 인덱싱)."""
    from _pd_build_test_caches import _make_loader
    ld = _make_loader(sensor, seq)
    files = getattr(ld, "frame_files", None)
    if files is None:
        files = ld.scan_files
    return [Path(f) for f in files]


def free_bytes(path: Path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize
