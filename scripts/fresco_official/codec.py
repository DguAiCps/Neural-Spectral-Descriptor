"""원시 키프레임 스캔 <-> 공식 선택기 입력 파일 바이트, 무손실 전송 페이로드 (2026-11-07).

producer(로컬)는 원시 파일을 읽어 "공식 extractPointCloud가 읽을 파일 바이트"(official bytes)를
만들고 그 sha256과 압축 페이로드를 보낸다. receiver(서버)는 페이로드에서 같은 바이트를 다시 만들고
sha256이 같은지 확인한 뒤 공식 전처리 드라이버에 넘긴다.

가족별 official bytes (protocol.json G1~G4):
  kitti_hdl64  KITTI·KITTI-360 bin(x,y,z,i float32) 그대로, 단 intensity = 0.
               공식 KITTI 선택기는 구조화하는 모든 점의 intensity를 -1로 덮어쓰므로
               (KittiPointCloudSelect.cpp:237-238) 원래 값은 결과에 영향이 없다.
  mulran_os1   MulRan bin 그대로, intensity = 0. 공식 MulRan 경로에서 intensity는
               ==-1 검사(BatchMultiBevGen.cpp:146-166)에만 쓰인다. 원시 값에 -1이 없음을
               producer가 확인한다(있으면 중단).
  mcd_os1_transposed  MCD 추출 bin은 행우선(빔 64 x 열 1024). 공식 MulRan 선택기는 k%64를 빔으로
               보므로(MulranPointCloudSelect.cpp:121) 점 순서를 열우선(k = c*64 + r)으로 바꾼다.
               intensity = 0(-1 없음 확인).
  nclt_oxford_hdl32  NCLT velodyne_sync(u16 x,y,z, u8 i, u8 laser). 로더와 같은 float32 식
               x = f32(u)*0.005 - 100으로 원시 센서 좌표(z-up, 로더의 z 반전 전)를 만들고, 공식
               Oxford 선택기 파일 배치(x 전부, y 전부, z 전부, i 전부; 읽은 뒤 x=-x, z=-z)에 맞춰
               (-x, y, -z, 0)으로 쓴다. 공식 함수가 부호를 되돌려 원시 좌표를 얻는다.
  helipr_vlp16 HeLiPR 22바이트 점(x,y,z,i f32, ring u16, t f32) 배치, intensity = 0, t = 0.
               적응 함수 extractPointCloudVlp16(fresco_prep_driver.cpp)이 읽는다.
페이로드(zlib level 1): kitti/mulran/mcd xyz float32, nclt u16 xyz, helipr xyz float32 + ring u8.
"""
from __future__ import annotations

import hashlib
import struct
import zlib

import numpy as np

HELIPR_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"),
                      ("ring", "<u2"), ("t", "<f4")])
NCLT_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")])
REC = struct.Struct("<iiII32s")   # kf index, scan id, payload len, official len, sha256(official)
DRIVER = {"kitti_hdl64": "kitti", "mulran_os1": "mulran", "mcd_os1_transposed": "mulran",
          "nclt_oxford_hdl32": "oxford", "helipr_vlp16": "helipr"}


def _xyz0(xyz: np.ndarray) -> bytes:
    out = np.zeros((len(xyz), 4), dtype="<f4")
    out[:, :3] = xyz
    return out.tobytes()


def nclt_xyz(u: np.ndarray) -> np.ndarray:
    """src/data/nclt_loader.py:247-249와 같은 float32 식(z 반전 없음)."""
    return np.stack([u[:, 0].astype(np.float32) * 0.005 - 100.0,
                     u[:, 1].astype(np.float32) * 0.005 - 100.0,
                     u[:, 2].astype(np.float32) * 0.005 - 100.0], axis=1).astype(np.float32)


def nclt_official(u: np.ndarray) -> bytes:
    xyz = nclt_xyz(u)
    cols = [-xyz[:, 0], xyz[:, 1], -xyz[:, 2], np.zeros(len(xyz), np.float32)]
    return b"".join(np.ascontiguousarray(c, dtype="<f4").tobytes() for c in cols)


def encode(family: str, raw: bytes) -> tuple[bytes, bytes]:
    """원시 파일 바이트 -> (payload, official bytes)."""
    if family in ("kitti_hdl64", "mulran_os1", "mcd_os1_transposed"):
        a = np.frombuffer(raw, dtype="<f4")
        assert a.size % 4 == 0, family
        a = a.reshape(-1, 4)
        if family != "kitti_hdl64":
            assert not np.any(a[:, 3] == -1.0), "intensity -1 in raw (zeroing would not be lossless)"
        if family == "mcd_os1_transposed":
            assert len(a) == 64 * 1024, len(a)
            a = a.reshape(64, 1024, 4).transpose(1, 0, 2).reshape(-1, 4)
        xyz = np.ascontiguousarray(a[:, :3])
        return xyz.tobytes(), _xyz0(xyz)
    if family == "nclt_oxford_hdl32":
        r = np.frombuffer(raw, dtype=NCLT_DT)
        u = np.stack([r["x"], r["y"], r["z"]], axis=1).astype("<u2")
        return u.tobytes(), nclt_official(u)
    if family == "helipr_vlp16":
        r = np.frombuffer(raw, dtype=HELIPR_DT)
        assert r["ring"].max(initial=0) <= 15, "VLP-16 ring > 15"
        xyz = np.stack([r["x"], r["y"], r["z"]], axis=1).astype("<f4")
        return xyz.tobytes() + r["ring"].astype("u1").tobytes(), helipr_official(xyz, r["ring"])
    raise ValueError(family)


def helipr_official(xyz: np.ndarray, ring: np.ndarray) -> bytes:
    out = np.zeros(len(xyz), dtype=HELIPR_DT)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["ring"] = ring
    return out.tobytes()


def decode(family: str, payload: bytes) -> bytes:
    """payload -> official bytes (receiver)."""
    if family in ("kitti_hdl64", "mulran_os1", "mcd_os1_transposed"):
        return _xyz0(np.frombuffer(payload, dtype="<f4").reshape(-1, 3))
    if family == "nclt_oxford_hdl32":
        return nclt_official(np.frombuffer(payload, dtype="<u2").reshape(-1, 3))
    if family == "helipr_vlp16":
        n = len(payload) // 13
        assert n * 13 == len(payload)
        xyz = np.frombuffer(payload[:12 * n], dtype="<f4").reshape(-1, 3)
        ring = np.frombuffer(payload[12 * n:], dtype="u1").astype("<u2")
        return helipr_official(xyz, ring)
    raise ValueError(family)


def pack(idx: int, sid: int, family: str, raw: bytes) -> bytes:
    payload, official = encode(family, raw)
    z = zlib.compress(payload, 1)
    return REC.pack(idx, sid, len(z), len(official), hashlib.sha256(official).digest()) + z


def sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()
