"""NCLT 좌표 적응 수정 가족 nclt_flu_hdl32_inverted의 공식 입력 바이트와 무손실 전송 페이로드 (2026-11-08).

기존 codec.py(가족 nclt_oxford_hdl32, G3)는 바꾸지 않는다. NCLT가 아닌 가족은 기존 codec에 그대로 맡긴다.

1차 자료(nclt.pdf p6 §2.1a, p8 §3·표4, p15-16 §7.3):
  velodyne_sync 점 = Segway body 좌표 FRD(x 앞, y 오른쪽, z 아래), 원점 = 바퀴 축 중심, u16*0.005-100 m.
  HDL-32E는 거꾸로 장착, 라이다 원점 = body (0.002, -0.004, -0.957) m.
목표 좌표(G3'): body 축에 정렬한 라이다 원점 FLU(x 앞, y 왼쪽, z 위; 오른손계)
  x' = x_b - 0.002,  y' = -(y_b + 0.004),  z' = -(z_b + 0.957)       (float32, 아래 FLU_* 상수)
표4 회전은 넣지 않는다. 목표가 body 축 정렬 좌표로 정의되므로 회전 역변환이 필요 없다
(OVN/fresco_fix_audit/spec.json fresco_nclt_G3prime_v1, sha256 3aeb0b97d4ba87405de2dde80c2ac92fbe496a4fb61c243c83e8a80ccfd1cd3e).
공식 입력 바이트 = Oxford 선택기 파일 배치(x 전부, y 전부, z 전부, i 전부)의 (-x', y', -z', 0).
드라이버(cpp/fresco_prep_ncltflu.cpp)가 무수정 공식 Oxford extractPointCloud로 읽으면 x=-x, z=-z 뒤 점 = (x', y', z')이고,
row만 공식과 같은 float 식에 오프셋 30.67로 다시 계산한다(spec allowed_code_diff I2).
페이로드(zlib level 1) = 원시 u16 x,y,z 그대로(기존 G3와 같은 무손실 페이로드). receiver가 같은 float32 식으로
공식 입력 바이트를 다시 만들고 sha256을 대조한다.
"""
from __future__ import annotations

import hashlib
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fresco_official"))
import codec as _old  # noqa: E402

FAMILY = "nclt_flu_hdl32_inverted"
REC = _old.REC
NCLT_DT = _old.NCLT_DT
DRIVER = dict(_old.DRIVER, **{FAMILY: "ncltflu"})
FLU_T = (np.float32(0.002), np.float32(0.004), np.float32(0.957))   # nclt.pdf 표4 x_body,vel 평행이동 크기


def body_xyz(u: np.ndarray) -> np.ndarray:
    """u16 -> body FRD float32 (기존 codec.nclt_xyz와 같은 식)."""
    return _old.nclt_xyz(u)


def flu_xyz(u: np.ndarray) -> np.ndarray:
    b = body_xyz(u)
    return np.stack([b[:, 0] - FLU_T[0], -(b[:, 1] + FLU_T[1]), -(b[:, 2] + FLU_T[2])],
                    axis=1).astype(np.float32)


def nclt_flu_official(u: np.ndarray) -> bytes:
    p = flu_xyz(u)
    cols = [-p[:, 0], p[:, 1], -p[:, 2], np.zeros(len(p), np.float32)]
    return b"".join(np.ascontiguousarray(c, dtype="<f4").tobytes() for c in cols)


def encode(family: str, raw: bytes) -> tuple[bytes, bytes]:
    if family == FAMILY:
        r = np.frombuffer(raw, dtype=NCLT_DT)
        u = np.stack([r["x"], r["y"], r["z"]], axis=1).astype("<u2")
        return u.tobytes(), nclt_flu_official(u)
    return _old.encode(family, raw)


def decode(family: str, payload: bytes) -> bytes:
    if family == FAMILY:
        return nclt_flu_official(np.frombuffer(payload, dtype="<u2").reshape(-1, 3))
    return _old.decode(family, payload)


def pack(idx: int, sid: int, family: str, raw: bytes) -> bytes:
    payload, official = encode(family, raw)
    z = zlib.compress(payload, 1)
    return REC.pack(idx, sid, len(z), len(official), hashlib.sha256(official).digest()) + z


def sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()
