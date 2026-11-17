"""FreSCo NCLT 좌표 적응 수정 재평가의 격리 환경 (2026-11-08).

기존 공통46 실행 코드(scripts/fresco_official/)를 바꾸지 않고 다시 쓴다. 이 모듈은 기존 모듈을 부르기 전에
  * FRESCO_NAS_ROOT를 새 NAS 루트로 정하고(기존 common.NAS_ROOT는 이 환경 변수를 읽는다),
  * 기존 모듈 파일의 sha256이 동결 해시와 같은지 확인하고,
  * 결과 경로(RESULTS)·가드·가족 이름을 새 값으로 바꾼다(patch 목록은 PATCHES).
기존 NAS 루트·기존 결과 폴더·기존 원장에는 쓰지 않는다. 새 산출은 새 NAS 루트에만 쓴다.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD_PY = REPO / "scripts/fresco_official"
TAG = "fresco_nclt_fix_20261108"
NEW_NAS = Path("/mnt/aicps-nas/qkrrlans00/nsd_fresco_nclt_fix_20261108")
OLD_NAS = Path("/mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107")
NEW_RESULTS = NEW_NAS / "results"            # 원장·상태 json (서버 root에 쓰지 않는다)
LOCAL_OUT = Path("/mnt/d/NSD_local/results/pdv2_overnight_repro_20261108/fresco_fix_pipeline")
NAS_BUDGET = 8 * 2 ** 30
MIN_ROOT_FREE = 35 * 2 ** 30
FAMILY_NCLT = "nclt_flu_hdl32_inverted"
DRIVER_NAME = "ncltflu"
NCLT9 = ["nclt/2012-01-08", "nclt/2012-02-04", "nclt/2012-03-17", "nclt/2012-05-11",
         "nclt/2012-08-04", "nclt/2012-11-04", "nclt/2012-11-16", "nclt/2013-01-10",
         "nclt/2013-02-23"]
# 기존 공통46 실행에 쓰인 파일 해시(로컬·서버 같음, protocol.json·protocol_amendments.json과 대조)
OLD_SHA256 = {
    "common.py": "cdac38c185d7372ff2739da4701382cdb279b527d94c001f3b0467a13e1089c0",
    "codec.py": "45687cdb54d397af8ed04bf85fee10af69914d6856861846342004a6524943b4",
    "stream.py": "c470804994637441a6e2506b2c17d97735634ecba0f471bf034d092e6a6cc4bd",
    "evaluate.py": "fca8744f91a25c32660d6be3e1c8529093738105690d2bc613274cb234242b3f",
    "server_master.py": "2555ac89751b47d22483bf968bfd96df1c13c6f6c92e3188f63351df035016f2",
    "aggregate.py": "d068693bc5e340e3db1962218d01aa0788fa7e2e0fe6b25ab6d0d79207eb4b87",
    "cpp/fresco_prep_driver.cpp": "c568dfc0fe0354789d88307a3e91c247a08ae3910719600080421c3cd5f96758",
}
OLD_MATLAB_ADAPTER_SHA256 = "09b854d92a5da88ec08e247b1f3f829bda04518dffa9867b04e5530ff64d77c3"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_old_code() -> None:
    for f, pre in OLD_SHA256.items():
        got = sha256_file(OLD_PY / f)
        assert got == pre, f"reused file changed: {f} {got[:16]} != {pre[:16]}"


def setup():
    """기존 모듈을 새 루트로 불러 patch한 뒤 (common, codec_fix, stream, evaluate)를 돌려준다."""
    os.environ["FRESCO_NAS_ROOT"] = str(NEW_NAS)
    sys.dont_write_bytecode = True
    check_old_code()
    for p in (str(OLD_PY), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import common
    assert common.NAS_ROOT == NEW_NAS, common.NAS_ROOT
    import codec_fix
    fam = dict(common.FAMILY)
    fam["nclt"] = FAMILY_NCLT
    import stream
    stream.NAS_ROOT = NEW_NAS
    stream.RESULTS = NEW_RESULTS
    stream.MIN_ROOT_FREE = MIN_ROOT_FREE
    stream.FAMILY = fam
    stream.codec = codec_fix
    import evaluate
    evaluate.NAS_ROOT = NEW_NAS
    evaluate.RESULTS = NEW_RESULTS
    evaluate.REPO = NEW_NAS                            # build_ledger의 반환 경로 표기(relative_to)에만 쓰인다
    evaluate.MDIR = OLD_NAS / "build/matlab"          # 기존 공식 MATLAB 어댑터(읽기만, 해시 확인)
    evaluate.FAMILY = fam
    got = sha256_file(evaluate.MDIR / "fresco_adapter.m") if (evaluate.MDIR / "fresco_adapter.m").exists() else None
    if got is not None:
        assert got == OLD_MATLAB_ADAPTER_SHA256, got
    return common, codec_fix, stream, evaluate


PATCHES = {
    "env FRESCO_NAS_ROOT": str(NEW_NAS),
    "stream.NAS_ROOT/evaluate.NAS_ROOT": str(NEW_NAS),
    "stream.RESULTS/evaluate.RESULTS": str(NEW_RESULTS),
    "evaluate.REPO": str(NEW_NAS) + " (반환 dict의 원장 상대 경로 표기 전용)",
    "stream.MIN_ROOT_FREE": "35 GiB",
    "stream.FAMILY/evaluate.FAMILY['nclt']": FAMILY_NCLT,
    "stream.codec": "codec_fix (nclt_flu_hdl32_inverted 추가, 다른 가족은 기존 codec 그대로 위임)",
    "evaluate.MDIR": str(OLD_NAS / "build/matlab") + " (읽기만, fresco_adapter.m sha256 확인)",
}
