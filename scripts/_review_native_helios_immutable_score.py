#!/usr/bin/env python3
"""전체 공개 센서 평가에서 완료된 부분 예측을 다시 쓰지 않는다."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import _review_native_helios as native


def main(model: str) -> None:
    """공식 검색 결과를 기존 배열과 대조하고 신규 쌍만 저장한다."""
    root = native.OUT / model
    preserved = {str(p.relative_to(root)): native.sha(p)
                 for pattern in ["native_partial_report.json", "native_pairs/*.npz"]
                 for p in root.glob(pattern)}
    original = np.savez_compressed
    reused, created = [], []

    def save_new_only(file, *args, **kwargs):
        path = Path(file)
        assert not args, "이 보존 검사는 이름 있는 예측 배열만 지원한다"
        if path.exists():
            with np.load(path) as old:
                assert set(old.files) == set(kwargs), str(path)
                for key, value in kwargs.items():
                    array = np.asarray(value)
                    assert old[key].dtype == array.dtype, (path, key, "dtype")
                    np.testing.assert_array_equal(old[key], array, err_msg=str(path) + ":" + key)
            reused.append(str(path.relative_to(root)))
            return
        original(path, **kwargs)
        created.append(str(path.relative_to(root)))

    np.savez_compressed = save_new_only
    try:
        native.score(model)
    finally:
        np.savez_compressed = original
    for relative, expected in preserved.items():
        assert native.sha(root / relative) == expected, relative
    report = json.loads((root / "native_report.json").read_text())
    assert len(report["pairs"]) == 24
    native.write(root / "full_release_preservation.json", {
        "passed": True, "model": model, "unix_time": time.time(),
        "unchanged_existing_sha256": preserved, "reused_prediction_files": reused,
        "new_prediction_files": created, "protocol_sha256": native.sha(native.OUT / "protocol.json"),
        "wrapper_sha256": native.sha(Path(__file__)),
        "scope": "공식 score 연산 불변; 기존 배열 전수일치 후 파일쓰기 생략. 신규 쌍만 저장."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["author", "v2_s1", "v2_s2", "v2_s3"])
    main(parser.parse_args().model)
