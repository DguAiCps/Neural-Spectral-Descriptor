#!/usr/bin/env python3
"""기존 표본을 포함해46시퀀스 모두의 원시 yaw 평가를 고정 실행한다."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import _review_direction_yaw as yaw

base = yaw.report
ROOT, PARENT, MAIN = yaw.ROOT, yaw.OUT, yaw.MAIN
OUT = PARENT / "expanded46"


def freeze() -> dict:
    """표본·각도·물리 변환·모델을 결과 전에 고정하고 재사용을 검산한다."""
    path = OUT / "yaw_protocol.json"
    if path.exists():
        spec = base.read(path)
        for name, digest in spec["sources"].items():
            assert base.digest(ROOT / name) == digest, name
        return spec
    old = base.read(PARENT / "yaw_protocol.json")
    original = base.read(MAIN / "protocol.json")
    samples, support, sources = {}, {}, dict(old["sources"])
    paths = [Path(__file__), Path(yaw.__file__), PARENT / "yaw_protocol.json",
             PARENT / "completion_validation.json"]
    for sequence in original["test"]:
        folder = MAIN / "test" / sequence.replace("/", "_")
        p = folder / "diff_s1.npz"
        with np.load(p) as data:
            q = data["query_idx"]
        take = q[np.linspace(0, len(q)-1, 64).astype(int)]
        samples[sequence] = take.tolist()
        support[sequence] = len(np.unique(np.concatenate([np.arange(t-15, t+16) for t in take])))
        if sequence in old["queries"]:
            assert samples[sequence] == old["queries"][sequence]
        paths += [p, base.cache_path(sequence), folder / "reference_validation.json"]
        ref = base.read(folder / "reference_validation.json")
        for name, digest in ref["input_sha256"].items():
            if "testspec_" in name or "/cache_" in name:
                assert base.digest(ROOT / name) == digest
                sources[name] = digest
    sources.update({str(p.relative_to(ROOT)): base.digest(p) for p in paths})
    spec = dict(old)
    spec.update(created_unix=time.time(), queries=samples, sources=sources,
        sample_rule="46개 시험 시퀀스 모두에서 같은 시간축 등간격64질의. 기존8시퀀스512질의를 정확히 포함. GT·점수·성능으로 표본·각도 선택 없음.",
        baseline="2944질의×3시작점 각도0 승자 정확일치, 점수오차<=8e-6. 기존8개 검산 재사용, 추가38개 모두 재검산한 뒤 신규 회전을 실행한다.",
        scope="46시퀀스를 빠짐없이 포함하는2944질의 원시 회전 표본. 전체195726질의의 원시 회전 전수평가는 아니다. 독립 query 좌표회전이며 gallery 전체는 원본 고정.",
        support_frames=support, total_raw_support_frames=sum(support.values()),
        reused_sequences=list(old["queries"]), parent_protocol_sha256=base.digest(PARENT / "yaw_protocol.json"))
    base.write(path, spec)
    reuse = {}
    for sequence in old["queries"]:
        name = sequence.replace("/", "_")
        src, dst = PARENT / "yaw" / name, OUT / "yaw" / name
        assert base.read(src / "rotate.json")["status"] == "passed"
        hashes = {}
        for p in src.glob("*.npz"):
            digest = base.digest(p)
            assert digest == base.read(p.with_suffix(".json"))["sha256"]
            hashes[p.name] = digest
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=True)
        reuse[sequence] = dict(source=str(src.relative_to(ROOT)), sha256=hashes)
    base.write(OUT / "reuse_manifest.json", reuse)
    return spec


def activate() -> dict:
    """기존 동결 실행기의 출력 경로·프로토콜 공급만 바꾼다."""
    spec = freeze()
    yaw.OUT = OUT
    yaw.SEQUENCES = tuple(spec["queries"])
    yaw.freeze = freeze
    return spec


def validate() -> None:
    """46개 시퀀스의 원래 승자와 최종점수 재생을 검사한다."""
    spec = activate()
    rows = {s: base.read(OUT / "yaw" / s.replace("/", "_") / "baseline.json") for s in spec["queries"]}
    checks = [r["baseline_check"] for v in rows.values() for r in v["records"].values()]
    assert len(checks) == 46*3
    assert all(r["winner_mismatches"] == 0 and r["score_max_error"] <= yaw.TOLERANCE for r in checks)
    base.write(OUT / "yaw_baseline_validation.json", dict(status="passed", queries=2944,
        query_seed_checks=8832, winner_mismatches=0,
        max_score_error=max(r["score_max_error"] for r in checks), tolerance=yaw.TOLERANCE, sequences=rows))


def main() -> None:
    """각도0 전체 재현을 먼저 끝낸 뒤 사전고정한 회전을 모두 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "baseline", "validate", "rotate", "run"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = activate()
    if args.stage in ("baseline", "run"):
        for sequence in spec["queries"]: yaw.run_sequence(sequence, "baseline")
        validate()
    if args.stage == "validate": validate()
    if args.stage in ("rotate", "run"):
        for sequence in spec["queries"]: yaw.run_sequence(sequence, "rotate")


if __name__ == "__main__":
    main()
