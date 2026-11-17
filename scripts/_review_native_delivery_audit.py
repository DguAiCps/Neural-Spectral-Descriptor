#!/usr/bin/env python3
"""RING++ 46개 입력의 전송 완료 마커·목록·크기를 양쪽에서 대조한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[1]
RELATIVE = "results/pdv2_ringpp_maxfix_20261105"


def inventory(root: Path) -> dict:
    """큰 점군을 다시 읽지 않고 전송 목록과 실제 파일 상태를 확인한다."""
    rows = {}
    for directory in sorted((root / "points").iterdir()):
        manifest = directory / "done.json"
        if not manifest.exists():
            continue
        value = json.loads(manifest.read_text())
        marker = directory / "transferred.json"
        rows[directory.name] = {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "frames": value["frames"],
            "marker": json.loads(marker.read_text()),
            "shard_sizes": {name: (directory / name).stat().st_size for name in value["shards"]}}
    return rows


def main() -> None:
    """복구 전 실패는 보존하고 현재 배달 상태를 별도 기록한다."""
    import inspect
    code = "import json, hashlib\nfrom pathlib import Path\n" + inspect.getsource(inventory)
    code += "\nprint(json.dumps(inventory(Path('/home/qkrrlans00/Neural-Spectral-Codec/" + RELATIVE + "'))))"
    remote = json.loads(subprocess.check_output(["ssh", "joms-gpu", "python3 -c " + shlex.quote(code)], text=True))
    local = inventory(REPO / RELATIVE)
    assert len(local) == len(remote) == 46
    assert local == remote
    assert sum(row["frames"] for row in local.values()) == 197796
    assert all(row["marker"]["stage"] == "transferred" for row in local.values())
    report = {"passed": True, "sequences": 46, "frames": 197796,
              "shards": sum(len(row["shard_sizes"]) for row in local.values()),
              "payload_bytes": sum(sum(row["shard_sizes"].values()) for row in local.values()),
              "scope": "양쪽 완료 metadata SHA256·전송 마커·모든 묶음 파일 크기 동일. payload 전체 SHA 재읽기 검사는 아니며 각 정규화 점군 해시는 GPU 생성기가 사용 직전 검사한다.",
              "original_failure_preserved": "producer_status.json 및 failure_records/의 과거 권한 오류는 보존한다.",
              "inventory": local}
    (REPO / RELATIVE / "delivery_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "inventory"})


if __name__ == "__main__":
    main()
