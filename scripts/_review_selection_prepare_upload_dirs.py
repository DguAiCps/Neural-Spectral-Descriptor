#!/usr/bin/env python3
"""고정한 로컬 분담 결과의 정확한 디렉터리에만 업로드 권한을 준비한다."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def main(role: str, uid: int, gid: int) -> None:
    """새 실험의 배정된 후보 디렉터리만 다루고 파일 소유권은 보존한다."""
    plan_path = OUT / ("local_helios_plan.json" if role == "val" else "local_helios_test_plan.json")
    plan = read(plan_path)
    assert plan.get("role", "val") == role
    base = OUT / "retained" / role / "helios"
    changes = []
    for seed, candidate in sorted({(job["seed"], job["candidate"]) for job in plan["jobs"]}):
        assert "/" not in candidate and ".." not in candidate
        leaf = base / f"s{seed}" / candidate
        for directory in [*reversed(leaf.parents), leaf]:
            try:
                directory.relative_to(OUT)
            except ValueError:
                continue
            assert not directory.is_symlink(), directory
            if not directory.exists():
                directory.mkdir()
                os.chown(directory, uid, gid)
                changes.append(dict(path=str(directory.relative_to(ROOT)), created=True))
        previous = leaf.stat()
        if previous.st_uid != uid or previous.st_gid != gid:
            os.chown(leaf, uid, gid)
            changes.append(dict(path=str(leaf.relative_to(ROOT)), created=False,
                                previous_uid=previous.st_uid, previous_gid=previous.st_gid))
    write(OUT / f"local_helios_upload_dirs_{role}.json", dict(changes=changes,
        uid=uid, gid=gid, individual_files_changed=False, original_data_changed=False,
        plan_sha256=sha(plan_path), completed_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("val", "test"), required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    args = parser.parse_args()
    main(args.role, args.uid, args.gid)
