#!/usr/bin/env python3
"""v2.4를 빌드·검산하고 새 드롭박스 버전 폴더에 반영한다."""

import argparse
import datetime
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

from _paper_publish_v22 import cloud_states, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper/body_v2.4"
TARGET = Path("/mnt/c/Users/qkrrl/Dropbox/Fourier-Compressed LiDAR/body_v2.4")
STEMS = ("cvpr2027_submission", "cvpr2027_supplement", "cvpr2027_anonymous")
# 빌드에 쓰이지 않는 편집용 원본만 보존 대상으로 허용한다.
PRESERVABLE_SUFFIXES = (".pptx",)
PRESERVE_REASON = ("명시 보존: 기존 드롭박스 파일이 원천과 달라 사용자 편집일 수 있으므로 "
                   "수정·백업 쓰기 없이 읽기 전용 해시만 기록한다")


def files() -> list:
    """새 버전을 독립적으로 읽고 빌드하는 데 필요한 파일을 나열한다."""
    result = [p for p in PAPER.iterdir() if p.is_file() and
              (p.suffix in (".tex", ".bib", ".sty", ".bst", ".json", ".txt")
               or p.name == "Makefile")]
    result += [PAPER / (name + ".pdf") for name in STEMS]
    for name in ("figs", "tables_v3", "baseline_fragments"):
        result += [p for p in (PAPER / name).rglob("*") if p.is_file()]
    return sorted(set(result))


def preserved_paths(relatives: list) -> list:
    """보존할 상대 경로를 검사한다. 누락·잘못된 경로·경로 탈출은 실패시킨다."""
    sources = {str(p.relative_to(PAPER)) for p in files()}
    result = []
    for relative in relatives:
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts or str(path) != relative:
            raise SystemExit(f"보존 경로는 정규화된 상대 경로여야 한다: {relative!r}")
        if relative not in sources:
            raise SystemExit(f"게시 대상 파일이 아니다: {relative}")
        if path.suffix not in PRESERVABLE_SUFFIXES:
            raise SystemExit(f"빌드와 무관한 편집용 원본({PRESERVABLE_SUFFIXES})만 보존한다: {relative}")
        target = TARGET / path
        if target.is_symlink() or not target.is_file():
            raise SystemExit(f"보존할 드롭박스 파일이 없거나 일반 파일이 아니다: {target}")
        if TARGET.resolve() not in target.resolve().parents:
            raise SystemExit(f"드롭박스 폴더를 벗어난다: {target}")
        if relative not in result:
            result.append(relative)
    return result


def snapshot(relative: str) -> dict:
    """보존 파일을 읽기만 해서 원천·대상 해시를 기록한다."""
    source, target = PAPER / relative, TARGET / relative
    stat = target.stat()
    return {"file": relative, "source_sha256": digest(source), "target_sha256": digest(target),
            "target_size": stat.st_size, "target_mtime_ns": stat.st_mtime_ns,
            "reason": PRESERVE_REASON}


def publish(sync_timeout: int, preserve: tuple = ()) -> dict:
    """이전 버전을 보존하고 해시·클라우드 동기화까지 확인한다."""
    preserve = preserved_paths(list(preserve))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "results/paper_publish_v24" / stamp
    output.mkdir(parents=True)
    state = {"status": "building", "target": str(TARGET), "version": "2.4"}
    if preserve:
        state["preserve_target_files"] = preserve
    write_json(output / "status.json", state)
    with Path("/tmp/nsd_paper_publish_v24.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # 표는 v2.4용 생성기가 따로 만든다. 이전 생성기(make tables)로 덮어쓰지 않는다.
        for index, command in enumerate((["make"],)):
            with (output / f"build_{index}.log").open("w") as log:
                subprocess.run(command, cwd=PAPER, stdout=log, stderr=subprocess.STDOUT, check=True)
        for stem in STEMS:
            log = (PAPER / (stem + ".log")).read_text()
            assert "Output written on" in log, stem
            assert not any(s in log for s in ("undefined references", "multiply defined",
                                              "Overfull", "Fatal error", "Citation `")), stem
        manifest = {"version": "2.4", "built_at_kst": stamp,
                    "target": str(TARGET), "figure_assets_preserved": True,
                    "files": {str(p.relative_to(PAPER)): digest(p) for p in files()
                              if p.name != "publication_manifest.json"}}
        kept = [snapshot(relative) for relative in preserve]
        for entry in kept:
            entry["identical"] = entry["source_sha256"] == entry["target_sha256"]
        if kept:
            manifest["preserved_target_files"] = kept
        write_json(PAPER / "publication_manifest.json", manifest)
        TARGET.mkdir(parents=True, exist_ok=True)
        changed = []
        for source in files():
            relative = source.relative_to(PAPER)
            target = TARGET / relative
            if str(relative) in preserve:
                continue
            if target.exists() and digest(target) == digest(source):
                continue
            if target.exists():
                backup = output / "dropbox_before" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert digest(source) == digest(target), relative
            changed.append(str(relative))
        verify = [str(p.relative_to(PAPER)) for p in files() if str(p.relative_to(PAPER)) not in preserve]
        state.update(status="copied", copied=len(changed), files=len(verify), hashes_verified=True)
        if kept:
            state.update(hashes_verified_scope="보존 파일을 뺀 나머지 파일",
                         preserved_target_files=kept)
        write_json(output / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        deadline = time.monotonic() + sync_timeout
        while True:
            states = []
            for start in range(0, len(verify), 40):
                states += cloud_states(TARGET, verify[start:start + 40])
            write_json(output / "cloud_states.json", {"files": states})
            pending = [x["file"] for x in states if not x["in_sync"]]
            if not pending:
                state.update(status="synced", cloud_verified=len(states))
                break
            if time.monotonic() >= deadline:
                state.update(status="sync_pending", pending=pending)
                break
            time.sleep(5)
        moved = []
        for entry in kept:
            now = snapshot(entry["file"])
            entry.update(target_sha256_after=now["target_sha256"],
                         target_mtime_ns_after=now["target_mtime_ns"],
                         target_changed_during_publish=(now["target_sha256"] != entry["target_sha256"]
                                                        or now["target_mtime_ns"] != entry["target_mtime_ns"]))
            if entry["target_changed_during_publish"]:
                moved.append(entry["file"])
        if moved:
            state.update(status="preserved_target_changed", preserved_target_changed=moved,
                         retry="보존 파일이 게시 중 바뀌었다. 덮어쓰지 않았으며 매니페스트 해시를 맞추려면 다시 실행한다")
        write_json(output / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return state


def main() -> None:
    """동기화 확인 제한 시간을 받아 게시한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-timeout", type=int, default=180)
    parser.add_argument("--preserve-target-file", action="append", default=[],
                        help="덮어쓰지 않을 드롭박스 파일의 원고 폴더 기준 상대 경로(.pptx만)")
    args = parser.parse_args()
    result = publish(args.sync_timeout, tuple(args.preserve_target_file))
    if result["status"] != "synced":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
