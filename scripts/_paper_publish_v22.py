#!/usr/bin/env python3
"""검증된 표·그림으로 원고를 빌드하고 승인된 드롭박스 사본을 갱신한다."""

import argparse
import base64
import contextlib
import datetime
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "docs/paper/body_v2.2"
DROPBOX = Path("/mnt/c/Users/qkrrl/Dropbox/Fourier-Compressed LiDAR/body_v2.2")
LOCK = Path("/tmp/nsd_paper_publish_v22.lock")
PDF_STEMS = ("cvpr2027_submission", "cvpr2027_supplement", "cvpr2027_anonymous")
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")


def digest(path: Path) -> str:
    """파일 본문의 SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """중간 JSON이 읽히지 않도록 원자적으로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def publication_lock():
    """표·그림 담당자가 같은 원고를 동시에 빌드하거나 복사하지 않게 한다."""
    with LOCK.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def cloud_states(target: Path, relative_paths: List[str]) -> list:
    """Windows Cloud Files의 읽기 전용 API로 각 파일의 동기화 상태를 읽는다."""
    windows_root = subprocess.check_output(["wslpath", "-w", str(target)], text=True).strip()
    names = "@(" + ",".join("'" + name.replace("/", "\\").replace("'", "''") + "'"
                             for name in relative_paths) + ")"
    script = r'''[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$ProgressPreference='SilentlyContinue'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class CloudFileCheck {
 [StructLayout(LayoutKind.Sequential)] public struct AttrTag {public uint Attributes; public uint ReparseTag;}
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern SafeFileHandle CreateFileW(string name,uint access,uint share,IntPtr security,uint disposition,uint flags,IntPtr template);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GetFileInformationByHandleEx(SafeFileHandle handle,int kind,out AttrTag info,uint size);
 [DllImport("cldapi.dll")] public static extern uint CfGetPlaceholderStateFromAttributeTag(uint attributes,uint tag);
 public static uint State(string path) {
  using(var h=CreateFileW(path,0,7,IntPtr.Zero,3,0,IntPtr.Zero)) {
   if(h.IsInvalid) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
   AttrTag info; if(!GetFileInformationByHandleEx(h,9,out info,8)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
   return CfGetPlaceholderStateFromAttributeTag(info.Attributes,info.ReparseTag);
  }
 }
}
'@
$root='ROOT_PATH'
$out=@()
foreach($name in FILE_NAMES){$state=[CloudFileCheck]::State((Join-Path $root $name)); $out+=@{file=$name;state=$state;in_sync=(($state -ne 4294967295) -and (($state -band 8) -ne 0))}}
ConvertTo-Json -InputObject $out -Compress
'''.replace("ROOT_PATH", windows_root.replace("'", "''")).replace("FILE_NAMES", names)
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run([str(POWERSHELL), "-NoProfile", "-NonInteractive",
                             "-EncodedCommand", encoded], capture_output=True, check=True)
    return json.loads(result.stdout.decode("utf-8-sig"))


def paper_files() -> List[Path]:
    """원고 배포 파일만 나열한다. 기존 그림 1–3은 복사하지 않는다."""
    paths = list(PAPER.glob("*.tex"))
    paths += [PAPER / (name + ".pdf") for name in PDF_STEMS]
    paths += [PAPER / name for name in ("references.bib", "cvpr.sty", "ieeenat_fullname.bst",
                                      "Makefile", "main_revision_manifest.json",
                                      "publication_manifest.json")]
    paths += list((PAPER / "figs").glob("fig_efficiency.*"))
    paths += list((PAPER / "figs").glob("fig_efficiency_*.json"))
    paths += [p for p in (PAPER / "figs/pair_analysis").rglob("*") if p.is_file()]
    paths += [p for p in (PAPER / "figs/review_current").rglob("*") if p.is_file()]
    for directory in ("tables_v2", "tables_rev"):
        paths += [p for p in (PAPER / directory).rglob("*") if p.is_file()
                  and (p.suffix in (".tex", ".json") or p.name.endswith(".tex.in"))]
    return sorted(set(p for p in paths if p.is_file()))


def publish(args: argparse.Namespace) -> Dict[str, object]:
    """공통 잠금 아래 생성·빌드·해시 검사·드롭박스 동기화를 실행한다."""
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    output = REPO / "results/paper_publish_v22" / stamp
    output.mkdir(parents=True)
    state = {"status": "started", "reason": args.reason, "output": str(output.relative_to(REPO))}
    write_json(output / "status.json", state)
    try:
        with publication_lock():
            commands = [] if args.skip_tables else [["make", "tables"]]
            if args.refresh_figure:
                commands.append([args.figure_python, str(REPO / "scripts/_paper_figure4.py"),
                                 "--summary", str(REPO / "results/pdv2_table3_20261029/summary.json"),
                                 "--render"])
            commands.append(["make"])
            for index, command in enumerate(commands):
                state.update(status="building", command=command)
                write_json(output / "status.json", state)
                with (output / ("build_%d.log" % index)).open("w") as log:
                    subprocess.run(command, cwd=PAPER, stdout=log, stderr=subprocess.STDOUT, check=True)
            for stem in PDF_STEMS:
                log = (PAPER / (stem + ".log")).read_text()
                failures = ("undefined references", "multiply defined", "Overfull", "Fatal error")
                if "Output written on" not in log or any(token in log for token in failures):
                    raise RuntimeError("원고 로그 검증 실패: " + stem)
            manifest = {"built_at_kst": stamp, "reason": args.reason,
                        "pdf_sha256": {s + ".pdf": digest(PAPER / (s + ".pdf")) for s in PDF_STEMS},
                        "existing_figures_not_published": True,
                        "sources": {str(p.relative_to(PAPER)): digest(p) for p in paper_files()
                                    if p.name != "publication_manifest.json"}}
            write_json(PAPER / "publication_manifest.json", manifest)
            if args.check_only:
                state.update(status="build_checked", pdf_sha256=manifest["pdf_sha256"])
                write_json(output / "status.json", state)
                return state
            if not DROPBOX.is_dir():
                raise FileNotFoundError("승인된 Dropbox 폴더가 없다: " + str(DROPBOX))
            changed = []
            for source in paper_files():
                relative = source.relative_to(PAPER)
                target = DROPBOX / relative
                if target.exists() and digest(source) == digest(target):
                    continue
                if target.exists():
                    backup = output / "dropbox_before" / relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if digest(source) != digest(target):
                    raise RuntimeError("Dropbox 복사 해시 불일치: " + str(relative))
                changed.append(str(relative))
            # 변경 파일뿐 아니라 핵심 PDF도 매번 클라우드 완료를 확인한다.
            verify = sorted(set(changed + [name + ".pdf" for name in PDF_STEMS]))
            state.update(status="copied", copied=changed, hashes_verified=True)
            write_json(output / "status.json", state)
            deadline = time.monotonic() + args.sync_timeout
            while True:
                states = cloud_states(DROPBOX, verify)
                write_json(output / "cloud_states.json", {"files": states})
                if all(item["in_sync"] for item in states):
                    state.update(status="synced", cloud_verified=len(states))
                    break
                if time.monotonic() >= deadline:
                    state.update(status="sync_pending", pending=[x["file"] for x in states if not x["in_sync"]])
                    break
                time.sleep(5)
            write_json(output / "status.json", state)
            return state
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        write_json(output / "status.json", state)
        raise


def main() -> None:
    """공통 게시 명령줄 옵션을 읽는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", required=True, help="이번 반영의 측정 단계 또는 그림 변경")
    parser.add_argument("--refresh-figure", action="store_true")
    parser.add_argument("--figure-python", default=sys.executable)
    parser.add_argument("--check-only", action="store_true", help="생성과 빌드까지만 검증하고 복사하지 않음")
    parser.add_argument("--skip-tables", action="store_true", help="그림·본문 편집에서는 기존 표를 유지하고 빌드")
    parser.add_argument("--sync-timeout", type=int, default=180)
    args = parser.parse_args()
    result = publish(args)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "sync_pending":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
