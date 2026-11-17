#!/usr/bin/env python3
"""로컬 CPU·전송·GPU 제어 상태를 서버의 별도 복구 폴더에 보존한다."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_review_selection_full_20261105"
NAMES = (
    "local_gpu_state.json", "local_gpu_control.json", "local_gpu_plan.json",
    "local_test_riv_state.json", "local_test_riv_plan.json", "test_external_input_owners.json",
    "test_relay_status.json", "test_relay_control.json", "test_request_gate.json",
    "transfer_drain_plan.json", "transfer_drain_state.json", "restart_runtime_latest.json",
    "campaign_scope.json", "execution_hosts.json", "transfer_parameter_guard.json",
    "local_helios_plan.json", "local_helios_assignment_proposal.json",
    "helios_numerical_margin_request.json", "helios_numerical_override.json",
    "helios_test_partition_policy.json", "local_helios_test_plan.json",
)
DIRECTORIES = (
    "local_riv_campaign", "test_input_requests", "test_relay_status", "transfer_drain_backups",
    "gpu_recovery_20261106_0634",
    "local_helios_inputs", "helios_local_execution_smoke", "local_helios_queue",
    "relay_cleanup_recovery",
    "helios_test_partition", "local_helios_test_queue", "local_helios_test_inputs",
)


def main() -> None:
    """실행 중 서버의 제어 파일을 덮지 않는 복구 사본만 동기화한다."""
    lock = (OUT / "runtime_backup.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = OUT / "local_runtime_snapshot"
    snapshot.mkdir(exist_ok=True)
    while True:
        manifest = {}
        skipped = []
        paths = [OUT / n for n in NAMES if (OUT / n).exists()]
        paths += [p for d in DIRECTORIES for p in (OUT / d).rglob("*.json")]
        for source in paths:
            try:
                raw = source.read_bytes()
                json.loads(raw)
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                skipped.append(str(source.relative_to(OUT)))
                continue
            relative = source.relative_to(OUT)
            dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            manifest[str(relative)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        record = dict(captured_unix=time.time(), files=manifest, retry_unstable_files=skipped,
            note="각 파일의 원자적 JSON 사본; 복구 시 실제 PID·원장과 대조. 서버 실행 제어를 덮어쓰지 않음")
        (snapshot / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        listing = OUT / "runtime_backup_files.txt"
        listing.write_text("\n".join([*manifest, "manifest.json"]) + "\n")
        result = subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms",
            "--files-from=" + str(listing), str(snapshot) + "/",
            "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/results/"
            "pdv2_review_selection_full_20261105/local_runtime_backup/"], check=False)
        (OUT / "runtime_backup_status.json").write_text(json.dumps(dict(pid=os.getpid(),
            status="synced" if result.returncode == 0 else "retry_pending",
            returncode=result.returncode, file_count=len(manifest), updated_unix=time.time()), indent=2))
        time.sleep(60)


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    main()
