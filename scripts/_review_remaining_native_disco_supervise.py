#!/usr/bin/env python3
"""서버 호스트에서 DiSCO 자료 확보→GPU0 락 추론→CPU 집계→독립 검산을 감독한다.

cd /home/qkrrlans00/Neural-Spectral-Codec && setsid nohup python3 -u \
    scripts/_review_remaining_native_disco_supervise.py \
    > results/pdv2_reproduction_orchestration_20261106/disco_protocol/logs/supervisor.log 2>&1 &
호스트 표준 라이브러리만 쓴다. 실행 중인 다운로더는 건드리지 않고, 완료 표시 없이 끝난 날짜만
같은 fetch 명령으로 다시 띄운다(부분 색인·추출 파일 재사용). GPU 단계만 flock으로 직렬화한다.
추론은 `flock -x` 자식 프로세스로 줄을 서서 기다린다(대기 중에는 락을 갖지 않는다). 다른 작업이
조건 사이에 락을 곧바로 다시 잡아도 해제 순간 깨어나며, 그동안 감독기는 다운로드 감시를 계속한다.
락을 얻은 뒤 `encode --dates auto`가 그 시점에 준비된 날짜를 모두 추론한다.
다시 실행하면 상태 파일과 완료 표시를 읽고 남은 단계부터 이어 간다. 이미 줄 선 추론 자식이 있으면
새로 띄우지 않는다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results/pdv2_remaining_native_20261106/disco"
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/disco_protocol"
LOGS = OUT / "logs"
LOCK = Path("/tmp/nsd_native_reproduction_gpu0.lock")
DATES = ["2012-03-17", "2012-05-26", "2012-06-15", "2012-08-20", "2012-09-28", "2012-10-28"]
FETCHED = DATES[1:]
MAX_RESTARTS = 6
MAX_STAGE_FAILURES = 2
STALL_S = 1800
POLL_S = 30
EXEC = ["docker", "exec", "-w", "/workspace/Neural-Spectral-Codec"]
THREADS = ["-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1"]
CPU_ENV = ["-e", "CUDA_VISIBLE_DEVICES="] + THREADS
GPU_ENV = ["-e", "CUDA_VISIBLE_DEVICES=0", "-e", "NVIDIA_TF32_OVERRIDE=0"] + THREADS
FETCH_ENV = ["-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1"]
EVAL = ["ring_baseline", "python3", "-u", "scripts/_review_remaining_native_disco_eval.py"]
HELD = {"340070": "_review_selection_full_queue (보류, 건드리지 않음)",
        "1083011": "_review_selection_full_server_riv (보류, 건드리지 않음)"}


def write(path: Path, value: dict) -> None:
    """JSON을 원자적으로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read(path: Path) -> dict | None:
    """쓰는 중인 파일은 건너뛰고 JSON을 읽는다."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def processes(script: str) -> dict[int, list[str]]:
    """호스트에서 보이는 파이썬 프로세스 중 해당 스크립트를 실행하는 것을 찾는다."""
    found = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if args and Path(args[0]).name.startswith("python") and any(a.endswith(script) for a in args):
            found[int(proc.name)] = args
    return found


def queued_encoders() -> list[int]:
    """락을 기다리거나 추론 중인 flock 자식(이전 감독기가 띄운 것 포함)."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if args and Path(args[0]).name == "flock" and str(LOCK) in args \
                and any(a.endswith("_review_remaining_native_disco_eval.py") for a in args):
            found.append(int(proc.name))
    return found


def fetchers() -> dict[str, int]:
    """날짜별 fetch 프로세스 PID."""
    alive = {}
    for pid, args in processes("_review_remaining_native_disco_data.py").items():
        if "fetch" in args and "--dates" in args:
            alive[args[args.index("--dates") + 1]] = pid
    return alive


def tail(path: Path, size: int = 1500) -> str:
    """로그 끝부분만 읽는다."""
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, path.stat().st_size - size))
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


class Supervisor:
    """감독 상태와 단계 실행을 관리한다."""

    def __init__(self) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        self.state_path = OUT / "supervisor_state.json"
        self.state = read(self.state_path) or {"restarts": {d: 0 for d in FETCHED}, "next_restart": {},
                                                "stage_failures": {}, "events": [], "stall_reported": {}}
        self.children: dict[str, subprocess.Popen] = {}
        self.encoder: subprocess.Popen | None = None
        self.gpu_note = None

    def event(self, kind: str, **detail) -> None:
        self.state["events"].append({"unix": time.time(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                     "kind": kind, **detail})
        self.state["events"] = self.state["events"][-60:]
        write(self.state_path, self.state)
        print(json.dumps(self.state["events"][-1], ensure_ascii=False), flush=True)

    def restart_fetch(self, date: str) -> None:
        """완료 표시 없이 끝난 날짜를 같은 명령으로 다시 띄운다."""
        n = self.state["restarts"].get(date, 0) + 1
        self.state["restarts"][date] = n
        self.state["next_restart"][date] = time.time() + 120 * n
        log = open(LOGS / f"fetch_{date}.restart{n}.log", "ab")
        command = EXEC + FETCH_ENV + ["ring_baseline", "python3", "-u",
                                      "scripts/_review_remaining_native_disco_data.py", "fetch", "--dates", date]
        self.children[date] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
        error = read(DATA / f"download_error_{date}.json")
        self.event("fetch_restarted", date=date, attempt=n, docker_exec_pid=self.children[date].pid,
                   previous_error=error)

    def run_stage(self, name: str, command: list[str], timeout: int) -> bool:
        """한 단계를 실행하고 실패 원인을 상태에 남긴다."""
        log_path = LOGS / f"{name}.log"
        started = time.time()
        with open(log_path, "ab") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {' '.join(command)}\n".encode())
            log.flush()
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        ok = code == 0
        if ok:
            self.event(f"{name}_complete", elapsed_s=round(time.time() - started, 1))
        else:
            failures = self.state["stage_failures"].get(name, 0) + 1
            self.state["stage_failures"][name] = failures
            self.event(f"{name}_failed", returncode=code, failures=failures, log_tail=tail(log_path),
                       pipeline_error=read(DATA / "pipeline_error.json"))
        return ok

    def queue_encode(self, pending: list[str]) -> None:
        """GPU0 락 줄에 추론 자식을 세운다. 감독기 자신은 락을 기다리며 멈추지 않는다."""
        if queued_encoders():
            self.gpu_note = "이전 감독기가 띄운 추론 자식이 락 대기 또는 실행 중; 중복으로 띄우지 않음"
            return
        if processes("_review_remaining_native_disco_eval.py"):
            self.gpu_note = "다른 DiSCO 평가기 실행 중; 중복 추론하지 않음"
            return
        if self.state["stage_failures"].get("encode", 0) >= MAX_STAGE_FAILURES:
            self.gpu_note = "encode 반복 실패; 원인 수정 필요"
            return
        attempt = self.state.get("encode_attempts", 0) + 1
        self.state["encode_attempts"] = attempt
        marker = LOGS / f"encode_lock_acquired_{attempt}.txt"
        # 락을 얻은 순간을 표시 파일에 남긴 뒤 같은 프로세스로 docker exec를 실행한다.
        command = ["flock", "-x", str(LOCK), "bash", "-c", 'date +%s.%N > "$0"; exec "$@"', str(marker)] \
            + EXEC + GPU_ENV + EVAL + ["encode", "--dates", "auto"]
        with open(LOGS / "encode.log", "ab") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} queued attempt {attempt}: "
                      f"{' '.join(command)}\n".encode())
            log.flush()
            self.encoder = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.state["last_gpu_run"] = {"attempt": attempt, "pending_at_queue": pending, "queued_unix": time.time(),
                                      "flock_pid": self.encoder.pid, "lock_marker": str(marker.relative_to(REPO))}
        self.gpu_note = f"GPU0 락 줄 대기(flock 자식 PID {self.encoder.pid}); 대기 중 락 미보유"
        self.event("encode_queued", attempt=attempt, flock_pid=self.encoder.pid, pending=pending)

    def lock_acquired(self) -> float | None:
        """현재 추론 자식이 락을 얻은 시각."""
        run = self.state.get("last_gpu_run") or {}
        try:
            return float((REPO / run["lock_marker"]).read_text().strip())
        except (KeyError, OSError, ValueError):
            return None

    def poll_encoder(self) -> bool:
        """줄 선 추론 자식을 확인한다. 아직 대기·실행 중이면 True."""
        if self.encoder is None:
            return False
        run = self.state["last_gpu_run"]
        acquired = self.lock_acquired()
        code = self.encoder.poll()
        if code is None:
            if acquired is None:
                self.gpu_note = f"GPU0 락 줄 대기(flock 자식 PID {self.encoder.pid}); 대기 중 락 미보유"
            else:
                self.gpu_note = f"GPU0 락 획득({time.strftime('%H:%M:%S', time.localtime(acquired))}), encode --dates auto 실행 중"
                if time.time() - acquired > 3600 and not run.get("overdue_reported"):
                    # docker exec 클라이언트를 죽여도 컨테이너 안 추론은 남고 락만 풀리므로 끊지 않고 알린다.
                    run["overdue_reported"] = True
                    self.event("encode_overdue", attempt=run["attempt"], held_s=round(time.time() - acquired))
            return True
        self.encoder = None
        run.update(returncode=code, lock_acquired_unix=acquired, finished_unix=time.time())
        if code == 0 and acquired is not None:
            self.event("encode_complete", attempt=run["attempt"], waited_s=round(acquired - run["queued_unix"], 1),
                       held_s=round(run["finished_unix"] - acquired, 1),
                       encoded=[d for d in DATES if (DATA / "encoded" / d / "complete.json").exists()])
        else:
            failures = self.state["stage_failures"].get("encode", 0) + 1
            self.state["stage_failures"]["encode"] = failures
            self.event("encode_failed", attempt=run["attempt"], returncode=code, failures=failures,
                       log_tail=tail(LOGS / "encode.log"), pipeline_error=read(DATA / "pipeline_error.json"))
        self.gpu_note = "GPU0 락 해제"
        return False

    def status(self, stage: str, dates: dict | None = None) -> None:
        """root가 읽는 worker_status.json을 쓴다."""
        encoded = [d for d in DATES if (DATA / "encoded" / d / "complete.json").exists()]
        remaining = []
        if len(encoded) < len(DATES):
            remaining.append(f"추론 대기: {[d for d in DATES if d not in encoded]}")
        if not (DATA / "report.json").exists():
            remaining.append("30쌍 집계(score)")
        if not (OUT / "verification.json").exists():
            remaining.append("독립 검산·보고서(verify)")
        write(OUT / "worker_status.json", {
            "worker": "disco_protocol supervisor (Opus 5.5)", "stage": stage,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "updated_unix": time.time(),
            "supervisor_host_pid": os.getpid(), "dates": dates or self.state.get("last_dates", {}),
            "encoded_sessions": encoded, "gpu": {"lock": str(LOCK), "lock_resolved": str(LOCK.resolve()), "note": self.gpu_note,
                                                  "last_run": self.state.get("last_gpu_run")},
            "stage_failures": self.state["stage_failures"],
            "paths": {"server_repo": str(REPO), "data": str(DATA.relative_to(REPO)), "protocol": str(OUT.relative_to(REPO)),
                      "logs": str(LOGS.relative_to(REPO)), "report": str((OUT / "report.json").relative_to(REPO))},
            "remaining_dependencies": remaining, "held_jobs_untouched": HELD,
            "resume": "cd /home/qkrrlans00/Neural-Spectral-Codec && setsid nohup python3 -u scripts/_review_remaining_native_disco_supervise.py >> results/pdv2_reproduction_orchestration_20261106/disco_protocol/logs/supervisor.log 2>&1 &",
            "recent_events": self.state["events"][-8:]})

    def downloads(self) -> dict:
        """날짜별 자료 상태를 확인하고 끝난 실패만 다시 띄운다."""
        alive, now, rows = fetchers(), time.time(), {}
        for date, child in list(self.children.items()):
            if child.poll() is not None:
                del self.children[date]
        for date in FETCHED:
            progress = read(DATA / f"download_status_{date}.json") or {}
            path = DATA / f"download_status_{date}.json"
            age = now - path.stat().st_mtime if path.exists() else None
            row = {"pid": alive.get(date), "restarts": self.state["restarts"].get(date, 0),
                   "stage": progress.get("stage"), "status_age_s": None if age is None else round(age),
                   "network_bytes": progress.get("network_bytes"), "block_retries": progress.get("block_retries"),
                   "compressed_fraction": round(progress["compressed_position"] / progress["archive_bytes"], 4)
                   if progress.get("archive_bytes") else None}
            if (DATA / "native_data" / date / "complete.json").exists():
                row["status"] = "ready"
            elif date in alive or date in self.children:
                row["status"] = "fetching"
                if age is not None and age > STALL_S:
                    row["status"] = "fetching_status_stale"
                    if not self.state["stall_reported"].get(date):
                        self.state["stall_reported"][date] = True
                        self.event("fetch_status_stale", date=date, age_s=round(age), pid=alive.get(date))
                else:
                    self.state["stall_reported"][date] = False
            elif self.state["restarts"].get(date, 0) >= MAX_RESTARTS:
                row["status"] = "failed_max_restarts"
            elif now < self.state["next_restart"].get(date, 0):
                row["status"] = "restart_backoff"
            else:
                self.restart_fetch(date)
                row["status"] = "restarted"
            rows[date] = row
        self.state["last_dates"] = rows
        return rows

    def loop(self) -> None:
        self.event("supervisor_started", pid=os.getpid())
        while True:
            rows = self.downloads()
            busy = self.poll_encoder() or bool(queued_encoders())
            ready = [d for d in DATES if (DATA / "native_data" / d / "complete.json").exists()]
            pending = [d for d in ready if not (DATA / "encoded" / d / "complete.json").exists()]
            if pending and not busy:
                self.queue_encode(pending)
                busy = self.encoder is not None
            encoded = all((DATA / "encoded" / d / "complete.json").exists() for d in DATES)
            failures = self.state["stage_failures"]
            if encoded and not busy and not (DATA / "report.json").exists() \
                    and failures.get("score", 0) < MAX_STAGE_FAILURES:
                self.status("scoring", rows)
                self.run_stage("score", EXEC + CPU_ENV + EVAL + ["score"], timeout=7200)
            if (DATA / "report.json").exists() and not (OUT / "verification.json").exists() \
                    and failures.get("verify", 0) < MAX_STAGE_FAILURES:
                self.status("verifying", rows)
                self.run_stage("verify", EXEC + CPU_ENV + EVAL + ["verify"], timeout=7200)
            if (OUT / "verification.json").exists() and (OUT / "report.json").exists():
                verification = read(OUT / "verification.json") or {}
                self.status("complete" if verification.get("passed") else "complete_verification_failed", rows)
                self.event("supervisor_finished", verification_passed=verification.get("passed"))
                return
            blocked = [n for n, c in failures.items() if c >= MAX_STAGE_FAILURES]
            failed = [d for d, r in rows.items() if r["status"] == "failed_max_restarts"]
            stage = "blocked_needs_fix" if blocked or failed else \
                "encoding" if busy and self.lock_acquired() is not None else \
                "waiting_for_data" if len(ready) < len(DATES) else \
                "waiting_for_gpu_lock" if busy or pending else "running"
            self.status(stage, rows)
            time.sleep(POLL_S)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-failures", action="store_true", help="원인 수정 뒤 단계 실패 횟수를 0으로 되돌린다")
    args = parser.parse_args()
    supervisor = Supervisor()
    if args.reset_failures:
        supervisor.state["stage_failures"] = {}
        supervisor.event("stage_failures_reset")
    supervisor.loop()
