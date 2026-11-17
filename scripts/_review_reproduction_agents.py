#!/usr/bin/env python3
"""명시적으로 승인된 Opus5.5 재현 작업자를 터미널에서 실행·점검한다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106"
CLI = Path("/mnt/d/NSD_tools/claude/2.1.280/claude")
MODEL = "claude-opus-5-5"
WORKERS = ("imlpr", "ring", "disco")


def write(path: Path, value: dict) -> None:
    """진행 정보를 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def live(pid: int) -> bool:
    """PID 재사용을 피하기 위해 실제 Claude 실행인지 확인한다."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        return bytes(str(CLI), "utf-8") in cmd and bytes(MODEL, "utf-8") in cmd
    except (OSError, ProcessLookupError):
        return False


def start(worker: str, followup: Path | None) -> dict:
    """허용된 역할 하나를 시작하고 로그·입력·PID를 보존한다."""
    probe = json.loads((OUT / "opus_probe_updated.json").read_text())
    assert probe["returncode"] == 0 and not probe["is_error"]
    assert MODEL in probe["modelUsage"]
    metadata = json.loads((OUT / "claude_cli_isolated.json").read_text())
    assert CLI.stat().st_size == metadata["bytes"]
    folder = OUT / worker
    current = folder / "launcher.json"
    if current.exists():
        prior = json.loads(current.read_text())
        if live(prior["pid"]):
            return {"worker": worker, "status": "already_running", "pid": prior["pid"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prompt = (OUT / "prompts" / f"{worker}.txt").read_text()
    if followup:
        prompt += "\n조정자의 후속 지시:\n" + followup.read_text()
    prompt_path = folder / f"prompt_{stamp}.txt"
    prompt_path.write_text(prompt)
    log_path = folder / f"events_{stamp}.jsonl"
    error_path = folder / f"stderr_{stamp}.log"
    allowed = "Read,Glob,Grep,Write,Edit,Bash,WebFetch,WebSearch"
    args = [str(CLI), "--safe-mode", "--model", MODEL,
            "--permission-mode", "dontAsk", "--tools", allowed,
            "--allowedTools", allowed, "--disallowedTools",
            "Bash(rm *)", "Bash(sudo *)", "Bash(git commit *)", "Bash(git push *)",
            "--no-session-persistence", "--verbose", "--output-format", "stream-json", "-p"]
    env = os.environ.copy()
    env["DISABLE_AUTOUPDATER"] = "1"
    with prompt_path.open("rb") as inp, log_path.open("wb") as out, error_path.open("wb") as err:
        proc = subprocess.Popen(args, stdin=inp, stdout=out, stderr=err, cwd=REPO,
                                env=env, start_new_session=True)
    record = {"worker": worker, "requested_model": MODEL, "pid": proc.pid,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "prompt": str(prompt_path.relative_to(REPO)),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "events": str(log_path.relative_to(REPO)), "stderr": str(error_path.relative_to(REPO)),
              "cli": str(CLI), "source": "사용자 지정 터미널 Opus5.5 에이전트"}
    write(current, record)
    return record


def status(worker: str) -> dict:
    """작업자의 실제 호출 모델·활동·결과를 작은 상태로 집계한다."""
    folder = OUT / worker
    path = folder / "launcher.json"
    if not path.exists():
        return {"worker": worker, "status": "not_started"}
    launch = json.loads(path.read_text())
    result = {"worker": worker, "pid": launch["pid"], "alive": live(launch["pid"]),
              "models": [], "tool_calls": 0, "last_tools": []}
    models, tool_names, final = set(), [], None
    log = REPO / launch["events"]
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                result["initialized_model"] = event.get("model")
            if event.get("type") == "assistant":
                message = event.get("message", {})
                if message.get("model"):
                    models.add(message["model"])
                for block in message.get("content", []):
                    if block.get("type") == "tool_use":
                        tool_names.append(block.get("name"))
            if event.get("type") == "result":
                final = {k: event.get(k) for k in ("is_error", "subtype", "result", "modelUsage", "total_cost_usd")}
    result.update(models=sorted(models), tool_calls=len(tool_names), last_tools=tool_names[-5:])
    if final:
        write(folder / "agent_result.json", final)
        result["agent_finished"] = True
        result["is_error"] = final["is_error"]
        result["result_preview"] = (final.get("result") or "")[:1200]
    progress = folder / "worker_status.json"
    if worker == "disco" and not progress.exists():
        progress = OUT / "disco_protocol/worker_status.json"
    if progress.exists():
        try:
            result["progress"] = json.loads(progress.read_text())
        except ValueError:
            result["progress_read"] = "write_in_progress"
    return result


def main() -> None:
    """중복 실행 없이 시작하거나 모든 작업자의 상태를 조회한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "watch"])
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--followup", type=Path)
    args = parser.parse_args()
    selected = [args.worker] if args.worker else list(WORKERS)
    if args.action == "watch":
        write(OUT / "supervisor.json", {"pid": os.getpid(), "stage": "monitoring",
              "workers": selected, "interval_s": 30,
              "scope": "실제 프로세스·호출 모델·도구 실행·최종 응답 감시; 작업자의 결과를 완료로 가정하지 않음"})
        while True:
            rows = [status(w) for w in selected]
            unexpected = [r["worker"] for r in rows if any(m != MODEL for m in r.get("models", []))]
            write(OUT / "agents_status.json", {"checked_utc": datetime.now(timezone.utc).isoformat(),
                  "workers": rows, "unexpected_models": unexpected})
            if not any(r.get("alive") for r in rows):
                write(OUT / "supervisor.json", {"pid": os.getpid(), "stage": "workers_exited",
                      "workers": selected, "checked_utc": datetime.now(timezone.utc).isoformat(),
                      "unfinished_or_error": [r["worker"] for r in rows if not r.get("agent_finished") or r.get("is_error")],
                      "note": "실험 완료와 작업자 종료는 다르다. 연결된 실행기의 실제 산출물은 별도 검산한다."})
                return
            time.sleep(30)
    rows = [start(w, args.followup) if args.action == "start" else status(w) for w in selected]
    if args.action == "status":
        write(OUT / "agents_status.json", {"checked_utc": datetime.now(timezone.utc).isoformat(), "workers": rows})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
