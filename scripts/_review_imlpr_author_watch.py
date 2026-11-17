#!/usr/bin/env python3
"""ImLPR 저자 data.tar.gz 다운로드 완료를 감지해 서버 추출·평가·검산까지 잇는다.

로컬 호스트(WSL)에서 torch 없이 실행한다. 무거운 압축 해제·추론은 서버 컨테이너가 맡는다.

    setsid nohup python3 -u scripts/_review_imlpr_author_watch.py watch \
        > results/pdv2_reproduction_orchestration_20261106/imlpr/watch.log 2>&1 &

단계: 다운로드 완료(이름 변경·크기 안정·gzip 머리) → SSH 스트림 추출
(scripts/_review_imlpr_author_extract.py) → 서버 평가 체인
(scripts/_review_imlpr_author_eval.py chain) → 결과 동기화 → 로컬 독립 검산·보고.
상태는 worker_status.json, 재개용 상태는 watch_state.json에 둔다.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REL_OUT = "results/pdv2_reproduction_orchestration_20261106/imlpr"
OUT = REPO / REL_OUT
RUN = OUT / "run"
DOWNLOADS = Path("/mnt/d/NSD_downloads")
FINAL_NAME = re.compile(r"^(data|imlpr_author_data)( \(\d+\))?\.tar\.gz$")
HOST = "joms-gpu"
REMOTE_REPO = "/home/qkrrlans00/Neural-Spectral-Codec"
CONTAINER_REPO = "/workspace/Neural-Spectral-Codec"
SCRIPTS = ["scripts/_review_imlpr_author_extract.py", "scripts/_review_imlpr_author_eval.py",
           "scripts/_review_imlpr_author_watch.py", "scripts/_review_imlpr_author_town_check.py"]
DRIVE_LISTED_SIZE = "27.82GB (공식 Drive 폴더 목록, 단위 표기 그대로)"
POLL_S = 60
STABLE_POLLS = 3
KST = timezone(timedelta(hours=9))
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=20", HOST]


def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ssh(cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)


STATE_PATH = OUT / "watch_state.json"
STATUS_PATH = OUT / "worker_status.json"


def load_state() -> dict:
    return read_json(STATE_PATH, {}) or {}


def save_state(state: dict) -> None:
    write_json(STATE_PATH, state)


def evidence() -> dict:
    """이미 끝난 근거 파일(프로토콜 고정·자체시험·공개 경로 조사)의 위치와 핵심 값."""
    proto = RUN / "protocol.json"
    st = read_json(OUT / "selftest_summary.json", {}) or {}
    return {"protocol": {"path": f"{REL_OUT}/run/protocol.json", "sha256": sha_file(proto) if proto.exists() else None,
                         "frozen_unix": (read_json(proto, {}) or {}).get("frozen_unix")},
            "selftest": {"path": f"{REL_OUT}/selftest_summary.json", "passed": st.get("passed")},
            "public_path_survey": f"{REL_OUT}/public_path_survey.json"}


def status(state: dict, stage: str, **extra) -> None:
    """root가 읽는 진행 상태를 한 파일로 갱신한다."""
    remaining = {
        "waiting_download": ["Chrome 다운로드 완료(.crdownload→data.tar.gz 이름 변경)", "서버 스트림 추출", "서버 평가 체인", "로컬 검산"],
        "streaming": ["스트림 추출 완료", "서버 평가 체인", "로컬 검산"],
        "waiting_protocol": ["서버 run/protocol.json 고정", "서버 평가 체인", "로컬 검산"],
        "server_chain": ["서버 평가 체인 완료", "로컬 검산"],
        "finalizing": ["로컬 검산·보고"],
        "done": [],
    }.get(stage, []) if "remaining" not in extra else extra.pop("remaining")
    value = {
        "worker": "imlpr_author_input",
        "stage": stage,
        "updated_kst": now_kst(),
        "watcher_pid": os.getpid(),
        "download_dir": str(DOWNLOADS),
        "expected_drive_listing": DRIVE_LISTED_SIZE,
        "attempt": state.get("attempt"),
        "server": {"host": HOST, "repo": REMOTE_REPO, "container_repo": CONTAINER_REPO,
                   "chain_pid": state.get("chain_pid"),
                   "held_processes_untouched": [340070, 1083011]},
        "paths": {"local_out": REL_OUT, "server_author_data": f"{REMOTE_REPO}/{REL_OUT}/author_data",
                  "server_run": f"{REMOTE_REPO}/{REL_OUT}/run", "watch_log": f"{REL_OUT}/watch.log"},
        "remaining_dependencies": remaining,
        "resume": ("setsid nohup python3 -u scripts/_review_imlpr_author_watch.py watch > "
                   f"{REL_OUT}/watch.log 2>&1 &  # watch_state.json의 단계부터 이어서 실행"),
        "evidence": evidence(),
    }
    value.update(extra)
    write_json(STATUS_PATH, value)


def sync_scripts() -> dict:
    """본인 스크립트만 서버 repo에 복사하고 해시를 대조한다."""
    files = [s for s in SCRIPTS if (REPO / s).exists()]
    subprocess.run(["rsync", "-a", "--chmod=F644"] + [str(REPO / s) for s in files] + [f"{HOST}:{REMOTE_REPO}/scripts/"],
                   check=True, capture_output=True, timeout=120)
    local = {s: sha_file(REPO / s) for s in files}
    r = ssh(f"cd {REMOTE_REPO} && sha256sum " + " ".join(files))
    remote = {line.split()[1]: line.split()[0] for line in r.stdout.splitlines() if line.strip()}
    if local != remote:
        raise RuntimeError(f"script hash mismatch {local} {remote}")
    return local


def find_download() -> dict:
    entries = []
    for p in DOWNLOADS.iterdir():
        try:
            st = p.stat()
        except OSError:
            continue
        entries.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    final = [e for e in entries if FINAL_NAME.match(e["name"])]
    partial = [e for e in entries if e["name"].endswith(".crdownload")]
    return {"final": final, "partial": partial, "checked_unix": time.time()}


def gzip_head_ok(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(3) == b"\x1f\x8b\x08"


def wait_download(state: dict) -> Path:
    history: dict[str, list] = {}
    last_partial: dict[str, tuple] = {}
    while True:
        d = find_download()
        rates = {}
        for e in d["partial"]:
            prev = last_partial.get(e["name"])
            if prev and d["checked_unix"] > prev[1]:
                rates[e["name"]] = (e["size"] - prev[0]) / (d["checked_unix"] - prev[1]) / 1e6
            last_partial[e["name"]] = (e["size"], d["checked_unix"])
        ready = None
        for e in d["final"]:
            h = history.setdefault(e["name"], [])
            h.append((e["size"], e["mtime"], d["checked_unix"]))
            del h[:-STABLE_POLLS]
            stable = (len(h) == STABLE_POLLS and len({x[0] for x in h}) == 1 and len({x[1] for x in h}) == 1
                      and h[-1][2] - h[0][2] >= (STABLE_POLLS - 1) * POLL_S * 0.9)
            growing_partial = any(rates.get(p["name"], 0) > 0 for p in d["partial"])
            if stable and not growing_partial and gzip_head_ok(DOWNLOADS / e["name"]):
                ready = e
        status(state, "waiting_download", download={"final_candidates": d["final"], "partial": d["partial"],
               "partial_rate_MBps": rates, "stable_history": history,
               "completion_rule": "최종 이름(data.tar.gz 또는 imlpr_author_data.tar.gz, 중복 번호 허용), "
                                  f"{STABLE_POLLS}회 연속 크기·수정시각 동일(약{(STABLE_POLLS - 1) * POLL_S}s), "
                                  "증가 중인 .crdownload 없음, gzip 머리 1f8b08"})
        if ready:
            path = DOWNLOADS / ready["name"]
            state["archive"] = {"path": str(path), "size": ready["size"], "mtime": ready["mtime"],
                                "detected_kst": now_kst()}
            save_state(state)
            return path
        time.sleep(POLL_S)


def next_attempt() -> str:
    r = ssh(f"mkdir -p {REMOTE_REPO}/{REL_OUT}/author_data && ls {REMOTE_REPO}/{REL_OUT}/author_data")
    nums = [int(m.group(1)) for m in re.finditer(r"attempt_(\d+)", r.stdout)]
    return f"{(max(nums) + 1) if nums else 1:02d}"


def stream(state: dict, path: Path) -> dict:
    """로컬 파일을 읽으며 해시하고 서버 추출기의 표준입력으로 보낸다."""
    attempt = next_attempt()
    state["attempt"] = attempt
    save_state(state)
    size = path.stat().st_size
    remote = (f"docker exec -i -w {CONTAINER_REPO} ring_baseline env OMP_NUM_THREADS=1 python3 -u "
              f"scripts/_review_imlpr_author_extract.py --attempt {attempt} --expect-bytes {size}")
    log = (OUT / f"stream_{attempt}.log").open("w")
    proc = subprocess.Popen(SSH + [remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
    h = hashlib.sha256()
    sent, start, last = 0, time.time(), 0.0
    try:
        with path.open("rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
                proc.stdin.write(b)
                sent += len(b)
                if time.time() - last > 30:
                    last = time.time()
                    el = last - start
                    status(state, "streaming", stream={"archive": str(path), "attempt": attempt, "sent_bytes": sent,
                           "total_bytes": size, "fraction": sent / size, "MBps": sent / el / 1e6,
                           "eta_min": (size - sent) / max(sent / el, 1) / 60, "ssh_pid": proc.pid})
        proc.stdin.close()
    except BrokenPipeError:
        pass
    out = proc.stdout.read().decode(errors="replace")
    rc = proc.wait()
    log.close()
    lines = [x for x in out.splitlines() if x.startswith("{")]
    remote_summary = json.loads(lines[-1]) if lines else {}
    result = {"attempt": attempt, "archive": str(path), "local_bytes": size, "local_sent_bytes": sent,
              "local_sha256": h.hexdigest(), "local_mtime": path.stat().st_mtime, "remote_rc": rc,
              "remote_summary": remote_summary, "elapsed_s": time.time() - start,
              "command": f"python stream {path} | ssh {HOST} '{remote}'",
              "sha256_equal": remote_summary.get("compressed_sha256") == h.hexdigest(),
              "bytes_equal": remote_summary.get("compressed_bytes") == size == sent}
    result["passed"] = bool(rc == 0 and remote_summary.get("passed") and result["sha256_equal"] and result["bytes_equal"])
    write_json(OUT / f"stream_result_{attempt}.json", result)
    state.setdefault("streams", []).append(result)
    if result["passed"]:
        state["stream_passed"] = attempt
        state["archive_sha256"] = h.hexdigest()
    save_state(state)
    return result


def remote_json(rel: str) -> dict | None:
    r = ssh(f"cat {REMOTE_REPO}/{REL_OUT}/{rel} 2>/dev/null")
    try:
        return json.loads(r.stdout) if r.stdout.strip() else None
    except ValueError:
        return None


def launch_chain(state: dict) -> None:
    attempt = state["attempt"]
    cmd = (f"cd {REMOTE_REPO} && mkdir -p {REL_OUT}/run && nohup setsid python3 -u scripts/_review_imlpr_author_eval.py "
           f"chain --attempt {attempt} > {REL_OUT}/run/chain_{attempt}.log 2>&1 < /dev/null & echo $!")
    r = ssh(cmd)
    state["chain_pid"] = int(r.stdout.strip().splitlines()[-1])
    state["chain_command"] = cmd
    state["chain_launched_kst"] = now_kst()
    save_state(state)


def chain_alive(pid: int) -> bool:
    r = ssh(f"ps -p {pid} -o cmd= 2>/dev/null")
    return "_review_imlpr_author_eval.py" in r.stdout


def pull_results(attempt: str) -> None:
    """작은 결과만 가져온다. 영상·서술자 조각은 서버에 둔다."""
    subprocess.run(["rsync", "-a", "--max-size=60m", "--exclude=desc/", "--exclude=*.npy", "--exclude=*.tmp",
                    "--exclude=dataset/", f"{HOST}:{REMOTE_REPO}/{REL_OUT}/run/", str(RUN) + "/"],
                   check=True, capture_output=True, timeout=600)
    dst = OUT / "author_data" / f"attempt_{attempt}"
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--max-size=60m", "--exclude=files/",
                    f"{HOST}:{REMOTE_REPO}/{REL_OUT}/author_data/attempt_{attempt}/", str(dst) + "/"],
                   check=True, capture_output=True, timeout=600)


def finalize(state: dict) -> dict:
    """서버 계산과 독립된 numpy 재집계로 최종 지표를 검산하고 보고서를 쓴다."""
    import numpy as np

    attempt = state["attempt"]
    pull_results(attempt)
    ar = RUN / f"attempt_{attempt}"
    server = read_json(ar / "server_report.json", {})
    checks = []
    for p in sorted((ar / "score").glob("*.npz")) if (ar / "score").exists() else []:
        z = np.load(p)
        first_pos, first_dist, evaluated = z["first_is_positive"], z["first_distance"], z["evaluated"]
        th = z["thresholds"]
        e = evaluated.astype(bool)
        r1 = float(first_pos[e].mean()) if e.any() else 0.0
        acc = first_dist[e][:, None] < th[None, :]
        pos = first_pos[e].astype(bool)[:, None]
        tp, fp = (acc & pos).sum(0), (acc & ~pos).sum(0)
        fn = (~acc & pos).sum(0)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = float((2 * prec * rec / (prec + rec + 1e-8)).max())
        js = read_json(p.with_suffix(".json"), {})
        checks.append({"file": str(p.relative_to(OUT)), "sha256": sha_file(p), "n_evaluated": int(e.sum()),
                       "r1_local": r1, "f1_local": f1, "r1_server": js.get("r1"), "f1_server": js.get("max_f1"),
                       "r1_equal": js.get("r1") is not None and abs(r1 - js["r1"]) < 1e-12,
                       "f1_equal": js.get("max_f1") is not None and abs(f1 - js["max_f1"]) < 1e-9})
    verification = {"local_numpy_recount": checks,
                    "all_equal": all(c["r1_equal"] and c["f1_equal"] for c in checks) if checks else None,
                    "not_applicable_reason": None if checks else "평가 원장이 없다(저자 압축본에 MulRan 입력 없음)",
                    "inclusion_independent_recheck": (read_json(OUT / "inclusion_list.json") or {}).get("mulran"),
                    "stream_result": read_json(OUT / f"stream_result_{attempt}.json"),
                    "server_report_sha256": sha_file(ar / "server_report.json") if (ar / "server_report.json").exists() else None}
    report = {"generated_kst": now_kst(), "scope": "ImLPR 저자 공개 가공 입력의 원 논문(CoRL 2025 표7, MulRan DCC01-03) 대표 평가. "
              "분할 v2 주표·우리 재학습 결과와 분리한다.", "attempt": attempt, "archive": state.get("archive"),
              "archive_sha256": state.get("archive_sha256"), "server_report": server,
              "result_categories": {
                  "public_model_public_input": "저자 공개 가중치+저자 공개 가공 입력(C1, 이 보고의 주 평가)",
                  "public_model_regenerated_input": "저자 공개 가중치+공식 생성기로 재생성한 입력(C5_k15/C5_k25, 기존 imlpr_official_replay.json)",
                  "paper_condition": "논문 표7 ImLPR(D) 수치(출판값, 재학습하지 않음)",
                  "our_split_retrain": "분할 v2 주표의 ImLPR 재학습 3시드(이 보고에 섞지 않음)"},
              "compare": read_json(ar / "compare.json"), "prepare": read_json(ar / "prepare.json"),
              "selftest": read_json(OUT / "selftest_summary.json"), "public_path_survey": read_json(OUT / "public_path_survey.json"),
              "inclusion_list": f"{REL_OUT}/inclusion_list.json",
              "author_helipr_rule_check": read_json(OUT / "author_helipr_rule_check.json"),
              "town_check": read_json(ar / "town_check.json"),
              "independent_verification": verification,
              "commands": {"watch": "python3 -u scripts/_review_imlpr_author_watch.py watch",
                           "stream": (verification["stream_result"] or {}).get("command"),
                           "chain": state.get("chain_command")},
              "script_sha256": {s: sha_file(REPO / s) for s in SCRIPTS if (REPO / s).exists()}}
    write_json(OUT / "report.json", report)
    (OUT / "report_ko.txt").write_text(report_text(report))
    return report


def report_text(rep: dict) -> str:
    s = rep.get("server_report") or {}
    lines = ["ImLPR 저자 가공 입력 원 논문 대표 평가 보고", f"생성: {rep['generated_kst']}", "",
             "범위: " + rep["scope"], f"압축본: {(rep.get('archive') or {}).get('path')} sha256={rep.get('archive_sha256')}", ""]
    inv = s.get("inventory_summary") or {}
    lines.append("[포함 여부] " + json.dumps(inv, ensure_ascii=False))
    for row in s.get("primary_rows", []):
        lines.append(f"[주 평가] {row['sequence']}: R@1 {row['r1'] * 100:.2f}% (논문 {row['published']['r1'] * 100:.1f}%), "
                     f"최대F1 {row['max_f1'] * 100:.2f}% (논문 {row['published']['f1'] * 100:.1f}%), 평가질의 {row['n_evaluated']}, 영상 {row['n_frames']}")
    if s.get("primary_macro"):
        m = s["primary_macro"]
        lines.append(f"[주 평가 평균] R@1 {m['r1'] * 100:.2f}% (논문86.5%), 최대F1 {m['max_f1'] * 100:.2f}% (논문94.3%)")
    for seq, rec in ((rep.get("compare") or {}).get("sequences") or {}).items():
        im, lab, dec = rec["image"], rec["labels_on_matched"], rec["decomposition"]
        lines.append(f"[원본/재생성 {seq}] 영상 저자{rec['n_author']}·재생성{rec['n_regenerated']}·공통{rec['n_matched']} "
                     f"(저자만{rec['author_only']}, 재생성만{rec['regenerated_only']}), 최대|dt| {rec['dt_ms']['max_abs']}ms, "
                     f"저자 영상 크기 {im['author_shapes']}, 법선 채널 근접 {im['normal_channel_closer']}, "
                     f"공통 영상 정답 동일 {lab['equal_sets']}/{rec['n_matched']}")
        for k in ("k15", "k25"):
            if k in im:
                lines.append(f"  - {k} 대비 채널별 평균|차이| {[round(x, 4) for x in im[k]['mean_abs_by_channel']]}, "
                             f"전채널 동일 영상 {im[k]['frames_all_channels_exact']}")
        lines.append("  - 분해 R@1/F1(%): " + ", ".join(f"{k} {v['r1'] * 100:.2f}/{v['max_f1'] * 100:.2f}(n{v['n_evaluated']})"
                                                     for k, v in dec.items()))
    for seq, v in ((rep.get("prepare") or {}).get("plan") or {}).items():
        if v.get("A_vs_B_labels"):
            lines.append(f"[정답 일치 {seq}] 저자 pickle 대 저자 poses 공식 생성기: {v['A_vs_B_labels']}")
    st = rep.get("selftest") or {}
    lines.append(f"[사전 자체시험] 합성 입력 전 단계 통과 {st.get('passed')} ({REL_OUT}/selftest_summary.json)")
    lines.append("[결과 구분] " + "; ".join(f"{k}: {v}" for k, v in rep["result_categories"].items()))
    v = rep["independent_verification"]
    if v.get("not_applicable_reason"):
        lines.append(f"[독립 검산] 지표 재집계 해당 없음: {v['not_applicable_reason']}")
        mr = v.get("inclusion_independent_recheck") or {}
        lines.append(f"  - 포함 여부 독립 재확인: 추출기 MulRan 선택 {mr.get('selected_mulran_files')}건, "
                     f"넓은 정규식 {mr.get('independent_wider_regex')} 일치 {mr.get('independent_wider_regex_hits')}건 "
                     f"(정확 포함 목록: {rep.get('inclusion_list')})")
    else:
        lines.append(f"[독립 검산] 로컬 numpy 재집계 {len(v['local_numpy_recount'])}건 모두 일치: {v['all_equal']}")
    rc = rep.get("author_helipr_rule_check") or {}
    if rc:
        lines.append("[보조 진단: 저자 HeLiPR 평가셋 규칙] " + rc.get("label_result", "") + " / " + rc.get("timestamp_unit", ""))
        lines.append("  - 표본: " + rc.get("sampling_rule", "") + "; 6세션 연속 3D 간격 최소 "
                     + ", ".join(f"{k} {x['min_3d_step_rounded4']}m" for k, x in rc.get("sampling_check_3d_consecutive_min_m", {}).items()))
        lines.append("  - 함의(가설): " + rc.get("implication", ""))
    sv = rep.get("public_path_survey") or {}
    for src in sv.get("sources", []):
        lines.append(f"[공개 경로 조사] {src['url']}: {src['finding']}")
    for row in s.get("primary_rows", []):
        ind = row.get("independent_float64") or {}
        lines.append(f"  - {row['sequence']}: 공식 _get_recall_for_pair 출력 일치 {row.get('official_pair_equal')}, "
                     f"float64 전수 L2 R@1 {ind.get('r1')} F1 {ind.get('max_f1')} 첫후보 일치 {ind.get('same_first_index')}/{row['n_evaluated']}")
    st = v.get("stream_result") or {}
    lines.append(f"[스트림 무결성] 바이트 일치 {st.get('bytes_equal')}, SHA-256 일치 {st.get('sha256_equal')}, "
                 f"gzip CRC/ISIZE {((st.get('remote_summary') or {}).get('passed'))}")
    ref = read_json(REPO / "results/pdv2_published_reproduction_20261106/imlpr_official_replay.json", {}) or {}
    for k, m in (ref.get("macro") or {}).items():
        lines.append(f"[참고: 기존 재생성 입력·저자 가중치 k{k}] 3시퀀스 평균 R@1 {m['r1'] * 100:.2f}%, 최대F1 {m['f1'] * 100:.2f}% "
                     "(재생성4275장, 논문4328장; results/pdv2_published_reproduction_20261106/imlpr_official_replay.json)")
    for note in s.get("notes", []):
        lines.append("- " + note)
    return "\n".join(lines) + "\n"


def watch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open("/tmp/imlpr_author_watch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another watcher is running")
    state = load_state()
    state.setdefault("started_kst", now_kst())
    state["watcher_pid"] = os.getpid()
    save_state(state)
    state["scripts_sha256"] = sync_scripts()
    save_state(state)
    while not state.get("stream_passed"):
        path = wait_download(state)
        result = stream(state, path)
        if not result["passed"]:
            fails = sum(1 for s in state.get("streams", []) if not s["passed"])
            status(state, "stream_failed_retry_wait", stream_failure=result, remaining=["스트림 재시도"])
            if fails >= 3:
                status(state, "stream_failed_stop", stream_failure=result, remaining=["사람 확인 필요: 스트림 3회 실패"])
                return
            time.sleep(300)
    while not state.get("chain_done"):
        if not state.get("chain_pid"):
            proto = remote_json("run/protocol.json")
            if not proto:
                status(state, "waiting_protocol")
                time.sleep(POLL_S)
                continue
            state["scripts_sha256"] = sync_scripts()
            launch_chain(state)
        cs = remote_json(f"run/attempt_{state['attempt']}/chain_status.json") or {}
        alive = chain_alive(state["chain_pid"])
        status(state, "server_chain", chain=cs, chain_alive=alive)
        if cs.get("finished"):
            state["chain_done"] = True
            state["chain_ok"] = bool(cs.get("ok"))
            save_state(state)
            break
        if not alive:
            time.sleep(20)
            cs = remote_json(f"run/attempt_{state['attempt']}/chain_status.json") or {}
            if not cs.get("finished"):
                status(state, "server_chain_died", chain=cs, remaining=["체인 로그 확인 후 재개"])
                return
            continue
        time.sleep(POLL_S)
    status(state, "finalizing")
    rep = finalize(state)
    status(state, "done", report=f"{REL_OUT}/report.json",
           verified=rep["independent_verification"]["all_equal"], chain_ok=state.get("chain_ok"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["watch", "sync", "finalize"])
    args = ap.parse_args()
    if args.stage == "watch":
        watch()
    elif args.stage == "sync":
        print(json.dumps(sync_scripts(), indent=2))
    else:
        st = load_state()
        rep = finalize(st)
        absent = bool(rep["independent_verification"].get("not_applicable_reason"))
        status(st, "done", report=f"{REL_OUT}/report.json", chain_ok=st.get("chain_ok"),
               verified=rep["independent_verification"]["all_equal"],
               remaining=(["저자 MulRan DCC01-03 가공 입력(공개 경로 없음)",
                           "새 압축본이 생기면: watch_state.json을 다른 이름으로 보관하고, 기존 data.tar.gz와 이름이 겹치지 않게 "
                           "새 파일만 D:\\NSD_downloads의 허용 이름(imlpr_author_data.tar.gz 등)으로 둔 뒤 watch를 실행하면 "
                           "attempt_02로 같은 추출·평가 체인이 돈다(현재 코드는 그 폴더의 허용 이름 파일을 자동 선택)"] if absent else []),
               outcome=("저자 압축본에 MulRan DCC01-03 입력 없음. 원본 입력 평가 미수행"
                        if rep["independent_verification"].get("not_applicable_reason") else "평가 완료"))
