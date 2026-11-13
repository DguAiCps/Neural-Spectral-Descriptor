#!/usr/bin/env python3
"""사용자에게 반환한 로컬 GPU의 선택실험 자동 재점유를 막는다."""
from __future__ import annotations

import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "results/pdv2_review_selection_full_20261105/local_gpu_reservation.json"


def held() -> bool:
    """다른 서버에는 적용하지 않는 영속 사용자 예약을 읽는다."""
    if not PATH.exists():
        return False
    value = json.loads(PATH.read_text())
    return bool(value.get("hold") and socket.gethostname() in value["container_hostnames"])


def require_available() -> None:
    """GPU 초기화 전 대기로 반환하며 기존 실행은 중단하지 않는다."""
    if held():
        print("WAITING_USER_LOCAL_GPU_RESERVATION", flush=True)
        raise SystemExit(75)


def require_background_available() -> None:
    """재부팅 뒤에도 로컬 CPU·전송·감시기 자동 시작을 차단한다."""
    path = PATH.with_name("local_cpu_reservation.json")
    if not path.exists():
        return
    value = json.loads(path.read_text())
    if value.get("hold") and socket.gethostname() in value["container_hostnames"]:
        print("WAITING_USER_LOCAL_CPU_IO_RESERVATION", flush=True)
        raise SystemExit(75)
