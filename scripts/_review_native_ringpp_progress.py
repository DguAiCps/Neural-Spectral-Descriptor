#!/usr/bin/env python3
"""전수 후보 쌍과 실제 완료 로그로 RING++ 남은 시간을 추정한다."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1] / "results/pdv2_ringpp_maxfix_20261105"


def read(path: Path) -> dict:
    """존재하는 JSON 상태만 읽는다."""
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    """경합 중 처리율은 정확도 작업 ETA에만 사용한다."""
    protocol = read(ROOT / "protocol.json")
    state = read(ROOT / "server_status.json")
    completed = set(state.get("completed", []))
    frame_total, generated, total_pairs, finished_pairs = 0, 0, 0, 0
    samples = []
    remaining_frames, remaining_pairs = 0, 0
    rows = []
    for item in protocol["sequences"]:
        sequence = item["sensor"] + "/" + item["sequence"]
        name, frames = sequence.replace("/", "_"), item["frames"]
        queries = frames - 30
        pairs = queries * (queries + 1) // 2
        frame_total += frames
        total_pairs += pairs
        bev = ROOT / "bev" / name
        done = (bev / "done.json").exists()
        generation = read(bev / "status.json")
        generated += frames if done else generation.get("persisted_frames", generation.get("frames", 0))
        log = ROOT / "logs" / ("dump_" + name + ".log")
        body = log.read_text() if log.exists() else ""
        result = re.search(r"DUMP_DONE .*?: (\d+) queries, (\d+) s \(\+(\d+) s TIRING\)", body)
        if sequence in completed:
            finished_pairs += pairs
            if result and queries >= 3000:
                samples.append({"sequence": sequence, "pairs": pairs,
                                "search_s": int(result[2]), "tiring_s": int(result[3]),
                                "pairs_per_s": pairs / int(result[2]),
                                "tiring_s_per_frame": int(result[3]) / frames})
            continue
        progress = re.findall(r"ringpp queries (\d+)/(\d+)", body)
        current = int(progress[-1][0]) if progress else 0
        seen_pairs = current * (current + 1) // 2
        finished_pairs += seen_pairs
        remaining_pairs += pairs - seen_pairs
        tiring_done = bool(progress)
        remaining_frames += 0 if tiring_done else frames
        rows.append({"sequence": sequence, "frames": frames, "bev_complete": done,
                     "search_queries_completed": current, "search_queries_total": queries})
    assert samples
    rates = sorted(s["pairs_per_s"] for s in samples)
    tiring = sorted(s["tiring_s_per_frame"] for s in samples)
    # 현재 수행 중인 TIRING 잔여량은 알 수 없으므로 전체를 남긴 보수적 추정이다.
    low = remaining_pairs / rates[-1] + remaining_frames * tiring[0]
    high = remaining_pairs / rates[0] + remaining_frames * tiring[-1]
    middle = remaining_pairs / statistics.median(rates) + remaining_frames * statistics.median(tiring)
    now = datetime.now().astimezone()
    report = {"generated_at": now.isoformat(), "completed_ledgers": len(completed),
              "total_ledgers": len(protocol["sequences"]), "generated_persisted_frames": generated,
              "total_frames": frame_total, "total_candidate_pairs": total_pairs,
              "completed_candidate_pairs": finished_pairs, "remaining_candidate_pairs": remaining_pairs,
              "throughput_samples": samples, "estimated_remaining_serial_seconds":
              {"fast_observed": low, "median_observed": middle, "slow_observed": high},
              "estimated_finish": {"fast_observed": (now + timedelta(seconds=low)).isoformat(),
                                   "median_observed": (now + timedelta(seconds=middle)).isoformat(),
                                   "slow_observed": (now + timedelta(seconds=high)).isoformat()},
              "limits": "최근 큰 완료 시퀀스 처리율 외삽이며 경합 변화·특징 생성 대기는 별도다. 진행 중 TIRING은 전체 잔여로 세므로 상향 편향 가능. Table3 효율 수치가 아니다.",
              "generation_counter_note": "묶음 저장 직후 status에는 persisted_frames 없이 frames만 있다. 이전 진행기록의 해당 구간은 0으로 과소계수됐으며 이 버전은 저장 완료 frames를 사용한다. 원장·지표·검색 ETA에는 영향 없음.",
              "active": state.get("active", {}), "errors": state.get("errors", {}), "remaining": rows}
    path = ROOT / "measured_progress.json"
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)
    with (ROOT / "measured_progress_history.jsonl").open("a") as stream:
        stream.write(json.dumps({k: v for k, v in report.items() if k not in ["remaining", "throughput_samples"]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["remaining", "throughput_samples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
