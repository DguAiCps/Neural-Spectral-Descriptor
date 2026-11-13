#!/usr/bin/env python3
"""RING·정확max RING++의 NCLT 2012-08-20 질의 → 2012-02-04 지도 평가(논문 그림11 문단의 날짜 쌍).

서버 컨테이너에서 단계별로 실행한다(GPU 구간만 호스트 flock).
    docker exec -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=4 -w /workspace/Neural-Spectral-Codec ring_baseline \
        python3 scripts/_review_ring_followup_datepair.py coverage
    ... fetch      (누락 스캔만 공식 S3 범위 수신; coverage의 자원 상한 판정이 허용할 때만)
    ... protocol   (성능을 보기 전에 조건·입력 해시 고정)
    flock -x /tmp/nsd_native_reproduction_gpu0.lock docker exec -e CUDA_VISIBLE_DEVICES=0 ... \
        python3 scripts/_review_ring_followup_datepair.py run --split all --bev occ --impl official
    ... run --split all --bev feat --impl maxfix
    ... verify / report

03-17 평가(scripts/_review_ring_followup_official.py)와 같은 공식 함수를 쓴다. 질의 집합은 공식
NCLTSequence(5m)·get_scans·filter_query_elements(20m)로 만들고, 08-20 velodyne_sync 목록은 공식
압축본 tar 멤버 목록(DiSCO 담당 members.json)에서 0바이트 자리 파일로 재현한다(생성기는 이름만 읽는다).
지도 집합은 질의 세션과 무관하므로 기존 공식 생성기 출력(evaluation_sets.json의 02-04 지도)을 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import pickle
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/date_pair_20261107"
NATIVE = REPO / "results/pdv2_review_native_20261105"
OFFICIAL = REPO / "_external/RING"
DISCO = REPO / "results/pdv2_remaining_native_20261106/disco/native_data/2012-08-20"
FETCHED = OUT / "native_data/2012-08-20/velodyne_sync"
LAYOUT = Path("/tmp/ring_datepair_layout")
QUERY_SEQ, MAP_SEQ = "2012-08-20", "2012-02-04"
RADII = [5.0, 10.0, 15.0, 20.0]
SPLITS = ("all", "test")
BLOCK = 16 * 1024 * 1024
PAPER_TEXT_SHA256 = "9a211f562e5756924f458c75ab6d5c040ce0998f2f36701e413be8974d8a15dd"   # 로컬 sha256sum
WORKERS = 3                      # 범위 요청 병렬 수(주 스레드 포함 4)
LIMIT_BYTES = 1024**3            # 이번 진단의 추가 저장 상한
LIMIT_SECONDS = 30 * 60          # 이번 진단의 수신 시간 상한
SOURCE_FILES = ("datasets/NCLTDataset.py", "evaluation/generate_evaluation_sets.py", "evaluation/evaluate.py",
                "evaluation/plot_PR_curve.py", "utils/core.py", "utils/circorr2.py", "utils/tools.py",
                "utils/config.py", "utils/poses.py")
sys.path.insert(0, str(REPO / "scripts"))


def digest(path: Path) -> str:
    """파일의 SHA256을 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    """JSON을 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def members() -> dict:
    """공식 08-20 압축본의 velodyne_sync 멤버 {이름: {offset, size}}와 원본 URL·ETag."""
    return json.loads((DISCO / "members.json").read_text())


def build_layout(names: list[str]) -> None:
    """공식 NCLTSequence가 읽는 경로 구조를 만든다. 스캔은 0바이트 자리 파일(이름만 읽힘)."""
    seq = LAYOUT / QUERY_SEQ
    sync = seq / "velodyne_data/velodyne_sync"
    if sync.exists():
        shutil.rmtree(seq)
    sync.mkdir(parents=True)
    (seq / "ground_truth").mkdir()
    (seq / "ground_truth" / f"groundtruth_{QUERY_SEQ}.csv").symlink_to(DISCO / "groundtruth.csv")
    for name in names:
        (sync / Path(name).name).touch()


def official_query_sets() -> dict:
    """공식 함수로 split별 08-20 질의를 만든다."""
    import _review_native_ring as native
    native.imports()
    from datasets.NCLTDataset import NCLTSequence
    from evaluation.generate_evaluation_sets import EvaluationTuple, get_scans, filter_query_elements

    ref = json.loads((NATIVE / "evaluation_sets.json").read_text())
    sets = {}
    for split in SPLITS:
        q = NCLTSequence(str(LAYOUT), sequence_name=QUERY_SEQ, split=split, sampling_distance=5.0)
        query_set = get_scans(q)
        map_set = [EvaluationTuple(e["timestamp"], e["raw_path"], np.asarray(e["xy"], np.float32),
                                   np.asarray(e["pose"], np.float32)) for e in ref[split]["map"]]
        kept = filter_query_elements(query_set, map_set, 20.0)
        sets[split] = {"query_before_filter": len(query_set), "query": [
            {"raw_path": f"{QUERY_SEQ}/velodyne_sync/{Path(e.rel_scan_filepath).name}",
             "timestamp": int(e.timestamp), "xy": e.position.tolist(), "pose": e.pose.tolist()} for e in kept]}
    return sets


def independent_query_sets(names: list[str]) -> dict:
    """공식 코드를 쓰지 않는 numpy 재구성(_review_ring_followup_sampling.py와 같은 규칙)."""
    import _review_ring_followup_sampling as smp
    raw = np.loadtxt(DISCO / "groundtruth.csv", delimiter=",", dtype=np.float64)[1:]   # header=0이 첫 행을 버림
    order = np.argsort(raw[:, 0].astype(np.int64), kind="quicksort")
    stamps, xy = raw[order, 0].astype(np.int64), raw[order, 1:3].astype(np.float32)
    lidar = sorted(int(Path(n).stem) for n in names)
    ref = json.loads((NATIVE / "evaluation_sets.json").read_text())
    out = {}
    for split in SPLITS:
        ts, pos = [], []
        for t in lidar:
            j = smp.nearest(t, stamps)
            if abs(int(stamps[j]) - t) > 1e9:
                continue
            ts.append(t)
            pos.append(xy[j])
        ts, pos = np.asarray(ts, np.int64), np.asarray(pos, np.float32)
        if split == "test":
            m = (np.linalg.norm(pos.astype(np.float64)[:, None] - smp.TEST_CENTER[None], axis=-1)
                 < smp.TEST_RADIUS).any(axis=1)
            ts, pos = ts[m], pos[m]
        chosen, prev = [], None
        for i, p in enumerate(pos):
            if prev is None or np.linalg.norm(prev - p) > 5.0:
                chosen.append(i)
                prev = p
        ts, pos = ts[chosen], pos[chosen]
        mxy = np.array([e["xy"] for e in ref[split]["map"]], np.float64)
        keep = (np.linalg.norm(pos.astype(np.float64)[:, None] - mxy[None], axis=-1) <= 20.0).any(axis=1)
        out[split] = {"ts": ts[keep], "xy": pos[keep], "before_filter": int(len(ts))}
    return out


def plan_blocks(need: list[str], info: dict) -> tuple[set, int, int]:
    """gzip 색인 시작점 기준으로 멤버 추출에 필요한 16MiB 압축 범위 블록 시작 위치·압축 바이트·시작점 수."""
    import indexed_gzip

    done = json.loads((DISCO / "complete.json").read_text())

    class NoNetwork:
        """원본 길이만 알려 주고 읽기는 막는 파일 객체(색인 일관성 검사용)."""
        pos = 0

        def read(self, *a):
            raise RuntimeError("계획 단계에서 원본을 읽지 않는다")

        def seek(self, offset, whence=0):
            self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else done["archive_bytes"])
            return self.pos

        def tell(self):
            return self.pos

        def seekable(self):
            return True

        def readable(self):
            return True

    gz = indexed_gzip.IndexedGzipFile(fileobj=NoNetwork(), spacing=16 * 1024 * 1024)
    gz.import_index(str(DISCO / "gzip.index"))
    points = np.array(list(gz.seek_points()), np.int64)    # (uncompressed, compressed)
    unc, cmp_ = points[:, 0], points[:, 1]
    blocks, compressed = set(), 0
    block = BLOCK
    for name in need:
        m = info[name]
        a = int(np.searchsorted(unc, m["offset"], side="right") - 1)
        b = int(np.searchsorted(unc, m["offset"] + m["size"], side="left"))
        c0 = int(cmp_[a]) - 1
        c1 = int(cmp_[b]) if b < len(cmp_) else int(cmp_[-1]) + block
        compressed += c1 - c0
        blocks.update(range(c0 // block, c1 // block + 1))
    return {i * block for i in blocks}, compressed, len(points)


def fetch_plan(need: list[str], info: dict) -> dict:
    """누락 멤버 추출에 필요한 압축 바이트·16MiB 범위 블록과 보수적 예상 시간(수신 없음)."""
    blocks, compressed, n_points = plan_blocks(need, info)
    done = json.loads((DISCO / "complete.json").read_text())
    rate = done["network_bytes"] / done["elapsed_s"]
    net = len(blocks) * BLOCK
    return {"missing_scans": len(need), "missing_raw_bytes": int(sum(info[n]["size"] for n in need)),
            "seek_points": n_points, "index_spacing_uncompressed_bytes": BLOCK,
            "compressed_span_bytes_upper": int(compressed), "range_blocks_16MiB": len(blocks),
            "network_bytes_estimate": int(net),
            "observed_disco_throughput_Bps": rate, "estimated_seconds": net / rate,
            "throughput_source": "DiSCO 08-20 complete.json network_bytes/elapsed_s(압축 해제·색인 시간 포함, 보수적)"}


def extraction_rate(info: dict) -> dict:
    """DiSCO가 같은 압축본에서 1,903장을 추출한 단계의 실측 처리량(첫·마지막 추출 파일 시각, 같은 블록 계산)."""
    pre = f"{QUERY_SEQ}/velodyne_sync/"
    held = sorted((p.name for p in (DISCO / "velodyne_sync").glob("*.bin")), key=lambda n: info[pre + n]["offset"])
    mt = [os.stat(DISCO / "velodyne_sync" / n).st_mtime for n in held]
    blocks, _, _ = plan_blocks([pre + n for n in held], info)
    seconds = mt[-1] - mt[0]
    return {"scans": len(held), "range_blocks_16MiB": len(blocks), "network_bytes": len(blocks) * BLOCK,
            "seconds_first_to_last_file": seconds, "throughput_Bps": len(blocks) * BLOCK / seconds,
            "mtime_monotonic_in_offset_order": all(b >= a for a, b in zip(mt, mt[1:]))}


def coverage() -> None:
    """필요 스캔 목록·보유 대조·누락 수신 비용을 기록한다."""
    info = members()
    names = list(info["members"])
    build_layout(names)
    sets = official_query_sets()
    alt = independent_query_sets(names)
    complete = json.loads((DISCO / "complete.json").read_text())
    hashes = {r["name"]: r["sha256"] for r in complete["raw_sha256"]}
    held = {p.name for p in (DISCO / "velodyne_sync").glob("*.bin")}
    ref = json.loads((NATIVE / "evaluation_sets.json").read_text())
    rows, need_all = {}, set()
    for split in SPLITS:
        q = sets[split]["query"]
        ts = np.array([e["timestamp"] for e in q], np.int64)
        xy = np.array([e["xy"] for e in q], np.float32)
        a = alt[split]
        names_q = [Path(e["raw_path"]).name for e in q]
        present = [n for n in names_q if n in held]
        missing = [n for n in names_q if n not in held]
        need_all.update(missing)
        mxy = np.array([e["xy"] for e in ref[split]["map"]], np.float64)
        geo = np.linalg.norm(xy.astype(np.float64)[:, None] - mxy[None], axis=-1)
        rows[split] = {
            "query_before_20m_filter": sets[split]["query_before_filter"], "queries": len(q),
            "maps": len(ref[split]["map"]),
            "positive_queries_10m": int((geo <= 10.0).any(axis=1).sum()),
            "positive_queries_5m": int((geo <= 5.0).any(axis=1).sum()),
            "independent_reconstruction": {
                "before_filter": a["before_filter"], "queries": int(len(a["ts"])),
                "timestamps_identical": bool(len(a["ts"]) == len(ts) and (a["ts"] == ts).all()),
                "xy_max_abs_diff_m": (float(np.abs(a["xy"] - xy).max())
                                      if len(a["ts"]) == len(ts) and (a["ts"] == ts).all() else None)},
            "present_in_disco_selection": len(present), "missing": len(missing),
            "missing_first10": missing[:10],
        }
    need_names = sorted(f"{QUERY_SEQ}/velodyne_sync/{n}" for n in need_all)
    plan = fetch_plan(need_names, info["members"]) if need_names else {"missing_scans": 0}
    plan["within_limits"] = bool(not need_names or (plan["missing_raw_bytes"] < LIMIT_BYTES
                                                    and plan["estimated_seconds"] < LIMIT_SECONDS))
    report = {
        "purpose": "08-20→02-04 RING 평가에 필요한 08-20 스캔과 보유 파일 대조(성능 계산 없음)",
        "rule": "공식 NCLTSequence(split, 5m 질의, 1e9 허용오차 그대로)·get_scans·filter_query_elements(20m); "
                "지도는 기존 공식 생성기 02-04 지도(20m)",
        "archive": {"url": info["url"], "etag": info["etag"], "velodyne_sync_members": len(names)},
        "groundtruth_sha256": digest(DISCO / "groundtruth.csv"),
        "groundtruth_first_row_dropped_by_header0": True,
        "disco_selection": {"held_files": len(held), "complete_json_hashes": len(hashes)},
        "splits": rows, "fetch_plan": plan,
        "limits": {"storage_bytes": LIMIT_BYTES, "seconds": LIMIT_SECONDS},
        "evaluation_sets_sha256": digest(NATIVE / "evaluation_sets.json"),
        "official_source_sha256": {p: digest(OFFICIAL / p) for p in SOURCE_FILES},
        "script_sha256": digest(Path(__file__)), "created_utc": utc(),
    }
    save_json(OUT / "query_sets.json", {s: sets[s] for s in SPLITS})
    save_json(OUT / "coverage.json", report)
    print(json.dumps({k: report[k] for k in ("splits", "fetch_plan")}, ensure_ascii=False, indent=1), flush=True)


def fetch() -> None:
    """누락 멤버만 공식 S3 범위 수신으로 내 결과 폴더에 저장한다. DiSCO 폴더는 읽기만 한다.

    수신 전 결정: 저장 < 1GiB이고 DiSCO 추출 단계 실측 처리량(연결 1개)으로 예상 시간 < 30분일 때만 시작.
    수신 중 결정: 실제 소비 블록 비율로 예상 총시간이 30분을 넘으면 중단(fetch_aborted.json).
    보유 질의 스캔도 같은 경로로 메모리에서 다시 추출해 DiSCO 해시와 비교한다(무결성 대조, 파일 쓰기 없음).
    """
    import bisect
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import indexed_gzip
    import _review_remaining_native_disco_data as dd

    info = members()
    pre = f"{QUERY_SEQ}/velodyne_sync/"
    sets = json.loads((OUT / "query_sets.json").read_text())
    held = {p.name for p in (DISCO / "velodyne_sync").glob("*.bin")}
    wanted = {Path(e["raw_path"]).name for s in SPLITS for e in sets[s]["query"]}
    need, controls = sorted(wanted - held), sorted(wanted & held)
    hashes = {r["name"]: r["sha256"] for r in json.loads((DISCO / "complete.json").read_text())["raw_sha256"]}
    raw_bytes = sum(info["members"][pre + n]["size"] for n in need)
    blocks, _, _ = plan_blocks([pre + n for n in need + controls], info["members"])
    rate = extraction_rate(info["members"])
    single = len(blocks) * BLOCK / rate["throughput_Bps"]
    decision = {"need": len(need), "controls": len(controls), "raw_bytes": raw_bytes,
                "planned_blocks": len(blocks), "planned_network_bytes": len(blocks) * BLOCK,
                "disco_extraction_phase": rate, "single_connection_estimate_s": single,
                "conservative_estimate_s_from_coverage": json.loads((OUT / "coverage.json").read_text())
                ["fetch_plan"]["estimated_seconds"],
                "workers": WORKERS, "limits": {"storage_bytes": LIMIT_BYTES, "seconds": LIMIT_SECONDS},
                "proceed": bool(raw_bytes < LIMIT_BYTES and single < LIMIT_SECONDS), "created_utc": utc()}
    save_json(OUT / "fetch_decision.json", decision)
    if not decision["proceed"]:
        raise SystemExit("자원 상한 초과 예상: 수신하지 않는다(fetch_decision.json)")
    FETCHED.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(OUT).free > 20 * 1024**3 + raw_bytes
    dd.OUT = OUT                                            # 상태 파일을 내 폴더에 쓰게 한다
    plan = sorted(blocks)

    class Prefetch(dd.HTTPRange):
        """계획된 다음 블록을 WORKERS개 연결로 미리 받는다(최대 8블록=128MiB 메모리)."""

        def __init__(self, url, date):
            super().__init__(url, date)
            self.pool, self.futures, self.mutex = ThreadPoolExecutor(WORKERS), {}, threading.Lock()

        def _block(self, start, end):
            i = bisect.bisect_left(plan, start)
            with self.mutex:
                for b in plan[i:i + 8]:
                    if b not in self.futures and b != start:
                        self.futures[b] = self.pool.submit(dd.HTTPRange._block, self, b, min(b + BLOCK, self.length) - 1)
                fut = self.futures.pop(start, None)
            return fut.result() if fut is not None else dd.HTTPRange._block(self, start, end)

    remote = Prefetch(info["url"], QUERY_SEQ)
    assert remote.etag == info["etag"], "원본 ETag가 색인 이후 바뀜"
    remote.stage = "ring_datepair_missing_scans"
    gz = indexed_gzip.IndexedGzipFile(fileobj=remote, spacing=BLOCK)
    gz.import_index(str(DISCO / "gzip.index"))
    kept, checked = [], []
    order = sorted(need + controls, key=lambda n: info["members"][pre + n]["offset"])
    for k, name in enumerate(order):
        m = info["members"][pre + name]
        path = FETCHED / name
        if name in held:
            gz.seek(m["offset"])
            blob = gz.read(m["size"])
            checked.append({"name": name, "match": hashlib.sha256(blob).hexdigest() == hashes[name]})
            assert checked[-1]["match"], f"보유 스캔 재추출 해시 불일치: {name}"
        elif not (path.exists() and path.stat().st_size == m["size"]):
            gz.seek(m["offset"])
            blob = gz.read(m["size"])
            assert len(blob) == m["size"]
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(blob)
            tmp.replace(path)
        if name not in held:
            kept.append({"name": name, "size": m["size"], "sha256": digest(path)})
        elapsed = time.time() - remote.started
        frac = remote.received / (len(plan) * BLOCK)
        status = {"stage": "fetching", "done": k + 1, "total": len(order), "network_bytes": remote.received,
                  "planned_network_bytes": len(plan) * BLOCK, "elapsed_s": elapsed,
                  "projected_total_s": elapsed / max(frac, 1e-9), "updated_utc": utc()}
        save_json(OUT / "fetch_status.json", status)
        if elapsed > 120 and status["projected_total_s"] > LIMIT_SECONDS:
            save_json(OUT / "fetch_aborted.json", status | {"reason": "실측 진행률 기준 예상 30분 초과"})
            raise SystemExit("수신 중단: 예상 30분 초과")
    assert all(c["match"] for c in checked), "보유 스캔 재추출 해시 불일치"
    save_json(OUT / "fetch_complete.json", {"fetched": len(kept), "network_bytes": remote.received,
                                            "planned_network_bytes": len(plan) * BLOCK,
                                            "block_retries": remote.retries, "etag": remote.etag,
                                            "elapsed_s": time.time() - remote.started,
                                            "control_reextraction": {"scans": len(checked),
                                                                     "all_match_disco_sha256": all(c["match"] for c in checked)},
                                            "raw_sha256": kept, "created_utc": utc()})
    gz.close()
    remote.pool.shutdown(wait=False)


def raw_path(rel: str) -> Path:
    """평가 행의 원시 스캔 위치(02-04는 기존 native 사본, 08-20은 DiSCO 확보분 또는 내 수신분)."""
    if rel.startswith(MAP_SEQ):
        return NATIVE / "native_data" / rel
    name = Path(rel).name
    return DISCO / "velodyne_sync" / name if (DISCO / "velodyne_sync" / name).exists() else FETCHED / name


def protocol() -> None:
    """성능을 보기 전에 조건과 입력을 고정한다. 이미 있으면 다시 쓰지 않는다."""
    path = OUT / "protocol.json"
    if path.exists():
        print("protocol 이미 고정됨", flush=True)
        return
    cov = json.loads((OUT / "coverage.json").read_text())
    sets = json.loads((OUT / "query_sets.json").read_text())
    ref = json.loads((NATIVE / "evaluation_sets.json").read_text())
    missing = [e["raw_path"] for s in SPLITS for e in sets[s]["query"] if not raw_path(e["raw_path"]).exists()]
    splits = list(SPLITS)
    value = {
        "purpose": "논문 표II RING 70.98%의 날짜 쌍 불확실성 진단: NCLT 08-20 질의→02-04 지도를 03-17과 같은 공개 규칙으로 평가. "
                   "주표·설정 선택·학습에 쓰지 않는다. 두 날짜 중 논문 값에 가까운 쪽을 정답으로 고르지 않는다",
        "paper_evidence": {
            "text": "results/pdv2_review_native_20261105/ringpp_paper.txt(로컬 사본, 서버에는 없음)",
            "text_sha256": PAPER_TEXT_SHA256,
            "08-20_pair": "778-783행: 03-17→02-04 외에 08-20→02-04 쌍으로 장기 장소 인식(그림11, 20m 지도 간격)",
            "table_II": "794행에서 표II 소개, 847-858행 표II NCLT RING 0.7098/0.7658/0.8246, RING++ 0.7321/0.7890/0.8374; 날짜 쌍 미기재",
            "table_III_VI": "837행(표III)·1019행(표VI) 03-17→02-04 명시, RING++ 73.21%는 표II·III·VI 동일",
        },
        "conditions": [{"split": "all", "bev": "occ", "impl": "official", "method": "RING"},
                       {"split": "all", "bev": "feat", "impl": "maxfix", "method": "RING++ 정확max"}]
                      + ([{"split": "test", "bev": "occ", "impl": "official", "method": "RING"},
                          {"split": "test", "bev": "feat", "impl": "maxfix", "method": "RING++ 정확max"}]
                         if "test" in splits else []),
        "primary": "split=all(03-17 비교 기준과 같은 전체 궤적). test는 같은 공개 영역 규칙의 보조 행",
        "fixed_rules": {"map": "02-04 20m(기존 공식 생성기 출력 그대로)", "query": "08-20 5m, 20m 안 지도 있는 질의만",
                        "revisit_thresholds_m": RADII, "primary_radius_m": 10.0, "num_k": 25,
                        "score": "공식 batch_circorr(circorr2 |corr| 합) 그대로; 부호 합 진단은 이번에 반복하지 않음",
                        "bev_impl": {"RING": "공식 occ 그대로", "RING++": "공식 점특징 + 결정적 격자 max(_review_native_ring_maxfix.py)"},
                        "metrics": ["공식 출력 R@1(전체 질의 분모)", "양성 질의 분모 R@1(10m 안 지도 있음)",
                                    "공식 compute_PR_pairs 격자 최대 F1", "양성 고정 분모 최대 F1(TP/양성 수)"],
                        "pose_icp": "pygicp 없음; ICP 대체(장소 검색 무관), 자세 오차는 정답 수락 질의 분위만 참고"},
        "not_repeated": ["비원자 max 원본 RING++(이미 확인한 오류)", "부호 합 점수 C1", "커널·라돈 대조"],
        "inputs": {"query_sets_sha256": digest(OUT / "query_sets.json"),
                   "coverage_sha256": digest(OUT / "coverage.json"),
                   "evaluation_sets_sha256": digest(NATIVE / "evaluation_sets.json"),
                   "map_rows_all": len(ref["all"]["map"]), "query_rows": {s: len(sets[s]["query"]) for s in SPLITS},
                   "missing_raw_at_freeze": len(set(missing)),
                   "missing_rule": "고정 시점 누락 스캔은 fetch 단계(공식 S3 압축본 범위 수신, ETag 고정)로만 채운다. "
                                   "한 질의라도 없으면 해당 split을 평가하지 않는다(질의 제외·대체 없음)",
                   "official_source_sha256": cov["official_source_sha256"]},
        "script_sha256": digest(Path(__file__)),
        "maxfix_sha256": digest(REPO / "scripts/_review_native_ring_maxfix.py"),
        "frozen_utc": utc(),
    }
    save_json(path, value)
    print(json.dumps(value["conditions"], ensure_ascii=False), flush=True)


def build_pickle(split: str, path: Path) -> list[str]:
    """공식 EvaluationSet.save와 같은 [query_l, map_l] 튜플 목록. 행 순서는 생성기 출력 그대로."""
    sets = json.loads((OUT / "query_sets.json").read_text())[split]
    ref = json.loads((NATIVE / "evaluation_sets.json").read_text())[split]

    absent = [e["raw_path"] for e in sets["query"] + ref["map"] if not raw_path(e["raw_path"]).exists()]
    assert not absent, f"원시 스캔 누락 {len(absent)}개: 평가하지 않는다"

    def tup(e):
        return (np.int64(e["timestamp"]), str(raw_path(e["raw_path"])),
                np.asarray(e["xy"], np.float32), np.asarray(e["pose"], np.float32))

    with path.open("wb") as stream:
        pickle.dump([[tup(e) for e in sets["query"]], [tup(e) for e in ref["map"]]], stream)
    return [str(raw_path(e["raw_path"])) for e in sets["query"]]


def run(split: str, bev: str, impl: str) -> None:
    """공식 evaluate()를 실행하고 pair_dists·지도 TIRING·일부 질의 TIRING을 관찰 저장한다."""
    import torch
    import _review_native_ring as native

    proto = json.loads((OUT / "protocol.json").read_text())
    assert any(c["split"] == split and c["bev"] == bev and c["impl"] == impl for c in proto["conditions"])
    assert proto["script_sha256"] == digest(Path(__file__)), "스크립트가 고정 이후 바뀜"
    for p, h in proto["inputs"]["official_source_sha256"].items():
        assert digest(OFFICIAL / p) == h, p
    name = f"{split}_{bev}_{impl}"
    cond = OUT / name
    cond.mkdir(parents=True, exist_ok=True)
    if (cond / "report.json").exists():
        print(f"{name} 완료됨", flush=True)
        return
    pkl = cond / f"eval_{split}_0820.pickle"
    qpaths = build_pickle(split, pkl)
    for mod in ("pygicp", "seaborn"):
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            stub.__spec__ = importlib.machinery.ModuleSpec(mod, None)
            sys.modules[mod] = stub
    native.imports()
    import evaluation.evaluate as ev
    import utils.core as core

    if impl == "maxfix":
        import _review_native_ring_maxfix as maxfix
        maxfix.install_maxfix()
        ev.generate_bev_feats = core.generate_bev_feats
    ev.fast_gicp = lambda source, target, max_correspondence_distance=1.0, init_pose=np.eye(4): (
        float("nan"), np.asarray(init_pose, dtype=np.float64))
    ev.plot_cdf = lambda *a, **k: None
    original_rep, original_pr = ev.generate_representations, ev.compute_PR_pairs
    captured = {"reps": []}

    def rep_hook(dataset, eval_subset, bev_type="occ"):
        out = original_rep(dataset, eval_subset, bev_type)
        captured["reps"].append(np.stack([t.detach().cpu().numpy() for t in out[3]]))
        return out

    def pr_hook(pair_dists, query_positions, map_positions, thresholds, save_path, revisit_threshold=10.0):
        if "pair_dists" not in captured:
            captured["pair_dists"] = np.array(pair_dists, copy=True)
            np.save(cond / "pair_dists.npy", captured["pair_dists"])
        return original_pr(pair_dists, query_positions, map_positions, thresholds, save_path, revisit_threshold)

    ev.generate_representations, ev.compute_PR_pairs = rep_hook, pr_hook
    torch.set_num_threads(4)
    start = time.time()
    save_json(cond / "status.json", {"stage": "evaluate_running", "pid": os.getpid(), "started_utc": utc()})
    cwd = os.getcwd()
    os.chdir(cond)
    try:
        ev.evaluate("nclt", str(pkl), RADII, num_k=25, bev_type=bev)
    finally:
        os.chdir(cwd)
    elapsed = time.time() - start
    maps_t, query_t = captured["reps"]
    picks = sorted(set(range(0, len(query_t), 25)) | {len(query_t) - 1})
    np.savez_compressed(cond / "tirings.npz", maps=maps_t, query_subset=query_t[picks], query_subset_index=picks)
    folder = cond / "results/nclt" / pkl.stem / ("revisit" + "".join(f"_{r}" for r in RADII))
    load = lambda n: np.loadtxt(folder / f"{n}_{bev}.txt")
    report = {"condition": name, "split": split, "bev": bev, "impl": impl,
              "queries": int(captured["pair_dists"].shape[0]), "maps": int(captured["pair_dists"].shape[1]),
              "elapsed_s": elapsed, "gpu": torch.cuda.get_device_name(0),
              "printed_recalls_k": load("recalls_k").tolist(), "printed_f1s": load("f1s").tolist(),
              "printed_precisions": load("precisions").tolist(), "printed_recalls": load("recalls").tolist(),
              "rot_errors": load("rot_errors").tolist(), "trans_errors": load("trans_errors").tolist(),
              "pickle_sha256": digest(pkl), "pair_dists_sha256": digest(cond / "pair_dists.npy"),
              "tirings_sha256": digest(cond / "tirings.npz"),
              "query_raw_sha256_first_last": [digest(Path(qpaths[0])), digest(Path(qpaths[-1]))],
              "created_utc": utc()}
    save_json(cond / "report.json", report)
    save_json(cond / "status.json", {"stage": "complete", "pid": os.getpid(), "elapsed_s": elapsed})
    print(json.dumps({"condition": name, "R@1_10m": report["printed_recalls_k"][1][0]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["coverage", "fetch", "protocol", "run"])
    parser.add_argument("--split", choices=list(SPLITS), default="all")
    parser.add_argument("--bev", choices=["occ", "feat"])
    parser.add_argument("--impl", choices=["official", "maxfix"], default="official")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"coverage": coverage, "fetch": fetch, "protocol": protocol}.get(args.stage, lambda: None)()
    if args.stage == "run":
        assert not (args.bev == "occ" and args.impl == "maxfix")
        run(args.split, args.bev, args.impl)
