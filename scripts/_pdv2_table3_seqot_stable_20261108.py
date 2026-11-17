#!/usr/bin/env python3
"""논문 학습법·GeM 수치 안정화 SeqOT 시드1을 표3 공통 규약으로 다시 잰다.

`scripts/_pdv2_table3_seqot.py`의 증분 구현(`SeqOT`)과 측정 루프(`measure`)를
그대로 불러 쓰고 아래만 바꾼다.

- 가중치: 단계1 11에폭(e05465db…), 단계2 1에폭(d197f9ae…). GeM은 같은 함수를
  로그 영역으로 계산하는 `GeMLogDomain`(`_seqot_stable_gem_20261108_gem.py`,
  50cc4504…)이다. 모델·데이터·시드 선택은 바꾸지 않는다.
- 출력: D 작업 폴더(`WORK/out`). 기존 `results/pdv2_table3_20261029`의 원본,
  공통 summary·status는 쓰지 않는다.
- 참조 캐시: 모델과 무관한 기존 `range.npy`(해시 고정)를 읽기만 한다. 단계1이
  바뀌었으므로 부분·최종 특징은 같은 GPU에서 배치 1로 새로 만든다.

``spec``은 GPU 없이 기대 계측 원천을 동결하고, ``crosscheck``는 서버에 이미 있는
VAL 부분 특징과 새 캐시를 CPU에서 대조한다. ``cache``와 ``measure``는
ring_baseline 안에서 직렬로 실행한다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _pdv2_table3_seqot as base

common, official, torch = base.common, base.official, base.torch
REPO = base.REPO
OLD_OUT = REPO / "results/pdv2_table3_20261029"
WORK = Path("/workspace/data/.nsd_work/seqot_stable_cost_20261108")
OUT = WORK / "out"
FROZEN = WORK / "frozen"
EXPECTED = FROZEN / "seqot_stable_protocol.expected.json"
LABEL = "paper recipe + numerical stabilization of official GeM"
COMMON_PROTOCOL_SHA256 = "16ed8bf28cc91815cc728e92c6fe32d24dcc1a4639dc6477d6fb7e6fcb88e35c"
BASE_SEQOT_PROTOCOL_SHA256 = "09b862c240dd81f3ffd1fccdd11df22420866bd33629c57488ae5e66e70862a8"
BASE_CACHE_MANIFEST_SHA256 = {
    1000: "86f2dc920739d92e24e5846956769e792a9022a4ca3cc6f08e5c6648e5b2ed9e",
    5000: "0e69f3d4520f53d7877e5211a1e5887e8cfcd6a52fcde513c93ba9186dd37ffd",
    10000: "10afda5d9243ec62355d4a448c1beaae93eef79f7ab89d0f18160bd2bdf636a1",
}
GEM_FILE = REPO / "scripts/_seqot_stable_gem_20261108_gem.py"
GEM_SHA256 = "50cc4504c6982a11d0a5c94cacec086408b7f30cb39e7f704206352ed9650563"
SERVER_REPO = "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"
MODELS = {
    "stage1": {"epoch": 11, "metric": "l2", "checkpoint": str(WORK / "weights/amodel_seqot11.pth.tar"),
               "server_checkpoint": SERVER_REPO + "_external/seqot_paper_recipe_20261107/s1/runs/stage1/amodel_seqot11.pth.tar",
               "sha256": "e05465db4498fee6a80da3bfd63e4770f2b4a1ab52fa97cba6bffb0cbe49b407"},
    "stage2": {"epoch": 1, "metric": "l2", "checkpoint": str(WORK / "weights/amodel_gem1.pth.tar"),
               "server_checkpoint": SERVER_REPO + "_external/seqot_stable_gem_20261108/s1/runs/stage2/amodel_gem1.pth.tar",
               "sha256": "d197f9ae67d57afeeea04b5e6f5d23ef6cf670e7b7a999b56454b03f3a979340"},
}
SERVER_SUB = {"path": str(FROZEN / "server_val_subdesc_s1e11_mulran_Sejong01.npz"),
              "server_path": SERVER_REPO + "_external/seqot_paper_recipe_20261107/s1/centered/subdesc/val/mulran_Sejong01.npz",
              "sha256": "e1c5558d351804e1ace95e78e16bdd4643fd0a0120d459a6b4ab1e989769a37a",
              "device": "server GPU0 RTX PRO 6000 Blackwell, select2 batch 64"}
SOURCES = ("scripts/_pdv2_table3_seqot_stable_20261108.py", "scripts/_pdv2_table3_seqot.py",
           "scripts/_pdv2_table3_benchmark.py", "scripts/_seqot_official_train.py",
           "scripts/_seqot_official_prep.py", "scripts/_seqot_stable_gem_20261108_gem.py",
           "_external/SeqOT/modules/seqTransformerCat.py", "_external/SeqOT/modules/netvlad.py",
           "_external/SeqOT/modules/gem.py", "_external/SeqOT/tools/utils/utils.py")


def chosen(stage: str) -> tuple:
    """`official._chosen` 대체: 고정 해시의 대표 시드1 가중치를 돌려준다."""
    c = dict(MODELS[stage])
    path = Path(c["checkpoint"])
    assert common.sha(path) == c["sha256"], f"{path} 해시 불일치"
    return path, c


def install() -> None:
    """원본 모듈의 출력 위치·가중치·GeM 클래스만 바꾼다."""
    assert common.sha(GEM_FILE) == GEM_SHA256, f"{GEM_FILE}는 승인된 GeM 패치가 아니다"
    spec = importlib.util.spec_from_file_location("_seqot_stable_gem_20261108_gem", GEM_FILE)
    gem_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gem_module)
    table = official.official()
    if "GeM_official" not in table:
        table["GeM_official"] = table["GeM"]
        table["GeM"] = gem_module.make_gem_logdomain(table["GeM_official"])
    assert issubclass(table["GeM"], table["GeM_official"]) and table["GeM"] is not table["GeM_official"]
    official._chosen = chosen
    base.OUT = common.OUT = OUT
    base.PROTOCOL = OUT / "seqot_protocol.json"
    base.prepare = prepare
    base.SeqOTWorkload = Workload
    base.verify = verify
    common.summarize = summarize


def addon() -> dict:
    """기존 SeqOT 계측 원천에서 가중치·GeM·캐시 규칙만 바꾼 동결 내용이다."""
    old = common.read(OLD_OUT / "seqot_protocol.json")
    value = {k: v for k, v in old.items() if k != "sources"}
    value.update({
        "schema": "pdv2-table3-seqot-stable-protocol-v1", "revision": 1, "label": LABEL,
        "base_method_protocol": {"path": "results/pdv2_table3_20261029/seqot_protocol.json",
                                 "sha256": BASE_SEQOT_PROTOCOL_SHA256},
        "stage1": MODELS["stage1"], "stage2": MODELS["stage2"],
        "gem": {"class": "GeMLogDomain", "file": str(GEM_FILE.relative_to(REPO)), "sha256": GEM_SHA256,
                "note": "same GeM function evaluated in the log domain; official modules/gem.py unchanged"},
        "initialization": old["initialization"] + "; model-independent range.npy of the base reference cache reused read-only (sha pinned), subdescriptor/final caches regenerated with the new weights on the measuring GPU, batch1",
        "reference_cache_manifests": {str(n): v for n, v in BASE_CACHE_MANIFEST_SHA256.items()},
        "outputs": str(OUT),
        "untouched": "results/pdv2_table3_20261029 originals, common summary.json/status.json",
        "sources": {rel: common.sha(REPO / rel) for rel in SOURCES},
    })
    return value


def prepare() -> dict:
    """공통 규약 사본과 동결 기대 원천을 확인하고 이 방법의 원천을 한 번 쓴다."""
    assert common.sha(OLD_OUT / "protocol.json") == COMMON_PROTOCOL_SHA256
    assert common.sha(OUT / "protocol.json") == COMMON_PROTOCOL_SHA256, "공통 규약 사본이 다르다"
    assert common.sha(OLD_OUT / "seqot_protocol.json") == BASE_SEQOT_PROTOCOL_SHA256
    spec = common.read(OUT / "protocol.json")
    assert spec["sequence"] == "mulran/Sejong01"
    assert spec["samples"] == common.SAMPLES and spec["repeats"] == common.REPEATS
    value = addon()
    assert common.read(EXPECTED) == value, "동결 기대 원천과 다르다: 새 개정 필요"
    if base.PROTOCOL.exists():
        assert common.read(base.PROTOCOL) == value, "SeqOT 계측 원천 변경: 새 개정 필요"
    else:
        common.write(base.PROTOCOL, value)
    return spec


class RangeRows:
    """필요한 거리 영상 행만 pread로 읽는다. mmap 페이지가 RSS에 쌓이지 않는다."""

    def __init__(self, path: Path) -> None:
        image = np.load(path, mmap_mode="r")
        assert image.dtype == np.float32 and image.shape[1:] == base.RANGE_SHAPE
        self.shape, self.offset = image.shape, int(image.offset)
        self.row = int(np.prod(base.RANGE_SHAPE)) * 4
        del image
        self.fd = os.open(path, os.O_RDONLY)

    def rows(self, ids) -> np.ndarray:
        out = np.empty((len(ids), *base.RANGE_SHAPE), np.float32)
        for k, r in enumerate(ids):
            assert 0 <= int(r) < self.shape[0]
            data = os.pread(self.fd, self.row, self.offset + int(r) * self.row)
            assert len(data) == self.row
            out[k] = np.frombuffer(data, np.float32).reshape(base.RANGE_SHAPE)
        return out


def base_cache(n_map: int) -> tuple:
    """기존 참조 캐시 목록과 재사용할 range.npy를 해시로 확인한다."""
    folder = OLD_OUT / "seqot_batch1" / f"n{n_map}"
    assert common.sha(folder / "complete.json") == BASE_CACHE_MANIFEST_SHA256[n_map]
    meta = common.read(folder / "complete.json")
    assert meta["seqot_protocol_sha256"] == BASE_SEQOT_PROTOCOL_SHA256
    assert common.sha(folder / "range.npy") == meta["files"]["range.npy"]
    return folder, meta


def cache(n_map: int) -> None:
    """원본 `cache`와 같은 창·배치1로 부분·최종 특징을 새 가중치로 만든다."""
    base.configure()
    spec = prepare()
    target = base.cache_dir(n_map)
    done = target / "complete.json"
    if done.exists():
        meta = common.read(done)
        assert meta["seqot_protocol_sha256"] == common.sha(base.PROTOCOL)
        for name, digest in meta["files"].items():
            assert common.sha(target / name) == digest
        return
    folder, old = base_cache(n_map)
    n_raw = int(old["n_raw"])
    ids = base.keyframe_ids(spec, n_map)
    np.testing.assert_array_equal(np.load(folder / "scan_ids.npy"), ids)
    windows = np.stack([base.gem_window(int(s), n_raw) for s in ids])
    last_sub = int(windows[windows < n_raw].max())
    last_raw = max(base.required_raw(int(s), n_raw) for s in ids)
    assert (last_raw + 1, last_sub + 1) == (old["range_scans"], old["fe_centers"])
    ranges = RangeRows(folder / "range.npy")
    assert ranges.shape[0] == last_raw + 1
    started = time.perf_counter()
    fe = official._load_fe(official._chosen("stage1")[0], "cuda")
    gem = official._load_gem(official._chosen("stage2")[0], "cuda")
    assert type(gem) is official.official()["GeM"]
    load_s = time.perf_counter() - started
    sub = np.empty((last_sub + 1, base.DIM), np.float32)
    common.sync()
    tick = time.perf_counter()
    with torch.no_grad():
        for r in range(last_sub + 1):
            w = official.subdesc_windows(
                [r], lambda x: 0 <= x < n_raw, 3, "centered")[0]
            sub[r] = fe(torch.from_numpy(ranges.rows(w)[None]).cuda())[0].cpu().numpy()
            if r % 1000 == 0:
                print("SEQOT_STABLE_FE_BATCH1", n_map, r, last_sub + 1, flush=True)
    common.sync()
    sub_s = time.perf_counter() - tick
    final = np.empty((len(ids), base.DIM), np.float32)
    tick = time.perf_counter()
    with torch.no_grad():
        for k, w in enumerate(windows):
            rows = np.zeros((20, base.DIM), np.float32)
            valid = w < n_raw
            rows[valid] = sub[w[valid]]
            final[k] = gem(torch.from_numpy(rows[None]).cuda())[0, 0].cpu().numpy()
    common.sync()
    gem_s = time.perf_counter() - tick
    assert np.isfinite(sub).all() and np.isfinite(final).all()
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "sub.npy", sub)
    np.save(target / "final.npy", final)
    np.save(target / "scan_ids.npy", ids)
    common.write(done, {
        "status": "passed", "seqot_protocol_sha256": common.sha(base.PROTOCOL), "label": LABEL,
        "n_map": n_map, "n_raw": n_raw, "keyframes": len(ids),
        "range_scans": last_raw + 1, "fe_centers": last_sub + 1, "batch": 1,
        "model_load_s": load_s, "fe_s": sub_s, "gem_s": gem_s,
        "seconds": time.perf_counter() - started,
        "files": {name: common.sha(target / name) for name in ("sub.npy", "final.npy", "scan_ids.npy")},
        "reused_range": {"path": str((folder / "range.npy").relative_to(REPO)),
                         "sha256": old["files"]["range.npy"],
                         "base_manifest_sha256": BASE_CACHE_MANIFEST_SHA256[n_map],
                         "base_projection_and_disk_read_s": old["projection_and_disk_read_s"]},
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda},
        "scope": "reference fixture for initial state plus measured arrivals; model-independent range reused read-only; excluded from incremental timings",
    })
    print("CACHE_COMPLETE", n_map, round(sub_s, 2), round(gem_s, 2), flush=True)


class Workload(base.SeqOTWorkload):
    """원본 workload와 같고 거리 영상만 기존 참조 캐시에서 읽기 전용으로 연다."""

    def __init__(self, spec: dict, n_map: int) -> None:
        from _pd_build_test_caches import _make_loader

        self.n_map = n_map
        self.initial = n_map + 44
        self.end = self.initial + common.SAMPLES
        self.ids = base.keyframe_ids(spec, n_map)
        target = base.cache_dir(n_map)
        self.meta = common.read(target / "complete.json")
        assert self.meta["seqot_protocol_sha256"] == common.sha(base.PROTOCOL)
        for name, digest in self.meta["files"].items():
            assert common.sha(target / name) == digest
        reused = REPO / self.meta["reused_range"]["path"]
        assert common.sha(reused) == self.meta["reused_range"]["sha256"]
        np.testing.assert_array_equal(np.load(target / "scan_ids.npy"), self.ids)
        self.n_raw = int(self.meta["n_raw"])
        self.ranges = np.load(reused, mmap_mode="r")
        self.sub = np.load(target / "sub.npy", mmap_mode="r")
        self.final = np.load(target / "final.npy", mmap_mode="r")
        self.raw_batches = []
        loader = _make_loader("mulran", "Sejong01")
        self.input_bytes = 0
        all_hash, keyframe_hash = base.hashlib.sha256(), base.hashlib.sha256()
        for t in range(self.initial, self.end):
            batch = []
            for r in range(int(self.ids[t - 1]) + 1, int(self.ids[t]) + 1):
                points = np.ascontiguousarray(loader[r]["points"])
                batch.append((r, points))
                self.input_bytes += points.nbytes
                all_hash.update(np.int64(r).tobytes())
                all_hash.update(points.tobytes())
            keyframe_hash.update(batch[-1][1].tobytes())
            self.raw_batches.append(batch)
        del loader
        self.input_hash = all_hash.hexdigest()
        self.keyframe_input_hash = keyframe_hash.hexdigest()
        self.reference_fixture_bytes = int(self.ids.nbytes + self.ranges.nbytes
                                           + self.sub.nbytes + self.final.nbytes)


def verify(runtime, work) -> dict:
    """원본 `verify`와 같다. 캐시 위치가 저장소 밖이라 경로 기록만 다르다."""
    runtime.capture = True
    queries = []
    for i, batch in enumerate(work.raw_batches[:common.WARMUP]):
        t = work.initial + i
        runtime.encode(batch)
        runtime.append()
        value = runtime.query(t - 15)
        index = runtime.faiss.IndexFlatL2(base.DIM)
        index.add(np.asarray(work.final[:t - 44]))
        distances, indices = index.search(np.asarray(work.final[t - 15:t - 14]), 1)
        assert value[0] == int(indices[0, 0])
        score_error = abs(value[1] + float(np.sqrt(max(float(distances[0, 0]), 0.))))
        assert score_error < 2e-5
        queries.append({"arrival": t, "winner": value[0], "score_error": score_error})
    n_raw = sum(map(len, work.raw_batches[:common.WARMUP]))
    assert runtime.counters["raw_projection"] == runtime.counters["fe"] == n_raw
    result = {"status": "passed", "official_prefix_checks": queries,
              "streaming_reference_checks": runtime.checks.copy(),
              "new_raw_scans": n_raw, "counters": runtime.counters.copy(),
              "tolerance": "range exact; batch1 FE/GeM atol1e-7 rtol1e-6; winner exact; L2 score abs2e-5",
              "reference_cache": str(base.cache_dir(work.n_map)),
              "reference_cache_manifest_sha256": common.sha(base.cache_dir(work.n_map) / "complete.json")}
    runtime.capture = False
    runtime.reset()
    return result


def summarize() -> None:
    """이 작업 폴더의 완료 행만 모은다. 공통 summary.json은 건드리지 않는다."""
    records = []
    for path in sorted((OUT / "methods").glob("seqot_n*.json")):
        row = common.read(path)
        assert row["status"] == "passed" and row["verification"]["status"] == "passed"
        assert row["protocol_sha256"] == COMMON_PROTOCOL_SHA256
        assert row["method_protocol_sha256"] == common.sha(base.PROTOCOL)
        records.append({k: v for k, v in row.items() if k not in ("raw_samples", "memory_repeats")})
    common.write(OUT / "summary.json", {
        "schema": "pdv2-table3-seqot-stable-v1", "label": LABEL,
        "protocol_sha256": COMMON_PROTOCOL_SHA256, "method_protocol_sha256": common.sha(base.PROTOCOL),
        "adoption": "not adopted until root verification pins these file hashes",
        "measurements": records})


def crosscheck() -> None:
    """서버 VAL 부분 특징(다른 GPU·배치64)과 새 캐시를 CPU에서 대조한다."""
    torch.set_num_threads(1)
    assert common.sha(Path(SERVER_SUB["path"])) == SERVER_SUB["sha256"]
    with np.load(SERVER_SUB["path"]) as z:
        server_sub, server_raw = z["sub"], z["raw"]
        assert str(z["stage1_sha256"]) == MODELS["stage1"]["sha256"]
    spec = common.read(OUT / "protocol.json")
    gem = official.official()["GeM"]().eval()
    gem.load_state_dict(torch.load(MODELS["stage2"]["checkpoint"], map_location="cpu",
                                   weights_only=False)["state_dict"])
    report = {"server_sub": SERVER_SUB, "rule": CROSS_RULE, "sizes": {}}
    for n_map in (1000, 5000, 10000):
        target = base.cache_dir(n_map)
        meta = common.read(target / "complete.json")
        for name, digest in meta["files"].items():
            assert common.sha(target / name) == digest
        sub, final = np.load(target / "sub.npy"), np.load(target / "final.npy")
        ids, n_raw = base.keyframe_ids(spec, n_map), int(meta["n_raw"])
        pos = np.full(n_raw + 1, -1, np.int64)
        pos[server_raw] = np.arange(len(server_raw))
        windows = np.stack([base.gem_window(int(s), n_raw) for s in ids])
        need = np.unique(windows[windows < n_raw])
        assert np.all(pos[need] >= 0), "서버 캐시가 필요한 부분 특징을 덮지 않는다"
        a, b = sub[need], server_sub[pos[need]]
        diff = np.abs(a - b)
        allowed = 2e-6 + 2e-6 * np.abs(b)

        def gem_rows(source) -> np.ndarray:
            out = np.empty((len(ids), base.DIM), np.float32)
            with torch.no_grad():
                for k, w in enumerate(windows):
                    rows = np.zeros((20, base.DIM), np.float32)
                    valid = w < n_raw
                    rows[valid] = source(w[valid])
                    out[k] = gem(torch.from_numpy(rows[None]))[0, 0].numpy()
            return out

        local_cpu = gem_rows(lambda w: sub[w])
        server_cpu = gem_rows(lambda w: server_sub[pos[w]])
        initial = n_map + 44
        winners, max_score = 0, 0.
        for t in range(initial, initial + common.SAMPLES):
            q = t - 15
            results = []
            for bank in (final, server_cpu):
                d = ((bank[:t - 44].astype(np.float64) - bank[q].astype(np.float64)) ** 2).sum(1)
                j = int(np.argmin(d))
                results.append((j, -float(np.sqrt(d[j]))))
            winners += results[0][0] == results[1][0]
            max_score = max(max_score, abs(results[0][1] - results[1][1]))
        report["sizes"][str(n_map)] = {
            "subdescriptor_rows": int(len(need)),
            "sub_max_abs": float(diff.max()), "sub_max_rel": float((diff / np.maximum(np.abs(b), 1e-12)).max()),
            "sub_rows_beyond_2e-6": int((diff > allowed).any(1).sum()),
            "final_gpu_vs_cpu_gem_max_abs": float(np.abs(final - local_cpu).max()),
            "final_local_vs_server_sub_max_abs": float(np.abs(local_cpu - server_cpu).max()),
            "measured_queries": common.SAMPLES, "winners_equal": int(winners),
            "l2_score_max_abs": max_score,
        }
    report["passed"] = all(v["winners_equal"] == v["measured_queries"] and v["l2_score_max_abs"] < 2e-5
                           for v in report["sizes"].values())
    report["elementwise_within_2e-6"] = all(v["sub_rows_beyond_2e-6"] == 0 for v in report["sizes"].values())
    common.write(OUT / "crosscheck_server_val_subdesc.json", report)
    print("CROSSCHECK", json.dumps(report["sizes"]), "passed", report["passed"], flush=True)


CROSS_RULE = ("diagnostic only, never a substitute for the same-GPU batch1 reference: inputs bitwise equal "
              "(server range rows spot-checked); report sub/final max abs; pass = 288 measured-query "
              "exact-L2 winners equal and |L2 score diff| < 2e-5 (base SeqOT prefix tolerance); elementwise "
              "|d| <= 2e-6 + 2e-6|ref| (base table3 NSD cached-encoder tolerance) reported separately")


def spec() -> None:
    """GPU·가중치 없이 기대 계측 원천을 한 번 동결한다."""
    value = addon()
    if EXPECTED.exists():
        assert common.read(EXPECTED) == value, "동결 기대 원천과 다르다"
    else:
        common.write(EXPECTED, value)
    print("SPEC", common.sha(EXPECTED), flush=True)


def selftest() -> None:
    """CPU에서 연결만 확인한다. CUDA를 초기화하지 않는다."""
    install()
    table = official.official()
    checks = {
        "gem_class_logdomain": table["GeM"].__name__ == "GeMLogDomain",
        "gem_subclass_of_official": issubclass(table["GeM"], table["GeM_official"]),
        "chosen_patched": official._chosen is chosen,
        "out_redirected": base.OUT == OUT and common.OUT == OUT and base.PROTOCOL.parent == OUT,
        "summarize_redirected": common.summarize is summarize,
        "workload_verify_prepare_patched": (base.SeqOTWorkload is Workload and base.verify is verify
                                            and base.prepare is prepare),
        "measure_unchanged": base.measure.__module__ == "_pdv2_table3_seqot",
        "expected_spec_equal": common.read(EXPECTED) == addon(),
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    print("SELFTEST", json.dumps(checks), flush=True)
    assert all(v for k, v in checks.items() if k != "cuda_initialized") and not checks["cuda_initialized"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("spec", "selftest", "prepare", "cache", "measure", "crosscheck"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    if args.action in ("spec", "selftest"):
        globals()[args.action]()
    else:
        install()
        if args.action == "prepare":
            prepare()
        elif args.action == "crosscheck":
            crosscheck()
        elif args.action == "cache":
            cache(args.n_map)
        else:
            base.measure(args.n_map)
