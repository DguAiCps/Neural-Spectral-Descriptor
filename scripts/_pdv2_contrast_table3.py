#!/usr/bin/env python3
"""현재 복소 차이 모델의 실제 증분 상태·검색을 표3 규약으로 측정한다."""
import argparse
import gc
import os
import platform
import resource
import time
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import _pdv2_spectral_power_timing as pilot
import _pdv2_contrast_pipeline as current
from _pdv2_input_alignment_lib import make_graph
from _pdv2_table3_model_memory import audit_numeric_state
from gnn.spectral_transport import attach_transport, lidar_yaw

bench = pilot.bench
ROOT = bench.REPO
OUT = ROOT / "results/pdv2_contrast_table3_20261105"
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
LEDGER = current.OUT / "val/mulran_Sejong01/diff_s1.npz"


class Runtime(pilot.Runtime):
    """원색인은 고정하고 최근31노드에서 수정 복소 계수만 갱신한다."""

    def __init__(self, spec: dict, work: bench.Workload):
        from _pd_build_test_caches import _effective_config, _make_desc_encoder
        import _pdv2_spectral_whitening as white
        self.white = white
        case = bench.read(MAIN / "protocol.json")["cases"]["diff_s1"]
        assert bench.sha(ROOT / case["checkpoint"]) == case["checkpoint_sha256"]
        self.net = current.load_model(case).cpu().eval()
        self.encoder = _make_desc_encoder(_effective_config(), "mulran", "cuda").eval()
        self.matrix = np.load(white.OUT / "statistics.npz")["full"].copy()
        self.scale = np.load(current.run.source.OUT / "remetric_octave288.npy").astype(np.float32)
        self.mode = bench.read(current.OUT / "protocol.json")["key_transform"]
        self.conf = bench.read(current.OUT / "protocol.json")["confidence"]
        self.affine = bench.read(current.e.prior.base.prior.PARENT / "calibration_recenter.json")["spectral"]
        self.work, self.capacity, self.n = work, work.end, work.initial
        self.keys = np.empty((self.capacity, 288), np.float32)
        self.hkey = np.empty((self.capacity, 128), np.float32)
        self.coeff = np.empty((self.capacity, 16, 8), np.complex64)
        self.poses = np.empty((self.capacity, 4, 4), work.poses.dtype)
        self.travel = np.empty(self.capacity, np.float64)
        self.recent = np.empty((31, 288), np.float32)
        self.recent_coeff = np.empty((31, 16, 8), np.complex64)
        self.transient = 0
        self.reset()

    def raw_key(self, x: np.ndarray) -> np.ndarray:
        """선택된 원키 색인을 스캔별로 생성한다."""
        raw = np.log1p(x) if self.mode == "log1p" else x.copy()
        raw = raw.astype(np.float32) * self.scale[None]
        return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)

    @torch.no_grad()
    def updated(self, raw: np.ndarray, z: np.ndarray, poses: np.ndarray) -> np.ndarray:
        """전체와 동일한 첫 자세 기준의 yaw·기하 시간 그래프를 계산한다."""
        sequence = np.zeros(len(raw), np.int64)
        graph = make_graph(raw, poses, sequence)
        relative = self.poses[0, :3, :3].astype(np.float64).T[None] @ poses[:, :3, :3].astype(np.float64)
        yaw = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
        attach_transport(graph, z, yaw, sequence)
        return self.net(graph).numpy()

    def reset(self) -> None:
        """측정 밖에서 동일한 초기 상태를 복원한다."""
        n = self.work.initial
        self.n = n
        self.poses[:n] = self.work.poses[:n]
        self.keys[:n] = self.raw_key(self.work.raw[:n])
        self.hkey[:n] = pilot.current.base.index_key(self.work.coeff[:n])
        self.coeff[:n] = self.updated(self.work.raw[:n], self.work.coeff[:n], self.poses[:n])
        self.travel[:n] = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.poses[:n, :3, 3], axis=0), axis=1))]
        self.recent[:] = self.work.raw[n-31:n]
        self.recent_coeff[:] = self.work.coeff[n-31:n]

    def append(self, raw: np.ndarray, coeff: np.ndarray, pose: np.ndarray) -> None:
        """한 도착을 추가하고 영향받는 마지막11개 복소 상태를 재계산한다."""
        t = self.n
        self.poses[t] = pose
        self.travel[t] = self.travel[t-1] + np.linalg.norm(pose[:3, 3] - self.poses[t-1, :3, 3])
        self.keys[t] = self.raw_key(raw[None])[0]
        self.hkey[t] = pilot.current.base.index_key(coeff[None])[0]
        self.recent[:-1] = self.recent[1:]
        self.recent[-1] = raw
        self.recent_coeff[:-1] = self.recent_coeff[1:]
        self.recent_coeff[-1] = coeff
        local = self.updated(self.recent, self.recent_coeff, self.poses[t-30:t+1])
        self.coeff[t-10:t+1] = local[-11:]
        self.n += 1

    def query(self, q: int) -> tuple:
        """현재 원장의 공통 양의 척도로 최종 승자·판정 점수를 반환한다."""
        j, value, count = self.query_power(q, "p050__aux_half__equal")
        f = self.affine
        return j, (value-f["mean"][0])*f["coefficients"][0]/f["scale"][0]+f["intercept"], count

    def memory(self) -> dict:
        """입력 재생 자료를 제외하고 배포 상주 배열과 모델을 실제 주소로 센다."""
        arrays = {k: getattr(self, k) for k in ("keys", "hkey", "coeff", "poses", "travel", "recent", "recent_coeff", "matrix", "scale")}
        audit = audit_numeric_state(arrays, self.net, self.encoder)
        assert audit["status"] == "passed", audit["unknown_capacity_paths"]
        gpu = torch.cuda.memory_allocated()
        return dict(cpu=audit["cpu_unique_bytes"], gpu=gpu, total=audit["cpu_unique_bytes"]+gpu,
                    breakdown={k: v.nbytes for k, v in arrays.items()}, model_numeric_audit=audit,
                    logical_feature_payload_bytes_per_frame=2688,
                    recent_original_coefficients=31, full_original_coefficients_retained=False,
                    graph="최근31노드 임시 그래프; 마지막11개 복소 상태 갱신",
                    derived_banks="질의별 필요 행만 계산; 상주하지 않음")


@torch.no_grad()
def verify(runtime: Runtime, work: bench.Workload) -> dict:
    """새 원시 입력·CPU 증분 상태를 독립 GPU 전체 접두 계산과 대조한다."""
    reference = current.load_model(bench.read(MAIN / "protocol.json")["cases"]["diff_s1"])
    records = []
    for i in range(bench.WARMUP):
        t = work.initial+i
        raw, coeff = runtime.encode(work.points[i])
        np.testing.assert_allclose(raw, work.raw[t], atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(coeff, work.coeff[t])
        runtime.append(raw, coeff, work.poses[t])
        graph = make_graph(work.raw[:t+1], work.poses[:t+1], np.zeros(t+1, np.int64)).cuda()
        attach_transport(graph, work.coeff[:t+1], lidar_yaw(work.poses[:t+1]), np.zeros(t+1, np.int64))
        z = reference(graph)
        error = float(abs(z.cpu().numpy()-runtime.coeff[:t+1]).max())
        np.testing.assert_allclose(runtime.coeff[:t+1], z.cpu().numpy(), atol=4e-6, rtol=0)
        q = t-15
        bank = current.run.base.variant_inputs(z.cpu().numpy(), 8)
        search = (runtime.hkey[:t+1], *bank[1:])
        weights = current.run.cal.fusion_weights(runtime.conf)[current.run.cal.FUSIONS.index("aux_half")]
        winner = current.run.cal.candidates(runtime.keys[:t+1], search, weights[None], q)
        plan = current.e.prior.base.prior.plan(runtime.travel[:t+1], np.array([q]), winner)
        value = current.e.prior.score(z, {k: torch.from_numpy(v).cuda() for k, v in plan.items()}).item()
        result = runtime.query(q)
        assert result[0] == int(winner[0]), (q, result, winner)
        assert abs(result[1]-value) < 8e-6, (q, result[1], value)
        records.append(dict(query=q, coefficient_max_error=error, score_error=abs(result[1]-value), winner=result[0]))
        del graph, z, bank, search
    del reference
    runtime.reset()
    return dict(status="passed", fullprefix_checks=records, raw_regeneration=len(records))


def measure(n_map: int) -> None:
    """같은96도착을 세 번 실행하고 고정 검증 원장까지 일치해야 공개한다."""
    current.e.configure()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(exist_ok=True)
    target = OUT / f"nsd_n{n_map}.json"
    assert not target.exists(), target
    spec = bench.read(bench.OUT / "protocol.json")
    sources = [Path(__file__), Path(pilot.__file__), Path(current.__file__), MAIN / "protocol.json", LEDGER,
               ROOT / "src/gnn/contrast_spectral_difference.py", ROOT / "src/gnn/spectral_difference.py",
               ROOT / "scripts/_pdv2_table3_model_memory.py", ROOT / "scripts/_pdv2_input_alignment_lib.py"]
    protocol = dict(method="diff_s1/matching_only", warmup=12, samples=96, repeats=3, threads=1,
                    sources={str(p.relative_to(ROOT)): bench.sha(p) for p in sources},
                    checkpoint=bench.read(MAIN / "protocol.json")["cases"]["diff_s1"],
                    workload_protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                    scope="인코딩+실제11개복소상태갱신+두색인후보검색+재정렬+최종시간판정. 초기구축·I/O·공통KISS 제외. 시험 선택 없음.")
    pp = OUT / "protocol.json"
    if pp.exists():
        assert bench.read(pp) == protocol
    else:
        bench.write(pp, protocol)
    work = bench.Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = Runtime(spec, work)
    build_s = time.perf_counter()-tick
    print("BUILT", n_map, build_s, flush=True)
    checked = verify(runtime, work)
    print("VERIFIED", checked, flush=True)
    # 전체 접두 검산 함수의 전역 추론 캐시는 배포 상태가 아니다.
    current.previous.source.COEFFICIENT_CACHE[:] = [None, None]
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    for i in range(bench.WARMUP):
        t = work.initial+i
        runtime.append(*runtime.encode(work.points[i]), work.poses[t])
        runtime.query(t-15)
    rows, memory = [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for i, points in enumerate(work.points):
            t = work.initial+i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(*encoded, work.poses[t]))
            result, query_ms = bench.timed(lambda: runtime.query(t-15))
            rows.append(dict(repeat=repeat, arrival=t, query=t-15, encode=encode_ms, state_update=state_ms,
                             update=encode_ms+state_ms, query_ms=query_ms, processing=encode_ms+state_ms+query_ms,
                             winner=result[0], confidence=result[1], rerank_candidates=result[2]))
        info = runtime.memory()
        info.update(process_rss=bench.rss(), process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    cuda_reserved=torch.cuda.memory_reserved(), cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved=torch.cuda.max_memory_reserved())
        memory.append(info)
    ledger = dict(np.load(LEDGER))
    ids = np.searchsorted(ledger["query_idx"], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["query_idx"][ids], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["j_star"][ids], [r["winner"] for r in rows])
    error = float(abs(ledger["score"][ids]-[r["confidence"] for r in rows]).max())
    assert error < 8e-6 and len(rows) == 288, error
    checked["final_ledger"] = dict(queries=len(rows), winners_exact=True, score_max_error=error, sha256=bench.sha(LEDGER))
    result = dict(schema="pdv2-table3-measurement-v1", method="nsd", selected="diff/matching_only", n_map=n_map,
                  status="passed", publish_ready=True, seed=1, sequence=bench.SEQUENCE,
                  samples=96, repeats=3, warmup=12, candidate_range=[n_map,n_map+95], mean_candidates=n_map+47.5,
                  method_protocol_sha256=bench.sha(pp), protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                  timings_ms={k: bench.stats([r[f] for r in rows]) for k,f in
                              (("encode","encode"),("state_update","state_update"),("update","update"),("query","query_ms"),("processing","processing"))},
                  resident_bytes=memory[-1], memory_repeats=memory, verification=checked,
                  initial_cached_state_build_s=build_s, raw_input_sha256=work.input_hash, raw_input_fixture_bytes=work.input_bytes,
                  raw_samples=rows, environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      cuda=torch.version.cuda, platform=platform.platform(), threads=1), completed_unix=time.time())
    bench.write(target, result)
    print("COMPLETE", n_map, result["timings_ms"], result["resident_bytes"]["total"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-map", type=int, choices=(1000,5000,10000), default=10000)
    measure(parser.parse_args().n_map)
