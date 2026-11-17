#!/usr/bin/env python3
"""새 보고모델(처음부터 학습, 시드4·u512)의 증분 비용을 주 결과표·부록 지도 크기별 비용표와 같은 규약으로 잰다.

기존 측정(`_pdv2_contrast_table3_opt.py`의 window 구현)과 같은 인코딩·21노드 창 상태 갱신·두 색인 후보 검색·
재정렬·최종 시간 판정 경로를 쓰고 모델과 고정 설정만 바꾼다.
- 모델: S.load(CFG, 4, 512). 입력 log1p, 정규화 층 LayerNorm, 게이트 고정(eta=-0.5).
- 고정 설정: P.selected() → 융합 (2,1,2,2), lambda=0.25(최종 판정의 .75/.25 혼합과 같다).
- 게이트: `gate=computed`는 현재 코드처럼 게이트 MLP를 계산하고, `gate=skipped`는 상수 eta=-0.5를 쓰며
  게이트 매개변수를 모델에서 뺀다(전체 그래프 계산에는 매개변수 없는 0 출력 모듈을 둔다). 두 변형의 복소 계수가
  비트 단위로 같은지 처음 12도착에서 검사한다.
- 검증: 처음 12도착을 GPU 전체 접두(q+15) 계산과 대조하고, 측정한 288질의의 승자·점수를 새 NSD 원장
  (results/pdv2_scratch_search_20261116/ledgers/val/mulran_Sejong01/<CFG>@s4@u512.npz)과 대조한다.

결과는 results/pdv2_report_model_20261116/cost/ 아래에만 쓴다. 기존 스크립트·결과는 바꾸지 않는다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_report_model_cost_20261116.py measure --gate skipped --n-map 10000 --folder .
docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_report_model_cost_20261116.py continuity
"""
import argparse
import gc
import platform
import resource
import subprocess
import sys
import time
import types
from pathlib import Path

import _pdv2_contrast_table3_opt as O   # 스레드 환경변수를 numpy·torch보다 먼저 고정하는 순서를 따른다

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from _pdv2_input_alignment_lib import make_graph, temporal_edges, pose_attributes
from _pdv2_table3_model_memory import audit_numeric_state
from gnn.spectral_transport import attach_transport, lidar_yaw

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _pdv2_scratch_search_20261116 as S  # noqa: E402

T, bench, pilot, current = O.T, O.bench, O.pilot, O.current
P = S.P
ROOT = O.ROOT
OUT = ROOT / "results/pdv2_report_model_20261116/cost"
CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
SEED, U = 4, 512
TAG = f"{CFG}@s{SEED}@u{U}"
CHECKPOINT = S.RUNS / CFG / f"s{SEED}" / f"u{U}.pth"
LEDGER = S.LEDGERS / "val" / "mulran_Sejong01" / f"{TAG}.npz"
W = O.W
GATE_CHANGE = {
    "computed": "gate: gate MLP evaluated on the first and last hidden states as in the current forward (frozen weights give eta=-0.5 exactly).",
    "skipped": "gate: constant eta=-0.5 used directly; gate MLP not evaluated and its parameters removed from the stored model (a parameter-free zero-output module stands in for full-graph initial state build).",
}


class ZeroGate(nn.Module):
    """매개변수 없는 0 출력. 고정 게이트의 마지막 선형층(가중치·편향 0)과 같은 출력이다."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(len(x), 1)


def environment_check() -> dict:
    """부하와 컨테이너 안의 다른 파이썬 프로세스 상태를 기록한다."""
    ps = subprocess.run(["ps", "-eo", "pid,stat,etime,args"], capture_output=True, text=True).stdout
    others = [line.strip()[:200] for line in ps.splitlines()[1:]
              if "python" in line and "_report_model_cost" not in line and " ps " not in line]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"loadavg": Path("/proc/loadavg").read_text().strip(), "other_python_processes": others, "nvidia_smi": gpu}


def gate_parameters(net: nn.Module) -> int:
    return sum(p.numel() for n, p in net.named_parameters() if n.startswith("gate."))


def model_state_counts(net: nn.Module) -> dict:
    """저장되는 모델 상태(매개변수·버퍼)를 센다. BatchNorm 통계는 없어야 한다."""
    params = sum(p.numel() for p in net.parameters())
    buffers = {n: b.numel() for n, b in net.named_buffers()}
    gate = gate_parameters(net)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
    return dict(parameters=params, gate_parameters=gate, parameters_without_gate=params - gate,
                buffers=buffers, buffer_elements=sum(buffers.values()),
                float32_bytes_with_gate=4 * (params + sum(buffers.values())),
                float32_bytes_without_gate=4 * (params - gate + sum(buffers.values())),
                batchnorm_modules=0, layernorm_modules=sum(isinstance(m, nn.LayerNorm) for m in net.modules()))


class Runtime(O.Runtime):
    """기존 window 측정기와 같은 상태·검색 규칙에 새 모델과 새 고정 설정만 넣는다."""

    def __init__(self, spec: dict, work: bench.Workload, skip_gate: bool):
        from _pd_build_test_caches import _effective_config, _make_desc_encoder
        import _pdv2_spectral_whitening as white
        self.white = white
        self.nodewise, self.skip_gate = False, skip_gate
        net = S.load(CFG, SEED, U).cpu().eval()
        last = [v for v in net.gate.modules() if isinstance(v, nn.Linear)][-1]
        assert last.out_features == 1 and not last.weight.any() and not last.bias.any()
        self.state_counts = model_state_counts(net)
        if skip_gate:
            net.gate = ZeroGate()
        self.net = net
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
        self.recent = np.empty((W, 288), np.float32)
        self.recent_coeff = np.empty((W, 16, 8), np.complex64)
        self.transient = 0
        self.reset()
        edges = temporal_edges(np.zeros(W, np.int64))
        src, tgt = edges
        index = np.broadcast_to(np.arange(W)[:, None], (W, 11)).copy()
        mask = np.zeros((W, 11), np.float32)
        index[tgt, src - tgt + 5], mask[tgt, src - tgt + 5] = src, 1.
        weights, lam = P.selected()
        assert lam == 0.25 and list(weights) == [2., 1., 2., 2.]   # 최종 판정의 .75/.25 혼합 = lambda 0.25
        self.consts = types.SimpleNamespace(
            edges=edges, edge_index=torch.from_numpy(edges), edge_type=torch.zeros(edges.shape[1], dtype=torch.long),
            transport_indices=torch.from_numpy(index), transport_mask=torch.from_numpy(mask),
            slots=torch.tensor(O.SLOTS), dense=O.DENSE, freq=O.FREQ, fusion=weights)
        if skip_gate:
            self.consts.strengths = torch.full((2, W, 1), -0.5)
        self.capture_encoder()

    def local(self, raw: np.ndarray, z: np.ndarray, poses: np.ndarray) -> np.ndarray:
        """21노드 창의 수정 복소 계수. 게이트 생략 변형은 상수 eta=-0.5를 쓴다."""
        if not self.skip_gate:
            return super().local(raw, z, poses)
        c, net = self.consts, self.net
        relative = self.poses[0, :3, :3].astype(np.float64).T[None] @ poses[:, :3, :3].astype(np.float64)
        yaw = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
        phase = torch.from_numpy(np.exp(1j * yaw[:, None] * c.freq).astype(np.complex64))
        with torch.inference_mode():
            h = F.relu(net.input_norm(net.input_proj(torch.from_numpy(raw.astype(np.float32)))))
            attr = net.edge_encoder(torch.from_numpy(pose_attributes(poses, c.edges)), c.edge_type)
            weights = []
            src, tgt = c.edge_index
            for layer, (conv, norm) in enumerate(zip(net.convs, net.batch_norms)):
                previous = h
                h = norm(conv(h, c.edge_index, edge_attr=attr))
                if layer == 0:
                    h = F.relu(h)
                h = h + previous
                dense = torch.zeros(len(h), 11, 4, dtype=h.dtype)
                dense[tgt, src - tgt + 5] = conv.attention
                weights.append(dense)
            net.last_strengths = c.strengths   # 원본 forward와 같은 내부 기록(초기 구축 때의 전체 그래프 기록을 남기지 않는다)
            spectrum = torch.from_numpy(np.ascontiguousarray(z, dtype=np.complex64))
            return O.transport_vectorized(spectrum, c.transport_indices, c.transport_mask, phase,
                                          c.strengths, torch.stack(weights), c.slots).numpy()


def identity_checks(runtime: Runtime, work: bench.Workload) -> dict:
    """처음 12도착: 인코딩 비트 일치, 21노드 창 대 31노드 갱신, 게이트 생략 대 게이트 계산의 비트 일치."""
    original_encode = bench.NSD.encode
    reference = S.load(CFG, SEED, U).cpu().eval()   # 게이트를 포함한 원래 모델
    records = []
    raw_seq = work.raw[:work.initial + bench.WARMUP].copy()
    coeff_seq = work.coeff[:work.initial + bench.WARMUP].copy()
    for i in range(bench.WARMUP):
        t = work.initial + i
        raw, coeff = runtime.encode(work.points[i])
        ref_raw, ref_coeff = original_encode(runtime, work.points[i])
        assert np.array_equal(raw, ref_raw) and np.array_equal(coeff, ref_coeff), t
        raw_seq[t], coeff_seq[t] = raw, coeff
        runtime.append(raw, coeff, work.poses[t])
        net, runtime.net = runtime.net, reference
        full = T.Runtime.updated(runtime, raw_seq[t-30:t+1], coeff_seq[t-30:t+1], runtime.poses[t-30:t+1])[-11:]
        skip, runtime.skip_gate = runtime.skip_gate, False
        gated = runtime.local(runtime.recent, runtime.recent_coeff, runtime.poses[t-W+1:t+1])[-11:]
        runtime.net, runtime.skip_gate = net, skip
        coefficient_error = float(abs(full - runtime.coeff[t-10:t+1]).max())
        assert coefficient_error < 4e-6, coefficient_error
        bitwise = bool(np.array_equal(gated, runtime.coeff[t-10:t+1]))
        assert bitwise, t
        records.append(dict(arrival=t, encode_bitwise=True, coefficient_max_error_vs_31node=coefficient_error,
                            coefficient_bitwise_vs_gate_computed_window=bitwise))
    del reference
    runtime.reset()
    return dict(status="passed", records=records,
                coefficient_error_max=max(r["coefficient_max_error_vs_31node"] for r in records),
                gate_variant_bitwise=all(r["coefficient_bitwise_vs_gate_computed_window"] for r in records))


@torch.no_grad()
def verify(runtime: Runtime, work: bench.Workload) -> dict:
    """새 원시 입력·CPU 증분 상태를 S.pipe와 같은 경로의 GPU 전체 접두(q+15) 계산과 대조한다."""
    assert S.pipeline is current
    reference = S.load(CFG, SEED, U)
    weights, lam = P.selected()
    records = []
    for i in range(bench.WARMUP):
        t = work.initial + i
        raw, coeff = runtime.encode(work.points[i])
        np.testing.assert_allclose(raw, work.raw[t], atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(coeff, work.coeff[t])
        runtime.append(raw, coeff, work.poses[t])
        graph = make_graph(work.raw[:t+1], work.poses[:t+1], np.zeros(t+1, np.int64)).cuda()
        attach_transport(graph, work.coeff[:t+1], lidar_yaw(work.poses[:t+1]), np.zeros(t+1, np.int64))
        z = reference(graph)
        assert bool((reference.last_strengths == -0.5).all())
        diff = abs(z.cpu().numpy() - runtime.coeff[:t+1])
        error = float(diff.max())
        # 기존 규약의 4e-6을 넘는 원소를 세어 기록한다(CPU 증분 대 GPU 전체 접두의 부동소수 차이). 판정은 1e-5와
        # 아래의 승자·최종 점수(8e-6) 일치로 한다.
        over = np.nonzero((diff > 4e-6).any(axis=(1, 2)))[0]
        assert error < 1e-5, error
        q = t - 15
        bank = S.run.base.variant_inputs(z.cpu().numpy(), 8)
        winner = S.run.cal.candidates(runtime.keys[:t+1], (runtime.hkey[:t+1], *bank[1:]), weights[None], q)
        plan = S.prior.base.prior.plan(runtime.travel[:t+1], np.array([q]), winner)
        value = P.score(z, {k: torch.from_numpy(v).cuda() for k, v in plan.items()}, lam).item()
        result = runtime.query(q)
        assert result[0] == int(winner[0]), (q, result, winner)
        assert abs(result[1] - value) < 8e-6, (q, result[1], value)
        records.append(dict(query=q, coefficient_max_error=error, elements_over_4e6=int((diff > 4e-6).sum()), nodes_over_4e6=over.tolist(), score_error=abs(result[1] - value), winner=result[0]))
        del graph, z, bank
    del reference
    runtime.reset()
    return dict(status="passed", fullprefix_checks=records, raw_regeneration=len(records))


def measure(n_map: int, gate: str, folder: str) -> None:
    """같은 96도착을 세 번 실행하고 새 NSD 원장과 일치하는지 검사한다."""
    current.e.configure()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = OUT / folder
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"nsd_n{n_map}_window.json"
    assert not target.exists(), target
    before = environment_check()
    sources = [Path(__file__), Path(O.__file__), Path(T.__file__), Path(S.__file__), Path(P.__file__), CHECKPOINT, LEDGER,
               P.OUT / "step1/selection.json",
               ROOT / "src/gnn/contrast_spectral_difference.py", ROOT / "src/gnn/spectral_difference.py",
               ROOT / "src/gnn/learned_spectral_difference.py", ROOT / "src/encoding/spectral_encoder.py",
               ROOT / "src/encoding/range_image.py", ROOT / "src/encoding/bev_image.py",
               ROOT / "scripts/_pdv2_table3_model_memory.py", ROOT / "scripts/_pdv2_input_alignment_lib.py"]
    weights, lam = P.selected()
    protocol = dict(method=f"{TAG}/matching_only", variant="window", gate=gate, warmup=12, samples=96, repeats=3, threads=1,
                    changes=O.CHANGES + [GATE_CHANGE[gate]],
                    sources={str(p.relative_to(ROOT)): bench.sha(p) for p in sources},
                    checkpoint=dict(kind="scratch", config=CFG, seed=SEED, updates=U,
                                    checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=bench.sha(CHECKPOINT)),
                    fusion_weights=[float(x) for x in weights], lam=lam,
                    workload_protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                    scope="기존 주 결과표 비용 열(window 구현)과 같음: 인코딩+11개복소상태갱신+두색인후보검색+재정렬+최종시간판정. "
                          "초기구축·I/O·공통KISS·CUDA graph 캡처 제외. 모델과 고정 설정만 새 보고모델로 바꿈. 시험 선택 없음.")
    path = out / "protocol_window.json"
    if path.exists():
        old = bench.read(path)
        assert old == protocol, "protocol mismatch"
    else:
        bench.write(path, protocol)
    spec = bench.read(bench.OUT / "protocol.json")
    assert n_map in spec["map_sizes"]
    work = bench.Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = Runtime(spec, work, skip_gate=gate == "skipped")
    build_s = time.perf_counter() - tick
    print("BUILT", gate, n_map, build_s, runtime.graph_pool, flush=True)
    identity = identity_checks(runtime, work)
    print("IDENTITY", identity["coefficient_error_max"], identity["gate_variant_bitwise"], flush=True)
    checked = verify(runtime, work)
    print("VERIFIED", checked["status"], max(r["score_error"] for r in checked["fullprefix_checks"]), flush=True)
    current.previous.source.COEFFICIENT_CACHE[:] = [None, None]
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    for i in range(bench.WARMUP):
        t = work.initial + i
        runtime.append(*runtime.encode(work.points[i]), work.poses[t])
        runtime.query(t - 15)
    rows, memory = [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for i, points in enumerate(work.points):
            t = work.initial + i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(*encoded, work.poses[t]))
            result, query_ms = bench.timed(lambda: runtime.query(t - 15))
            rows.append(dict(repeat=repeat, arrival=t, query=t - 15, encode=encode_ms, state_update=state_ms,
                             update=encode_ms + state_ms, query_ms=query_ms, processing=encode_ms + state_ms + query_ms,
                             winner=result[0], confidence=result[1], rerank_candidates=result[2],
                             transient_derived_bytes=runtime.transient))
        info = runtime.memory()
        info.update(process_rss=bench.rss(), process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    cuda_reserved=torch.cuda.memory_reserved(), cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved=torch.cuda.max_memory_reserved())
        memory.append(info)
    after = environment_check()
    ledger = dict(np.load(LEDGER))
    ids = np.searchsorted(ledger["query_idx"], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["query_idx"][ids], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["j_star"][ids], [r["winner"] for r in rows])
    error = float(abs(ledger["score"][ids] - [r["confidence"] for r in rows]).max())
    assert error < 8e-6 and len(rows) == 288, error
    checked["final_ledger"] = dict(queries=len(rows), winners_exact=True, score_max_error=error, sha256=bench.sha(LEDGER),
                                   path=str(LEDGER.relative_to(ROOT)))
    # 새 모델에는 이전 공개 측정이 없다. 표 생성기 호환 필드로 같은 원장 대조를 다시 적는다.
    checked["published_measurement"] = dict(
        winners_equal=True, reference="new NSD ledger (no earlier published measurement for this model)",
        confidence_max_difference=error)
    other = OUT / ("gate_computed" if gate == "skipped" else ".") / f"nsd_n{n_map}_window.json"
    if folder in (".", "gate_computed") and other.exists():
        o = bench.read(other)["raw_samples"]
        checked["other_gate_variant"] = dict(
            path=str(other.relative_to(ROOT)), winners_equal=[(r["query"], r["winner"]) for r in rows] == [(r["query"], r["winner"]) for r in o],
            confidence_bitwise_equal=all(a["confidence"] == b["confidence"] for a, b in zip(rows, o)))
    checked["identity_vs_original_implementation"] = identity
    result = dict(schema="pdv2-table3-measurement-v1", method="nsd", selected="diff/matching_only", model=TAG,
                  variant="window", gate=gate, n_map=n_map, status="passed", publish_ready=False, seed=SEED, sequence=bench.SEQUENCE,
                  samples=96, repeats=3, warmup=12, candidate_range=[n_map, n_map + 95], mean_candidates=n_map + 47.5,
                  method_protocol_sha256=bench.sha(path), protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                  timings_ms={k: bench.stats([r[f] for r in rows]) for k, f in
                              (("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                               ("query", "query_ms"), ("processing", "processing"))},
                  resident_bytes=memory[-1], memory_repeats=memory, model_state=runtime.state_counts,
                  model_state_stored=dict(gate_parameters_in_runtime=gate_parameters(runtime.net),
                                          parameters_in_runtime=sum(p.numel() for p in runtime.net.parameters())),
                  verification=checked,
                  initial_cached_state_build_s=build_s, raw_input_sha256=work.input_hash, raw_input_fixture_bytes=work.input_bytes,
                  raw_samples=rows, environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      cuda=torch.version.cuda, platform=platform.platform(), threads=1, before=before, after=after),
                  completed_unix=time.time())
    bench.write(target, result)
    print("COMPLETE", gate, n_map, {k: round(v["mean"], 4) for k, v in result["timings_ms"].items()},
          result["resident_bytes"]["total"], flush=True)


def continuity() -> None:
    """이전 보고모델의 기존 측정(nsd_n10000_window)을 같은 코드로 새 폴더에 다시 잰다(기계 상태 점검)."""
    O.OUT = OUT / "continuity_prev_model"
    O.environment_check = environment_check
    O.measure("window")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--gate", choices=("computed", "skipped"), required=True)
    m.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), required=True)
    m.add_argument("--folder", required=True)
    sub.add_parser("continuity")
    args = parser.parse_args()
    measure(args.n_map, args.gate, args.folder) if args.action == "measure" else continuity()
