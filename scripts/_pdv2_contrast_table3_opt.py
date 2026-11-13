#!/usr/bin/env python3
"""표3 NSD 측정의 파이썬 최적화판. 결과를 바꾸지 않는 구현 정리만 적용해 같은 규약으로 잰다.

- 인코딩: 거리 영상과 BEV가 극좌표를 공유하고, 빈 칸 보간을 행 루프 없이 np.interp와 같은 식으로 계산한다.
  상수 구간 경계를 한 번만 만들어 GPU 스펙트럼 통계를 CUDA graph로 재생하고, BEV는 GPU 대기 중에 계산한다.
  원시 키와 복소 계수는 원본과 비트 단위로 같아야 한다.
- 상태 갱신: 마지막 11개 계수의 수용 영역인 최근 21노드 창, 고정 간선 재사용, 복소 전달 벡터화.
  nodewise 변형은 DiffAttnConv의 K/V 투영을 W(h_j-h_i)=Wh_j-Wh_i+b로 노드에서 한 번만 한다.
- 질의: 최종 판정이 쓰는 arc_unique 계획 하나만 만들고 뱅크·채널마다 상관을 한 번 계산한다.

결과는 results/pdv2_contrast_table3_opt_20261106/ 에만 쓴다. 표3 원천(pdv2_contrast_table3_20261105)은 바꾸지 않는다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_pdv2_contrast_table3_opt.py baseline --tag before
docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_pdv2_contrast_table3_opt.py measure --variant window
"""
import argparse
import gc
import platform
import resource
import subprocess
import time
import types
from pathlib import Path

import _pdv2_contrast_table3 as T   # 스레드 환경변수를 numpy·torch보다 먼저 고정한다

import numpy as np
import torch
import torch.nn.functional as F

from _pdv2_input_alignment_lib import make_graph, temporal_edges, pose_attributes
from _pdv2_table3_model_memory import audit_numeric_state
from gnn.spectral_transport import attach_transport, lidar_yaw

bench, pilot, current = T.bench, T.pilot, T.current
ROOT = bench.REPO
OUT = ROOT / "results/pdv2_contrast_table3_opt_20261106"
PUBLISHED = T.OUT / "nsd_n10000.json"
W = 21                      # 마지막 11개 계수(t-10..t)의 입력 의존 범위 t-20..t
SLOTS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
DENSE = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
BETA = (2. / 3.) * (11. / 10.)
FREQ = np.arange(1, 9)[None]
CHANGES = [
    "encode: shared polar coordinates for range image and BEV; vectorized circular linear interpolation with np.interp's formula; constant octave bin edges built once; GPU spectral statistics replayed as a CUDA graph; BEV computed while the GPU runs. Raw keys and coefficients must be bitwise equal to the original encode.",
    "state update: 21-node recent window (receptive field of the last 11 coefficients), fixed edge/transport indices reused, complex difference propagation over the 10 neighbor slots vectorized.",
    "query: temporal decision builds only the arc_unique (arc_dense) plan used by the final score and computes the center and neighbor correlations once per bank and channel; the unused arc_sparse score and other neighbor plans are not computed.",
]
NODEWISE_CHANGE = "nodewise: DiffAttnConv key/value projections computed once per node as W h_j - W h_i + b instead of W(h_j - h_i) per edge."


def environment_check() -> dict:
    """부하와 컨테이너 안의 다른 파이썬 프로세스 상태를 기록한다."""
    ps = subprocess.run(["ps", "-eo", "pid,stat,etime,args"], capture_output=True, text=True).stdout
    others = [line.strip()[:200] for line in ps.splitlines()[1:]
              if "python" in line and "_pdv2_contrast_table3" not in line and " ps " not in line]
    return {"loadavg": Path("/proc/loadavg").read_text().strip(), "other_python_processes": others}


# ------------------------------------------------------------------ 인코딩
def interp_rows(image: np.ndarray) -> np.ndarray:
    """interpolate_range_image(method='linear')와 같은 값. 모든 행을 한 번에 보간한다."""
    result = image.copy()
    n_rows, n_az = image.shape
    valid = result > 0
    need = np.nonzero(valid.any(1) & ~valid.all(1))[0]
    if len(need):
        tiled_valid = np.tile(valid[need], 3)
        tiled_value = np.tile(result[need].astype(np.float64), 3)
        position = np.arange(3 * n_az)
        prev = np.maximum.accumulate(np.where(tiled_valid, position, -1), axis=1)
        nxt = np.minimum.accumulate(np.where(tiled_valid, position, 3 * n_az)[:, ::-1], axis=1)[:, ::-1]
        middle = slice(n_az, 2 * n_az)
        rows = np.arange(len(need))[:, None]
        x = (position[middle] - n_az).astype(np.float64)
        x0 = (prev[:, middle] - n_az).astype(np.float64)
        x1 = (nxt[:, middle] - n_az).astype(np.float64)
        f0 = tiled_value[rows, prev[:, middle]]
        f1 = tiled_value[rows, nxt[:, middle]]
        invalid = ~valid[need]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (f1 - f0) / (x1 - x0)
            value = slope * (x - x0) + f0
        block = result[need]
        block[invalid] = value[invalid]
        result[need] = block
    for row in range(n_rows):
        if not np.any(result[row] > 0):
            for offset in range(1, n_rows):
                if row - offset >= 0 and np.any(result[row - offset] > 0):
                    result[row] = result[row - offset]
                    break
                if row + offset < n_rows and np.any(result[row + offset] > 0):
                    result[row] = result[row + offset]
                    break
    return result


def polar(points: np.ndarray) -> tuple:
    """두 투영이 같은 점에서 따로 계산하던 제곱·거리·방위각을 한 번만 계산한다."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not finite.all():
        x, y, z = x[finite], y[finite], z[finite]
    x_sq, y_sq, z_sq = np.clip(x**2, 0, 1e10), np.clip(y**2, 0, 1e10), np.clip(z**2, 0, 1e10)
    # BEV 원본은 clip 없이 sqrt(x**2+y**2)를 쓴다. clip이 걸리지 않으면 같다.
    assert x_sq.max(initial=0.) < 1e10 and y_sq.max(initial=0.) < 1e10
    azimuth = (np.arctan2(y, x) + np.pi) % (2 * np.pi)
    return z, np.sqrt(x_sq + y_sq + z_sq), azimuth, np.sqrt(x_sq + y_sq)


def range_image(projector, z, distance, azimuth, horizontal) -> np.ndarray:
    """RangeImageProjector.project와 같은 식이다."""
    keep = (distance >= projector.min_range) & (distance <= projector.max_range) & np.isfinite(distance)
    elevation = np.arctan2(z[keep], horizontal[keep])
    norm = (elevation - projector.elevation_min) / (projector.elevation_max - projector.elevation_min)
    rows = np.clip(np.floor(norm * projector.n_elevation).astype(int), 0, projector.n_elevation - 1)
    cols = np.clip(np.floor(azimuth[keep] / (2 * np.pi) * projector.n_azimuth).astype(int), 0, projector.n_azimuth - 1)
    flat = np.full(projector.n_elevation * projector.n_azimuth, np.inf, dtype=np.float32)
    np.minimum.at(flat, rows * projector.n_azimuth + cols, distance[keep])
    image = flat.reshape(projector.n_elevation, projector.n_azimuth)
    image[image == np.inf] = 0.0
    return image


def bev_image(projector, z, azimuth, horizontal) -> np.ndarray:
    """BEVProjector.project(max 높이)와 같은 식이다."""
    keep = (horizontal >= projector.min_range) & (horizontal < projector.max_range) & (z > projector.z_min)
    rings = np.clip(np.floor(horizontal[keep] - projector.min_range).astype(int), 0, projector.n_rings - 1)
    sectors = np.clip(np.floor(azimuth[keep] / (2 * np.pi) * projector.n_sectors).astype(int), 0, projector.n_sectors - 1)
    flat = np.full(projector.n_rings * projector.n_sectors, -np.inf, dtype=np.float32)
    np.maximum.at(flat, rings * projector.n_sectors + sectors, z[keep].astype(np.float32))
    image = flat.reshape(projector.n_rings, projector.n_sectors)
    image[image == -np.inf] = 0.0
    return image


# ------------------------------------------------------------------ 상태 갱신
def transport_vectorized(spectrum, indices, mask, phase, strengths, attentions, slots):
    """transport_difference와 같은 식. 이웃 칸 10개를 한 번에 모은다(덧셈 순서만 다르다)."""
    norm = spectrum.abs().flatten(1).norm(dim=1).clamp_min(1e-12)
    world = spectrum / norm[:, None, None] * phase.conj()[:, None]
    neighbor = indices[:, slots]
    valid = mask[:, slots]
    for layer in range(2):
        a = attentions[layer][:, slots].repeat_interleave(2, dim=-1) * valid[..., None]
        update = (a[:, :, None, :] * (world[neighbor] - world[:, None])).sum(1)
        world = world + strengths[layer, :, :, None] * update
    return world * phase[:, None]


def conv_nodewise(conv, x, edge_index, edge_attr):
    """ControlledRelationConv(diff)와 같은 식. K/V 투영을 노드에서 한 번만 한다."""
    from torch_geometric.utils import softmax
    from gnn.geometry_interaction import combine_scores
    src, tgt = edge_index
    query = conv.W_q(x).view(-1, conv.heads, conv.head_dim)
    key_node = F.linear(x, conv.W_k.weight)
    value_node = F.linear(x, conv.W_v.weight)
    key_bias = conv.W_k.bias if conv.W_k.bias is not None else 0.
    value_bias = conv.W_v.bias if conv.W_v.bias is not None else 0.
    key = (key_node[src] - key_node[tgt] + key_bias).tanh().view(-1, conv.heads, conv.head_dim)
    content = (query[tgt] * key).sum(-1) / conv.scale
    logits = combine_scores(content, conv.bias_mlp(edge_attr), tgt, len(x), conv.geometry_mode)
    conv.attention = conv.dropout(softmax(logits, tgt, num_nodes=len(x)))
    value = (value_node[src] - value_node[tgt] + value_bias).tanh().view(-1, conv.heads, conv.head_dim)
    messages = (value * conv.attention[..., None]).flatten(1)
    return torch.zeros(len(x), conv.channels, dtype=x.dtype).index_add(0, tgt, messages)


class Runtime(T.Runtime):
    """현재 측정기와 같은 상태·검색 규칙에 위 구현 정리만 적용한다."""

    def __init__(self, spec: dict, work: bench.Workload, nodewise: bool):
        self.nodewise = nodewise
        super().__init__(spec, work)
        edges = temporal_edges(np.zeros(W, np.int64))
        src, tgt = edges
        index = np.broadcast_to(np.arange(W)[:, None], (W, 11)).copy()
        mask = np.zeros((W, 11), np.float32)
        index[tgt, src - tgt + 5], mask[tgt, src - tgt + 5] = src, 1.
        self.consts = types.SimpleNamespace(
            edges=edges, edge_index=torch.from_numpy(edges), edge_type=torch.zeros(edges.shape[1], dtype=torch.long),
            transport_indices=torch.from_numpy(index), transport_mask=torch.from_numpy(mask),
            slots=torch.tensor(SLOTS), dense=DENSE, freq=FREQ,
            fusion=pilot.cal.fusion_weights(self.conf)[pilot.cal.FUSIONS.index("aux_half")])
        self.capture_encoder()

    def capture_encoder(self) -> None:
        """상수 구간 경계로 GPU 통계를 CUDA graph에 담고 전용 풀 크기를 기록한다."""
        encoder = self.encoder
        edges = encoder._compute_bin_edges(encoder.alpha).clone()
        self.consts.bin_edges = edges
        encoder._compute_bin_edges = lambda alpha: edges
        static_in = torch.zeros(encoder.projector.n_elevation, encoder.projector.n_azimuth, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                encoder.encode_projected_image(static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        reserved, allocated = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            static_out = encoder.encode_projected_image(static_in)
        torch.cuda.synchronize()
        self.graph_pool = {"reserved": torch.cuda.memory_reserved() - reserved,
                           "allocated": torch.cuda.memory_allocated() - allocated}
        self.graph, self.consts.static_in, self.consts.static_out = graph, static_in, static_out

    def encode(self, points: np.ndarray) -> tuple:
        """원본 encode와 비트 단위로 같은 원시 키·복소 계수다."""
        from _pbev_branch_fusion import BP
        from encoding.phase_features import _phase_sketch, _pool_rows
        z, distance, azimuth, horizontal = polar(points)
        image = interp_rows(range_image(self.encoder.projector, z, distance, azimuth, horizontal))
        self.consts.static_in.copy_(torch.from_numpy(image))
        self.graph.replay()
        coeff = _phase_sketch(_pool_rows(np.maximum(bev_image(BP, z, azimuth, horizontal), 0.), 16), 8)
        return self.consts.static_out.float().cpu().numpy(), coeff

    def reset(self) -> None:
        """원본 reset과 같고 최근 버퍼만 21개다."""
        if self.recent.shape[0] != W:
            self.recent = np.empty((W, 288), np.float32)
            self.recent_coeff = np.empty((W, 16, 8), np.complex64)
        n = self.work.initial
        self.n = n
        self.poses[:n] = self.work.poses[:n]
        self.keys[:n] = self.raw_key(self.work.raw[:n])
        self.hkey[:n] = pilot.current.base.index_key(self.work.coeff[:n])
        self.coeff[:n] = T.Runtime.updated(self, self.work.raw[:n], self.work.coeff[:n], self.poses[:n])
        self.travel[:n] = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.poses[:n, :3, 3], axis=0), axis=1))]
        self.recent[:] = self.work.raw[n-W:n]
        self.recent_coeff[:] = self.work.coeff[n-W:n]

    def local(self, raw: np.ndarray, z: np.ndarray, poses: np.ndarray) -> np.ndarray:
        """21노드 창의 수정 복소 계수다. 간선과 이웃 색인은 고정이다."""
        c, net = self.consts, self.net
        relative = self.poses[0, :3, :3].astype(np.float64).T[None] @ poses[:, :3, :3].astype(np.float64)
        yaw = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
        phase = torch.from_numpy(np.exp(1j * yaw[:, None] * c.freq).astype(np.complex64))
        with torch.inference_mode():
            h = F.relu(net.input_norm(net.input_proj(torch.from_numpy(raw.astype(np.float32)))))
            first = h
            attr = net.edge_encoder(torch.from_numpy(pose_attributes(poses, c.edges)), c.edge_type)
            weights = []
            src, tgt = c.edge_index
            for layer, (conv, norm) in enumerate(zip(net.convs, net.batch_norms)):
                previous = h
                message = (conv_nodewise(conv, h, c.edge_index, attr) if self.nodewise
                           else conv(h, c.edge_index, edge_attr=attr))
                h = norm(message)
                if layer == 0:
                    h = F.relu(h)
                h = h + previous
                dense = torch.zeros(len(h), 11, 4, dtype=h.dtype)
                dense[tgt, src - tgt + 5] = conv.attention
                weights.append(dense)
            strengths = -torch.stack((net.gate(first).sigmoid(), net.gate(h).sigmoid()))
            net.last_strengths = strengths   # 원본 forward와 같은 내부 기록을 남긴다
            spectrum = torch.from_numpy(np.ascontiguousarray(z, dtype=np.complex64))
            return transport_vectorized(spectrum, c.transport_indices, c.transport_mask, phase,
                                        strengths, torch.stack(weights), c.slots).numpy()

    def append(self, raw: np.ndarray, coeff: np.ndarray, pose: np.ndarray) -> None:
        """한 도착을 추가하고 마지막11개 복소 상태를 21노드 창에서 다시 계산한다."""
        t = self.n
        self.poses[t] = pose
        self.travel[t] = self.travel[t-1] + np.linalg.norm(pose[:3, 3] - self.poses[t-1, :3, 3])
        self.keys[t] = self.raw_key(raw[None])[0]
        self.hkey[t] = pilot.current.base.index_key(coeff[None])[0]
        self.recent[:-1] = self.recent[1:]
        self.recent[-1] = raw
        self.recent_coeff[:-1] = self.recent_coeff[1:]
        self.recent_coeff[-1] = coeff
        self.coeff[t-10:t+1] = self.local(self.recent, self.recent_coeff, self.poses[t-W+1:t+1])[-11:]
        self.n += 1

    def query(self, q: int) -> tuple:
        """원본 query와 같은 승자·점수. 시간 판정은 쓰이는 arc_unique만 계산한다."""
        from _causal_pr_dump import prefix_topk
        from _xs_eval_lib import column_align_bank, minmax
        from retrieval.sequence_evidence import distance_indices, path_means, repeated_candidate_weights
        from retrieval.spectral_evidence import correlation_profiles
        v4, rows_cls = pilot.current, pilot.current.base.Rows
        self.transient = 0
        ids = np.unique(np.r_[prefix_topk(self.keys, q, q - 29, 100), prefix_topk(self.hkey, q, q - 29, 100)])
        a = v4.v3.spectral_scores(self.coeff, q, ids)
        c = v4.v3.column_scores(self.columns(np.union1d(ids, [q]), ("original",))["original"], q, ids)
        w = self.consts.fusion
        score = (w[0] * minmax(self.keys[ids] @ self.keys[q]) + w[1] * minmax(self.hkey[ids] @ self.hkey[q])
                 + w[2] * minmax(a) + w[3] * minmax(c))
        j = int(ids[np.argmax(score)])
        span = np.r_[np.arange(q - 5, q + 6), np.arange(j - 10, j + 11)]
        rows = np.unique(span[(span >= 0) & (span < self.n)])
        banks = []
        for values in (self.coeff[rows], self.white.transform(self.coeff[rows], self.matrix)):
            x = pilot.study.transform(values, pilot.study.POWERS["p050"])[:, :, :8]
            spectrum = np.zeros((len(x), 16, 31), np.complex64)
            spectrum[:, :, 1:9] = x
            norm = np.clip(np.linalg.norm(np.abs(x).reshape(len(x), -1), axis=1), 1e-8, None)
            column = column_align_bank(np.fft.irfft(spectrum, n=60, axis=-1).astype(np.float32))
            self.transient += x.nbytes + spectrum.nbytes + norm.nbytes + column[0].nbytes + column[1].nbytes
            banks.append((None, rows_cls(rows, spectrum), rows_cls(rows, norm),
                          (rows_cls(rows, column[0]), rows_cls(rows, column[1]))))
        travel = self.travel[:self.n]
        n = len(travel)
        qa, ja = np.array([q]), np.array([j])
        offsets = qa[:, None] + self.consts.dense[None]
        safe = np.clip(offsets, 0, n - 1)
        neighbors, valids = [], []
        for direction in (1, -1):
            jn = distance_indices(travel, ja[:, None], direction * (travel[safe] - travel[qa, None]))
            neighbors.append(jn)
            valids.append((offsets >= 0) & (offsets < n) & (jn >= 0) & (jn < n) & (jn != ja[:, None]))
        jn, valid = np.stack(neighbors), np.stack(valids)
        qn = np.stack([offsets, offsets])
        unique, inverse = np.unique((qn * n + jn)[valid], return_inverse=True)
        weights = repeated_candidate_weights(jn, valid)
        qs, js = np.r_[q, unique // n], np.r_[j, unique % n]
        ends = {}
        for channel in ("c_spec", "c_col"):
            mixed = []
            for index, bank in enumerate(banks):
                peaks = correlation_profiles(bank, qs, js, channel).max(1)
                center = peaks[:1]
                raw = center.astype(np.float64) if index == 0 else center
                values = np.zeros(qn.shape, dtype=peaks.dtype)
                values[valid] = peaks[1:][inverse]
                mixed.append((1 - BETA) * raw + BETA * path_means(center, values, valid, weights).max(0))
            ends[channel] = .75 * mixed[0] + .25 * mixed[1]
        value = float(pilot.cal.confidence(ends["c_spec"], ends["c_col"], "equal", self.conf["decision"])[0])
        f = self.affine
        return j, (value - f["mean"][0]) * f["coefficients"][0] / f["scale"][0] + f["intercept"], len(ids)

    def memory(self) -> dict:
        """원본과 같은 감사에 고정 상수·CUDA graph 전용 풀을 더한다."""
        arrays = {k: getattr(self, k) for k in ("keys", "hkey", "coeff", "poses", "travel", "recent", "recent_coeff", "matrix", "scale")}
        audit = audit_numeric_state(arrays, self.net, self.encoder, self.consts)
        assert audit["status"] == "passed", audit["unknown_capacity_paths"]
        allocated = torch.cuda.memory_allocated()
        pool_idle = max(self.graph_pool["reserved"] - self.graph_pool["allocated"], 0)
        gpu = allocated + pool_idle
        return dict(cpu=audit["cpu_unique_bytes"], gpu=gpu, total=audit["cpu_unique_bytes"] + gpu,
                    gpu_allocated=allocated, cuda_graph_pool=self.graph_pool, cuda_graph_pool_idle_added=pool_idle,
                    breakdown={k: v.nbytes for k, v in arrays.items()}, model_numeric_audit=audit,
                    logical_feature_payload_bytes_per_frame=2688,
                    recent_original_coefficients=W, full_original_coefficients_retained=False,
                    graph=f"최근{W}노드 고정 그래프; 마지막11개 복소 상태 갱신",
                    derived_banks="질의별 필요 행만 계산; 상주하지 않음")


def identity_checks(runtime: Runtime, work: bench.Workload) -> dict:
    """처음 12도착에서 인코딩·갱신·질의를 원본 구현과 같은 입력으로 대조한다."""
    original_encode = bench.NSD.encode
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
        reference = T.Runtime.updated(runtime, raw_seq[t-30:t+1], coeff_seq[t-30:t+1], runtime.poses[t-30:t+1])[-11:]
        coefficient_error = float(abs(reference - runtime.coeff[t-10:t+1]).max())
        assert coefficient_error < 4e-6, coefficient_error
        mine, theirs = runtime.query(t - 15), T.Runtime.query(runtime, t - 15)
        assert mine[0] == theirs[0], (t, mine, theirs)
        records.append(dict(arrival=t, encode_bitwise=True, coefficient_max_error_vs_31node=coefficient_error,
                            winner=mine[0], score_difference_vs_original_query=abs(mine[1] - theirs[1])))
    runtime.reset()
    return dict(status="passed", records=records,
                score_difference_max=max(r["score_difference_vs_original_query"] for r in records),
                coefficient_error_max=max(r["coefficient_max_error_vs_31node"] for r in records))


def measure(variant: str) -> None:
    """같은96도착을 세 번 실행하고 고정 검증 원장과 일치하는지 검사한다."""
    current.e.configure()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"nsd_n10000_{variant}.json"
    assert not target.exists(), target
    before = environment_check()
    sources = [Path(__file__), Path(T.__file__), Path(pilot.__file__), Path(current.__file__), T.MAIN / "protocol.json", T.LEDGER,
               ROOT / "src/gnn/contrast_spectral_difference.py", ROOT / "src/gnn/spectral_difference.py",
               ROOT / "src/gnn/learned_spectral_difference.py", ROOT / "src/encoding/spectral_encoder.py",
               ROOT / "src/encoding/range_image.py", ROOT / "src/encoding/bev_image.py",
               ROOT / "scripts/_pdv2_table3_model_memory.py", ROOT / "scripts/_pdv2_input_alignment_lib.py", PUBLISHED]
    protocol = dict(method="diff_s1/matching_only", variant=variant, warmup=12, samples=96, repeats=3, threads=1,
                    changes=CHANGES + ([NODEWISE_CHANGE] if variant == "nodewise" else []),
                    sources={str(p.relative_to(ROOT)): bench.sha(p) for p in sources},
                    checkpoint=bench.read(T.MAIN / "protocol.json")["cases"]["diff_s1"],
                    workload_protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                    scope="표3과 같음: 인코딩+11개복소상태갱신+두색인후보검색+재정렬+최종시간판정. 초기구축·I/O·공통KISS·CUDA graph 캡처 제외. 시험 선택 없음.",
                    publication="원고 표3 반영 여부는 저자가 정한다. 그 전까지 표3 원천을 바꾸지 않는다.")
    path = OUT / f"protocol_{variant}.json"
    if path.exists():
        assert bench.read(path) == protocol
    else:
        bench.write(path, protocol)
    spec = bench.read(bench.OUT / "protocol.json")
    work = bench.Workload(spec, 10000)
    tick = time.perf_counter()
    runtime = Runtime(spec, work, nodewise=variant == "nodewise")
    build_s = time.perf_counter() - tick
    print("BUILT", variant, build_s, runtime.graph_pool, flush=True)
    identity = identity_checks(runtime, work)
    print("IDENTITY", identity["score_difference_max"], identity["coefficient_error_max"], flush=True)
    checked = T.verify(runtime, work)
    print("VERIFIED", checked["status"], flush=True)
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
    ledger = dict(np.load(T.LEDGER))
    ids = np.searchsorted(ledger["query_idx"], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["query_idx"][ids], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["j_star"][ids], [r["winner"] for r in rows])
    error = float(abs(ledger["score"][ids] - [r["confidence"] for r in rows]).max())
    assert error < 8e-6 and len(rows) == 288, error
    checked["final_ledger"] = dict(queries=len(rows), winners_exact=True, score_max_error=error, sha256=bench.sha(T.LEDGER))
    published = bench.read(PUBLISHED)
    assert [(r["query"], r["winner"]) for r in rows] == [(r["query"], r["winner"]) for r in published["raw_samples"]]
    checked["published_measurement"] = dict(
        winners_equal=True,
        confidence_max_difference=float(max(abs(a["confidence"] - b["confidence"]) for a, b in zip(rows, published["raw_samples"]))))
    checked["identity_vs_original_implementation"] = identity
    result = dict(schema="pdv2-table3-measurement-v1", method="nsd", selected="diff/matching_only", variant=variant,
                  n_map=10000, status="passed", publish_ready=False, seed=1, sequence=bench.SEQUENCE,
                  samples=96, repeats=3, warmup=12, candidate_range=[10000, 10095], mean_candidates=10047.5,
                  method_protocol_sha256=bench.sha(path), protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                  timings_ms={k: bench.stats([r[f] for r in rows]) for k, f in
                              (("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                               ("query", "query_ms"), ("processing", "processing"))},
                  published_timings_ms=published["timings_ms"], published_resident_total=published["resident_bytes"]["total"],
                  resident_bytes=memory[-1], memory_repeats=memory, verification=checked,
                  initial_cached_state_build_s=build_s, raw_input_sha256=work.input_hash, raw_input_fixture_bytes=work.input_bytes,
                  raw_samples=rows, environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      cuda=torch.version.cuda, platform=platform.platform(), threads=1, before=before, after=after),
                  completed_unix=time.time())
    bench.write(target, result)
    print("COMPLETE", variant, {k: round(v["mean"], 4) for k, v in result["timings_ms"].items()},
          result["resident_bytes"]["total"], flush=True)


def baseline(tag: str) -> None:
    """현재 측정기를 그대로 새 폴더에 실행해 같은 장비 상태의 원본 값을 얻는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    folder = OUT / f"baseline_{tag}"
    folder.mkdir(exist_ok=True)
    (folder / "environment_before.json").write_text(__import__("json").dumps(environment_check(), ensure_ascii=False, indent=2))
    T.OUT = folder
    T.measure(10000)
    (folder / "environment_after.json").write_text(__import__("json").dumps(environment_check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--variant", choices=("window", "nodewise"), required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--tag", required=True)
    args = parser.parse_args()
    measure(args.variant) if args.action == "measure" else baseline(args.tag)
