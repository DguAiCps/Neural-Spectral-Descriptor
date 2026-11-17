#!/usr/bin/env python3
"""정제 그래프의 이웃 오프셋만 바꾼 NSD 변형(dilated 이웃, 2026-11-17).

보고 구성 x=log__n=ln__g=frozen__lr=1e-3__U=1536(u512)의 학습·검색 코드를 그대로 import해 쓰고,
정제 그래프(DiffAttnConv 두 층과 복소 차이 전달)의 시간 이웃 오프셋만 바꾼다. 시간 증거(±5 plan),
고정 설정(융합 (2,1,2,2), λ 0.25), 학습 표본(pool), 손실·학습 설정은 그대로다. 손실 척도는 같은 절차로
변형의 초기 출력(그 변형의 균등 감산)에 다시 맞춘다.

변형은 환경 변수 NSD_DIL로 고른다(import 때 패치가 걸린다).
  d1  : ±1..±5 (보고모델과 같음; 비트 단위 재현 확인용)
  d2  : ±2,±4,...,±10
  d3  : ±3,±6,...,±15
  n20 : ±1..±10 (이웃 20개)

바꾸는 곳(기존 파일은 수정하지 않고 이 프로세스 안에서만 바꾼다):
  _pdv2_input_alignment_lib.temporal_edges     간선 목록(make_graph가 부른다; edge_attr는 같은 함수로 새 쌍에서 계산)
  _pdv2_diffattn_complex_difference.attach_transport   이웃 칸 색인·마스크(칸 = 정렬한 오프셋, 가운데가 자기 칸)
  _pdv2_diffattn_complex_difference.transport_difference  칸 수 일반화(원래 11칸 하드코딩)
  ContrastSpectralDifference.forward           주의 dense 배열의 `src - tgt + 5` 11칸 가정을 칸 색인으로 교체
  ±1..±5에서는 칸 순서·간선 순서·합산 순서가 원래 코드와 같다.

docker exec -e NSD_DIL=d2 -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_dilated_neighbors_20261117.py <action> [...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _pdv2_scratch_search_20261116 as S  # noqa: E402
import _pdv2_input_alignment_lib as L  # noqa: E402
from gnn import contrast_spectral_difference as CSD  # noqa: E402
from gnn import spectral_transport as ST  # noqa: E402

P, prior, run = S.P, S.prior, S.run
ROOT = P.ROOT
CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
SEEDS = (4, 5, 6)
OUT = ROOT / "results/pdv2_dilated_20261117"
ORIG = dict(runs=S.RUNS, ledgers=S.LEDGERS, train_inputs=P.recal.OUT / "train_inputs",
            loss_fit=P.OUT / "step2/loss_fit.json", selection=P.OUT / "step1/selection.json", xstats=S.OUT / "xstats.pt")

VARIANTS = {
    "d1": list(range(1, 6)),
    "d2": list(range(2, 11, 2)),
    "d3": list(range(3, 16, 3)),
    "n20": list(range(1, 11)),
}
VAR = os.environ.get("NSD_DIL", "d1")
assert VAR in VARIANTS, VAR
POS_OFFS = VARIANTS[VAR]
OFFSETS = np.array(sorted([-o for o in POS_OFFS] + POS_OFFS), dtype=np.int64)   # 오름차순, 0 제외
R = int(OFFSETS.max())
SLOT_OFFS = sorted(OFFSETS.tolist() + [0])                                      # 칸 = 상대 위치(이웃 - 중심)
NSLOT = len(SLOT_OFFS)
SELF = SLOT_OFFS.index(0)
LOOKUP = np.full(2 * R + 1, -1, np.int64)
for k, o in enumerate(SLOT_OFFS):
    LOOKUP[o + R] = k
LAG = 2 * R + 5            # 정제 스펙트럼 i+2R, 시간 증거 +5
VD = OUT / VAR


# ----------------------------------------------------------------- 패치
def temporal_edges(sequences: np.ndarray) -> np.ndarray:
    """L.temporal_edges와 같은 순서 규칙(노드별, 오프셋 오름차순)으로 OFFSETS 이웃을 연결한다."""
    n = len(sequences)
    source = np.broadcast_to(np.arange(n)[:, None], (n, len(OFFSETS)))
    target = source + OFFSETS[None]
    valid = (target >= 0) & (target < n)
    safe = np.clip(target, 0, n - 1)
    valid &= sequences[source] == sequences[safe]
    return np.stack((source[valid], target[valid]))


def attach_transport(graph, spectrum: np.ndarray, yaw: np.ndarray, sequences: np.ndarray) -> dict:
    """ST.attach_transport의 칸 일반화. 칸 k는 상대 위치 SLOT_OFFS[k] (이웃 - 중심)."""
    n = graph.num_nodes
    spectrum = np.ascontiguousarray(spectrum, dtype=np.complex64)
    yaw = np.asarray(yaw, dtype=np.float64)
    sequences = np.asarray(sequences)
    assert spectrum.shape == (n, 16, 8) and yaw.shape == (n,)
    assert np.isfinite(spectrum).all() and np.isfinite(yaw).all()
    source, target = graph.edge_index.detach().cpu().numpy()
    rel = source - target
    assert np.all(np.abs(rel) <= R)
    slot = LOOKUP[rel + R]
    assert np.all(slot >= 0) and np.all(slot != SELF)
    assert not np.any(sequences[source] != sequences[target])
    assert bool((graph.edge_type == 0).all())
    index = np.broadcast_to(np.arange(n)[:, None], (n, NSLOT)).copy()
    mask = np.zeros((n, NSLOT), np.float32)
    index[target, slot], mask[target, slot] = source, 1.
    assert int(mask.sum()) == len(source)
    device = graph.x.device
    graph.transport_indices = torch.as_tensor(index, device=device)
    graph.transport_mask = torch.as_tensor(mask, device=device)
    graph.transport_phase = torch.as_tensor(np.exp(1j * yaw[:, None] * np.arange(1, 9)[None]).astype(np.complex64), device=device)
    graph.spectrum = torch.as_tensor(spectrum, device=device)
    return {"zero_spectrum_fraction": float(np.mean(np.linalg.norm(spectrum.reshape(n, -1), axis=1) == 0)),
            "isolated_fraction": float(np.mean(mask.sum(1) == 0))}


def transport_difference(spectrum, indices, mask, phase, strengths, attentions=None, transport: bool = True):
    """gnn.spectral_difference.transport_difference의 칸 수 일반화(자기 칸 SELF 제외, 칸 순서대로 합산)."""
    assert strengths.shape == (2, len(spectrum), 1)
    assert mask.shape[1] == NSLOT
    norm = spectrum.abs().flatten(1).norm(dim=1).clamp_min(1e-12)
    world = spectrum / norm[:, None, None]
    if transport:
        world = world * phase.conj()[:, None]
    if attentions is None:
        weights = mask / mask.sum(1, keepdim=True).clamp_min(1)
        weights = weights[None, :, :, None].expand(2, -1, -1, 4)
    else:
        weights = attentions
    for layer in range(2):
        update = torch.zeros_like(world)
        for slot in range(NSLOT):
            if slot == SELF:
                continue
            a = weights[layer, :, slot].repeat_interleave(2, dim=-1)
            a = a * mask[:, slot, None]
            delta = world[indices[:, slot]] - world
            update = update + a[:, None] * delta
        world = world + strengths[layer, :, :, None] * update
    return world * phase[:, None] if transport else world


_LOOKUP_T = {}


def slots_of(src, tgt):
    dev = src.device
    if dev not in _LOOKUP_T:
        _LOOKUP_T[dev] = torch.as_tensor(LOOKUP, device=dev)
    return _LOOKUP_T[dev][src - tgt + R]


def dense_attention(graph, values, n):
    src, tgt = graph.edge_index
    out = torch.zeros((n, NSLOT) + tuple(values.shape[1:]), dtype=values.dtype, device=values.device)
    out[tgt, slots_of(src, tgt)] = values
    return out


def csd_forward(self, graph) -> torch.Tensor:
    """ContrastSpectralDifference.forward와 같고 dense 칸만 일반화."""
    h = torch.nn.functional.relu(self.input_norm(self.input_proj(graph.x)))
    first = h
    attr = self.edge_encoder(graph.edge_attr, graph.edge_type) if self.edge_encoder is not None else None
    weights = []
    src, tgt = graph.edge_index
    slot = slots_of(src, tgt)
    for layer, (conv, norm) in enumerate(zip(self.convs, self.batch_norms)):
        previous = h
        h = norm(conv(h, graph.edge_index, edge_attr=attr))
        if layer == 0:
            h = torch.nn.functional.relu(h)
        h = h + previous
        if self.kind != "point_gate":
            dense = torch.zeros(len(h), NSLOT, 4, dtype=h.dtype, device=h.device)
            dense[tgt, slot] = conv.attention
            weights.append(dense)
    strengths = -torch.stack((self.gate(first).sigmoid(), self.gate(h).sigmoid()))
    self.last_strengths = strengths
    attention = torch.stack(weights) if weights else None
    return transport_difference(graph.spectrum, graph.transport_indices, graph.transport_mask,
                                graph.transport_phase, strengths, attention)


L.temporal_edges = temporal_edges
prior.attach_transport = attach_transport
prior.transport_difference = transport_difference
CSD.ContrastSpectralDifference.forward = csd_forward
# 결과 경로
S.OUT = VD
S.RUNS = VD / "runs"
S.LEDGERS = VD / "ledgers"
P.OUT = VD / "pilot"
P.recal = types.SimpleNamespace(OUT=VD)      # S.train이 P.recal.OUT/"train_inputs"를 읽는다


def write(path: Path, obj) -> None:
    P.write(path, obj)


def setup() -> None:
    """변형 폴더에 고정 설정(선택 결과)·xstats 사본을 둔다."""
    (VD / "pilot/step1").mkdir(parents=True, exist_ok=True)
    for src, dst in ((ORIG["selection"], VD / "pilot/step1/selection.json"), (ORIG["xstats"], VD / "xstats.pt")):
        if not dst.exists():
            shutil.copyfile(src, dst)
        assert run.HASH(src) == run.HASH(dst)


# ----------------------------------------------------------------- 학습 입력·손실 척도
def graphs() -> None:
    """학습8 그래프를 변형 오프셋으로 만든다(로컬; KITTI 보정 파일 필요). d1은 기존 입력과 비트 단위 비교."""
    setup()
    folder = VD / "train_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    rec = {}
    for s in S.TRAIN:
        name = s.replace("/", "_")
        path = folder / (name + ".pt")
        if not path.exists():
            graph, _, audit = prior.graph_inputs("train", s)
            torch.save(graph.cpu(), path)
            write(path.with_suffix(".json"), dict(audit=audit, graph_sha256=run.HASH(path), offsets=OFFSETS.tolist()))
        g = torch.load(path, map_location="cpu", weights_only=False)
        rec[s] = dict(nodes=int(g.num_nodes), edges=int(g.edge_index.shape[1]), slots=int(g.transport_mask.shape[1]),
                      interior_nodes=int((g.transport_mask.sum(1) == len(OFFSETS)).sum()))
        if VAR == "d1":
            ref = torch.load(ORIG["train_inputs"] / (name + ".pt"), map_location="cpu", weights_only=False)
            rec[s]["equal_to_existing"] = {k: bool(torch.equal(getattr(g, k), getattr(ref, k))) for k in
                                           ("x", "edge_index", "edge_attr", "edge_type", "transport_indices", "transport_mask",
                                            "transport_phase", "spectrum")}
        print("GRAPH", VAR, s, rec[s], flush=True)
    write(VD / "train_inputs_summary.json", dict(offsets=OFFSETS.tolist(), sequences=rec))


def calibrate() -> None:
    """P.calibrate와 같은 절차: 초기 출력(이 변형의 균등 감산)의 학습8 점수에 양의 로지스틱 척도를 맞춘다."""
    setup()
    _, lam = P.selected()
    xs, ys, ws = [], [], []
    for s in S.TRAIN:
        name = s.replace("/", "_")
        data = run.ARR(P.NEWPOOLS / "train" / name / "pool.npz")
        graph = torch.load(VD / "train_inputs" / (name + ".pt"), map_location="cuda", weights_only=False)
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            z = P.scratch_model()(graph)
            assert torch.equal(z, prior.filtered(graph, -0.5))
            v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(data["query_idx"])).astype(np.float64)
        keep = data["eligible"]
        y = data["correct"][keep].astype(np.float64)
        b = np.zeros(len(y))
        classes = np.unique(y)
        for c in classes:
            b[y == c] = 1.0 / (len(classes) * int((y == c).sum()))
        xs.append(v[keep]); ys.append(y); ws.append(b)
        del graph, z
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    fit = P.fit_score(x, y, w / w.sum())
    ref = run.READ(ORIG["loss_fit"])["fit"]
    rec = dict(fit=fit, lam=lam, report_model_fit=ref, offsets=OFFSETS.tolist(),
               equal_to_report_fit=all(fit[k] == ref[k] for k in ("mean", "scale", "coefficients", "intercept")), validation_used=False)
    write(VD / "pilot/step2/loss_fit.json", rec)
    print("FIT", VAR, {k: fit[k] for k in ("mean", "scale", "coefficients", "intercept", "loss")}, "equal_to_report", rec["equal_to_report_fit"], flush=True)


def train(seed: int) -> None:
    setup()
    S.train(CFG, seed)
    if VAR == "d1":
        mine = S.RUNS / CFG / f"s{seed}" / "u512.pth"
        ref = ORIG["runs"] / CFG / f"s{seed}" / "u512.pth"
        a = torch.load(mine, map_location="cpu", weights_only=False)["model_state_dict"]
        b = torch.load(ref, map_location="cpu", weights_only=False)["model_state_dict"]
        rec = dict(seed=seed, sha_equal=run.HASH(mine) == run.HASH(ref),
                   state_equal=all(torch.equal(a[k], b[k]) for k in b) and set(a) == set(b))
        write(VD / f"verify_train_s{seed}.json", rec)
        print("VERIFY_TRAIN", rec, flush=True)


# ----------------------------------------------------------------- 검색·접두 검사
def prefix(role: str, sequence: str, models: list, npoints: int) -> None:
    """S.pipe의 접두 검사를 새 지연 LAG로: 질의 t의 점수를 노드 t+LAG까지만 있는 그래프로 다시 계산."""
    name = sequence.replace("/", "_")
    target = VD / "prefix" / role / (name + ".json")
    if target.exists() or npoints <= 0:
        return
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, sequence)
    raw = S.pipeline.raw_key(cache["descriptors"])
    weights, lam = P.selected()
    out, small = [], {}
    for m in models:
        led = run.ARR(S.LEDGERS / role / name / (m + ".npz"))
        q, j, value = led["query_idx"], led["j_star"], led["score"]
        if m == "uniform":
            net = None
        else:
            c, sd, u = m.split("@")
            net = S.load(c, int(sd[1:]), int(u[1:]))
        with torch.no_grad():
            full = prior.graph_inputs(role, sequence)[0]
            z = prior.filtered(full, -0.5) if net is None else net(full)
            for k in range(npoints):
                at = (k + 1) * len(q) // (npoints + 1)
                t = int(q[at])
                end = t + 1 + LAG
                if end not in small:
                    small[end] = prior.graph_inputs(role, sequence, end)[0]
                short = prior.filtered(small[end], -0.5) if net is None else net(small[end])
                sb = run.base.variant_inputs(short.cpu().numpy(), 8)
                winner = run.cal.candidates(raw[:end], (bank[0][:end], *sb[1:]), weights[None], t)
                pp = {kk: torch.from_numpy(v).cuda() for kk, v in prior.base.prior.plan(travel[:end], np.array([t]), winner).items()}
                chk = P.score(short, pp, lam).cpu().numpy()
                # 한 칸 더 짧은 접두(t+LAG-1까지)에서는 점수가 달라지는지(지연이 꽉 찼는지) 참고로 본다
                shorter = prior.graph_inputs(role, sequence, end - 1)[0]
                s2 = prior.filtered(shorter, -0.5) if net is None else net(shorter)
                pp2 = {kk: torch.from_numpy(v).cuda() for kk, v in prior.base.prior.plan(travel[:end - 1], np.array([t]), np.array([j[at]])).items()}
                chk2 = P.score(s2, pp2, lam).cpu().numpy()
                out.append(dict(model=m, query=t, end=end, winner_equal=int(winner[0]) == int(j[at]),
                                score_error=float(abs(chk - value[at]).max()),
                                output_max_error=float((short[: t + 1] - z[: t + 1]).abs().max()),
                                one_shorter_score_change=float(abs(chk2 - value[at]).max())))
                del shorter, s2
        del net, z, full
    write(target, dict(lag=LAG, offsets=OFFSETS.tolist(), checks=out))
    print("PREFIX", role, sequence, [(c["winner_equal"], round(c["score_error"], 8)) for c in out], flush=True)


def pipe(role: str, sequence: str, models: list, nprefix: int, prefix_models: list) -> None:
    setup()
    S.pipe(role, sequence, models, 0)       # 원래 접두 검사(q+15)는 끄고 아래에서 새 지연으로 한다
    if nprefix:
        prefix(role, sequence, prefix_models, nprefix)


def fanout(commands: list, workers: int, logdir: Path) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    running, queue, failed = {}, list(commands), []
    while queue or running:
        while queue and len(running) < workers:
            label, cmd = queue.pop(0)
            env = dict(os.environ, NSD_DIL=VAR, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            f = (logdir / (label + ".log")).open("a")
            running[subprocess.Popen([sys.executable, __file__, *cmd], stdout=f, stderr=subprocess.STDOUT, env=env)] = (label, f)
        time.sleep(1)
        for proc in list(running):
            if proc.poll() is not None:
                label, f = running.pop(proc)
                f.close()
                if proc.returncode:
                    failed.append(label)
                print("DONE" if not proc.returncode else "FAIL", label, flush=True)
    assert not failed, failed


def compare_d1(role: str, seqs: list, models: list) -> None:
    """d1 원장이 보고모델 원장과 비트 단위로 같은지."""
    rec = []
    for s in seqs:
        name = s.replace("/", "_")
        for m in models:
            a, b = run.ARR(S.LEDGERS / role / name / (m + ".npz")), run.ARR(ORIG["ledgers"] / role / name / (m + ".npz"))
            rec.append(dict(sequence=s, model=m, keys_equal=set(a) == set(b), bitwise=all(np.array_equal(a[k], b[k]) for k in b)))
    write(VD / f"verify_ledgers_{role}.json", dict(checks=rec, all_bitwise=all(r["bitwise"] for r in rec)))
    print("VERIFY_LEDGERS", role, all(r["bitwise"] for r in rec), flush=True)


# ----------------------------------------------------------------- 주의 가중치
def attention() -> None:
    """시험46 내부 노드(이웃 전부 있음)의 칸별 평균 α(헤드 평균), 층별·시드별. _attention_analysis_20261117 방식."""
    setup()
    res = {}
    nets = {s: S.load(CFG, s, 512) for s in SEEDS}
    sums = {s: np.zeros((2, NSLOT)) for s in SEEDS}
    count = 0
    checks = []
    for seq in S.TESTSEQ:
        graph, _, _ = prior.graph_inputs("test", seq)
        n = len(graph.x)
        mask = graph.transport_mask
        interior = (mask.sum(1) == len(OFFSETS))
        count += int(interior.sum())
        with torch.no_grad():
            for s in SEEDS:
                nets[s](graph)
                a = torch.stack([dense_attention(graph, c.attention, n) for c in nets[s].convs])   # (2,N,S,4)
                checks.append(bool(((a.sum(-2) - 1).abs() < 1e-5).all()) and bool((a[:, :, SELF] == 0).all()) and bool((a[:, mask == 0] == 0).all()))
                sums[s] += a[:, interior].mean(-1).double().sum(1).cpu().numpy()
        del graph
    per_seed = {s: (v / count).tolist() for s, v in sums.items()}
    mean = np.mean([np.asarray(v) for v in per_seed.values()], 0)
    keep = [k for k in range(NSLOT) if k != SELF]
    offs = [SLOT_OFFS[k] for k in keep]
    far = [k for k in keep if abs(SLOT_OFFS[k]) == R]
    near = [k for k in keep if abs(SLOT_OFFS[k]) == min(POS_OFFS)]
    res = dict(variant=VAR, offsets=offs, uniform=1 / len(offs), interior_nodes=count, checks_all_ok=all(checks),
               layer_mean={f"L{l + 1}": dict(zip(map(str, offs), mean[l, keep].tolist())) for l in range(2)},
               per_seed={str(s): {f"L{l + 1}": dict(zip(map(str, offs), np.asarray(v)[l, keep].tolist())) for l in range(2)} for s, v in per_seed.items()},
               farthest_pair_sum={f"L{l + 1}": dict(mean=float(mean[l, far].sum()), seeds=[float(np.asarray(v)[l, far].sum()) for v in per_seed.values()]) for l in range(2)},
               nearest_pair_sum={f"L{l + 1}": dict(mean=float(mean[l, near].sum()), seeds=[float(np.asarray(v)[l, near].sum()) for v in per_seed.values()]) for l in range(2)},
               past_sum={f"L{l + 1}": float(mean[l, [k for k in keep if SLOT_OFFS[k] < 0]].sum()) for l in range(2)})
    write(VD / "attention.json", res)
    for l in range(2):
        print(f"L{l + 1}", " ".join(f"{o:+d}:{x:.3f}" for o, x in zip(offs, mean[l, keep])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--seed", type=int)
    p.add_argument("--role", default="test")
    p.add_argument("--sequence")
    p.add_argument("--sequences")
    p.add_argument("--models")
    p.add_argument("--prefix-models", default="")
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stamp", default="x")
    a = p.parse_args()
    P.e.configure()
    torch.set_num_threads(4)
    print("VARIANT", VAR, "offsets", OFFSETS.tolist(), "slots", NSLOT, "self", SELF, "lag", LAG, flush=True)
    if a.action == "graphs":
        graphs()
    elif a.action == "calibrate":
        calibrate()
    elif a.action == "train":
        train(a.seed)
    elif a.action == "pipe":
        pipe(a.role, a.sequence, a.models.split(","), a.prefix, [m for m in a.prefix_models.split(",") if m])
    elif a.action == "campaign":
        tick = time.time()
        seqs = a.sequences.split(",") if a.sequences else (S.VAL if a.role == "val" else S.TESTSEQ)
        fanout([(f"{a.role}_{s.replace('/', '_')}", ["pipe", "--role", a.role, "--sequence", s, "--models", a.models,
                                                     "--prefix", str(a.prefix), "--prefix-models", a.prefix_models]) for s in seqs],
               a.workers, VD / "logs/pipe")
        write(VD / "logs" / f"campaign_{a.role}_{a.stamp}.done.json", dict(models=a.models.split(","), sequences=seqs, seconds=time.time() - tick))
    elif a.action == "compare_d1":
        assert VAR == "d1"
        compare_d1(a.role, a.sequences.split(","), a.models.split(","))
    elif a.action == "agg":
        S.agg()
    elif a.action == "attention":
        attention()
    else:
        raise SystemExit(a.action)
