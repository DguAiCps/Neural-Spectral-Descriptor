#!/usr/bin/env python3
"""보고 모델(처음부터 학습, x=log__n=ln__g=frozen__lr=1e-3, 시드 4, 512갱신)의 학습 시간과
학습 전 1회성 전처리 시간을 잰다. 기존 결과·캐시는 읽기만 하고 새 결과는 OUT에만 쓴다.

docker exec -w <repo> ring_baseline env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python3 scripts/_train_timing_20261117.py <action> [...]

action
  train --rep R        : 512갱신 단독 학습 1회(프로세스 1개). 학습 루프를 기존 train()과 같게 따라가고
                         체크포인트를 기존 u512.pth와 비교한다.
  encode --sequence S  : (a) 원시 점군 → 키프레임 선택·288D 원 특징(전 스캔 읽기), 키프레임 BEV 복소 계수,
                         학습 그래프(시간 간선·자세 정렬 전달 색인). 기존 캐시와 비교한다.
  stats                : (b) 재계량 ω, 상관 보정 Γ_k(스펙트럼 2차 모멘트), log1p 입력 통계, 손실 척도 aS+b.
  pools                : (c) 학습8 표본(고정 원키·보조 색인 후보 검색·고정 융합 승자·라벨·시간 대응 plan).
"""
from __future__ import annotations

import time

T_PROCESS = time.time()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _pdv2_scratch_search_20261116 as S  # noqa: E402

T_IMPORT = time.time()
P, run, prior = S.P, S.run, S.prior
ROOT = S.ROOT
OUT = ROOT / "results/pdv2_train_timing_20261117"
NAME = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"  # 보고 구성(상수 학습률이라 512갱신 저장본 = 512갱신만 학습)
SEED = 4
UPDATES = 512
REF = S.RUNS / NAME / f"s{SEED}" / "u512.pth"


def sync() -> None:
    torch.cuda.synchronize()


def device() -> dict:
    return dict(name=torch.cuda.get_device_name(0), visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                order=os.environ.get("CUDA_DEVICE_ORDER"), torch=torch.__version__, cuda=torch.version.cuda,
                threads=torch.get_num_threads(), host=os.uname().nodename)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_avg() -> list:
    return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]


def rss_peak_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


# ----------------------------------------------------------------- 학습
def train(rep: int) -> None:
    """S.train(NAME, 4)의 학습 부분을 512갱신까지 같은 순서로 따라간다(검증9 고정 승자 평가는 뺀다)."""
    cfg = S.parse(NAME)
    t = {}
    tick = time.time()
    _, lam = P.selected()
    fit = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    stats = S.xstats()  # 기존 xstats.pt를 읽기만 한다
    t["setup_s"] = time.time() - tick
    # 학습 입력 적재(그래프 8개를 GPU로, 표본 8개와 plan을 GPU로)
    sync()
    tick = time.time()
    graphs = {s: torch.load(P.recal.OUT / "train_inputs" / (s.replace("/", "_") + ".pt"), map_location="cuda", weights_only=False) for s in S.TRAIN}
    pools = {s: run.ARR(P.NEWPOOLS / "train" / s.replace("/", "_") / "pool.npz") for s in S.TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pools.items()}
    sync()
    t["input_load_s"] = time.time() - tick
    # 모델 생성 ~ 마지막 갱신. 모델 초기화는 scratch_model 안의 manual_seed로 정해지므로 적재 순서를 바꿔도 같다.
    sync()
    t_model = time.time()
    net = S.build(cfg, SEED, stats)
    assert not [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    for n_, p in net.named_parameters():
        p.requires_grad_(S.trainable(cfg, n_))
    sync()
    t["model_build_s"] = time.time() - t_model
    tick = time.time()
    with torch.no_grad():
        assert torch.equal(net(graphs[S.TRAIN[0]]), prior.filtered(graphs[S.TRAIN[0]], -0.5))
    sync()
    t["init_check_s"] = time.time() - tick
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    batch, updates, losses_all = int(cfg["B"]), 0, []
    sync()
    t_loop = time.time()
    for epoch in range(1, UPDATES // 64 + 1):
        for ordinal, s in enumerate(S.TRAIN):
            data = pools[s]
            sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng((SEED, epoch, ordinal))
            for _ in range(8):
                n = batch // len(sets)
                chosen = rng.permutation(np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets]))
                ix = torch.from_numpy(chosen).cuda()
                for group in opt.param_groups:
                    group["lr"] = S.lr_at(cfg, updates)
                opt.zero_grad(set_to_none=True)
                z = net(graphs[s])
                value = P.score(z, {k: v[ix] for k, v in plans[s].items()}, lam)
                logit = (value - fit["mean"][0]) * fit["coefficients"][0] / fit["scale"][0] + fit["intercept"]
                loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(data["correct"][chosen]).cuda().float())
                assert torch.isfinite(loss)
                loss.backward()
                g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                assert torch.isfinite(g)
                opt.step()
                updates += 1
                losses_all.append(float(loss.detach()))
    sync()
    t_end = time.time()
    t["loop_s"] = t_end - t_loop
    t["model_to_last_update_s"] = t_end - t_model
    assert updates == UPDATES
    # 저장과 기존 체크포인트 비교(측정 구간 밖)
    folder = OUT / "train" / f"rep{rep}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "u512.pth"
    torch.save(dict(model_state_dict=net.state_dict(), updates=updates, cfg=cfg, seed=SEED), path)
    ref = torch.load(REF, map_location="cuda", weights_only=False)["model_state_dict"]
    mine = net.state_dict()
    assert set(ref) == set(mine)
    diff = {k: float((mine[k].double() - ref[k].double()).abs().max()) if mine[k].numel() else 0.0 for k in ref}
    rec = dict(
        device=device(), rep=rep, config=NAME, seed=SEED, updates=updates, timing=t,
        process_start_to_import_s=T_IMPORT - T_PROCESS, process_python_wall_s=time.time() - T_PROCESS,
        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), reference=str(REF.relative_to(ROOT)),
                        reference_sha256=sha(REF), file_identical=sha(path) == sha(REF),
                        tensors_bitwise_equal=all(torch.equal(mine[k], ref[k]) for k in ref),
                        max_abs_diff=max(diff.values()), max_abs_diff_tensor=max(diff, key=diff.get)),
        final_mean_loss_last8=float(np.mean(losses_all[-8:])), peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1024 ** 3,
        peak_rss_gb=rss_peak_gb(), loadavg_end=load_avg())
    P.write(folder / "timing.json", rec)
    print(json.dumps({k: rec[k] for k in ("device", "timing", "checkpoint")}, indent=1), flush=True)


# ----------------------------------------------------------------- (a) 인코딩
def encode(sequence: str) -> None:
    import train_multi_dataset as tmd
    from keyframe.selector import KeyframeSelector
    from _pd_build_test_caches import _make_desc_encoder, _make_loader
    from _pbev_branch_fusion import BP
    from encoding.phase_features import _phase_sketch, _pool_rows
    from gnn.spectral_transport import attach_transport, lidar_yaw
    from _pdv2_matching_mine import cache_path, spec_path

    name = sequence.replace("/", "_")
    sensor, seq = sequence.split("/")
    folder = OUT / "preproc" / "encode"
    folder.mkdir(parents=True, exist_ok=True)
    rec = dict(device=device(), sequence=sequence, loadavg_start=load_avg())
    cfg = tmd.apply_encoder_preset(tmd.load_config(str(ROOT / "configs/training_place_disjoint_v2.yaml")), "no_interdiff")
    old_cache = cache_path(sequence)
    assert old_cache.name.startswith(f"cache_{tmd.compute_cache_key(cfg, sensor)}_"), old_cache
    # a1: 전 스캔 읽기 → 키프레임 선택 → 288D 원 특징(train_multi_dataset.process_dataset 그대로)
    sync()
    tick = time.time()
    encoder = _make_desc_encoder(cfg, sensor, "cuda")
    selector = KeyframeSelector(distance_threshold=cfg["keyframe"]["distance_threshold"], rotation_threshold=cfg["keyframe"]["rotation_threshold"],
                                overlap_threshold=cfg["keyframe"]["overlap_threshold"], temporal_threshold=cfg["keyframe"]["temporal_threshold"])
    loader = _make_loader(sensor, seq)
    n_scans = len(loader)
    keyframes = tmd.process_dataset(loader, encoder, selector, "cuda", dataset_name=name)
    sync()
    rec["a1_key288_s"] = time.time() - tick
    rec["a1_profile_s"] = {k: v for k, v in tmd.profiler.timings.items() if k.endswith(name)}
    rec["scans"], rec["keyframes"] = n_scans, len(keyframes)
    target = folder / f"cache_{name}.npz"
    tmd.save_keyframes_cache(target, keyframes)
    del loader, keyframes
    new = run.ARR(target)
    with np.load(old_cache) as old:
        rec["a1_check"] = dict(scan_ids_equal=bool(np.array_equal(new["scan_ids"], old["scan_ids"])),
                               poses_equal=bool(np.array_equal(new["poses"], old["poses"])),
                               descriptors_bitwise_equal=bool(np.array_equal(new["descriptors"], old["descriptors"])),
                               descriptors_max_abs_diff=float(np.abs(new["descriptors"].astype(np.float64) - old["descriptors"]).max())
                               if new["descriptors"].shape == old["descriptors"].shape else None)
    # a2: 키프레임만 다시 읽기 → 자른 BEV → 16×12 복소 계수(_pdv2_matching_mine.spectra의 BEV 부분. 현재 경로가
    #     쓰지 않는 높이 극좌표 격자는 뺀다)
    ids = new["scan_ids"]
    loader = _make_loader(sensor, seq)
    re = np.empty((len(ids), 16, 12), np.float32)
    im = np.empty_like(re)
    t_load = 0.0
    tick = time.time()
    for row, sid in enumerate(ids):
        t0 = time.perf_counter()
        points = loader[int(sid)]["points"]
        t_load += time.perf_counter() - t0
        bev, _ = BP.project(points, keep_intensity=False)
        sketch = _phase_sketch(_pool_rows(np.maximum(bev, 0.), 16), 12)
        re[row], im[row] = sketch.real, sketch.imag
    rec["a2_bev_s"] = time.time() - tick
    rec["a2_bev_load_s"] = t_load
    del loader
    np.savez_compressed(folder / f"spec_{name}.npz", scan_ids=ids, sk_re=re, sk_im=im)
    with np.load(spec_path(sequence)) as old:
        rec["a2_check"] = dict(sk_re_bitwise_equal=bool(np.array_equal(re, old["sk_re"])), sk_im_bitwise_equal=bool(np.array_equal(im, old["sk_im"])),
                               max_abs_diff=float(max(np.abs(re - old["sk_re"]).max(), np.abs(im - old["sk_im"]).max())))
    # a3: 학습 그래프(_pdv2_diffattn_complex_difference.graph_inputs와 같은 계산, 새 배열 사용)
    sync()
    tick = time.time()
    x, poses = new["descriptors"], new["poses"]
    z = (re[:, :, :8].astype(np.float32) + 1j * im[:, :, :8].astype(np.float32))
    extrinsic = None
    if sensor == "kitti":
        line = [l for l in (Path("/workspace/data/kitti/dataset/sequences") / seq / "calib.txt").read_text().splitlines() if l.startswith("Tr:")]
        extrinsic = np.eye(4)
        extrinsic[:3] = np.fromstring(line[0].split(":", 1)[1], sep=" ").reshape(3, 4)
    yaw = lidar_yaw(poses, extrinsic)
    graph = run.make_graph(x, poses, np.zeros(len(x), np.int64)).cuda()
    attach_transport(graph, z, yaw, np.zeros(len(x), np.int64))
    sync()
    rec["a3_graph_s"] = time.time() - tick
    old = torch.load(P.recal.OUT / "train_inputs" / (name + ".pt"), map_location="cuda", weights_only=False)
    keys = [k for k in (old.keys() if callable(old.keys) else old.keys) if torch.is_tensor(old[k])]
    rec["a3_check"] = {k: bool(torch.equal(graph[k], old[k])) for k in keys}
    rec["a_total_s"] = rec["a1_key288_s"] + rec["a2_bev_s"] + rec["a3_graph_s"]
    rec["peak_rss_gb"], rec["loadavg_end"] = rss_peak_gb(), load_avg()
    P.write(folder / f"{name}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "device"}, indent=1), flush=True)


# ----------------------------------------------------------------- (b) 고정 통계
def stats() -> None:
    from scipy.spatial import cKDTree
    from _pdv2_matching_mine import cache_path, spec_path
    from _pdv2_spectral_whitening import coefficients, whitening_matrix
    from _pdv2_contrast_calibration import fit_score

    rec = dict(device=device())
    # b1: 재계량 ω (scripts/_pdv2_spectral_keys.py:158-187과 같은 계산)
    tick = time.time()
    arrays, pairs = {}, {}
    for s in sorted(S.TRAIN, key=lambda s: str(cache_path(s))):
        with np.load(cache_path(s)) as c:
            arrays[s] = c["descriptors"].astype(np.float64)
            xyz = c["poses"][:, :3, 3].astype(np.float64)
        tree, selected = cKDTree(xyz), []
        for q in range(30, len(xyz)):
            ids = np.asarray(sorted(tree.query_ball_point(xyz[q], 5.0)), np.int64)
            ids = ids[ids < q - 30]
            if not len(ids):
                continue
            d = np.linalg.norm(xyz[ids] - xyz[q], axis=1)
            ids = ids[d < 5.0]
            if len(ids):
                selected.append((q, int(ids[np.argmin(np.linalg.norm(xyz[ids] - xyz[q], axis=1))])))
        pairs[s] = np.asarray(selected, np.int64).reshape(-1, 2)
    deltas = np.concatenate([arrays[s][p[:, 0]] - arrays[s][p[:, 1]] for s, p in pairs.items()])
    variance = 0.5 * np.mean(deltas ** 2, axis=0)
    omega = 1.0 / np.sqrt(variance + 1e-6 * variance.mean())
    omega /= omega.mean()
    rec["b1_omega_s"] = time.time() - tick
    ref = np.load(ROOT / "results/pdv2_spectral_keys_20261025/remetric_octave288.npy")
    rec["b1_check"] = dict(pairs=len(deltas), bitwise_equal=bool(np.array_equal(omega, ref)), max_abs_diff=float(np.abs(omega - ref).max()))
    np.save(OUT / "preproc" / "omega.npy", omega)
    # b2: Γ_k (주파수별 반경 행 2차 모멘트 → 수축 역제곱근, scripts/_pdv2_spectral_whitening.py:83-101)
    tick = time.time()
    moments = []
    for s in S.TRAIN:
        x = coefficients(spec_path(s)).astype(np.complex128)
        x /= np.maximum(np.sqrt(np.sum(abs(x) ** 2, axis=(1, 2))), 1e-12)[:, None, None]
        moments.append(np.einsum("nik,njk->kij", x, x.conj(), optimize=True) / len(x))
    moment = np.mean(moments, axis=0)
    full = whitening_matrix(moment, True)
    rec["b2_gamma_s"] = time.time() - tick
    ref = run.ARR(ROOT / "results/pdv2_spectral_whitening_20261026/statistics.npz")
    rec["b2_check"] = dict(moment_bitwise_equal=bool(np.array_equal(moment, ref["moment"])), full_bitwise_equal=bool(np.array_equal(full, ref["full"])),
                           full_max_abs_diff=float(np.abs(full - ref["full"]).max()))
    # b3: log1p 입력의 좌표별 평균·표준편차(S.xstats와 같은 계산; x=log 구성은 버퍼로만 저장되고 log 변환만 쓴다)
    tick = time.time()
    acc = {k: [] for k in ("lin_m", "lin_q", "log_m", "log_q")}
    for s in S.TRAIN:
        g = torch.load(P.recal.OUT / "train_inputs" / (s.replace("/", "_") + ".pt"), map_location="cpu", weights_only=False)
        x = g.x.double()
        y = torch.log1p(x)
        acc["lin_m"].append(x.mean(0)); acc["lin_q"].append((x * x).mean(0))
        acc["log_m"].append(y.mean(0)); acc["log_q"].append((y * y).mean(0))
    xs = {}
    for k in ("lin", "log"):
        m = torch.stack(acc[k + "_m"]).mean(0)
        q = torch.stack(acc[k + "_q"]).mean(0)
        xs[k] = dict(mean=m.float(), std=(q - m * m).clamp_min(1e-12).sqrt().float())
    rec["b3_xstats_s"] = time.time() - tick
    ref = torch.load(S.OUT / "xstats.pt", map_location="cpu", weights_only=False)
    rec["b3_check"] = {f"{k}_{kk}": bool(torch.equal(xs[k][kk], ref[k][kk])) for k in xs for kk in ("mean", "std")}
    # b4: 손실 척도 aS+b (P.calibrate와 같은 계산: 균등 감산 초기 출력의 학습8 점수에 양의 로지스틱 적합)
    sync()
    tick = time.time()
    _, lam = P.selected()
    xv, yv, wv = [], [], []
    for s in S.TRAIN:
        name = s.replace("/", "_")
        data = run.ARR(P.NEWPOOLS / "train" / name / "pool.npz")
        graph = torch.load(P.recal.OUT / "train_inputs" / (name + ".pt"), map_location="cuda", weights_only=False)
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            z = P.scratch_model()(graph)
            v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(data["query_idx"])).astype(np.float64)
        keep = data["eligible"]
        y = data["correct"][keep].astype(np.float64)
        b = np.zeros(len(y))
        classes = np.unique(y)
        for c in classes:
            b[y == c] = 1.0 / (len(classes) * int((y == c).sum()))
        xv.append(v[keep]); yv.append(y); wv.append(b)
    x, y, w = np.concatenate(xv), np.concatenate(yv), np.concatenate(wv)
    fit = fit_score(x, y, w / w.sum())
    sync()
    rec["b4_loss_fit_s"] = time.time() - tick
    ref = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    rec["b4_check"] = dict(equal=json.dumps(fit, sort_keys=True) == json.dumps(ref, sort_keys=True), fit=fit, reference=ref)
    rec["b_total_s"] = sum(rec[k] for k in ("b1_omega_s", "b2_gamma_s", "b3_xstats_s", "b4_loss_fit_s"))
    rec["peak_rss_gb"], rec["loadavg_end"] = rss_peak_gb(), load_avg()
    P.write(OUT / "preproc" / "stats.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "device"}, indent=1, default=str), flush=True)


# ----------------------------------------------------------------- (c) 학습 표본
def pools() -> None:
    rec = dict(device=device(), sequences={})
    w, _ = P.selected()
    folder = OUT / "preproc" / "pools"
    for s in S.TRAIN:
        target = folder / "train" / s.replace("/", "_") / "pool.npz"
        assert not target.exists(), target  # build_pool은 있으면 읽기만 하므로 새로 만들어야 잰다
        sync()
        tick = time.time()
        d = P.build_pool("train", s, w, folder)
        sync()
        sec = time.time() - tick
        ref = run.ARR(P.NEWPOOLS / "train" / s.replace("/", "_") / "pool.npz")
        rec["sequences"][s] = dict(seconds=sec, rows=len(d["query_idx"]), equal={k: bool(np.array_equal(d[k], ref[k])) for k in ref})
        print("POOL", s, round(sec, 1), all(rec["sequences"][s]["equal"].values()), flush=True)
    rec["c_total_s"] = sum(v["seconds"] for v in rec["sequences"].values())
    rec["peak_rss_gb"], rec["loadavg_end"] = rss_peak_gb(), load_avg()
    P.write(folder / "pools.json", rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("train", "encode", "stats", "pools"))
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--sequence")
    a = ap.parse_args()
    P.e.configure()
    torch.set_num_threads(4)  # 탐색 스크립트 __main__과 같은 설정
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preproc").mkdir(exist_ok=True)
    {"train": lambda: train(a.rep), "encode": lambda: encode(a.sequence), "stats": stats, "pools": pools}[a.action]()
