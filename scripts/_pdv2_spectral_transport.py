#!/usr/bin/env python3
"""시간 간선의 복소 전달을 같은 학습·최종 판정에서 비교한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -u scripts/_pdv2_spectral_transport_run.py
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_derivatives as base
from _pdv2_matching_mine import cache_path, spec_path
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.spectral_transport import SpectralTransportGNN, attach_transport, lidar_yaw

OUT = REPO / "results/pdv2_spectral_transport_20261020"
CONFIG = REPO / "configs/pdv2_spectral_transport.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["arms"])
COMPLEX_ARMS = ("spectral_local", "untransported", "transported")
EPOCHS, METHOD = tuple(CFG["checkpoint_epochs"]), CFG["final_score"]
REFERENCE_FACTORY = base.model
REFERENCE_SUBGRAPH = base.temporal_subgraph
VALID_SPECTRUM = None
read = base.read


def protocol(inputs: bool = False) -> dict:
    """새 실험의 학습 전에 고정한 소스와 입력을 확인한다."""
    spec = read(OUT / "protocol.json")
    for field in ("runtime", "inputs") if inputs else ("runtime",):
        for p, h in spec[field].items():
            assert sha256(REPO / p) == h, p
    return spec


def model(arm: str):
    """같은 입출력 초기값으로 표준·투영·복소 전달 모델을 만든다."""
    net = REFERENCE_FACTORY("point" if arm == "point" else "standard")
    if arm in COMPLEX_ARMS:
        net.gnn = SpectralTransportGNN(net.gnn, arm)
    return net


def read_spectrum(path: Path) -> tuple:
    """기존 NSD가 쓰는 조감도 16줄·주파수 1~8을 읽는다."""
    with np.load(path) as z:
        spectrum = (z["sk_re"][:, :, :8] + 1j*z["sk_im"][:, :, :8]).astype(np.complex64)
        return z["scan_ids"], spectrum


def training_phase(state: dict, sequences: list) -> tuple:
    """학습 스캔의 복소 특징과 라이다 축의 회전각을 순서대로 조립한다."""
    spectra, yaws, sources, stats = [], [], set(), {}
    for seqid, name in enumerate(sequences):
        cp, sp = cache_path(name), spec_path(name)
        metadata = read(sp.with_suffix(".json"))
        assert sha256(cp) == metadata["cache_sha256"] and sha256(sp) == metadata["sha256"]
        sources.update((cp, sp, sp.with_suffix(".json")))
        with np.load(cp) as cache:
            ids, z = read_spectrum(sp)
            np.testing.assert_array_equal(ids, cache["scan_ids"])
            indices = np.flatnonzero(state["sequences"] == seqid)
            np.testing.assert_array_equal(state["descriptors"][indices], cache["descriptors"])
            np.testing.assert_array_equal(state["poses"][indices], cache["poses"])
            poses = cache["poses"].copy()
        extrinsic = None
        if name.startswith("kitti/"):
            source = Path("/workspace/data/kitti/dataset/sequences") / name.split("/")[1] / "calib.txt"
            text = source.read_text()
            lines = [line for line in text.splitlines() if line.startswith("Tr:")]
            assert len(lines) == 1
            extrinsic = np.eye(4)
            extrinsic[:3] = np.fromstring(lines[0].split(":", 1)[1], sep=" ").reshape(3, 4)
            np.testing.assert_allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3],
                                       np.eye(3), atol=1e-6)
            target = OUT / "calibration" / (name.replace("/", "_") + ".txt")
            target.parent.mkdir(exist_ok=True)
            target.write_text(text)
            sources.add(target)
        yaw = lidar_yaw(poses, extrinsic)
        spectra.append(z)
        yaws.append(yaw)
        # 학습 자료만의 회전 보정 진단. 모델·설정 선택에는 사용하지 않는다.
        norm = np.maximum(np.linalg.norm(z.reshape(len(z), -1), axis=1), 1e-12)
        unit = z / norm[:, None, None]
        q = np.arange(1, len(z))
        delta = yaw[q] - yaw[q-1]
        aligned = unit[q-1] * np.exp(1j*delta[:, None, None]*np.arange(1, 9))
        wrong = unit[q-1] * np.exp(-1j*delta[:, None, None]*np.arange(1, 9))
        error = lambda a: np.sum(np.abs(a-unit[q])**2, axis=(1, 2))
        turn = np.abs(np.angle(np.exp(1j*delta))) >= np.deg2rad(5)
        stats[name] = {"nodes": len(z), "camera_to_lidar_applied": extrinsic is not None,
                       "zero_spectrum_fraction": float((norm <= 1e-12).mean()),
                       "yaw_step_degrees_quantiles": np.rad2deg(np.quantile(
                           np.abs(np.angle(np.exp(1j*delta))), [.5, .9, .99, 1])).tolist(),
                       "mean_squared_complex_error": {
                           "plain": float(error(unit[q-1]).mean()),
                           "transported": float(error(aligned).mean()),
                           "wrong_sign": float(error(wrong).mean())},
                       "turns_at_least_5deg": int(turn.sum()),
                       "turn_squared_complex_error": {
                           k: float(error(a)[turn].mean()) if turn.any() else None
                           for k, a in (("plain", unit[q-1]), ("transported", aligned),
                                        ("wrong_sign", wrong))}}
    return np.concatenate(spectra), np.concatenate(yaws), sources, stats


def prepare() -> None:
    """모델·표본·최종 평가·확장 기준을 결과 관측 전에 동결한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    previous = base.protocol(True)
    state = torch.load(base.PREPARED, map_location="cpu", weights_only=False)
    original = torch.load(base.GRAPH, map_location="cpu", weights_only=False)
    graph = REFERENCE_SUBGRAPH(original, np.arange(original.num_nodes))
    hashes = {k: array_hash(getattr(graph, k).numpy())
              for k in ("x", "edge_index", "edge_attr", "edge_type")}
    assert hashes == previous["graph_hashes"]
    z, yaw, sources, geometry = training_phase(state, previous["train"])
    plan = attach_transport(graph, z, yaw, state["sequences"])
    torch.save(graph, OUT / "training_graph.pt")
    sources.add(OUT / "training_graph.pt")
    # 불필요한 거리 차분 그래프를 제외하고 실제 공통 입력을 그대로 고정한다.
    sources.update(REPO / p for p in previous["inputs"]
                   if "temporal_derivatives_20261020/" not in p)
    counts, initial = {}, {}
    for arm in ARMS:
        torch.manual_seed(1)
        net = model(arm)
        counts[arm] = sum(p.numel() for p in net.parameters())
        initial[arm] = {k: array_hash(v.numpy()) for k, v in net.state_dict().items()}
    common = set.intersection(*(set(v) for v in initial.values()))
    assert all(len({initial[a][k] for a in ARMS}) == 1 for k in common)
    assert initial["spectral_local"] == initial["untransported"] == initial["transported"]
    assert counts["transported"] == 167469
    runtime = dict(previous["runtime"])
    for p in (CONFIG, Path(__file__), REPO / "scripts/_pdv2_spectral_transport_run.py",
              REPO / "src/gnn/spectral_transport.py",
              REPO / "tests/gnn/test_spectral_transport.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / "training_geometry.json", geometry)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "val": previous["val"],
        "train": previous["train"], "confidence": previous["confidence"],
        "parameters": counts, "initial_common_tensors": len(common),
        "graph_hashes": hashes, "graph_nodes": graph.num_nodes,
        "graph_edges": graph.edge_index.shape[1], "plan": plan, "test_used": False,
        "training": "같은 GT학습8·72306노드·183249삼중쌍·순열·20epoch900업데이트. no_interdiff288·게이트초기.0625. 기존 직접GNNTrainer·InfoNCE·Adam·AMP. 표준/투영 포함 다섯 조건 새 학습. 선택checkpoint10/20.",
        "operator": "NSD BEV16x8복소계수·노드별RMS정규화. exp(ik(yaw_i-yaw_j))로±5이웃을회전보정해평균. 주파수별복소16→16의자기/이웃선형변환과잔차후RMS. 중심/새복소크기+h를384→86→128로변환,두층. 복소위상은층사이보존. raw/입출력/BN/잔차/게이트유지. old attention/간선부호기대체. 최대미래10.",
        "pose": "학습GT: KITTI카메라자세에공식Tr(velo→camera)우곱후첫라이다좌표에서yaw. HeLiPR Velodyne_gt도첫센서좌표에서yaw. 검증:KISS의라이다자세만읽고GT는정답용. 전체SE3보정아닌yaw근사,병진/roll/pitch/가림보정없음. 광학LiDAR위상아닌각도DFT위상.",
        "controls": "standard/point는288입력. 복소3조건은동일가중치초기/파라미터/입력이며local은mean=z로이웃무시,untransported는위상보정없이평균. local의두복소행렬은합으로작용해식별중복이있음. 용량근사,연산량동일아님. 저장NSD스펙트럼재사용하나추론문맥비용추가.",
        "evaluation": previous["evaluation"],
        "selection": "각조건10/20중최종arc_unique_b100 AP 최대. 사전 주조건transported가standard/point/local각각AP+.1%p,R100손실<=.1%p,계열AP손실<=.25%p이고untransported보다AP엄격양수면같은5조건시드2/3확장. 시험/원고교체없음.",
        "runtime": runtime,
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in sorted(sources)},
    })
    print("PREPARED", counts, plan, flush=True)


def configure(arm: str = None) -> None:
    """검산된 공통 학습·평가 함수를 새 실험의 경로와 모델에 연결한다."""
    base.OUT, base.ARMS, base.CFG, base.EPOCHS = OUT, ARMS, CFG, EPOCHS
    base.protocol, base.model, base.make_keys = protocol, model, make_keys
    if arm in COMPLEX_ARMS:
        base.temporal_subgraph = lambda unused, ids: torch.load(
            OUT / "training_graph.pt", map_location="cpu", weights_only=False)
    else:
        base.temporal_subgraph = REFERENCE_SUBGRAPH


def make_keys(net, cache: dict, arm: str, end: int = None) -> tuple:
    """GT를 쓰지 않고 현재 관측한 라이다 자세와 복소 특징으로 키를 만든다."""
    raw = cache["descriptors"][:end].astype(np.float32)
    poses = cache["poses"][:end]
    sequence = np.zeros(len(raw), np.int64)
    graph = base.make_graph(raw, poses, sequence)
    if arm in COMPLEX_ARMS:
        assert VALID_SPECTRUM is not None
        attach_transport(graph, VALID_SPECTRUM[:end], lidar_yaw(poses), sequence)
    with torch.no_grad():
        key = base.normalize(net(graph.cuda()).float().cpu().numpy())
    assert key.shape == (len(raw), 416) and np.isfinite(key).all()
    return key, None


def validation(seed: int, arm: str, epoch: int, sequence: str) -> None:
    """같은 스캔의 복소 특징을 읽어 기존 최종 평가를 실행한다."""
    global VALID_SPECTRUM
    _, _, paths, _ = base.exp.old.base.inputs("val", sequence)
    ids, VALID_SPECTRUM = read_spectrum(paths["spec"])
    with np.load(paths["cache"]) as z:
        np.testing.assert_array_equal(ids, z["scan_ids"])
    configure(arm)
    base.validation(seed, arm, epoch, sequence)


def report() -> dict:
    """최종 AP로 선택하고 표준·단일 프레임·무보정 대조의 손익을 잰다."""
    spec = protocol(True)
    per_seed, effects, checks = {}, {}, {}
    for seed in (1, 2, 3):
        models = {}
        for arm in ARMS:
            candidates = []
            for epoch in EPOCHS:
                folder = OUT / f"s{seed}/{arm}/epoch{epoch}"
                paths = [folder / (s.replace("/", "_") + ".json") for s in spec["val"]]
                if all(p.exists() for p in paths):
                    rows = {s: read(p)["metrics"] for s, p in zip(spec["val"], paths)}
                    candidates.append({"epoch": epoch, **{m: base.macro(
                        {s: row[m] for s, row in rows.items()}) for m in ("raw", METHOD)}})
            if len(candidates) == len(EPOCHS):
                selected = max(candidates, key=lambda r: (round(r[METHOD]["macro"]["AP"], 12),
                                                          -r["epoch"]))
                models[arm] = {"selected": selected, "candidates": candidates}
        if not models:
            continue
        assert set(models) == set(ARMS)
        per_seed[str(seed)] = models
        effects[str(seed)], checks[str(seed)] = {}, {}
        chosen = models["transported"]["selected"][METHOD]
        for arm in ARMS:
            ref = models[arm]["selected"][METHOD]
            effects[str(seed)][arm] = {m: 100*(chosen["macro"][m]-ref["macro"][m])
                                      for m in base.exp.METRICS}
            if arm in CFG["promotion"]["comparators"]:
                checks[str(seed)][arm] = {
                    "AP": chosen["macro"]["AP"] >= ref["macro"]["AP"]+.001,
                    "R100": chosen["macro"]["R@100P"] >= ref["macro"]["R@100P"]-.001,
                    "family_AP": all(chosen["families"][f]["AP"] >= ref["families"][f]["AP"]-.0025
                                     for f in ref["families"])}
        checks[str(seed)]["alignment"] = {
            "AP": chosen["macro"]["AP"] > models["untransported"]["selected"][METHOD]["macro"]["AP"]}
    eligible = all(all(v.values()) for v in checks["1"].values())
    result = {"per_seed": per_seed, "transported_minus_other_pp": effects,
              "promotion_checks": checks, "pilot_eligible": eligible, "test_used": False,
              "protocol_sha256": sha256(OUT / "protocol.json")}
    write_json(OUT / "summary.json", result)
    for seed, models in per_seed.items():
        for arm, v in models.items():
            print("FINAL", seed, arm, v["selected"]["epoch"], v["selected"][METHOD]["macro"], flush=True)
    print("PILOT_ELIGIBLE", eligible, checks, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "report"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epoch", type=int, choices=EPOCHS)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action == "train" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare":
        prepare()
    elif args.action == "report":
        report()
    elif args.action == "train":
        configure(args.arm)
        base.train(args.seed, args.arm)
    else:
        validation(args.seed, args.arm, args.epoch, args.sequence)
