#!/usr/bin/env python3
"""시간 전용 attention·반대 방향 쌍 대조의 규약, 입력, 최종 평가다."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal3 as old
from _pdv2_no_context_ablation import sha256, write_json

base = old.base
CONFIG = REPO / "configs/pdv2_temporal_focus.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = REPO / "results/pdv2_temporal_focus_20261018"
ARMS, SEEDS = tuple(CFG["arms"]), tuple(CFG["seeds"])
EPOCHS = tuple(CFG["checkpoint_epochs"])
SAMPLES = OUT / "samples"
REUSED = {"diff_original": "temporal", "point_original": "point"}


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def arm_dir(seed: int, arm: str) -> Path:
    """재사용 조건과 신규 조건의 보존 위치를 반환한다."""
    return (old.OUT / f"s{seed}/{REUSED[arm]}" if arm in REUSED else OUT / f"s{seed}/{arm}")


def folder(seed: int, arm: str, epoch: int) -> Path:
    """과거 산출물을 덮어쓰지 않고 평가 폴더를 정한다."""
    assert arm in ARMS and seed in SEEDS and epoch in EPOCHS
    path = arm_dir(seed, arm) / f"epoch{epoch}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(seed: int, arm: str) -> Path:
    """조건별 모델 경로다."""
    return (old.checkpoint_dir(seed, REUSED[arm]) if arm in REUSED else
            REPO / f"checkpoints/pdv2_temporal_focus_s{seed}_{arm}")


def training_config(arm: str) -> Path:
    """표본 정책과 무관한 집계 방식별 설정이다."""
    return OUT / ("standard.yaml" if arm.startswith("standard") else "diff.yaml")


def protocol(data: bool = False) -> dict:
    """동결한 실행 코드·입력 해시를 확인한다."""
    spec = read(OUT / "protocol.json")
    for field in ("runtime_sha256", "input_sha256") if data else ("runtime_sha256",):
        for p, h in spec[field].items():
            assert sha256(REPO / p) == h, p
    return spec


def opposite_replace(original: np.ndarray, pools: list, negative: np.ndarray,
                     seed: int, fraction: float) -> tuple:
    """고정 표본의 일부만 실제 반대 방향 쌍으로 교체한다."""
    rng = np.random.default_rng(seed)
    trips = original.copy()
    eligible = np.asarray([i for i, p in enumerate(pools) if len(p) and negative[i] >= 0])
    count = int(round(len(trips) * fraction))
    assert len(eligible) and 0 <= count <= len(trips)
    rows = rng.choice(len(trips), count, replace=False)
    a = rng.choice(eligible, count, replace=True)
    p = np.array([rng.choice(pools[i]) for i in a])
    trips[rows] = np.stack((a, p, negative[a]), axis=1)
    return trips, rows


def prepare() -> None:
    """기존 학습 자료에서 실제 반대 방향 쌍을 만들고 실험을 동결한다."""
    import torch
    from gnn.pair_sampling import past_heading
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    original_protocol = old.protocol(True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    data = {REPO / p for p in original_protocol["input_sha256"]}
    prepared = REPO / "results/pdv2_pair_sampling_20261013/prepared.pt"
    state = torch.load(prepared, map_location="cpu", weights_only=False)
    with np.load(old.SAMPLES / "training_inputs.npz") as z:
        for name in ("poses", "descriptors", "sequences"):
            np.testing.assert_array_equal(state[name], z[name])
    pos, seq, frames = state["poses"][:, :3, 3], state["sequences"], state["frames"]
    heading, valid = past_heading(pos, seq)
    records = {}
    for ep in CFG["sample_refresh_epochs"]:
        with np.load(old.SAMPLES / f"original_epoch{ep}.npz") as z:
            original = z["triplets"]
        opposite, rows = opposite_replace(original, state["opposite"], state["negative"], CFG["sample_seed"] + ep, CFG["opposite_fraction"])
        unchanged = np.ones(len(original), bool)
        unchanged[rows] = False
        np.testing.assert_array_equal(original[unchanged], opposite[unchanged])
        records[str(ep)] = {}
        for label, trips in (("original", original), ("opposite", opposite)):
            a, p, n = trips.T
            assert np.all(frames[a] == frames[p]) and np.all(seq[a] == seq[n])
            assert np.all(np.linalg.norm(pos[a] - pos[p], axis=1) <= 5)
            assert np.all(np.linalg.norm(pos[a] - pos[n], axis=1) > 10)
            assert np.all((seq[a] != seq[p]) | (np.abs(a-p) >= 30))
            isopp = valid[a] & valid[p] & ((heading[a] * heading[p]).sum(1) <= -.5)
            if label == "opposite":
                assert isopp[rows].all()
            counts = np.unique(a[isopp], return_counts=True)[1]
            records[str(ep)][label] = {"triplets": len(trips), "opposite_count": int(isopp.sum()),
                "unique_opposite_anchors": len(counts), "unique_opposite_pairs": len(np.unique(trips[isopp, :2], axis=0)),
                "max_opposite_anchor_repeats": int(counts.max()) if len(counts) else 0,
                "cross_sequence_pairs": int((seq[a] != seq[p]).sum())}
            path = SAMPLES / f"{label}_epoch{ep}.npz"
            np.savez_compressed(path, triplets=trips)
            data.add(path)
        print("SAMPLES", ep, records[str(ep)], flush=True)
    runtime = {REPO / p for p in original_protocol["runtime_sha256"]}
    for kind in ("diff", "standard"):
        cfg = yaml.safe_load(old.training_config().read_text())
        cfg["gnn"]["diffattn_value_source"] = kind
        cfg["data"]["datasets"]["test"] = []
        training_config(kind).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        runtime.add(training_config(kind))
    runtime.update(REPO / p for p in ("configs/pdv2_temporal_focus.yaml", "scripts/_pdv2_temporal_focus.py",
        "scripts/_train_temporal_focus.py", "scripts/_pdv2_temporal_focus_run.py", "scripts/_pdv2_temporal_focus_verify.py",
        "tests/gnn/test_temporal_focus.py", "src/gnn/pair_sampling.py"))
    data.update((prepared, old.OUT / "summary.json", old.OUT / "protocol.json"))
    for seed in SEEDS:
        for arm in REUSED:
            for name in ("initial_model.json", "training_graph.json", "batch_order.json", "training_complete.json"):
                data.add(arm_dir(seed, arm) / name)
            for ep in EPOCHS:
                data.add(checkpoint_dir(seed, arm) / f"checkpoint_epoch_{ep}.pth")
    write_json(OUT / "sample_audit.json", records)
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "config": CFG, "samples": records,
      "train_roster": original_protocol["train_roster"], "val_roster": original_protocol["val_roster"], "test_used": False,
      "training": "GT 키프레임·시간 간선만·처음부터20에포크900업데이트. 기존 InfoNCE 그대로이며 배치 같은장소 마스킹 없음. 네 표본 묶음과 에포크별 순열을 모든 모델에 공유. 원래 표본은75%보존,25%를 실제 반대방향쌍으로교체.",
      "reuse": "diff_original/point_original은 temporal3 동결 코드·자료·초기화·배치순서·예산을 재사용. 새 표준/반대방향과 같은 시간그래프를 검사. 기존 경로에 쓰지 않음.",
      "evaluation": "KISS입력·q+15·j<=q-30·100+100·기존시드별재계량/융합/판정 고정. 10/20중최종NSD/H계열평균AP로선택. 같은에포크도보고.",
      "expansion": "시드1예비에서 point 동일표본 대비 두운영점 AP가 모두양수인 시간 후보 중 jointAP최대(동률사전arms순서)를 시드2/3로확장. R100손익도보고하며AP만으로원고채택안함. 통과없으면예비에서중단.",
      "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
      "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(data)}})


def load_model(seed: int, arm: str, epoch: int, device: str = "cuda") -> tuple:
    """학습 때와 같은 집계 연산으로 저장 가중치를 읽는다."""
    import torch
    from gnn.model import create_spectral_gnn
    from gnn.graph_free import create_graph_free_model
    cfg = yaml.safe_load(training_config(arm).read_text())
    factory = create_graph_free_model if arm.startswith("point") else create_spectral_gnn
    model = factory(input_dim=288, hidden_dim=128, context_dim=128, n_layers=2, n_heads=4, dropout=.1,
        norm_type="batch_norm", use_residual_gate=True, gate_hidden_dim=64, gate_initial_alpha=.0625,
        diffattn_value_source=cfg["gnn"]["diffattn_value_source"], edge_encoder_config=cfg["gnn"]["edge_encoding"],
        key_remetrize_config={"enabled": True, "learnable": False, "init_path": str(base.RVEC)}).to(device).eval()
    path = checkpoint_dir(seed, arm) / f"checkpoint_epoch_{epoch}.pth"
    assert read(arm_dir(seed, arm) / "training_complete.json")["checkpoints"][path.name] == sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    missing, extra = model.load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra
    return cfg, model


def keys(seed: int, arm: str, epoch: int, sequence: str) -> None:
    """시간 전용 또는 투영 키를 만들고 실제 사용 그래프를 기록한다."""
    import torch
    from _pdv2_pair_evaluate import ArrayCache
    protocol()
    assert arm not in REUSED
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq, paths, _ = base.inputs("val", sequence)
    target = folder(seed, arm, epoch) / f"keys_val_{sensor}_{seq}.npz"
    if target.exists():
        return
    with np.load(paths["cache"]) as z:
        cache = ArrayCache({k: z[k] for k in z.files})
    cfg, model = load_model(seed, arm, epoch)
    emb, graph = old.embedding(cache, sensor, cache["descriptors"].astype(np.float32), cfg, model,
                               "point" if arm.startswith("point") else "temporal")
    assert emb.shape == (len(cache["descriptors"]), 416) and np.isfinite(emb).all()
    np.savez_compressed(target, embedding=emb, scan_ids=cache["scan_ids"], meta=json.dumps({
        "graph": graph, "protocol_sha256": sha256(OUT / "protocol.json"),
        "checkpoint_sha256": sha256(checkpoint_dir(seed, arm) / f"checkpoint_epoch_{epoch}.pth")}))


def select(seed: int, arms: tuple) -> dict:
    """선택된 각 조건의 최종 지표와 표본을 맞춘 문맥 효과를 보고한다."""
    protocol()
    base.ARMS = arms
    base.select(seed)
    chosen = read(OUT / f"s{seed}/validation_summary.json")["arms"]
    effects = {}
    for arm in arms:
        if arm.startswith("point"):
            continue
        ref = "point_" + arm.split("_")[1]
        effects[arm] = {op: {m: chosen[arm]["ops"][op]["macro"][m] - chosen[ref]["ops"][op]["macro"][m]
                            for m in base.PRIMARY} for op in base.OPS}
    write_json(OUT / f"s{seed}/effects.json", effects)
    return chosen


base.OUT, base.ARMS, base.EPOCHS = OUT, ARMS, EPOCHS
base.protocol, base.checkpoint_dir, base.folder = protocol, checkpoint_dir, folder

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "keys", "score"))
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--epoch", type=int, choices=EPOCHS)
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "keys":
        keys(a.seed, a.arm, a.epoch, a.sequence)
    else:
        assert a.arm not in REUSED
        base.score(a.seed, a.arm, a.epoch, "val", a.sequence)
