#!/usr/bin/env python3
"""시간 문맥·이웃 없는 투영의 3시드 최종 판정 대조.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_pdv2_temporal3_run.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_matching_eval as base
from _pdv2_no_context_ablation import no_context_key, sha256, write_json

CONFIG = REPO / "configs/pdv2_temporal3.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = REPO / "results/pdv2_temporal3_20261017"
ARMS, SEEDS = tuple(CFG["arms"]), tuple(CFG["seeds"])
EPOCHS = tuple(CFG["checkpoint_epochs"])
SAMPLES = REPO / CFG["sample_source"]


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def checkpoint_dir(seed: int, arm: str) -> Path:
    """주모델과 분리된 새 체크포인트 위치를 반환한다."""
    assert seed in SEEDS and arm in ARMS
    return REPO / f"checkpoints/pdv2_temporal3_s{seed}_{arm}"


def training_config() -> Path:
    """두 조건에 공통인 시간 그래프 학습 설정 위치다."""
    return OUT / "training.yaml"


def protocol(data: bool = False) -> dict:
    """동결한 코드와 필요시 자료 해시를 검사한다."""
    spec = read(OUT / "protocol.json")
    for name, digest in spec["runtime_sha256"].items():
        assert sha256(REPO / name) == digest, name
    if data:
        for name, digest in spec["input_sha256"].items():
            assert sha256(REPO / name) == digest, name
    return spec


def prepare() -> None:
    """학습 전에 비교 범위·선택 규칙·코드·자료를 고정한다."""
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    cfg = yaml.safe_load((REPO / CFG["base_training_config"]).read_text())
    cfg["gnn"]["diffattn_value_source"] = "diff"
    cfg["keyframe"]["graph"]["edge_type_filter"] = "temporal_only"
    cfg["data"]["datasets"]["test"] = []
    OUT.mkdir(parents=True, exist_ok=True)
    training_config().write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    old = read(REPO / "results/pdv2_attention_neighbors_20261015/protocol.json")
    runtime = {REPO / p for p in old["runtime_sha256"]}
    runtime.update({CONFIG, training_config(), REPO / "src/gnn/graph_free.py"})
    runtime.update(REPO / p for p in (
        "scripts/_pdv2_temporal3.py", "scripts/_train_temporal3.py",
        "scripts/_pdv2_temporal3_run.py", "scripts/_pdv2_temporal3_verify.py",
        "tests/gnn/test_temporal3.py", "scripts/_pdv2_attention_neighbors_verify_reference.py"))
    data = {SAMPLES / "audit.json", SAMPLES / "training_inputs.npz", base.RVEC}
    data.update(SAMPLES / f"original_epoch{e}.npz" for e in (1, 6, 11, 16))
    data.update(REPO / p for p in read(SAMPLES / "audit.json")["sha256"]
                if p.startswith("data/"))
    for seed in SEEDS:
        data.add(base.BASE / f"s{seed}/confidence_selection.json")
        for sensor, seq in base.VAL:
            _, _, paths, _ = base.inputs("val", sensor + "/" + seq)
            data.update(paths.values())
            data.update(base.BASE / f"s{seed}/val_selected_symmetric_{op}_{sensor}_{seq}.npz"
                        for op in base.OPS)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "seeds": list(SEEDS), "arms": list(ARMS),
        "train_roster": [s + "/" + q for s, q in base.TRAIN],
        "val_roster": [s + "/" + q for s, q in base.VAL], "test_used": False,
        "training": "각 조건·시드 처음부터20에포크900업데이트. no_interdiff·문맥게이트초기0.0625·기존InfoNCE·학습률 유지. 공유 층은 동일 초기화. 시간 모델은 차이 attention2층, point는 같은 차원·게이트·잔차의 노드별 투영으로 파라미터 수는 다름.",
        "samples": "기존 학습8 원래 삼중쌍4묶음을 여섯 실행에 고정. 시드는 초기화·드롭아웃·배치순서에 적용하며 표본 재추출의 변동은 포함하지 않음. 같은 시드 두 조건의 매 에포크 배치순서 해시를 대조.",
        "graph": "학습 두 조건 모두 동일 시간그래프·속성. point는 간선 입력을 사용하지 않음. 추론 temporal은 시간그래프 한 통과, point는 간선없는 한 통과. 간선 분류기 사용 안 함.",
        "inference": "KISS-ICP 동일 키프레임, q+15 판정·j<=q-30·100+100·재계량·정렬 유지. 같은 시드 두 조건에 기존 주모델의 동일 융합·수락점수·임계값. 조건별 재선택 없음. 지연 단축·실시간 성능 측정은 이번 범위 아님.",
        "selection": "각 시드·조건10/20에포크 중 최종 검증 AP를 시퀀스→3계열→NSD/H 평균하여 최대 선택.1e-12동률 이른에포크. 동일 에포크 차이도 보고.",
        "scope": "반복 노출된 검증9의3학습시드 비교. 독립 시험·유의성 확증 아님. 최종 AP/F1/R100/R1, 방향·자료별 결과와 비재방문 오수락을 보고. 원고 모델·주표 자동 교체 없음.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(data)},
    })
    print("PREPARED", len(runtime), len(data), flush=True)


def load_model(seed: int, arm: str, epoch: int, device: str = "cuda") -> tuple:
    """새 모델을 원고 가중치·간선 분류기를 읽지 않고 구성한다."""
    import torch
    from gnn.model import create_spectral_gnn
    from gnn.graph_free import create_graph_free_model
    factory = create_spectral_gnn if arm == "temporal" else create_graph_free_model
    cfg = yaml.safe_load(training_config().read_text())
    net = factory(
        input_dim=288, hidden_dim=128, context_dim=128, n_layers=2, n_heads=4,
        dropout=.1, norm_type="batch_norm", use_residual_gate=True,
        gate_hidden_dim=64, gate_initial_alpha=.0625,
        edge_encoder_config=cfg["gnn"]["edge_encoding"],
        key_remetrize_config={"enabled": True, "learnable": False, "init_path": str(base.RVEC)},
    ).to(device).eval()
    path = checkpoint_dir(seed, arm) / f"checkpoint_epoch_{epoch}.pth"
    assert read(OUT / f"s{seed}/{arm}/training_complete.json")["checkpoints"][path.name] == sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    missing, extra = net.load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra, (missing, extra)
    assert all(torch.equal(net.state_dict()[k].cpu(), v) for k, v in state.items())
    return cfg, net


def embedding(cache, sensor: str, raw: np.ndarray, cfg: dict,
              net, arm: str, device: str = "cuda") -> tuple:
    """시간 간선만 또는 간선 없이 한 번 계산하고 그래프 사용을 기록한다."""
    import torch
    from torch_geometric.data import Data
    from _lag_branch_fusion import lag_pass1_graph
    from _xs_eval_lib import normalize
    if arm == "temporal":
        cfg = copy.deepcopy(cfg)
        cfg["keyframe"]["graph"]["similarity_max_k"] = 0
        cfg["keyframe"]["graph"]["similarity_threshold"] = 2.
        graph = lag_pass1_graph(cache, raw, cfg, device, sensor)
        assert bool((graph.edge_type == 0).all())
    else:
        assert arm == "point"
        graph = Data(x=torch.from_numpy(raw).to(device),
                     edge_index=torch.empty((2, 0), dtype=torch.long, device=device))
    with torch.no_grad():
        result = normalize(net(graph).float().cpu().numpy())
    return result, {"nodes": len(raw), "edges": int(graph.edge_index.shape[1]),
                    "similarity": 0, "passes": 1}


def keys(seed: int, arm: str, epoch: int, sequence: str) -> None:
    """검증 키를 저장하고 원래 키 블록·차원·유한성을 확인한다."""
    import torch
    from _pdv2_pair_evaluate import ArrayCache
    from _xs_eval_lib import normalize
    protocol()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq, paths, _ = base.inputs("val", sequence)
    target = base.folder(seed, arm, epoch) / f"keys_val_{sensor}_{seq}.npz"
    if target.exists():
        return
    with np.load(paths["cache"]) as z:
        cache = ArrayCache({k: z[k] for k in z.files})
    raw = cache["descriptors"].astype(np.float32)
    cfg, net = load_model(seed, arm, epoch)
    result, graph = embedding(cache, sensor, raw, cfg, net, arm)
    assert result.shape == (len(raw), 416) and result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(normalize(result[:, :288]),
                               no_context_key(raw, np.load(base.RVEC))[:, :288], atol=3e-7)
    meta = {"protocol_sha256": sha256(OUT / "protocol.json"), "graph": graph,
            "checkpoint_sha256": sha256(checkpoint_dir(seed, arm) / f"checkpoint_epoch_{epoch}.pth"),
            "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()}}
    np.savez_compressed(target, embedding=result, scan_ids=cache["scan_ids"], meta=json.dumps(meta))
    print("KEYS", seed, arm, epoch, sequence, graph, flush=True)


def report() -> None:
    """시드별 최종 결과·짝지은 차이·에포크별 차이를 저장한다."""
    protocol()
    for seed in SEEDS:
        base.select(seed)
    selected = {str(s): read(OUT / f"s{s}/validation_summary.json")["arms"] for s in SEEDS}
    means, effects, rows = {}, {}, []
    for op in base.OPS:
        means[op], effects[op] = {}, {}
        for arm in ARMS:
            means[op][arm] = {}
            for m in base.PRIMARY:
                vals = [selected[str(s)][arm]["ops"][op]["macro"][m] for s in SEEDS]
                means[op][arm][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
        for m in base.PRIMARY:
            vals = [selected[str(s)]["temporal"]["ops"][op]["macro"][m]
                    - selected[str(s)]["point"]["ops"][op]["macro"][m] for s in SEEDS]
            effects[op][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
        for seed in SEEDS:
            for arm in ARMS:
                r = selected[str(seed)][arm]
                rows.append({"seed": seed, "arm": arm, "op": op, "epoch": r["epoch"], **r["ops"][op]["macro"]})
    fixed = {}
    for e in EPOCHS:
        fixed[str(e)] = {str(s): {a: base.aggregate(s, a, e, "val") for a in ARMS} for s in SEEDS}
    write_json(OUT / "summary.json", {"per_seed": selected, "mean_std": means,
               "temporal_minus_point": effects, "fixed_epoch": fixed, "test_used": False,
               "protocol_sha256": sha256(OUT / "protocol.json")})
    with (OUT / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("SUMMARY", means, effects, flush=True)


base.OUT, base.ARMS, base.EPOCHS = OUT, ARMS, EPOCHS
base.protocol, base.checkpoint_dir = protocol, checkpoint_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "keys", "score", "report"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epoch", type=int, choices=EPOCHS)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "keys":
        keys(args.seed, args.arm, args.epoch, args.sequence)
    elif args.action == "score":
        base.score(args.seed, args.arm, args.epoch, "val", args.sequence)
    else:
        report()
