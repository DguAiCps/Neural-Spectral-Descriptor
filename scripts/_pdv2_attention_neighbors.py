#!/usr/bin/env python3
"""attention 값 전달×이웃 종류의 같은 예산 학습·최종 검증 도구.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_pdv2_attention_neighbors_run.py
"""
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
import _pdv2_matching_eval as base
from _pdv2_no_context_ablation import sha256, write_json, no_context_key

CONFIG = REPO / "configs/pdv2_attention_neighbors.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = REPO / "results/pdv2_attention_neighbors_20261015"
ARMS = tuple(CFG["arms"])
EPOCHS = tuple(CFG["checkpoint_epochs"])
SAMPLES = REPO / CFG["sample_source"]


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def checkpoint_dir(seed: int, arm: str) -> Path:
    """기존 모델과 분리한 체크포인트 위치다."""
    assert seed == CFG["seed"] and arm in ARMS
    return REPO / f"checkpoints/pdv2_attention_s{seed}_{arm}"


def training_config(arm: str) -> Path:
    """실험 전에 생성·고정한 조건별 설정 위치다."""
    assert arm in ARMS
    return OUT / "configs" / (arm + ".yaml")


def protocol(data: bool = False) -> dict:
    """실행 코드와 필요시 입력 자료 해시를 검사한다."""
    value = read(OUT / "protocol.json")
    for name, expected in value["runtime_sha256"].items():
        assert sha256(REPO / name) == expected, name
    if data:
        for name, expected in value["input_sha256"].items():
            assert sha256(REPO / name) == expected, name
    return value


def prepare() -> None:
    """학습 전 네 조건과 선택 규칙·공통 입력을 고정한다."""
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    original = yaml.safe_load((REPO / CFG["base_training_config"]).read_text())
    for arm in ARMS:
        config = copy.deepcopy(original)
        config["gnn"]["diffattn_value_source"] = arm.split("_")[0]
        config["keyframe"]["graph"]["edge_type_filter"] = (
            "temporal_only" if arm.endswith("temporal") else "both")
        config["data"]["datasets"]["test"] = []
        path = training_config(arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    runtime = {CONFIG, REPO / CFG["base_training_config"]}
    runtime.update(training_config(a) for a in ARMS)
    runtime.update(REPO / p for p in (
        "scripts/_pdv2_attention_neighbors.py", "scripts/_train_attention_neighbors.py",
        "scripts/_pdv2_attention_neighbors_run.py", "scripts/_pdv2_attention_neighbors_verify.py",
        "scripts/_pdv2_matching_eval.py", "scripts/_pdv2_matching_mine.py",
        "scripts/_pdv2_lag15_campaign.py", "scripts/_pdv2_lag15_report.py",
        "scripts/_pdv2_no_context_ablation.py", "scripts/_pdv2_checkpoint_select.py",
        "scripts/_lag_branch_fusion.py", "scripts/_causal_branch_fusion.py",
        "scripts/_causal_edge_features.py", "scripts/_causal_pr_dump.py",
        "scripts/_height_candidate_fusion.py", "scripts/build_and_train_edge_classifier.py",
        "scripts/_xs_eval_lib.py", "scripts/_pr_eval_lib.py", "scripts/_pdv2_pilot.py",
        "scripts/_pdv2_pair_evaluate.py", "scripts/run_kitti_operating_point.py",
        "train_multi_dataset.py", "src/keyframe/graph_manager.py",
        "tests/gnn/test_attention_neighbors.py"))
    runtime.update((REPO / "src/gnn").glob("*.py"))
    data = {base.RVEC, base.classifier_path(1), base.BASE / "s1/confidence_selection.json",
            SAMPLES / "audit.json", SAMPLES / "training_inputs.npz"}
    data.update(SAMPLES / f"original_epoch{e}.npz" for e in (1, 6, 11, 16))
    for sensor, seq in base.VAL:
        _, _, paths, _ = base.inputs("val", sensor + "/" + seq)
        data.update(paths.values())
        data.update(base.BASE / f"s1/val_selected_symmetric_{op}_{sensor}_{seq}.npz" for op in base.OPS)
    value = {
        "created_unix": time.time(), "seed": 1, "arms": list(ARMS),
        "train_roster": [s + "/" + q for s, q in base.TRAIN],
        "val_roster": [s + "/" + q for s, q in base.VAL], "test_used": False,
        "training": "4조건 처음부터 동일 시드·초기 텐서·원래 삼중쌍·학습률·20에포크900업데이트. InfoNCE 유지. 기준 자세의 기존 한 통과 학습이며 최근 최종 순위 미세조정과 다르다. 필수 no_interdiff·게이트초기값0.0625 유지.",
        "factors": "diff/standard는 K,V 입력의 (h_j-h_i)/h_j 교체이며 모듈·파라미터 수 동일. full은 현재 시간+유사 간선, temporal은 학습·검증 모두 시간 간선만. 시간 간선과 속성은 동일하다.",
        "inference": "full은 KISS-ICP 기존 두 통과·고정 간선 분류기. temporal은 시간 그래프 한 통과(같은 원시 입력으로 두 번 계산해도 같음). 모든 조건 공통 q+15에서 판정, j<=q-30, 100+100 후보·재계량·정렬·융합·수락점수·임계값 고정. 분류기를 조건별 재적합한 비교가 아니다.",
        "selection": "10/20에포크 중 검증 시퀀스→3계열 최종 AP→NSD/H 평균 최대. 양성없는 KITTI09 공통 제외,1e-12동률 이른에포크. 같은 에포크의 교차효과도 별도 보고.",
        "scope": "시드1 검증9의 구조 진단. 시험·주표 교체 없음. 구조 효과가 학습손실/고정 간선분류기와 상호작용할 수 있다.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(data)},
    }
    write_json(OUT / "protocol.json", value)
    print("PREPARED", len(runtime), len(data), flush=True)


def embedding(cache, sensor: str, raw: np.ndarray, layout: np.ndarray,
              bundle: tuple, arm: str) -> tuple:
    """동일 모델에서 지정된 정보 전달 방식·이웃 조건으로 키를 만든다."""
    import torch
    from gnn.model import DiffAttnConv
    from keyframe.graph_manager import filter_edges_by_type
    from _lag_branch_fusion import lag_pass1_graph, nsd_embedding_lag
    from _xs_eval_lib import normalize
    cfg, net, clf, blob, rvec, _ = bundle
    mode = arm.split("_")[0]
    convs = [m for m in net.modules() if isinstance(m, DiffAttnConv)]
    assert len(convs) == 2
    for conv in convs:
        conv.value_source = mode
        assert conv.W_v_abs is None
    net.eval()
    counts = []

    def audit_graph(module, args):
        graph = args[0]
        counts.append({"nodes": int(graph.num_nodes), "temporal": int((graph.edge_type == 0).sum()),
                       "similarity": int((graph.edge_type == 1).sum())})

    hook = net.register_forward_pre_hook(audit_graph)
    try:
        if arm.endswith("temporal"):
            graph = filter_edges_by_type(lag_pass1_graph(cache, raw, cfg, "cuda", sensor), "temporal_only")
            assert int((graph.edge_type != 0).sum()) == 0
            with torch.no_grad():
                result = normalize(net(graph).float().cpu().numpy())
        else:
            result = nsd_embedding_lag(cache, sensor, raw, layout, cfg, net, clf, blob, rvec, "cuda")
    finally:
        hook.remove()
    return result, counts


def load_bundle(arm: str, epoch: int) -> tuple:
    """선택 전에도 지정된 학습 스냅샷만 읽는다."""
    import torch
    bundle = base.model(1)
    ckpt = checkpoint_dir(1, arm) / f"checkpoint_epoch_{epoch}.pth"
    assert read(OUT / f"s1/{arm}/training_complete.json")["checkpoints"][ckpt.name] == sha256(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    missing, extra = bundle[1].load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra
    assert all(torch.equal(bundle[1].state_dict()[k].cpu(), v) for k, v in state.items())
    return bundle


def keys(arm: str, epoch: int, sequence: str) -> None:
    """검증 전체 키와 그래프 종류·입력 연결 정보를 저장한다."""
    import torch
    from _pdv2_pair_evaluate import ArrayCache
    from _xs_eval_lib import normalize
    protocol()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq, paths, _ = base.inputs("val", sequence)
    target = base.folder(1, arm, epoch) / f"keys_val_{sensor}_{seq}.npz"
    if target.exists():
        return
    with np.load(paths["cache"]) as z:
        cache = ArrayCache({k: z[k] for k in z.files})
    with np.load(paths["layouts"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        layout = z["nsd_layouts"].astype(np.float32)
    raw = cache["descriptors"].astype(np.float32)
    result, counts = embedding(cache, sensor, raw, layout, load_bundle(arm, epoch), arm)
    assert result.shape == (len(raw), 416) and result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(normalize(result[:, :288]), no_context_key(raw, np.load(base.RVEC))[:, :288], atol=3e-7)
    ckpt = checkpoint_dir(1, arm) / f"checkpoint_epoch_{epoch}.pth"
    meta = {"protocol_sha256": sha256(OUT / "protocol.json"), "checkpoint_sha256": sha256(ckpt),
            "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()},
            "graphs": counts, "value_source": arm.split("_")[0]}
    np.savez_compressed(target, embedding=result, scan_ids=cache["scan_ids"], meta=json.dumps(meta))
    print("KEYS", arm, epoch, sequence, counts, flush=True)


def report() -> None:
    """조건별 최종 AP 선택과 에포크를 맞춘 교차효과를 함께 기록한다."""
    protocol()
    base.select(1)
    selected = read(OUT / "s1/validation_summary.json")["arms"]
    rows = {"scope": "시드1 검증9 최종 판정, 시험 미사용", "arms": selected,
            "protocol_sha256": sha256(OUT / "protocol.json"), "effects": {}}
    for label, epoch in (("selected", None), ("epoch10", 10), ("epoch20", 20)):
        cells = {a: selected[a]["ops"] if epoch is None else base.aggregate(1, a, epoch, "val") for a in ARMS}
        rows["effects"][label] = {}
        for op in base.OPS:
            rows["effects"][label][op] = {}
            for metric in base.PRIMARY:
                d = {a: cells[a][op]["macro"][metric] for a in ARMS}
                rows["effects"][label][op][metric] = {
                    "standard_minus_diff_full": d["standard_full"] - d["diff_full"],
                    "standard_minus_diff_temporal": d["standard_temporal"] - d["diff_temporal"],
                    "temporal_minus_full_diff": d["diff_temporal"] - d["diff_full"],
                    "temporal_minus_full_standard": d["standard_temporal"] - d["standard_full"],
                    "interaction": d["standard_temporal"] - d["diff_temporal"] - d["standard_full"] + d["diff_full"],
                }
    write_json(OUT / "summary.json", rows)
    print("SUMMARY", {a: (r["epoch"], {o: r["ops"][o]["macro"] for o in base.OPS}) for a, r in selected.items()}, flush=True)


base.OUT, base.ARMS, base.EPOCHS = OUT, ARMS, EPOCHS
base.protocol, base.checkpoint_dir = protocol, checkpoint_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "keys", "score", "report"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epoch", type=int, choices=EPOCHS)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "keys":
        keys(args.arm, args.epoch, args.sequence)
    elif args.action == "score":
        base.score(1, args.arm, args.epoch, "val", args.sequence)
    else:
        report()
