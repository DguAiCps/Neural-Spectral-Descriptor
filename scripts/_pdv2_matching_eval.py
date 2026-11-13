#!/usr/bin/env python3
"""어려운 오답×이웃 정보 대조의 고정 규칙과 최종 판정 평가다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_matching_mine import OUT, BASE, RVEC
from _pdv2_no_context_ablation import evaluate_keys, no_context_key, sha256, write_json
from _pdv2_lag15_campaign import bank_for, inputs, model, classifier_path
from _pdv2_lag15_report import read as read_ledger, sequence_metrics
from _pdv2_checkpoint_select import aggregate_rows, PRIMARY
from _pdv2_pilot import TRAIN, VAL, TEST, OPS

ARMS = ("gnn_original", "gnn_matching", "point_original", "point_matching")
EPOCHS = (10, 20)
CONFIG = REPO / "configs/pdv2_matching_mining_pilot.yaml"


def read_json(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def checkpoint_dir(seed: int, arm: str) -> Path:
    """원고 체크포인트와 분리된 학습 출력 위치다."""
    assert arm in ARMS
    return REPO / f"checkpoints/pdv2_matching_s{seed}_{arm}"


def protocol() -> dict:
    """최초 학습 전에 고정한 실행 코드를 검사한다."""
    value = read_json(OUT / "protocol.json")
    for name, expected in value["runtime_sha256"].items():
        assert sha256(REPO / name) == expected, name
    return value


def prepare() -> None:
    """파일·표본·평가 규칙과 3시드 확장 기준을 학습 전에 고정한다."""
    if (OUT / "protocol.json").exists():
        protocol()
        return
    files = [CONFIG, RVEC]
    files += [REPO / path for path in (
        "scripts/_pdv2_matching_mine.py", "scripts/_pdv2_matching_prepare.py",
        "scripts/_train_matching_context.py", "scripts/_pdv2_matching_eval.py",
        "scripts/_pdv2_matching_run.py", "src/gnn/matching_mining.py",
        "src/gnn/model.py", "src/gnn/graph_free.py", "src/gnn/trainer.py",
        "src/gnn/triplet_miner.py", "src/keyframe/graph_manager.py", "train_multi_dataset.py",
        "scripts/_pdv2_lag15_campaign.py", "scripts/_lag_branch_fusion.py",
        "scripts/_pdv2_no_context_ablation.py", "scripts/_pdv2_lag15_report.py",
        "scripts/_pdv2_checkpoint_select.py", "scripts/_xs_eval_lib.py",
        "scripts/_pr_eval_lib.py", "scripts/_pdv2_pilot.py")]
    for seed in (1, 2, 3):
        files.extend((BASE / f"s{seed}/confidence_selection.json", classifier_path(seed)))
    data = {}
    for path in sorted((OUT / "mining").glob("*.json")):
        data[str(path.relative_to(REPO))] = sha256(path)
    samples = OUT / "s1/samples/audit.json"
    assert samples.exists() and len(data) == 16
    data[str(samples.relative_to(REPO))] = sha256(samples)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "arms": list(ARMS), "pilot_seed": 1,
        "train_roster": [s + "/" + q for s, q in TRAIN],
        "val_roster": [s + "/" + q for s, q in VAL],
        "epochs": 20, "checkpoint_epochs": list(EPOCHS),
        "training": "네 조건 모두 처음부터 학습. 183249 삼중쌍/에포크, 배치1024×누적4, 총900업데이트. 원래 InfoNCE/학습률/그래프, 공통20에포크 스케줄. AMP 초기scale128·성장간격1000000. 초기화·정답쌍·배치순서·그래프를 같은 모델의 두 오답 조건에서 공유.",
        "mining": "학습셋 재계량 d와 보조 크기 100+100 후보, 시드1의 고정 융합과 정렬. j<=q-30, 거리>10m인 높은 융합 점수 오답을 운영점마다 최대5개 수집·중복제거. 적격 삼중쌍의50%를 풀에서 균등 추출한 오답으로 교체. 정답·앵커·총량 불변. 기존 양성은 양방향 시간 제외 규칙 유지.",
        "fixed_inference": "시드별 원고 간선 분류기·재계량·보조색인·100+100·정렬·융합·판정점수·임계값 유지. 새 모델에 다시 맞추는 전체최적화 아님. 그래프없는 투영은 파라미터수 대조가 아님.",
        "checkpoint_selection": "각 조건·시드에서10/20에포크 중 검증 시퀀스→3계열 AP→NSD/NSD-H 평균 최대,1e-12동률이면이른에포크. 공통30<=q<N-15,후보j<=q-30. KITTI09 양성없음 제외.",
        "expansion_rule": {"joint_AP_gain_min": 0.001,
                           "per_op_AP_loss_tolerance": 0.001,
                           "same_direction_R1_loss_tolerance": 0.002,
                           "comparators": ["gnn_original", "point_matching"],
                           "meaning": "시드1 새GNN이 두 대조보다 공동AP>=0.001 높고, 각AP하락<=0.001, 같은방향최종R1하락<=0.002이면 같은설정 시드2·3 확장. 탐색 예산 기준이며 유의성 검정 아님. 미달이면 검증 예비실험에서 종료하고 시험 미사용."},
        "test_rule": "확장한 경우 세시드 네조건의 검증선택을 고정한 뒤 공통시험46시퀀스에서 최종 지표를 평가. 시험 성능으로 규칙/설정 변경 금지.",
        "scope": "보유 시험의 개발 노출 이력이 있는 추가 진단. 원고 모델/수치 교체 안 함.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in files},
        "prepared_data_sha256": data,
    })


def folder(seed: int, arm: str, epoch: int) -> Path:
    """시드·조건·학습 시점별 결과를 분리한다."""
    assert arm in ARMS and epoch in EPOCHS
    path = OUT / f"s{seed}" / arm / f"epoch{epoch}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def guard(seed: int, arm: str, epoch: int, role: str) -> None:
    """선택 완료 전 시험 실행을 차단한다."""
    protocol()
    if role == "test":
        frozen = read_json(OUT / "freeze_test.json")
        assert frozen["protocol_sha256"] == sha256(OUT / "protocol.json")
        assert frozen["selected"][str(seed)][arm]["epoch"] == epoch


def keys(seed: int, arm: str, epoch: int, role: str, sequence: str) -> None:
    """저장된 가중치로 동일 주평가 조건의416차원 키를 생성한다."""
    import torch
    from torch_geometric.data import Data
    from gnn.graph_free import create_graph_free_model
    from _pdv2_pair_evaluate import ArrayCache
    from _lag_branch_fusion import nsd_embedding_lag
    from _xs_eval_lib import normalize
    guard(seed, arm, epoch, role)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq, paths, _ = inputs(role, sequence)
    target = folder(seed, arm, epoch) / f"keys_{role}_{sensor}_{seq}.npz"
    if target.exists():
        return
    ckpt = checkpoint_dir(seed, arm) / f"checkpoint_epoch_{epoch}.pth"
    completed = read_json(OUT / f"s{seed}/{arm}/training_complete.json")
    assert completed["checkpoints"][ckpt.name] == sha256(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    with np.load(paths["cache"]) as z:
        cache = ArrayCache({k: z[k] for k in z.files})
    raw = cache["descriptors"].astype(np.float32)
    if arm.startswith("gnn_"):
        cfg, net, clf, blob, rvec, _ = model(seed)
        missing, extra = net.load_state_dict(state, strict=False)
        assert missing == ["gnn.key_scale"] and not extra
        with np.load(paths["layouts"]) as z:
            np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
            layout = z["nsd_layouts"].astype(np.float32)
        embedding = nsd_embedding_lag(cache, sensor, raw, layout, cfg, net, clf, blob, rvec, "cuda")
    else:
        net = create_graph_free_model(
            input_dim=288, hidden_dim=128, context_dim=128, n_layers=2, n_heads=4,
            dropout=0.1, norm_type="batch_norm", use_residual_gate=True,
            gate_hidden_dim=64, gate_initial_alpha=0.0625,
            key_remetrize_config={"enabled": True, "learnable": False, "init_path": str(RVEC)},
        ).to("cuda")
        missing, extra = net.load_state_dict(state, strict=False)
        assert missing == ["gnn.key_scale"] and not extra
        net.eval()
        graph = Data(x=torch.from_numpy(raw).to("cuda"),
                     edge_index=torch.empty((2, 0), dtype=torch.long, device="cuda"))
        with torch.no_grad():
            embedding = normalize(net(graph).float().cpu().numpy())
    assert all(torch.equal(net.state_dict()[k].cpu(), v) for k, v in state.items())
    assert embedding.shape == (len(raw), 416) and embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    np.testing.assert_allclose(normalize(embedding[:, :288]),
                               no_context_key(raw, np.load(RVEC))[:, :288], atol=3e-7)
    meta = {"protocol_sha256": sha256(OUT / "protocol.json"), "checkpoint_sha256": sha256(ckpt),
            "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()}}
    np.savez_compressed(target, embedding=embedding, scan_ids=cache["scan_ids"], meta=json.dumps(meta))


def score(seed: int, arm: str, epoch: int, role: str, sequence: str) -> None:
    """키 검색부터 최종 판정까지 계산하고 기준과 같은 질의인지 확인한다."""
    guard(seed, arm, epoch, role)
    sensor, seq, paths, min_past = inputs(role, sequence)
    output = folder(seed, arm, epoch)
    target = output / f"{role}_{sensor}_{seq}.json"
    if target.exists():
        return
    with np.load(output / f"keys_{role}_{sensor}_{seq}.npz") as z:
        embedding, ids = z["embedding"], z["scan_ids"]
    with np.load(paths["cache"]) as z:
        np.testing.assert_array_equal(ids, z["scan_ids"])
        positions = z["poses_gt"][:, :3, 3]
    selection = read_json(BASE / f"s{seed}/confidence_selection.json")["ops"]
    result = {}
    for op in OPS:
        with np.load(paths["spec"]) as z:
            np.testing.assert_array_equal(ids, z["scan_ids"])
            bank = bank_for(op, z)
        selected = selection[op]
        arrays = evaluate_keys(embedding, bank, positions, min_past,
                               {seed: tuple(float(w) for w in selected["tuple"])})[seed]
        keep = arrays["query_idx"] < len(embedding) - 15
        arrays = {k: v[keep] for k, v in arrays.items()}
        ref, _ = read_ledger(BASE / f"s{seed}/{role}_selected_symmetric_{op}_{sensor}_{seq}.npz")
        for k in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(arrays[k], ref[k])
        arrays["group"] = ref["group"]
        metrics = sequence_metrics(arrays, selected["decision"], selected["tau_star"])
        directions = {}
        for i, group in enumerate(("same_available", "opposite_only", "other")):
            mask = (arrays["min_past_gt_dist"] <= 5.) & (arrays["group"] == i)
            directions[group] = {"n": int(mask.sum()),
                                 "correct": int((arrays["top1_geo_dist"][mask] <= 5.).sum())}
        dest = output / f"{role}_{op}_{sensor}_{seq}.npz"
        np.savez_compressed(dest, **arrays, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        result[op] = {"metrics": metrics, "directions": directions,
                      "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest)}
    write_json(target, {"seed": seed, "arm": arm, "epoch": epoch, "role": role,
               "sequence": sequence, "ops": result, "protocol_sha256": sha256(OUT / "protocol.json")})


def aggregate(seed: int, arm: str, epoch: int, role: str) -> dict:
    """최종 지표는 계열/도시 평균, 방향 정답률은 양성 질의 합산으로 구한다."""
    rows = {s + "/" + q: read_json(folder(seed, arm, epoch) / f"{role}_{s}_{q}.json")
            for s, q in (VAL if role == "val" else TEST)}
    result = {}
    for op in OPS:
        value = aggregate_rows(rows, role, op)
        directions = {}
        for group in ("same_available", "opposite_only", "other"):
            n = sum(r["ops"][op]["directions"][group]["n"] for r in rows.values())
            correct = sum(r["ops"][op]["directions"][group]["correct"] for r in rows.values())
            directions[group] = {"n": n, "correct": correct, "R@1": correct / n if n else None}
        value["directions"] = directions
        result[op] = value
    return result


def select(seed: int) -> None:
    """검증 최종 AP로 조건별 스냅샷을 고른다."""
    protocol()
    result = {}
    for arm in ARMS:
        candidates = []
        for epoch in EPOCHS:
            ops = aggregate(seed, arm, epoch, "val")
            candidates.append({"epoch": epoch, "ops": ops,
                               "joint_AP": float(np.mean([r["macro"]["AP"] for r in ops.values()]))})
        best = max(r["joint_AP"] for r in candidates)
        chosen = min((r for r in candidates if best - r["joint_AP"] <= 1e-12), key=lambda r: r["epoch"])
        path = checkpoint_dir(seed, arm) / f"checkpoint_epoch_{chosen['epoch']}.pth"
        result[arm] = {**chosen, "checkpoint": str(path.relative_to(REPO)),
                       "checkpoint_sha256": sha256(path), "candidates": candidates}
    write_json(OUT / f"s{seed}/validation_summary.json", {"seed": seed, "arms": result,
               "protocol_sha256": sha256(OUT / "protocol.json")})
    print("SELECTED", seed, {arm: (r["epoch"], r["joint_AP"]) for arm, r in result.items()}, flush=True)


def pilot() -> None:
    """사전에 고정한 탐색 예산 기준으로 확장 여부를 판정한다."""
    spec = protocol()
    rows = read_json(OUT / "s1/validation_summary.json")["arms"]
    rule = spec["expansion_rule"]
    new = rows["gnn_matching"]
    checks = {}
    for reference in rule["comparators"]:
        old = rows[reference]
        joint_delta = new["joint_AP"] - old["joint_AP"]
        delta_ap = {op: new["ops"][op]["macro"]["AP"] - old["ops"][op]["macro"]["AP"] for op in OPS}
        direction_delta = {op: new["ops"][op]["directions"]["same_available"]["R@1"]
                           - old["ops"][op]["directions"]["same_available"]["R@1"] for op in OPS}
        passed = (joint_delta >= rule["joint_AP_gain_min"]
                  and min(delta_ap.values()) >= -rule["per_op_AP_loss_tolerance"]
                  and min(direction_delta.values()) >= -rule["same_direction_R1_loss_tolerance"])
        checks[reference] = {"joint_AP_delta": joint_delta, "AP_delta": delta_ap,
                            "same_direction_R1_delta": direction_delta, "passed": passed}
    effects = {}
    for op in OPS:
        values = {arm: row["ops"][op]["macro"]["AP"] for arm, row in rows.items()}
        effects[op] = {
            "gnn_mining_delta": values["gnn_matching"] - values["gnn_original"],
            "point_mining_delta": values["point_matching"] - values["point_original"],
            "graph_delta_original": values["gnn_original"] - values["point_original"],
            "graph_delta_matching": values["gnn_matching"] - values["point_matching"],
            "interaction": (values["gnn_matching"] - values["gnn_original"]
                            - values["point_matching"] + values["point_original"]),
        }
    write_json(OUT / "pilot_summary.json", {"arms": rows, "effects": effects,
               "expansion_checks": checks, "expand": all(c["passed"] for c in checks.values()),
               "selection_sha256": sha256(OUT / "s1/validation_summary.json"),
               "protocol_sha256": sha256(OUT / "protocol.json"),
               "scope": "시드1 검증셋 최종 판정 결과. 시험셋 수치가 아니며 통계적 유의성 검정은 하지 않음."})
    print("PILOT", {"checks": checks, "effects": effects}, flush=True)


def freeze() -> None:
    """세 시드 네 조건을 고정한 뒤에만 시험을 허용한다."""
    protocol()
    assert read_json(OUT / "pilot_summary.json")["expand"]
    selected = {str(seed): read_json(OUT / f"s{seed}/validation_summary.json")["arms"] for seed in (1, 2, 3)}
    write_json(OUT / "freeze_test.json", {"selected": selected, "created_unix": time.time(),
               "protocol_sha256": sha256(OUT / "protocol.json"),
               "validation_sha256": {str(seed): sha256(OUT / f"s{seed}/validation_summary.json") for seed in (1, 2, 3)}})


def report_test() -> None:
    """고정된 시험 결과의 세 시드 평균과 차이를 기록한다."""
    frozen = read_json(OUT / "freeze_test.json")
    rows = {str(seed): {arm: aggregate(seed, arm, frozen["selected"][str(seed)][arm]["epoch"], "test")
                       for arm in ARMS} for seed in (1, 2, 3)}
    means = {op: {arm: {m: {"mean": float(np.mean([rows[str(s)][arm][op]["macro"][m] for s in (1, 2, 3)])),
                            "std": float(np.std([rows[str(s)][arm][op]["macro"][m] for s in (1, 2, 3)], ddof=1))}
                       for m in PRIMARY} for arm in ARMS} for op in OPS}
    write_json(OUT / "test_summary.json", {"per_seed": rows, "mean_std": means,
               "freeze_sha256": sha256(OUT / "freeze_test.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "keys", "score", "select", "pilot", "freeze", "report-test"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epoch", type=int, choices=EPOCHS)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "select":
        select(args.seed)
    elif args.action == "pilot":
        pilot()
    elif args.action == "freeze":
        freeze()
    elif args.action == "report-test":
        report_test()
    else:
        (keys if args.action == "keys" else score)(args.seed, args.arm, args.epoch, args.role, args.sequence)
