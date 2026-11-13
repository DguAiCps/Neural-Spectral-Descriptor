#!/usr/bin/env python3
"""저장된 학습 상태와 검증 원장으로 문맥 기여의 병목을 진단한다."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
OUT = REPO / "results/pdv2_context_mechanism_20261014"
PILOT = REPO / "results/pdv2_matching_mining_20261014"
BASE = REPO / "results/pdv2_symmetric3_20261013"
ARMS = ("gnn_original", "gnn_matching", "point_original", "point_matching")


def sha(path: Path) -> str:
    """입력 파일의 해시를 구한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, data: dict) -> None:
    """분석 결과를 유한한 JSON으로 저장한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    data["script_sha256"] = sha(Path(__file__))
    path = OUT / name
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def graph() -> None:
    """실제 학습 진입점에서 동일 그래프를 재구성하고 학습 전에 중단한다."""
    import torch
    import keyframe.graph_manager as gm
    from _train_matching_context import array_hash
    torch.set_num_threads(4)
    dest = OUT / "graph_cache"
    dest.mkdir(parents=True, exist_ok=True)
    source = REPO / "checkpoints/pdv2_matching_s1_gnn_original/similarity_dist.npz"
    shutil.copyfile(source, dest / source.name)
    original = gm.build_graph_from_keyframes_batch

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        assert value.x.shape == (72306, 288)
        hashes = {k: array_hash(getattr(value, k).cpu().numpy())
                  for k in ("x", "edge_index", "edge_attr")}
        expected = json.loads((PILOT / "s1/gnn_original/training_graph.json").read_text())
        assert hashes == expected, (hashes, expected)
        torch.save(value.cpu(), OUT / "train_graph.pt")
        write("graph_audit.json", {"hashes": hashes, "reference": expected,
              "nodes": value.num_nodes, "edges": value.num_edges,
              "similarity_dist_sha256": sha(source), "training_executed": False})
        print("GRAPH_EXACT", flush=True)
        raise SystemExit(0)

    gm.build_graph_from_keyframes_batch = capture
    sys.argv = [str(REPO / "train_multi_dataset.py"), "--seed", "1", "--config",
                str(REPO / "configs/pdv2_matching_mining_pilot.yaml"),
                "--checkpoint-dir", str(dest), "--data-root", "/workspace/data",
                "--encoder-preset", "no_interdiff", "--use-gated-context",
                "--gate-initial-alpha", "0.0625"]
    runpy.run_path(sys.argv[0], run_name="__main__")


def loss() -> None:
    """고정 배치에서 명시 오답과 같은 장소의 배치 오답이 받는 확률 질량을 잰다."""
    import torch
    import torch.nn.functional as F
    import yaml
    from gnn.model import create_spectral_gnn
    from gnn.graph_free import create_graph_free_model
    from gnn.trainer import InfoNCELoss
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = torch.load(OUT / "train_graph.pt", map_location="cuda", weights_only=False)
    samples = PILOT / "s1/samples"
    with np.load(samples / "training_inputs.npz") as z:
        positions = torch.tensor(z["poses"][:, :3, 3], dtype=torch.float64, device="cuda")
        sequences = torch.tensor(z["sequences"], device="cuda")
        np.testing.assert_array_equal(data.x.cpu().numpy(), z["descriptors"])
    # 두 조건에 같은 순서와 같은 기준·정답을 사용한다. 역사적 학습 배치 재현은 아니다.
    trips = [np.load(samples / f"{policy}_epoch6.npz")["triplets"]
             for policy in ("original", "matching")]
    np.testing.assert_array_equal(trips[0][:, :2], trips[1][:, :2])
    chosen = np.random.default_rng(20261014).permutation(len(trips[0]))[:32 * 1024]
    old, new = [torch.tensor(t[chosen], device="cuda") for t in trips]
    changed = (old[:, 2] != new[:, 2]).cpu().numpy()
    cfg = yaml.safe_load((REPO / "configs/pdv2_matching_mining_pilot.yaml").read_text())
    result = {"scope": "학습셋·고정 체크포인트의 eval 모드 FP32 진단. 실제 학습 시점의 평균이나 파라미터 기울기 비율이 아니다.",
              "batch_size": 1024, "n_batches": 32, "sample_seed": 20261014,
              "n_rows": len(chosen), "n_changed": int(changed.sum()), "temperature": .1,
              "sources": {str(samples / f"{p}_epoch6.npz"): sha(samples / f"{p}_epoch6.npz")
                          for p in ("original", "matching")}, "models": {}}
    for arm, epoch in itertools.product(ARMS, (10, 20)):
        path = REPO / f"checkpoints/pdv2_matching_s1_{arm}/checkpoint_epoch_{epoch}.pth"
        factory = create_graph_free_model if arm.startswith("point_") else create_spectral_gnn
        model = factory(input_dim=288, hidden_dim=128, context_dim=128, n_layers=2,
                        n_heads=4, dropout=.1, norm_type="batch_norm",
                        edge_encoder_config=cfg["gnn"]["edge_encoding"],
                        use_residual_gate=True, gate_hidden_dim=64,
                        gate_initial_alpha=.0625).to("cuda")
        model.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            embedding = F.normalize(model(data).float(), dim=1)
        assert embedding.shape == (72306, 416) and torch.isfinite(embedding).all()
        store = {policy: {} for policy in ("original", "matching")}
        pair_stats = {}
        for start in range(0, len(chosen), 1024):
            a, p, n0 = old[start:start + 1024].T
            n1 = new[start:start + 1024, 2]
            x, y = embedding[a], embedding[p]
            cross = x @ y.T / .1
            geo = torch.cdist(positions[a], positions[p], compute_mode="donot_use_mm_for_euclid_dist")
            same = sequences[a, None] == sequences[p][None, :]
            bridge = (sequences[a, None] >= 4) & (sequences[p][None, :] >= 4)
            diagonal = torch.eye(1024, dtype=torch.bool, device="cuda")
            false_same = same & (geo <= 5) & ~diagonal
            false_cross = ~same & bridge & (geo <= 5) & ~diagonal
            duplicate = (a[:, None] == p[None, :]) & ~diagonal
            poslogit = cross.diag()
            negatives = [(x * embedding[n]).sum(1) / .1 for n in (n0, n1)]
            for key, value in {"new_negative_harder": negatives[1] > negatives[0],
                               "new_minus_old_cosine": (negatives[1] - negatives[0]) * .1,
                               "same_place_inbatch_count": false_same.sum(1),
                               "cross_bridge_same_place_count": false_cross.sum(1)}.items():
                pair_stats.setdefault(key, []).append(value.cpu().numpy())
            for policy, neg in zip(store, negatives):
                logits = torch.cat([cross, neg[:, None]], dim=1)
                prob = logits.softmax(1)
                target = prob[:, :1024].diag()
                ce = torch.logsumexp(logits, dim=1) - poslogit
                # d(CE)/d(logit_neg)=p_neg。분모는 모든 오답 logit의 기울기 합이다.
                values = {"loss": ce, "explicit_negative_probability": prob[:, -1],
                          "explicit_fraction_of_negative_logit_gradient": prob[:, -1] / (1 - target),
                          "same_place_same_sequence_probability": (prob[:, :1024] * false_same).sum(1),
                          "same_place_cross_bridge_probability": (prob[:, :1024] * false_cross).sum(1),
                          "same_scan_inbatch_probability": (prob[:, :1024] * duplicate).sum(1),
                          "target_probability": target,
                          "explicit_negative_above_positive": neg >= poslogit}
                reference = InfoNCELoss(.1)(x, y, embedding[n0 if policy == "original" else n1])
                torch.testing.assert_close(ce.mean(), reference, atol=2e-6, rtol=2e-6)
                for key, value in values.items():
                    store[policy].setdefault(key, []).append(value.cpu().numpy())
        summary = {"checkpoint_sha256": sha(path), "policies": {}}
        for policy, values in store.items():
            merged = {k: np.concatenate(v) for k, v in values.items()}
            summary["policies"][policy] = {
                subset: {k: float(v[mask].mean()) for k, v in merged.items()}
                for subset, mask in (("all", np.ones(len(chosen), bool)), ("changed_only", changed))}
        summary["pair_statistics_changed_only"] = {
            k: float(np.concatenate(v)[changed].mean()) for k, v in pair_stats.items()}
        result["models"][f"{arm}_epoch{epoch}"] = summary
        write("loss_diagnosis.json", result)
        print("LOSS_DONE", arm, epoch, summary["policies"]["matching"]["all"], flush=True)
        del model, embedding


def ledger() -> None:
    """검증 AP 차이를 질의 유형별 교체의 모든 순서에 걸쳐 분해한다."""
    from _pdv2_pilot import VAL, FAMILY
    from _pr_eval_lib import classify_queries, pr_sweep
    categories = ("recovered_positive", "regressed_positive", "both_correct_positive",
                  "both_wrong_positive", "negative_query")
    result = {"scope": "기존 검증 원장의 사후 분석. 정답을 이용한 AP 차이 회계이며 실행 가능한 새 방법이나 인과 추정이 아니다.",
              "categories": categories, "comparisons": {}, "sources": {}}
    for name, seeds in (("main_no_context_to_context", (1, 2, 3)),
                        ("matching_point_to_gnn", (1,)), ("original_to_matching_gnn", (1,))):
        result["comparisons"][name] = {}
        for seed in seeds:
            selection = json.loads((BASE / f"s{seed}/confidence_selection.json").read_text())["ops"]
            for op in ("pbev128_main", "hspec12"):
                rows, counts, directions = {}, {}, {}
                for sensor, seq in VAL:
                    if name == "main_no_context_to_context":
                        paths = [BASE / f"s{seed}/val_selected_{arm}_{op}_{sensor}_{seq}.npz"
                                 for arm in ("no_context", "symmetric")]
                    else:
                        arms = (("point_matching", 20), ("gnn_matching", 10)) if name == "matching_point_to_gnn" else (("gnn_original", 10), ("gnn_matching", 10))
                        paths = [PILOT / f"s1/{arm}/epoch{ep}/val_{op}_{sensor}_{seq}.npz" for arm, ep in arms]
                    arrays = []
                    for path in paths:
                        with np.load(path) as z:
                            arrays.append({k: z[k] for k in z.files if k != "meta"})
                        result["sources"][str(path.relative_to(REPO))] = sha(path)
                    a, b = arrays
                    for field in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[field], b[field])
                    qc = classify_queries(a["min_past_gt_dist"])
                    good0, good1 = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                    masks = [qc["positive"] & ~good0 & good1, qc["positive"] & good0 & ~good1,
                             qc["positive"] & good0 & good1, qc["positive"] & ~good0 & ~good1,
                             qc["negative"]]
                    decision = selection[op]["decision"]
                    same_pair = a["j_star"] == b["j_star"]
                    np.testing.assert_allclose(a[decision][same_pair], b[decision][same_pair], atol=2e-6, rtol=0)
                    for key, mask in zip(categories, masks):
                        counts[key] = counts.get(key, 0) + int(mask.sum())
                    for key, value in {"queries": len(same_pair), "same_pair": int(same_pair.sum()),
                                       "negative_false_accepts_before": int((qc["negative"] & (a[decision] >= selection[op]["tau_star"])).sum()),
                                       "negative_false_accepts_after": int((qc["negative"] & (b[decision] >= selection[op]["tau_star"])).sum())}.items():
                        counts[key] = counts.get(key, 0) + value
                    for group in range(3):
                        mask = qc["positive"] & (a["group"] == group)
                        d = directions.setdefault(str(group), {"n": 0, "before": 0, "after": 0, "recovered": 0, "regressed": 0})
                        for key, v in {"n": mask.sum(), "before": (mask & good0).sum(), "after": (mask & good1).sum(),
                                       "recovered": (mask & ~good0 & good1).sum(), "regressed": (mask & good0 & ~good1).sum()}.items():
                            d[key] += int(v)
                    if not qc["n_pos"]:
                        continue
                    values = {}
                    for subset in range(32):
                        mask = np.zeros(len(same_pair), bool)
                        for i, m in enumerate(masks):
                            if subset & (1 << i):
                                mask |= m
                        scores = np.where(mask, b[decision], a[decision])
                        distances = np.where(mask, b["top1_geo_dist"], a["top1_geo_dist"])
                        values[subset] = float(pr_sweep(scores, distances, qc)["AP"])
                    shares = {}
                    for i, category in enumerate(categories):
                        total = 0.
                        for subset in range(32):
                            if subset & (1 << i):
                                continue
                            k = bin(subset).count("1")
                            weight = math.factorial(k) * math.factorial(4 - k) / math.factorial(5)
                            total += weight * (values[subset | (1 << i)] - values[subset])
                        shares[category] = total
                    assert abs(sum(shares.values()) - (values[31] - values[0])) < 1e-12
                    rows[f"{sensor}/{seq}"] = {"group": FAMILY.get(sensor, sensor), "before": values[0],
                                               "after": values[31], **shares}
                groups = sorted(set(r["group"] for r in rows.values()))
                macro = {key: float(np.mean([np.mean([r[key] for r in rows.values() if r["group"] == g])
                                            for g in groups])) for key in ("before", "after", *categories)}
                result["comparisons"][name][f"s{seed}_{op}"] = {"macro": macro, "counts": counts,
                                                                  "directions": directions, "sequences": rows}
                write("ledger_diagnosis.json", result)
                print("LEDGER_DONE", name, seed, op, macro, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("graph", "loss", "ledger"))
    globals()[parser.parse_args().action]()
