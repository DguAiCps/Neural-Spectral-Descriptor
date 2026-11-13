#!/usr/bin/env python3
"""학습·검색의 원 특징 거리 불일치와 최종 성능을 대조한다."""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _pdv2_spectral_keys as source
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.trainer import InfoNCELoss

REPO = source.REPO
OUT = REPO / "results/pdv2_key_metric_alignment_20261026"
PRIOR = source.OUT
BASE = source.base
SEEDS = (1, 2, 3)
EPOCHS = (10, 20)
ARMS = ("control", "aligned")
RVEC = PRIOR / "remetric_octave288.npy"
PREPARED = PRIOR / "prepared_octave288.pt"
GRAPH = PRIOR / "graph_octave288.pt"
FACTORY = source.factory
DESCRIPTORS = source.descriptors
EVALUATE = source.evaluate
TRAIN = source.ORIGINAL_TRAIN
METHOD = source.METHOD
METRICS = source.METRICS
read = source.read


def root(seed: int, arm: str) -> Path:
    """대조군은 원본을 보존하고 정합 학습만 새 폴더에 쓴다."""
    return (
        PRIOR / f"s{seed}/octave288" if arm == "control" else OUT / f"s{seed}/aligned"
    )


def model(arm: str, seed: int) -> torch.nn.Module:
    """같은 초기값에서 학습 시 원 특징 재계량 유무만 달리한다."""
    assert arm in ARMS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = FACTORY(288)
    if arm == "aligned":
        net.gnn.key_scale = torch.nn.Parameter(
            torch.from_numpy(np.load(RVEC)).float(), requires_grad=False
        )
    return net


def protocol(check_inputs: bool = False) -> dict:
    """실행 소스와 선택적으로 학습·검증 자료의 해시를 검사한다."""
    spec = read(OUT / "protocol.json")
    for group in (("runtime", "inputs") if check_inputs else ("runtime",)):
        for path, digest in spec[group].items():
            assert sha256(REPO / path) == digest, path
    return spec


def prepare() -> None:
    """학습 전에 가설·동일 조건·선택 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    old = source.protocol(True)
    initial, counts = {}, {}
    for seed in SEEDS:
        initial[str(seed)] = {}
        for arm in ARMS:
            net = model(arm, seed)
            initial[str(seed)][arm] = {
                k: array_hash(v.numpy()) for k, v in net.state_dict().items()
            }
            counts[arm] = sum(p.numel() for p in net.parameters() if p.requires_grad)
        control = initial[str(seed)]["control"]
        assert control == read(root(seed, "control") / "initial_model.json")
        assert all(initial[str(seed)]["aligned"][k] == v for k, v in control.items())
    assert counts == {"control": 167561, "aligned": 167561}
    runtime = dict(old["runtime"])
    for path in (
        Path(__file__),
        REPO / "scripts/_pdv2_key_metric_alignment_run.py",
        REPO / "tests/gnn/test_key_metric_alignment.py",
    ):
        runtime[str(path.relative_to(REPO))] = sha256(path)
    inputs = dict(old["inputs"])
    paths = {PRIOR / "validation_summary.json", RVEC, PREPARED, GRAPH}
    for seed in SEEDS:
        folder = root(seed, "control")
        paths.update((folder / "training_complete.json", folder / "initial_model.json"))
        for epoch in EPOCHS:
            paths.add(folder / "checkpoints" / f"checkpoint_epoch_{epoch}.pth")
        for arm in source.ARMS:
            for epoch in EPOCHS:
                for sequence in old["val"]:
                    p = (
                        PRIOR
                        / f"s{seed}/{arm}/epoch{epoch}"
                        / (sequence.replace("/", "_") + ".npz")
                    )
                    paths.update(
                        (p, p.with_suffix(".json"), p.with_name("coverage_" + p.name))
                    )
    inputs.update({str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)})
    write_json(
        OUT / "protocol.json",
        {
            "created_unix": time.time(),
            "arms": ARMS,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "train": old["train"],
            "val": old["val"],
            "confidence": old["confidence"],
            "graph_hashes": old["graph_hashes"]["octave288"],
            "initial": initial,
            "trainable_parameters": counts,
            "remetric": {a: old["remetric"]["octave288"] for a in ARMS},
            "hypothesis": "학습 때 미적용한 원 특징 재계량을 검색 때 적용하면 문맥이 보완해야 할 오답과 기울기가 달라진다. 학습 처음부터 동일 고정 재계량을 사용하면 최종 성능이 개선되는지 확인한다.",
            "training": "원288 입력·차이 K/V-tanh·기하 상호작용·시간±5·2층·4head·gate 초기.0625. GT학습8·183249삼중쌍·동일순서/초기텐서·20에포크900갱신·기존GNNTrainer InfoNCE/AMP. 재계량은 원 특징 출력에만 적용하며 GNN 입력과 간선은 그대로다. 새 학습 매개변수0. control은 직전 동일 실행 보존 가중치다.",
            "selection": "각조건/시드의10·20에포크 중 검증 최종 arc_unique_b100 AP 최대, 1e-12 반올림 동률은 이른 에포크. 같은 에포크 결과도 모두 보존한다.",
            "promotion": "3시드 평균 최종 AP +0.1%p 이상, 모든 시드 AP 개선, 평균 R100 손실0.1%p 이하, 계열별 평균 AP 손실0.25%p 이하일 때만 후속 후보로 인정한다. 이번 실행은 시험 평가와 주표 교체를 하지 않는다.",
            "evaluation": "KISS검증9·공통99236질의·q+15·j<=q-30·100+100후보 재검색·동일 재계량/시드별융합/판정. 시퀀스→3계열 평균. 원고 표2와 같은 최종 단계이나 검증 분할이다.",
            "diagnosis": "선택된3키의 검증 원장 비교. 학습183249삼중쌍의 재계량 전후 순위와, 기존 시드별10에포크에서 동일 FP32/eval 1024쌍씩3배치의 InfoNCE 기울기 비교. 진단 기울기는 AMP 학습 기울기와 구분한다.",
            "test_used": False,
            "runtime": runtime,
            "inputs": inputs,
        },
    )
    print("FROZEN", counts, len(inputs), flush=True)


def describe(values: np.ndarray) -> dict:
    """수치 분포를 고정 분위수로 요약한다."""
    return {
        "mean": float(np.mean(values)),
        "quantiles": np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist(),
    }


def diagnose() -> None:
    """최종 선택의 공통 부분과 학습 거리의 변경 규모를 측정한다."""
    spec = protocol(True)
    selected = read(PRIOR / "validation_summary.json")["per_seed"]
    frontend = {}
    for seed in SEEDS:
        frontend[str(seed)] = {}
        for arm in ("trunc288", "trunc352"):
            rows = {}
            for sequence in spec["val"]:
                loaded = []
                for name in ("octave288", arm):
                    epoch = selected[str(seed)][name]["selected"]["epoch"]
                    p = (
                        PRIOR
                        / f"s{seed}/{name}/epoch{epoch}"
                        / (sequence.replace("/", "_") + ".npz")
                    )
                    loaded.append(
                        (
                            BASE.exp.arrays(p),
                            BASE.exp.arrays(p.with_name("coverage_" + p.name)),
                        )
                    )
                (a, ac), (b, bc) = loaded
                np.testing.assert_array_equal(a["query_idx"], b["query_idx"])
                same = a["j_star"] == b["j_star"]
                np.testing.assert_array_equal(a[METHOD][same], b[METHOD][same])
                pos = a["min_past_gt_dist"] <= 5
                ca, cb = pos & (a["top1_geo_dist"] <= 5), pos & (
                    b["top1_geo_dist"] <= 5
                )
                rows[sequence] = {
                    "queries": len(same),
                    "positive": int(pos.sum()),
                    "same_winner": int(same.sum()),
                    "gained_correct": int((cb & ~ca).sum()),
                    "lost_correct": int((ca & ~cb).sum()),
                    "octave_missing_candidate": int(
                        (pos & ~ac["union_hit"].astype(bool)).sum()
                    ),
                    "changed_missing_candidate": int(
                        (pos & ~bc["union_hit"].astype(bool)).sum()
                    ),
                    "octave_wrong_despite_candidate": int(
                        (pos & ac["union_hit"].astype(bool) & ~ca).sum()
                    ),
                    "changed_wrong_despite_candidate": int(
                        (pos & bc["union_hit"].astype(bool) & ~cb).sum()
                    ),
                    "same_winner_score_exact": True,
                }
            fields = [
                k for k in next(iter(rows.values())) if k != "same_winner_score_exact"
            ]
            total = {k: sum(r[k] for r in rows.values()) for k in fields}
            frontend[str(seed)][arm] = {"sequences": rows, "total": total}
    state = torch.load(PREPARED, map_location="cpu", weights_only=False)
    graph = torch.load(GRAPH, map_location="cpu", weights_only=False).cuda()
    triplets = BASE.triplets(10)
    x = state["descriptors"].astype(np.float32)
    raw, scaled = BASE.normalize(x), BASE.normalize(
        x * np.load(RVEC).astype(np.float32)
    )

    def margins(feature):
        a, p, n = (feature[triplets[:, k]] for k in range(3))
        return np.sum(a * p, 1) - np.sum(a * n, 1)

    before, after = margins(raw), margins(scaled)
    pair_stats = {
        "triplets": len(triplets),
        "raw_margin": describe(before),
        "scaled_margin": describe(after),
        "raw_positive_beats_negative": int((before > 0).sum()),
        "scaled_positive_beats_negative": int((after > 0).sum()),
        "rescued": int(((before <= 0) & (after > 0)).sum()),
        "harmed": int(((before > 0) & (after <= 0)).sum()),
    }
    gradients = {}
    for seed in SEEDS:
        net = model("control", seed).cuda().eval()
        cp = torch.load(
            root(seed, "control") / "checkpoints/checkpoint_epoch_10.pth",
            map_location="cpu",
            weights_only=False,
        )
        net.load_state_dict(cp["model_state_dict"], strict=True)
        order = np.random.RandomState(seed * 100000 + 10).permutation(len(triplets))
        records = []
        for number in range(3):
            batch = triplets[order[number * 1024 : (number + 1) * 1024]]
            outputs, grads, losses, alphas = [], [], [], []
            for enabled in (False, True):
                net.gnn.key_scale = (
                    torch.nn.Parameter(
                        torch.from_numpy(np.load(RVEC)).float().cuda(),
                        requires_grad=False,
                    )
                    if enabled
                    else None
                )
                net.zero_grad(set_to_none=True)
                output = net(graph)
                loss = InfoNCELoss(0.1)(*(output[batch[:, k]] for k in range(3)))
                loss.backward()
                grad = torch.cat(
                    [
                        p.grad.flatten()
                        for p in net.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                )
                assert torch.isfinite(grad).all()
                grads.append(grad.detach().cpu())
                losses.append(float(loss))
                outputs.append(output[:, 288:].detach().cpu())
                alphas.append(net.gnn._last_alpha.detach().cpu().numpy())
                del output, loss, grad
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            cosine = float(F.cosine_similarity(grads[0][None], grads[1][None]))
            records.append(
                {
                    "batch": number,
                    "triplet_sha256": array_hash(batch),
                    "loss_unscaled": losses[0],
                    "loss_scaled": losses[1],
                    "gradient_cosine": cosine,
                    "gradient_norm_ratio": float(grads[1].norm() / grads[0].norm()),
                    "context_exactly_equal": True,
                    "alpha": describe(alphas[0]),
                }
            )
        gradients[str(seed)] = records
        del net
    write_json(
        OUT / "diagnosis.json",
        {
            "frontend": frontend,
            "training_pairs": pair_stats,
            "gradients": gradients,
            "test_used": False,
            "protocol_sha256": sha256(OUT / "protocol.json"),
        },
    )
    print(
        "DIAGNOSED",
        pair_stats,
        {s: [r["gradient_cosine"] for r in v] for s, v in gradients.items()},
        flush=True,
    )


def train(seed: int) -> None:
    """기존 학습 루프에 고정 재계량만 추가한다."""
    BASE.PREPARED, BASE.GRAPH = PREPARED, GRAPH
    BASE.root, BASE.protocol, BASE.EPOCHS = root, protocol, EPOCHS
    BASE.model = lambda arm: model(arm, seed)
    TRAIN(seed, "aligned")
    spec = protocol()
    assert (
        read(root(seed, "aligned") / "initial_model.json")
        == spec["initial"][str(seed)]["aligned"]
    )
    complete = read(root(seed, "aligned") / "training_complete.json")
    assert (
        complete["orders"]
        == read(root(seed, "control") / "training_complete.json")["orders"]
    )
    for epoch in EPOCHS:
        cp = torch.load(
            root(seed, "aligned") / "checkpoints" / f"checkpoint_epoch_{epoch}.pth",
            map_location="cpu",
            weights_only=False,
        )
        np.testing.assert_array_equal(
            cp["model_state_dict"]["gnn.key_scale"].numpy(),
            np.load(RVEC).astype(np.float32),
        )


def evaluate(seed: int, epoch: int, sequence: str) -> None:
    """기존 최종 평가를 같은 입력과 검색 예산으로 실행한다."""
    source.root, source.model, source.protocol = root, model, protocol
    source.dimension = lambda arm: 288
    source.descriptors = lambda role, seq, arm: DESCRIPTORS(role, seq, "octave288")
    EVALUATE(seed, "aligned", epoch, sequence, "val")


def finish() -> None:
    """모든 원장과 지표를 재검산하고 사전 승격 기준을 적용한다."""
    spec = protocol(True)
    selected, all_rows, checks = {}, {}, []
    for seed in SEEDS:
        selected[str(seed)], all_rows[str(seed)] = {}, {}
        for arm in ARMS:
            candidates = []
            complete = read(root(seed, arm) / "training_complete.json")
            assert complete["updates"] == 900
            assert (
                complete["orders"]
                == read(root(seed, "control") / "training_complete.json")["orders"]
            )
            for epoch in EPOCHS:
                rows = {}
                for sequence in spec["val"]:
                    p = (
                        root(seed, arm)
                        / f"epoch{epoch}"
                        / (sequence.replace("/", "_") + ".npz")
                    )
                    info, row = read(p.with_suffix(".json")), BASE.exp.arrays(p)
                    assert sha256(p) == info["sha256"]
                    assert (
                        sha256(p.with_name("coverage_" + p.name))
                        == info["coverage_sha256"]
                    )
                    assert sha256(p.with_name("keys_" + p.name)) == info["keys_sha256"]
                    checkpoint = (
                        root(seed, arm)
                        / "checkpoints"
                        / f"checkpoint_epoch_{epoch}.pth"
                    )
                    assert (
                        sha256(checkpoint)
                        == info["checkpoint_sha256"]
                        == complete["checkpoints"][checkpoint.name]
                    )
                    ref = BASE.exp.arrays(
                        root(seed, "control") / f"epoch{epoch}" / p.name
                    )
                    for name in ("query_idx", "min_past_gt_dist"):
                        np.testing.assert_array_equal(row[name], ref[name])
                    assert (
                        (row["j_star"] >= 0) & (row["j_star"] <= row["query_idx"] - 30)
                    ).all()
                    assert all(
                        r["winner_equal"] and r["score_error"] == 0
                        for r in info["prefixes"]
                    )
                    metric = independent_metrics(row, row[METHOD])
                    for key in METRICS:
                        if metric[key] is not None:
                            np.testing.assert_allclose(
                                metric[key],
                                info["metrics"][METHOD][key],
                                atol=1e-12,
                                rtol=0,
                            )
                    rows[sequence] = metric
                    checks.append(
                        {
                            "path": str(p.relative_to(REPO)),
                            "sha256": sha256(p),
                            "rows": len(row["query_idx"]),
                            "prefixes": len(info["prefixes"]),
                        }
                    )
                candidates.append(
                    {"epoch": epoch, **BASE.macro(rows), "sequences": rows}
                )
            best = max(
                candidates, key=lambda r: (round(r["macro"]["AP"], 12), -r["epoch"])
            )
            selected[str(seed)][arm] = best
            all_rows[str(seed)][arm] = candidates
    stats = {}
    for arm in ARMS:
        stats[arm] = {
            m: {
                "mean": float(
                    np.mean([selected[str(s)][arm]["macro"][m] for s in SEEDS])
                ),
                "std": float(
                    np.std([selected[str(s)][arm]["macro"][m] for s in SEEDS], ddof=1)
                ),
            }
            for m in METRICS
        }
    deltas = {
        str(s): {
            m: 100
            * (
                selected[str(s)]["aligned"]["macro"][m]
                - selected[str(s)]["control"]["macro"][m]
            )
            for m in METRICS
        }
        for s in SEEDS
    }
    family_delta = {
        f: 100
        * float(
            np.mean(
                [
                    selected[str(s)]["aligned"]["families"][f]["AP"]
                    - selected[str(s)]["control"]["families"][f]["AP"]
                    for s in SEEDS
                ]
            )
        )
        for f in ("kitti", "helipr", "mulran")
    }
    mean_delta = {
        m: 100 * (stats["aligned"][m]["mean"] - stats["control"][m]["mean"])
        for m in METRICS
    }
    criteria = {
        "AP_mean": mean_delta["AP"] >= 0.1,
        "AP_all_seeds": all(v["AP"] > 0 for v in deltas.values()),
        "R100": mean_delta["R@100P"] >= -0.1,
        "family_AP": min(family_delta.values()) >= -0.25,
    }
    write_json(
        OUT / "summary.json",
        {
            "selected": selected,
            "all_epochs": all_rows,
            "stats": stats,
            "delta_pp": mean_delta,
            "per_seed_delta_pp": deltas,
            "family_AP_delta_pp": family_delta,
            "promotion_checks": criteria,
            "promoted": all(criteria.values()),
            "test_used": False,
            "protocol_sha256": sha256(OUT / "protocol.json"),
        },
    )
    write_json(
        OUT / "completion_validation.json",
        {
            "status": "passed",
            "ledgers": len(checks),
            "query_rows": sum(r["rows"] for r in checks),
            "prefix_checks": sum(r["prefixes"] for r in checks),
            "checks": checks,
            "summary_sha256": sha256(OUT / "summary.json"),
            "diagnosis_sha256": sha256(OUT / "diagnosis.json"),
            "new_updates": 2700,
            "test_used": False,
        },
    )
    print("FINAL", stats, mean_delta, family_delta, criteria, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "diagnose", "train", "val", "finish")
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action == "train" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare":
        prepare()
    elif args.action == "diagnose":
        diagnose()
    elif args.action == "train":
        train(args.seed)
    elif args.action == "val":
        evaluate(args.seed, args.epoch, args.sequence)
    else:
        finish()
