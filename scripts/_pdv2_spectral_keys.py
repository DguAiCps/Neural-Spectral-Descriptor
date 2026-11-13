#!/usr/bin/env python3
"""집계288·절단288·절단352를 같은 차이 문맥과 최종 판정으로 대조한다."""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_derivatives as base
import _pdv2_spectral_keys_cache as cache_io
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from _train_attention_neighbors import array_hash
from encoding.frequency_key import frequency_key
from gnn.geometry_interaction import attach_geometry_interaction
from gnn.model import create_spectral_gnn

OUT = cache_io.OUT
ARMS = ("octave288", "trunc288", "trunc352")
SEEDS = (1, 2, 3)
EPOCHS = (10, 20)
METHOD = "arc_unique_b100"
METRICS = ("AP", "max_F1", "R@100P", "R@1")
ORIGINAL_PREPARED, ORIGINAL_GRAPH = base.PREPARED, base.GRAPH
ORIGINAL_TRAIN = base.train
STANDARD_CONFIG = base.exp.old.OUT / "standard.yaml"
ORIGINAL_RVEC = base.exp.old.base.RVEC
read = base.read


def dimension(arm: str) -> int:
    """원 특징의 차원을 반환한다."""
    assert arm in ARMS
    return 352 if arm == "trunc352" else 288


def root(seed: int, arm: str) -> Path:
    """시드와 특징 조건의 독립 결과 경로다."""
    return OUT / f"s{seed}" / arm


def factory(dim: int) -> torch.nn.Module:
    """차이 K/V-tanh와 기하 상호작용의 시간 전용 모델을 만든다."""
    cfg = yaml.safe_load(STANDARD_CONFIG.read_text())
    net = create_spectral_gnn(
        input_dim=dim,
        hidden_dim=128,
        context_dim=128,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        norm_type="batch_norm",
        use_residual_gate=True,
        gate_hidden_dim=64,
        gate_initial_alpha=0.0625,
        diffattn_value_source="diff",
        edge_encoder_config=cfg["gnn"]["edge_encoding"],
    )
    attach_geometry_interaction(net, "interaction")
    return net


def model(arm: str, seed: int) -> torch.nn.Module:
    """입력 크기 외의 공통 텐서를 동일한 시드 초기값으로 맞춘다."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        canonical = factory(288)
        if dimension(arm) == 288:
            return canonical
        torch.manual_seed(seed)
        net = factory(352)
        ref, own = canonical.state_dict(), net.state_dict()
        different = []
        for name, value in own.items():
            if value.shape == ref[name].shape:
                own[name] = ref[name].clone()
            else:
                different.append(name)
        assert different == ["gnn.input_proj.weight"], different
        net.load_state_dict(own, strict=True)
        return net


def descriptors(role: str, sequence: str, arm: str) -> tuple:
    """같은 키프레임 순서의 원 특징과 원본 자료를 읽는다."""
    original = base.exp.arrays(cache_io.source_path(role, sequence))
    if arm == "octave288":
        value = original["descriptors"].astype(np.float32)
    else:
        path = cache_io.output_path(role, sequence)
        info = read(path.with_suffix(".json"))
        assert sha256(path) == info["sha256"]
        assert info["cache_protocol_sha256"] == sha256(OUT / "cache_protocol.json")
        assert sha256(cache_io.source_path(role, sequence)) == info["source_sha256"]
        with np.load(path) as z:
            np.testing.assert_array_equal(z["scan_ids"], original["scan_ids"])
            value = frequency_key(z["low22"], 22 if arm == "trunc352" else 18)
    assert value.shape == (len(original["scan_ids"]), dimension(arm))
    assert np.isfinite(value).all()
    return value, original


def protocol(check_inputs: bool = False) -> dict:
    """고정한 소스와 선택적으로 학습·검증 입력을 검사한다."""
    spec = read(OUT / "protocol.json")
    for group in (("runtime", "inputs") if check_inputs else ("runtime",)):
        for name, digest in spec[group].items():
            assert sha256(REPO / name) == digest, name
    return spec


def prepare() -> None:
    """시험을 읽기 전에 학습·선택·최종 판정 조건을 고정한다."""
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    parent = read(REPO / "results/pdv2_geometry_interaction_20261023/protocol.json")
    cache_spec = cache_io.freeze_cache()
    state = torch.load(ORIGINAL_PREPARED, map_location="cpu", weights_only=False)
    original_graph = torch.load(ORIGINAL_GRAPH, map_location="cpu", weights_only=False)
    graph = base.temporal_subgraph(original_graph, np.arange(original_graph.num_nodes))
    assert graph.num_nodes == 72306 and (graph.edge_type == 0).all()
    inputs = {
        ORIGINAL_PREPARED,
        ORIGINAL_GRAPH,
        STANDARD_CONFIG,
        ORIGINAL_RVEC,
        OUT / "cache_protocol.json",
        base.SAMPLES / "audit.json",
        base.SAMPLES / "training_inputs.npz",
    }
    for epoch in (1, 6, 11, 16):
        inputs.add(base.SAMPLES / f"original_epoch{epoch}.npz")
        assert base.triplets(epoch).shape == (183249, 3)
    matrices, graph_hashes, remetric = {}, {}, {}
    train_arrays = {}
    for arm in ARMS:
        pieces = []
        for number, sequence in enumerate(cache_spec["train"]):
            value, original = descriptors("train", sequence, arm)
            ids = np.flatnonzero(state["sequences"] == number)
            np.testing.assert_array_equal(
                state["descriptors"][ids], original["descriptors"]
            )
            np.testing.assert_array_equal(state["poses"][ids], original["poses"])
            assert len(ids) == len(value)
            pieces.append(value)
            train_arrays[(arm, sequence)] = value.astype(np.float64)
        matrices[arm] = np.concatenate(pieces)
    np.testing.assert_array_equal(matrices["octave288"], state["descriptors"])
    # 원 재계량의 파일 순서·엄격한 <5 m·q-30 미포함·가장 가까운 쌍을 재현한다.
    pairs = {}
    for sequence in sorted(
        cache_spec["train"], key=lambda s: str(cache_io.source_path("train", s))
    ):
        original = base.exp.arrays(cache_io.source_path("train", sequence))
        xyz = original["poses"][:, :3, 3].astype(np.float64)
        tree, selected = cKDTree(xyz), []
        for q in range(30, len(xyz)):
            ids = np.asarray(sorted(tree.query_ball_point(xyz[q], 5.0)), np.int64)
            ids = ids[ids < q - 30]
            if not len(ids):
                continue
            distance = np.linalg.norm(xyz[ids] - xyz[q], axis=1)
            ids = ids[distance < 5.0]
            if len(ids):
                j = ids[np.argmin(np.linalg.norm(xyz[ids] - xyz[q], axis=1))]
                selected.append((q, int(j)))
        pairs[sequence] = np.asarray(selected, np.int64).reshape(-1, 2)
    np.savez_compressed(
        OUT / "remetric_pairs.npz", **{s.replace("/", "_"): p for s, p in pairs.items()}
    )
    inputs.add(OUT / "remetric_pairs.npz")
    for arm in ARMS:
        deltas = np.concatenate(
            [
                train_arrays[(arm, s)][p[:, 0]] - train_arrays[(arm, s)][p[:, 1]]
                for s, p in pairs.items()
            ]
        )
        variance = 0.5 * np.mean(deltas**2, axis=0)
        scale = 1.0 / np.sqrt(variance + 1e-6 * variance.mean())
        scale /= scale.mean()
        if arm == "octave288":
            np.testing.assert_array_equal(scale, np.load(ORIGINAL_RVEC))
        p = OUT / f"remetric_{arm}.npy"
        np.save(p, scale)
        inputs.add(p)
        remetric[arm] = {
            "path": str(p.relative_to(REPO)),
            "sha256": sha256(p),
            "pairs": len(deltas),
            "dimension": len(scale),
        }
        modified = dict(state)
        modified["descriptors"] = matrices[arm]
        arm_graph = graph.clone()
        arm_graph.x = torch.from_numpy(matrices[arm])
        modified["graph"] = arm_graph
        pp, gp = OUT / f"prepared_{arm}.pt", OUT / f"graph_{arm}.pt"
        torch.save(modified, pp)
        torch.save(arm_graph, gp)
        inputs.update((pp, gp))
        graph_hashes[arm] = {
            k: array_hash(getattr(arm_graph, k).numpy())
            for k in ("x", "edge_index", "edge_attr", "edge_type")
        }
    counts, initial = {}, {}
    for seed in SEEDS:
        initial[str(seed)] = {}
        for arm in ARMS:
            net = model(arm, seed)
            counts[arm] = sum(p.numel() for p in net.parameters())
            initial[str(seed)][arm] = {
                k: array_hash(v.numpy()) for k, v in net.state_dict().items()
            }
        assert initial[str(seed)]["octave288"] == initial[str(seed)]["trunc288"]
        for k, h in initial[str(seed)]["octave288"].items():
            if k != "gnn.input_proj.weight":
                assert h == initial[str(seed)]["trunc352"][k]
    for role in ("train", "val"):
        for sequence in cache_spec[role]:
            p = cache_io.output_path(role, sequence)
            inputs.add(cache_io.source_path(role, sequence))
            if role == "train":
                info = read(p.with_suffix(".json"))
                assert sha256(p) == info["sha256"]
                inputs.update((p, p.with_suffix(".json")))
            if role == "val":
                inputs.update(base.exp.old.base.inputs(role, sequence)[2].values())
                for seed in SEEDS:
                    inputs.add(
                        base.fixed.VAL
                        / f"s{seed}"
                        / ("standard_" + sequence.replace("/", "_") + ".npz")
                    )
    runtime = dict(parent["runtime"])
    runtime.update(cache_spec["runtime"])
    for p in (
        Path(__file__),
        REPO / "scripts/_pdv2_spectral_keys_run.py",
        REPO / "scripts/_pdv2_spectral_keys_finish.py",
        REPO / "scripts/_pdv2_context_tail_finish.py",
    ):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(
        OUT / "protocol.json",
        {
            "created_unix": time.time(),
            "frontend": "모든조건동일360방향·16높이행. log_magnitude=false, "
            "zero_center=false, normalize_channels=false. 절단은DC포함0..17/0..21. "
            "cache_protocol의로그크기변환표현은동일인코더설정분기를뜻하며실제로로그변환하지않음.",
            "arms": ARMS,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "train": cache_spec["train"],
            "val": cache_spec["val"],
            "test": cache_spec["test"],
            "confidence": parent["confidence"],
            "parameters": counts,
            "initial": initial,
            "graph_hashes": graph_hashes,
            "remetric": remetric,
            "method": METHOD,
            "training": "GT학습8·같은183249삼중쌍/에포크순서·20에포크900갱신. "
            "Adam5e-4/1e-5·InfoNCE·AMP·3epoch warmup/cosine. no_interdiff·gate초기.0625. "
            "차이K/V-tanh+기하상호작용·시간±5·2층·4head·문맥128. 352는입력투영만차원/초기값다름.",
            "selection": "각시드/각조건10·20에포크의동일최종검증AP(시퀀스→3계열)최대. "
            "1e-12반올림동률이면이른에포크. 재계량은동일학습쌍/기존공식으로조건별적합. "
            "융합가중치와최종판정채널은기존시드별값으로모든조건동일하게고정.",
            "test_policy": "선택완료후9모델과소스를동결하고46시퀀스평가. "
            "KISS·q+15·j<=q-30·100+100후보를조건별다시검색. 최종arc_unique_b100. "
            "표2와동일195726질의·45양성시퀀스→6도시→3시드평균/표본표준편차. "
            "세조건모두보고하며시험성능으로설정을선택하지않음. 기존시험개발노출한계유지.",
            "comparison": "BEVPlace++공식재학습3시드공통질의원장과대조. "
            "원고기존NSD행과현재연구대조를구분. 원고자동교체안함.",
            "test_used_for_selection": False,
            "runtime": runtime,
            "inputs": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)},
        },
    )
    print("PREPARED_SPECTRAL_KEYS", counts, remetric, flush=True)


def train(seed: int, arm: str) -> None:
    """기존 검증된 학습 루프에 조건별 특징·그래프만 공급한다."""
    spec = protocol()
    base.PREPARED, base.GRAPH = OUT / f"prepared_{arm}.pt", OUT / f"graph_{arm}.pt"
    base.root = root
    base.EPOCHS = EPOCHS
    base.model = lambda selected: model(selected, seed)
    base.protocol = lambda *args: {
        **protocol(),
        "graph_hashes": spec["graph_hashes"][arm],
    }
    ORIGINAL_TRAIN(seed, arm)
    assert (
        read(root(seed, arm) / "initial_model.json") == spec["initial"][str(seed)][arm]
    )


def make_keys(
    net: torch.nn.Module, features: np.ndarray, poses: np.ndarray, end: int = None
) -> np.ndarray:
    """현재 주행 추정 자세의 시간 간선으로 조건별 검색 키를 만든다."""
    x, pose = features[:end], poses[:end]
    graph = base.make_graph(x, pose, np.zeros(len(x), np.int64))
    with torch.no_grad():
        key = base.normalize(net(graph.cuda()).float().cpu().numpy())
    assert key.shape == (len(x), x.shape[1] + 128) and np.isfinite(key).all()
    return key


def evaluate(seed: int, arm: str, epoch: int, sequence: str, role: str) -> None:
    """최종 후보·승자·시간 판정을 모두 다시 계산한다."""
    spec = protocol()
    if role == "test":
        frozen = read(OUT / "test_freeze.json")
        assert frozen["protocol_sha256"] == sha256(OUT / "protocol.json")
        assert epoch == frozen["models"][str(seed)][arm]["epoch"]
    folder = root(seed, arm) / (f"epoch{epoch}" if role == "val" else "test")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (sequence.replace("/", "_") + ".npz")
    kp, cp = target.with_name("keys_" + target.name), target.with_name(
        "coverage_" + target.name
    )
    if target.with_suffix(".json").exists():
        info = read(target.with_suffix(".json"))
        for p, field in (
            (target, "sha256"),
            (kp, "keys_sha256"),
            (cp, "coverage_sha256"),
        ):
            assert sha256(p) == info[field]
        return
    checkpoint = root(seed, arm) / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
    assert (
        sha256(checkpoint)
        == read(root(seed, arm) / "training_complete.json")["checkpoints"][
            checkpoint.name
        ]
    )
    net = model(arm, seed).cuda().eval()
    net.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ],
        strict=True,
    )
    net.gnn.key_scale = torch.nn.Parameter(
        torch.from_numpy(np.load(REPO / spec["remetric"][arm]["path"])).float().cuda(),
        requires_grad=False,
    )
    features, cache = descriptors(role, sequence, arm)
    sensor, seq, paths, min_past = base.exp.old.base.inputs(role, sequence)
    started = time.time()
    key = make_keys(net, features, cache["poses"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = base.exp.old.base.bank_for(base.exp.OP, z)
    conf = spec["confidence"][str(seed)]
    ledger = evaluate_keys(
        key, bank, cache["poses_gt"][:, :3, 3], min_past, {seed: tuple(conf["tuple"])}
    )[seed]
    keep = ledger["query_idx"] < len(key) - 15
    ledger = {k: v[keep] for k, v in ledger.items()}
    refpath = (
        base.fixed.VAL / f"s{seed}/standard_{sensor}_{seq}.npz"
        if role == "val"
        else base.fixed.OUT / f"test/s{seed}/{sensor}_{seq}.npz"
    )
    reference = base.exp.arrays(refpath)
    for name in ("query_idx", "min_past_gt_dist"):
        np.testing.assert_array_equal(ledger[name], reference[name])
    ledger["group"] = reference["group"]
    q, j = ledger["query_idx"], ledger["j_star"]
    assert ((j >= 0) & (j <= q - 30)).all()
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    chunks = [
        base.fixed.frozen_scores(
            bank,
            travel,
            q[i : i + 256],
            j[i : i + 256],
            conf["decision"],
            ledger[conf["decision"]][i : i + 256],
        )
        for i in range(0, len(q), 256)
    ]
    scores = {k: np.concatenate([r[k] for r in chunks]) for k in ("raw", METHOD)}
    np.testing.assert_allclose(
        ledger["top1_geo_dist"],
        np.linalg.norm(
            cache["poses_gt"][q, :3, 3] - cache["poses_gt"][j, :3, 3], axis=1
        ),
        atol=2e-5,
        rtol=2e-6,
    )
    prefixes = []
    for ix in (len(q) // 3, 2 * len(q) // 3):
        t, end = int(q[ix]), int(q[ix] + 16)
        small = make_keys(net, features, cache["poses"], end)
        np.testing.assert_allclose(small[: t + 1], key[: t + 1], atol=4e-6, rtol=4e-5)
        ids = np.unique(
            np.r_[
                base.prefix_topk(small, t, t - 29, 100),
                base.prefix_topk(bank[0], t, t - 29, 100),
            ]
        )
        channels = (
            small[ids] @ small[t],
            bank[0][ids] @ bank[0][t],
            base.align_scores(bank[1], bank[2], t, ids),
            base.column_align_scores(bank[3], t, ids),
        )
        fused = sum(float(w) * base.minmax(v) for w, v in zip(conf["tuple"], channels))
        winner = int(ids[np.argsort(-fused, kind="stable")[0]])
        assert winner == int(j[ix]), (sequence, t, winner, j[ix])
        tiny = base.fixed.frozen_scores(
            base.exp.truncate_bank(bank, end),
            travel[:end],
            q[ix : ix + 1],
            j[ix : ix + 1],
            conf["decision"],
            ledger[conf["decision"]][ix : ix + 1],
        )
        for name in scores:
            np.testing.assert_array_equal(tiny[name], scores[name][ix : ix + 1])
        prefixes.append(
            {
                "q": t,
                "key_max_error": float(abs(small[: t + 1] - key[: t + 1]).max()),
                "winner_equal": True,
                "score_error": 0.0,
            }
        )
    metrics = {name: base.exp.metrics(value, ledger) for name, value in scores.items()}
    for name, value in scores.items():
        independently = independent_metrics(ledger, value)
        for m in (*METRICS, "n_pos"):
            if metrics[name][m] is None:
                assert independently[m] is None
            else:
                np.testing.assert_allclose(
                    metrics[name][m], independently[m], atol=1e-12, rtol=0
                )
    np.savez_compressed(target, **{k: ledger[k] for k in base.FIELDS}, **scores)
    np.savez_compressed(kp, embedding=key, scan_ids=cache["scan_ids"])
    np.savez_compressed(cp, **ledger)
    write_json(
        target.with_suffix(".json"),
        {
            "sha256": sha256(target),
            "keys_sha256": sha256(kp),
            "coverage_sha256": sha256(cp),
            "metrics": metrics,
            "prefixes": prefixes,
            "seconds": time.time() - started,
            "checkpoint_sha256": sha256(checkpoint),
            "input_dimension": dimension(arm),
            "test_used": role == "test",
            "test_used_for_selection": False,
        },
    )
    print(
        "EVALUATED",
        role,
        seed,
        arm,
        epoch,
        sequence,
        round(time.time() - started, 2),
        flush=True,
    )


def select_and_freeze() -> None:
    """검증 최종 AP만으로 선택한 모든 조건을 시험 전에 고정한다."""
    spec = protocol(True)
    if (OUT / "test_freeze.json").exists():
        assert read(OUT / "test_freeze.json")["protocol_sha256"] == sha256(
            OUT / "protocol.json"
        )
        return
    results, models, inputs = {}, {}, {}
    for seed in SEEDS:
        results[str(seed)], models[str(seed)] = {}, {}
        orders = None
        for arm in ARMS:
            folder = root(seed, arm)
            complete = read(folder / "training_complete.json")
            assert complete["updates"] == 900
            assert orders is None or orders == complete["orders"]
            orders = complete["orders"]
            candidates = []
            for epoch in EPOCHS:
                rows = {
                    s: read(folder / f"epoch{epoch}" / (s.replace("/", "_") + ".json"))
                    for s in spec["val"]
                }
                candidates.append(
                    {
                        "epoch": epoch,
                        **{
                            m: base.macro({s: r["metrics"][m] for s, r in rows.items()})
                            for m in ("raw", METHOD)
                        },
                    }
                )
            chosen = max(
                candidates,
                key=lambda r: (round(r[METHOD]["macro"]["AP"], 12), -r["epoch"]),
            )
            results[str(seed)][arm] = {"selected": chosen, "candidates": candidates}
            ck = folder / "checkpoints" / f"checkpoint_epoch_{chosen['epoch']}.pth"
            assert sha256(ck) == complete["checkpoints"][ck.name]
            models[str(seed)][arm] = {
                "epoch": chosen["epoch"],
                "checkpoint": str(ck.relative_to(REPO)),
                "sha256": sha256(ck),
            }
            inputs[str(ck.relative_to(REPO))] = sha256(ck)
            for p in folder.glob("epoch*/*"):
                inputs[str(p.relative_to(REPO))] = sha256(p)
    write_json(
        OUT / "validation_summary.json", {"per_seed": results, "test_used": False}
    )
    baseline = REPO / "results/pdv2_baseline_seeds/summary.json"
    inputs[str(baseline.relative_to(REPO))] = sha256(baseline)
    bevplace = read(baseline)["methods"]["bevplace"]
    for sequence in spec["val"]:
        p = cache_io.output_path("val", sequence)
        info = read(p.with_suffix(".json"))
        assert sha256(p) == info["sha256"]
        assert info["cache_protocol_sha256"] == sha256(OUT / "cache_protocol.json")
        inputs[str(p.relative_to(REPO))] = sha256(p)
        inputs[str(p.with_suffix(".json").relative_to(REPO))] = sha256(
            p.with_suffix(".json")
        )
    for sequence in spec["test"]:
        sensor, seq, paths, _ = base.exp.old.base.inputs("test", sequence)
        for p in paths.values():
            inputs[str(p.relative_to(REPO))] = sha256(p)
        for seed in SEEDS:
            p = base.fixed.OUT / f"test/s{seed}/{sensor}_{seq}.npz"
            inputs[str(p.relative_to(REPO))] = sha256(p)
            p = (
                REPO
                / bevplace["per_seed"][str(seed)]["source"]
                / (f"dump_bevplace_{sensor}_{seq}.npz")
            )
            inputs[str(p.relative_to(REPO))] = sha256(p)
    write_json(
        OUT / "test_freeze.json",
        {
            "created_unix": time.time(),
            "models": models,
            "protocol_sha256": sha256(OUT / "protocol.json"),
            "validation_sha256": sha256(OUT / "validation_summary.json"),
            "inputs": inputs,
            "test": spec["test"],
            "cities": base.fixed.CITY,
            "test_used_for_selection": False,
            "baseline_summary": str(baseline.relative_to(REPO)),
        },
    )
    print("TEST_FROZEN", models, flush=True)


def main() -> None:
    """개별 준비·학습·평가·선택 작업을 실행한다."""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "train", "eval", "select"))
    p.add_argument("--seed", type=int, choices=SEEDS, default=1)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--epoch", type=int, choices=EPOCHS)
    p.add_argument("--role", choices=("val", "test"), default="val")
    p.add_argument("--sequence")
    a = p.parse_args()
    torch.set_num_threads(4 if a.action == "train" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if a.action == "prepare":
        prepare()
    elif a.action == "train":
        train(a.seed, a.arm)
    elif a.action == "eval":
        evaluate(a.seed, a.arm, a.epoch, a.sequence, a.role)
    else:
        select_and_freeze()


if __name__ == "__main__":
    main()
