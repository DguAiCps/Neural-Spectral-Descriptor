#!/usr/bin/env python3
"""선택된 시간 모델의 메시지 상쇄와 고정 최종 오류의 관계를 진단한다."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import softmax

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_nonlinear_difference as previous
from _pdv2_no_context_ablation import sha256, write_json
from gnn.message_cancellation import cancellation_statistics
from gnn.nonlinear_relation import NonlinearRelationConv

OUT = REPO / "results/pdv2_message_cancellation_20261021"
ARMS = ("diff", "diff_tanh", "standard", "standard_tanh")
FIELDS = ("mass", "surviving", "cancellation", "within", "between", "side_cosine", "valid", "both_sides")
run = previous.base


def describe(values: np.ndarray) -> dict:
    """유한한 수치의 개수와 분포를 요약한다."""
    x = np.asarray(values, dtype=np.float64)
    return {"n": len(x), "mean": float(x.mean()) if len(x) else None,
            "quantiles": np.quantile(x, [0, .1, .5, .9, 1]).tolist() if len(x) else []}


def capture(net: torch.nn.Module, graph) -> tuple:
    """실제 층 입력·출력에 훅을 걸어 메시지를 재계산하고 검산한다."""
    measured, errors = {}, []

    def hook(index):
        def measure(conv, args, kwargs, output):
            h, edge = args[:2]
            src, tgt = edge
            relation = h[src] - h[tgt] if conv.value_source == "diff" else h[src]
            q = conv.W_q(h[tgt]).view(-1, conv.heads, conv.head_dim)
            k = conv.W_k(relation).view(-1, conv.heads, conv.head_dim)
            if isinstance(conv, NonlinearRelationConv):
                k = k.tanh()
            scores = torch.einsum("ehd,ehd->eh", q, k) / conv.scale
            attr = args[2] if len(args) > 2 else kwargs.get("edge_attr")
            if attr is not None:
                scores = scores + conv.bias_mlp(attr)
            if kwargs.get("edge_logit") is not None:
                scores = scores + kwargs["edge_logit"]
            assert kwargs.get("edge_value_gate") is None
            attention = softmax(scores, tgt, num_nodes=len(h))
            value = conv.W_v(relation).view(-1, conv.heads, conv.head_dim)
            messages = (attention[..., None] * value).reshape(-1, conv.channels)
            stats = cancellation_statistics(messages, edge, len(h))
            torch.testing.assert_close(stats["total"], output, atol=2e-5, rtol=2e-5)
            error = float((stats["total"] - output).abs().max())
            errors.append(error)
            torch.testing.assert_close(stats["cancellation"], stats["within"] + stats["between"],
                                       atol=1e-6, rtol=0)
            measured[index + 1] = {k: stats[k].cpu().numpy() for k in FIELDS}
        return measure

    handles = [conv.register_forward_hook(hook(i), with_kwargs=True)
               for i, conv in enumerate(net.gnn.convs)]
    try:
        with torch.no_grad():
            key = run.normalize(net(graph).float().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return measured, key, max(errors)


def main() -> None:
    """학습·검증 진단을 고정하고 오류와의 연관성을 같은 구간 안에서 비교한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = previous.protocol(True)
    old_summary = previous.read(previous.OUT / "summary.json")
    sources = {**spec["runtime"], **spec["inputs"]}
    tau = run.fixed.choose_tau(1)["tau"]
    for p in (Path(__file__), REPO / "src/gnn/message_cancellation.py",
              REPO / "tests/gnn/test_message_cancellation.py", previous.OUT / "summary.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for arm in ARMS:
        epoch = old_summary["per_seed"]["1"][arm]["selected"]["epoch"]
        checkpoint = run.root(1, arm) / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        expected = previous.read(run.root(1, arm) / "training_complete.json")["checkpoints"][checkpoint.name]
        assert sha256(checkpoint) == expected
        sources[str(checkpoint.relative_to(REPO))] = expected
        for sequence in spec["val"]:
            p = run.root(1, arm) / f"epoch{epoch}" / (sequence.replace("/", "_") + ".npz")
            info = previous.read(p.with_suffix(".json"))
            for path, expected in ((p, info["sha256"]), (p.with_name("keys_" + p.name), info["keys_sha256"])):
                assert sha256(path) == expected
                sources[str(path.relative_to(REPO))] = expected
    design = {"created_unix": time.time(), "sources": sources, "test_used": False,
        "models": ARMS, "definition": "C=1-L1(sum aV)/sum L1(aV). 방향 내부와 과거/미래 사이의 L1 상쇄로 가산 분해. 정보 손실률과 다름.",
        "outcome": "고정 검증 최종 승자의 양성 정답/오답. 주대조는 diff_tanh의 표준_tanh 대비 잃은 정답 대 유지 정답, 마지막 층.",
        "final_acceptance": "기존 표준 모델에서 이미 고른 공통 검증 최대 F1 임계값으로 참수락/오수락/누락을 함께 진단. 새 임계값 선택 없음.",
        "fixed_tau": tau,
        "strata": "시퀀스×재방문방향×회전(±5 yaw최대>=5도)×학습메시지질량4분위. 두 집단 각각>=3인 구간에서 min(n오답,n정답) 가중평균.",
        "followup_gate": "주조건 마지막 층의 과거/미래 상쇄가 학습8 중 2이상에서 중앙값>=.1. 층화한 잃은 정답의 상쇄 증가>=.02, 포함 손실>=20개, 포함된 2이상 계열에서 양수이면 같은 용량 값처리 대조로 진행. 이는 탐색 진입 기준이며 유의성/인과성 기준 아님.",
        "scope": "새 학습·선택·최종 성능 재평가 없음. 기존 원장과 체크포인트의 설명 진단. 시험 사용 없음."}
    freeze = OUT / "protocol.json"
    if freeze.exists():
        old = previous.read(freeze)
        assert old["sources"] == sources
    else:
        write_json(freeze, design)
    original = torch.load(run.GRAPH, map_location="cpu", weights_only=False)
    train_graph = run.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    train_state = torch.load(run.PREPARED, map_location="cpu", weights_only=False)
    train_sequences = np.asarray(train_state["sequences"])
    train_report, val_report, rows = {}, {}, {}
    max_output_error = max_key_error = max_prefix_error = 0.
    prefix_checks = 0
    for arm in ARMS:
        epoch = old_summary["per_seed"]["1"][arm]["selected"]["epoch"]
        net = previous.model(arm).cuda().eval()
        checkpoint = run.root(1, arm) / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(run.exp.old.base.RVEC)).float().cuda(), requires_grad=False)
        measured, _, error = capture(net, train_graph)
        max_output_error = max(max_output_error, error)
        train_report[arm] = {}
        edges = {}
        for layer, values in measured.items():
            edges[layer] = np.quantile(values["mass"][values["valid"]], [.25, .5, .75])
            train_report[arm][str(layer)] = {str(seq): {k: describe(v[train_sequences == seq])
                for k, v in values.items() if v.dtype != bool} for seq in np.unique(train_sequences)}
        path = OUT / f"train_{arm}.npz"
        np.savez_compressed(path, sequence=train_sequences, **{f"l{l}_{k}": v for l, vs in measured.items() for k, v in vs.items()})
        sources[str(path.relative_to(REPO))] = sha256(path)
        val_report[arm], rows[arm] = {}, {}
        print("TRAIN_DIAG", arm, flush=True)
        for sequence in spec["val"]:
            _, _, paths, _ = run.exp.old.base.inputs("val", sequence)
            cache = run.exp.arrays(paths["cache"])
            graph = run.make_graph(cache["descriptors"].astype(np.float32), cache["poses"],
                                   np.zeros(len(cache["descriptors"]), np.int64)).cuda()
            measured, key, error = capture(net, graph)
            max_output_error = max(max_output_error, error)
            p = run.root(1, arm) / f"epoch{epoch}" / (sequence.replace("/", "_") + ".npz")
            ledger = run.exp.arrays(p)
            expected = run.exp.arrays(p.with_name("keys_" + p.name))["embedding"]
            np.testing.assert_allclose(key, expected, atol=4e-6, rtol=4e-5)
            max_key_error = max(max_key_error, float(abs(key - expected).max()))
            q, j = ledger["query_idx"], ledger["j_star"]
            yaw = np.arctan2(cache["poses"][:, 1, 0], cache["poses"][:, 0, 0])
            heading = np.stack([np.angle(np.exp(1j * (yaw[q + d] - yaw[q]))) for d in range(-5, 6)])
            turning = abs(heading).max(0) >= np.deg2rad(5)
            packed = {**ledger, "turning": turning}
            for layer, values in measured.items():
                for name, v in values.items():
                    packed[f"l{layer}_{name}"] = v[q]
                    packed[f"j_l{layer}_{name}"] = v[j]
                packed[f"l{layer}_mass_bin"] = np.digitize(values["mass"][q], edges[layer])
            path = OUT / (arm + "_" + sequence.replace("/", "_") + ".npz")
            np.savez_compressed(path, **packed)
            sources[str(path.relative_to(REPO))] = sha256(path)
            rows[arm][sequence] = packed
            pos = ledger["min_past_gt_dist"] <= 5
            correct = ledger["top1_geo_dist"] <= 5
            negative = ledger["min_past_gt_dist"] > 25
            accepted = ledger[run.METHOD] >= tau
            val_report[arm][sequence] = {group: {str(layer): {name: describe(packed[f"l{layer}_{name}"][mask])
                for name in ("cancellation", "within", "between", "mass", "side_cosine")}
                for layer in (1, 2)} for group, mask in {
                    "positive_correct": pos & correct, "positive_wrong": pos & ~correct,
                    "negative": negative, "true_accept": pos & correct & accepted,
                    "false_accept": ((pos & ~correct) | negative) & accepted,
                    "missed_positive": pos & (~correct | ~accepted)}.items()}
            if arm == "diff_tanh":
                for t in (int(q[len(q)//3]), int(q[2*len(q)//3])):
                    small_graph = run.make_graph(cache["descriptors"][:t+16].astype(np.float32),
                        cache["poses"][:t+16], np.zeros(t+16, np.int64)).cuda()
                    small, small_key, _ = capture(net, small_graph)
                    np.testing.assert_allclose(small_key[t], key[t], atol=4e-6, rtol=4e-5)
                    for layer in (1, 2):
                        for field in ("cancellation", "within", "between"):
                            delta = abs(float(small[layer][field][t]) - float(measured[layer][field][t]))
                            assert delta < 2e-5
                            max_prefix_error = max(max_prefix_error, delta)
                    prefix_checks += 1
                    del small_graph
            print("VAL_DIAG", arm, sequence, flush=True)
            del graph
        del net
    associations = {}
    for arm, reference in (("diff", "standard"), ("diff_tanh", "standard_tanh")):
        associations[arm] = {}
        for layer in (1, 2):
            groups, matched = {}, []
            for sequence, r in rows[arm].items():
                ref = rows[reference][sequence]
                np.testing.assert_array_equal(r["query_idx"], ref["query_idx"])
                eligible = (r["min_past_gt_dist"] <= 5) & (ref["top1_geo_dist"] <= 5)
                lost = eligible & (r["top1_geo_dist"] > 5)
                kept = eligible & (r["top1_geo_dist"] <= 5)
                groups[sequence] = {g: {k: describe(r[f"l{layer}_{k}"][mask])
                    for k in ("cancellation", "between", "within", "mass")}
                    for g, mask in (("lost", lost), ("kept", kept))}
                strata = np.stack((r["group"], r["turning"], r[f"l{layer}_mass_bin"]), axis=1)
                for values in np.unique(strata[eligible], axis=0):
                    cell = (strata == values).all(1)
                    a, b = lost & cell, kept & cell
                    if min(a.sum(), b.sum()) < 3:
                        continue
                    matched.append({"sequence": sequence, "stratum": values.tolist(), "lost": int(a.sum()),
                        "kept": int(b.sum()), "weight": int(min(a.sum(), b.sum())),
                        **{k: float(r[f"l{layer}_{k}"][a].mean() - r[f"l{layer}_{k}"][b].mean())
                           for k in ("cancellation", "between", "within")}})
            family_delta = {family: float(np.average([v["between"] for v in matched if v["sequence"].startswith(family)],
                weights=[v["weight"] for v in matched if v["sequence"].startswith(family)]))
                for family in ("kitti", "helipr", "mulran") if any(v["sequence"].startswith(family) for v in matched)}
            associations[arm][str(layer)] = {"per_sequence": groups, "strata": matched,
                "matched_lost": sum(v["lost"] for v in matched), "family_between_difference": family_delta,
                "weighted_between_difference": float(np.average([v["between"] for v in matched], weights=[v["weight"] for v in matched])) if matched else None}
    primary = associations["diff_tanh"]["2"]
    support = sum(v["between"]["quantiles"][2] >= .1 for v in train_report["diff_tanh"]["2"].values())
    checks = {"train_cross_cancellation": support >= 2, "matched_loss_count": primary["matched_lost"] >= 20,
        "matched_between_difference": primary["weighted_between_difference"] is not None and primary["weighted_between_difference"] >= .02,
        "two_families_positive": sum(v > 0 for v in primary["family_between_difference"].values()) >= 2}
    write_json(OUT / "summary.json", {"train": train_report, "val": val_report, "associations": associations,
        "followup_checks": checks, "followup_eligible": all(checks.values()),
        "validation_queries_per_model": sum(len(r["query_idx"]) for r in rows["diff_tanh"].values()),
        "max_output_reconstruction_error": max_output_error, "max_saved_key_error": max_key_error,
        "prefix_checks": prefix_checks, "max_prefix_cancellation_error": max_prefix_error,
        "protocol_sha256": sha256(freeze), "sources": sources, "test_used": False})
    print("FOLLOWUP", checks, "PRIMARY", primary["weighted_between_difference"], primary["family_between_difference"], flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
