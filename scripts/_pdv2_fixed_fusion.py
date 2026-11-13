#!/usr/bin/env python3
"""최종 융합·수락 점수식을 고정한 채 표현의 순위 손실만 비교한다."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_linear_fusion as reuse
from retrieval.fixed_fusion_learning import fixed_fusion_scores, positive_ranking_loss
from _pdv2_no_context_ablation import sha256, write_json
from _pr_eval_lib import classify_queries, pr_sweep, operating_point
from _xs_eval_lib import minmax

base = reuse.base
PREVIOUS = reuse.PREVIOUS
OUT = REPO / "results/pdv2_fixed_fusion_20261015"
CONFIG = REPO / "configs/pdv2_fixed_fusion.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
SELECTION = base.read(base.BASE / "s1/confidence_selection.json")["ops"]


def prepare() -> None:
    """초기 풀을 공유하고 네 조건의 공통 양성 질의를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pools").mkdir(exist_ok=True)
    roster = {}
    for kind in ("gnn", "point"):
        source = PREVIOUS / "pools" / (kind + "_initial")
        target = OUT / "pools" / (kind + "_initial")
        if not target.exists():
            target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
        assert target.resolve() == source.resolve()
    for sensor, seq in base.TRAIN:
        masks, queries = [], None
        for kind in ("gnn", "point"):
            for op in base.OPS:
                with np.load(OUT / "pools" / f"{kind}_initial/train/{sensor}_{seq}/{op}.npz") as z:
                    if queries is not None:
                        np.testing.assert_array_equal(queries, z["queries"])
                    queries = z["queries"]
                    masks.append(z["correct"].any(1) & (z["min_past_gt_dist"] <= 5))
        chosen = queries[np.logical_and.reduce(masks)]
        roster[sensor + "/" + seq] = chosen.tolist()
    record = {"scope": "두 초기 표현·두 운영점의 실제 후보에 모두 정답이 있는 공통 학습 질의. GT는 표본 선택·손실에만 사용.",
              "queries": roster, "total": sum(map(len, roster.values())),
              "updates_per_epoch": sum((len(q) + CFG["queries_per_forward"] - 1) // CFG["queries_per_forward"] for q in roster.values())}
    path = OUT / "training_roster.json"
    if path.exists():
        assert base.read(path) == record
    else:
        write_json(path, record)
    print("ROSTER", record["total"], record["updates_per_epoch"], flush=True)


def protocol(create: bool = False) -> dict:
    """추가 학습 전에 손실·최종 규칙·예산과 원천 해시를 동결한다."""
    path = OUT / "protocol.json"
    if path.exists():
        result = base.read(path)
        for name, digest in result["code_sha256"].items():
            assert sha256(REPO / name) == digest, name
        return result
    assert create
    old = base.read(REPO / "results/pdv2_linear_fusion_20261015/protocol.json")
    for name, digest in old["code_sha256"].items():
        assert sha256(REPO / name) == digest, name
    code = dict(old["code_sha256"])
    for name in [str(CONFIG.relative_to(REPO)), "src/retrieval/fixed_fusion_learning.py",
                 "tests/test_fixed_fusion_learning.py", "scripts/_pdv2_fixed_fusion.py",
                 "scripts/_pdv2_fixed_fusion_run.py", "scripts/_pdv2_fixed_fusion_verify.py"]:
        code[name] = sha256(REPO / name)
    inputs = dict(old["input_sha256"])
    inputs[str((OUT / "training_roster.json").relative_to(REPO))] = sha256(OUT / "training_roster.json")
    result = {"created_unix": time.time(), "config": CFG, "scope": "시드1·검증9 예비 실험. 시험 실행·원고 교체 없음.",
              "train_sequences": old["train_sequences"], "val_sequences": old["val_sequences"],
              "training_roster": base.read(OUT / "training_roster.json"),
              "score": SELECTION,
              "training": "GNN/이웃없는투영 × 키순위/고정융합순위. 네조건같은초기표현·질의·예산. 새파라미터없음. 두번째통과만역전파, BN/dropout/key_scale고정. 1/5/9에포크간선·후보갱신.",
              "loss": "정답확률합의음의로그. 키대조는질의별minmax키코사인, 융합은기존네점수minmax가중합을키가중치로나눔. 양쪽키계수1·온도0.1로맞춤. 정렬점수는상수이며문맥만학습. 후보에정답없는행/오답없는행은0손실, 제외횟수기록. 거절학습없음.",
              "evaluation": "전체검증질의에서기존100+100후보·float32질의별minmax·고정융합가중치·기존c_col/c_spec수락점수·기존tau유지. 학습점수를판정점수로쓰지않음.",
              "selection": "4/8/12에포크의최종AP: 시퀀스→3계열→NSD/NSD-H평균최대,1e-12동률시이른모델. 양성없는KITTI09는PR평균제외·오수락별도. epoch0은대조표에만표시.",
              "limits": "첫통과와이산간선선택은미분하지않음,간선분류기고정. 학습8중KITTI01은공통양성질의0으로업데이트없음. 단일시드·모델크기대조아님. 이전거절학습과는질의/손실/252업데이트도달라직접일요인대조아님.",
              "code_sha256": code, "input_sha256": inputs}
    write_json(path, result)
    return result


def loss_for(embedding: torch.Tensor, pool: dict, rows: torch.Tensor,
             op: str, mode: str) -> tuple:
    """고정 후보의 현재 키 코사인에만 학습 기울기를 보낸다."""
    cand = pool["candidates"][rows]
    valid = cand >= 0
    key = (embedding[pool["queries"][rows], None] * embedding[cand.clamp_min(0)]).sum(-1)
    weights = SELECTION[op]["tuple"]
    scores = fixed_fusion_scores(key, pool["static"][rows], valid, weights, mode == "fusion") / weights[0]
    return positive_ranking_loss(scores, pool["correct"][rows], valid, CFG["temperature"])


def smoke() -> None:
    """실제 두 표현의 키·융합 손실 역전파를 초기 모델에서 확인한다."""
    records = []
    roster = base.read(OUT / "training_roster.json")["queries"]["kitti/06"]
    for kind in ("gnn", "point"):
        for mode in ("key", "fusion"):
            bundle = base.make_encoder(kind)
            for name, param in bundle[1].named_parameters():
                param.requires_grad_(not name.endswith("key_scale"))
            path = OUT / "pools" / f"{kind}_initial/train/kitti_06"
            graph = torch.load(path / "graph.pt", map_location="cuda", weights_only=False)
            embedding = F.normalize(bundle[1](graph), dim=1)
            total = torch.zeros((), device="cuda")
            for op in base.OPS:
                pool = base.load_pool(path, op)
                rows = torch.searchsorted(pool["queries"], torch.tensor(roster[:64], device="cuda"))
                loss, active = loss_for(embedding, pool, rows, op, mode)
                assert active.all()
                total = total + loss.mean() / len(base.OPS)
            total.backward()
            grads = [p.grad for p in bundle[1].parameters() if p.grad is not None]
            assert torch.isfinite(total) and all(torch.isfinite(g).all() for g in grads)
            norm = sum(float(g.abs().sum()) for g in grads)
            assert norm > 0
            records.append({"arm": kind + "_" + mode, "loss": float(total.detach()), "gradient_abs_sum": norm})
    write_json(OUT / "preflight_checks.json", {"status": "passed", "records": records})
    print("SMOKE_PASSED", records, flush=True)


def train(arm: str) -> None:
    """정해진 양성 질의와 같은 예산으로 표현만 추가 학습한다."""
    protocol()
    dest = OUT / arm
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "training_complete.json").exists():
        return
    assert not list(dest.glob("epoch*.pth")), "부분 학습을 자동 덮어쓰지 않는다"
    kind, mode = arm.split("_")
    bundle = base.make_encoder(kind)
    net = bundle[1]
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    for name, param in net.named_parameters():
        param.requires_grad_(not name.endswith("key_scale"))
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                 lr=CFG["encoder_learning_rate"], weight_decay=CFG["weight_decay"])
    roster = base.read(OUT / "training_roster.json")
    history, updates, started = [], 0, time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        losses, active_counts, zero_batches = [], {op: 0 for op in base.OPS}, 0
        refresh = max(e for e in CFG["candidate_refresh_epochs"] if e <= epoch)
        tag = kind + "_initial" if refresh == 1 else f"{arm}_refresh{refresh}"
        for sequence_index, (sensor, seq) in enumerate(base.TRAIN):
            sequence = sensor + "/" + seq
            queries = np.asarray(roster["queries"][sequence], dtype=np.int64)
            if not len(queries):
                continue
            path = base.snapshot(kind, tag, "train", sequence, bundle)
            pools = {op: base.load_pool(path, op) for op in base.OPS}
            rows = np.searchsorted(pools[base.OPS[0]]["queries"].cpu().numpy(), queries)
            for pool in pools.values():
                np.testing.assert_array_equal(pool["queries"][torch.tensor(rows, device="cuda")].cpu(), queries)
            order = np.random.default_rng(epoch * 100 + sequence_index).permutation(rows)
            graph = torch.load(path / "graph.pt", map_location="cuda", weights_only=False)
            for start in range(0, len(order), CFG["queries_per_forward"]):
                selected = order[start:start + CFG["queries_per_forward"]]
                optimizer.zero_grad(set_to_none=True)
                embedding = F.normalize(net(graph), dim=1)
                total = torch.zeros((), device="cuda")
                active_total = 0
                for op, pool in pools.items():
                    for offset in range(0, len(selected), CFG["batch_queries"]):
                        ix = torch.tensor(selected[offset:offset + CFG["batch_queries"]], device="cuda")
                        loss, active = loss_for(embedding, pool, ix, op, mode)
                        total = total + loss.sum() / len(selected) / len(base.OPS)
                        count = int(active.sum())
                        active_counts[op] += count
                        active_total += count
                assert torch.isfinite(total)
                if active_total:
                    total.backward()
                    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["gradient_clip"])
                    assert torch.isfinite(norm)
                    optimizer.step()
                    updates += 1
                else:
                    zero_batches += 1
                losses.append(float(total.detach()))
            del pools, graph, embedding
        row = {"epoch": epoch, "updates": updates, "loss": float(np.mean(losses)),
               "active_query_ops": active_counts, "zero_signal_batches": zero_batches,
               "runtime_s": time.time() - started}
        history.append(row)
        write_json(dest / "training_progress.json", row)
        print("EPOCH", arm, row, flush=True)
        if epoch in CFG["checkpoint_epochs"]:
            torch.save({"encoder": net.state_dict(), "epoch": epoch, "updates": updates,
                        "protocol_sha256": sha256(OUT / "protocol.json")}, dest / f"epoch{epoch}.pth")
    delta = {k: float((v.cpu().float() - initial[k].float()).abs().max()) for k, v in net.state_dict().items()}
    assert max(delta.values()) > 0
    for key in initial:
        if "running_" in key or "num_batches_tracked" in key or key.endswith("key_scale"):
            assert delta[key] == 0, key
    assert updates == CFG["epochs"] * roster["updates_per_epoch"], "실제 업데이트 예산 불일치"
    write_json(dest / "training_complete.json", {"history": history, "updates": updates,
               "encoder_max_change": max(delta.values()), "batch_norm_frozen": True,
               "added_parameters": 0, "checkpoints": {p.name: sha256(p) for p in dest.glob("epoch*.pth")}})


def fixed_decisions(embedding: np.ndarray, arrays: dict, op: str) -> tuple:
    """기존 float32 융합으로 후보를 고르고 원래 정렬 점수를 반환한다."""
    chosen, confidence, ranking = [], [], []
    weights = SELECTION[op]["tuple"]
    channel = 2 if SELECTION[op]["decision"] == "c_col" else 1
    for i, query in enumerate(arrays["queries"]):
        valid = arrays["candidates"][i] >= 0
        c = arrays["candidates"][i, valid]
        static = arrays["static"][i, valid].astype(np.float32)
        score = weights[0] * minmax(embedding[c] @ embedding[query])
        for k in range(3):
            score = score + weights[k + 1] * minmax(static[:, k])
        best = int(np.argmax(score))
        chosen.append(best)
        confidence.append(arrays["static"][i, best, channel])
        ranking.append(float(score[best]))
    return np.asarray(chosen), np.asarray(confidence), np.asarray(ranking)


def evaluate(arm: str, epoch: int, sequence: str) -> None:
    """전체 검증에서 기존 최종 판정 방식으로 원장을 작성한다."""
    protocol()
    kind = arm.split("_")[0]
    checkpoint = OUT / arm / f"epoch{epoch}.pth"
    assert base.read(OUT / arm / "training_complete.json")["checkpoints"][checkpoint.name] == sha256(checkpoint)
    dest = OUT / arm / f"val_epoch{epoch}"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / (sequence.replace("/", "_") + ".json")
    if marker.exists():
        return
    bundle = base.make_encoder(kind)
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    assert state["protocol_sha256"] == sha256(OUT / "protocol.json")
    bundle[1].load_state_dict(state["encoder"], strict=True)
    folder = base.snapshot(kind, f"{arm}_epoch{epoch}", "val", sequence, bundle)
    with np.load(folder / "keys.npz") as z:
        embedding = z["embedding"]
    result = {"sequence": sequence, "arm": arm, "epoch": epoch, "ops": {}}
    for op in base.OPS:
        with np.load(folder / (op + ".npz")) as z:
            a = {k: z[k] for k in z.files}
        chosen, score, ranking = fixed_decisions(embedding, a, op)
        rows = np.arange(len(chosen))
        ledger = {"query_idx": a["queries"], "min_past_gt_dist": a["min_past_gt_dist"], "group": a["group"],
                  "j_star": a["candidates"][rows, chosen], "top1_geo_dist": a["distances"][rows, chosen],
                  "score_fixed": score, "rank_score": ranking, "n_candidates": (a["candidates"] >= 0).sum(1)}
        assert (ledger["j_star"] <= ledger["query_idx"] - 30).all()
        assert all(np.isfinite(v).all() for v in ledger.values())
        qc = classify_queries(ledger["min_past_gt_dist"])
        metrics = None
        if qc["n_pos"]:
            sweep = pr_sweep(score, ledger["top1_geo_dist"], qc)
            metrics = {k: float(sweep[k]) for k in base.METRICS[:3]}
            metrics["R@1"] = float((ledger["top1_geo_dist"][qc["positive"]] <= 5).mean())
        target = dest / (sequence.replace("/", "_") + "_" + op + ".npz")
        np.savez_compressed(target, **ledger)
        result["ops"][op] = {"metrics": metrics, "ledger": str(target.relative_to(REPO)), "sha256": sha256(target)}
    result.update(checkpoint_sha256=sha256(checkpoint), protocol_sha256=sha256(OUT / "protocol.json"))
    write_json(marker, result)
    print("EVAL_DONE", arm, epoch, sequence, flush=True)


def report() -> None:
    """최종 검증 AP로 선택하고 기존 공통 임계값과 방향별 결과를 기록한다."""
    protocol()
    result = {"scope": "고정 추론 규칙의 시드1 검증9 최종 판정. 시험 실행·원고 변경 없음.",
              "arms": {}, "protocol_sha256": sha256(OUT / "protocol.json")}
    for arm in CFG["arms"]:
        candidates = []
        for epoch in CFG["checkpoint_epochs"]:
            summaries = [base.read(OUT / arm / f"val_epoch{epoch}" / f"{s}_{q}.json") for s, q in base.VAL]
            ops = {}
            for op in base.OPS:
                groups = {}
                for row in summaries:
                    if row["ops"][op]["metrics"] is not None:
                        sensor = row["sequence"].split("/")[0]
                        groups.setdefault(base.FAMILY.get(sensor, sensor), []).append(row["ops"][op]["metrics"])
                ops[op] = {k: float(np.mean([np.mean([r[k] for r in g]) for g in groups.values()])) for k in base.METRICS}
            candidates.append({"epoch": epoch, "ops": ops, "joint_AP": float(np.mean([r["AP"] for r in ops.values()]))})
        best = max(r["joint_AP"] for r in candidates)
        selected = copy.deepcopy(min((r for r in candidates if best - r["joint_AP"] <= 1e-12), key=lambda r: r["epoch"]))
        selected["candidates"], selected["decision"] = candidates, {}
        for op in base.OPS:
            tau = SELECTION[op]["tau_star"]
            per_sequence, groups = {}, {}
            directions = {str(i): {"n": 0, "correct": 0} for i in range(3)}
            for sensor, seq in base.VAL:
                with np.load(OUT / arm / f"val_epoch{selected['epoch']}" / f"{sensor}_{seq}_{op}.npz") as z:
                    a = {k: z[k] for k in z.files}
                qc = classify_queries(a["min_past_gt_dist"])
                row = operating_point(a["score_fixed"], a["top1_geo_dist"], qc, tau)
                row["negative_false_accepts"] = int((qc["negative"] & (a["score_fixed"] >= tau)).sum())
                per_sequence[sensor + "/" + seq] = row
                if qc["n_pos"]:
                    groups.setdefault(base.FAMILY.get(sensor, sensor), []).append(row["F1"])
                for i in range(3):
                    mask = qc["positive"] & (a["group"] == i)
                    directions[str(i)]["n"] += int(mask.sum())
                    directions[str(i)]["correct"] += int((mask & (a["top1_geo_dist"] <= 5)).sum())
            selected["decision"][op] = {"tau": tau, "macro_F1_at_tau": float(np.mean([np.mean(g) for g in groups.values()])),
                                        "sequences": per_sequence, "directions": directions}
        result["arms"][arm] = selected
    old = base.read(PREVIOUS / "summary.json")
    result["initial_fixed_pipeline"] = old["initial_fixed_pipeline"]
    result["effects"] = {op: {metric: {
        "gnn_fusion_minus_key": result["arms"]["gnn_fusion"]["ops"][op][metric] - result["arms"]["gnn_key"]["ops"][op][metric],
        "point_fusion_minus_key": result["arms"]["point_fusion"]["ops"][op][metric] - result["arms"]["point_key"]["ops"][op][metric],
        "gnn_minus_point_fusion": result["arms"]["gnn_fusion"]["ops"][op][metric] - result["arms"]["point_fusion"]["ops"][op][metric]
        } for metric in base.METRICS} for op in base.OPS}
    write_json(OUT / "summary.json", result)
    print("SUMMARY", {a: (v["epoch"], v["ops"]) for a, v in result["arms"].items()}, flush=True)


base.OUT, base.CFG, base.CONFIG = OUT, CFG, CONFIG
base.protocol = protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "protocol", "train", "evaluate", "report"))
    parser.add_argument("--arm", choices=CFG["arms"])
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    base.setup()
    if args.action == "prepare":
        prepare()
    elif args.action == "smoke":
        smoke()
    elif args.action == "protocol":
        protocol(True)
    elif args.action == "train":
        train(args.arm)
    elif args.action == "evaluate":
        evaluate(args.arm, args.epoch, args.sequence)
    else:
        report()
