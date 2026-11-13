#!/usr/bin/env python3
"""초기화·표본·매 업데이트 갱신·최종 원장과 q+15 입력 한계를 검사한다."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_graph_alignment as exp
from _pdv2_no_context_ablation import sha256, write_json


def document(summary: dict, validation: dict) -> None:
    """완료한 최종 지표와 검사 범위를 위키·실험 일지에 기록한다."""
    labels = {"diff_full": "현재 학습", "diff_online": "온라인 유사 간선으로 학습", "diff_temporal": "시간 이웃만"}
    table = ["| 조건 | 선택 에포크 | 운영점 | AP | 최대 F1 | R@100P |",
             "|---|---:|---|---:|---:|---:|"]
    for arm in exp.ARMS:
        row = summary["arms"][arm]
        for op, name in zip(exp.base.OPS, ("NSD", "NSD-H")):
            m = row["ops"][op]["macro"]
            table.append(f"| {labels[arm]} | {row['epoch']} | {name} | {m['AP']:.6f} | {m['max_F1']:.6f} | {m['R@100P']:.6f} |")
    text = ("## 유사 간선 학습·추론 정합 실험\n\n"
            "시드 1, 학습 8·검증 9의 최종 판정 비교를 완료했다. 같은 초기값·삼중쌍·InfoNCE·20에포크900업데이트를 쓴다. 현재 학습과 시간 이웃 대조군은 완료한 attention 실험의 파일을 해시로 고정해 재사용한다. 새 조건은 매 업데이트 직전 현재 가중치와 고정 간선 분류기로 온라인 유사 간선을 새로 계산한다. 간선 선택은 미분하지 않고 둘째 문맥 통과를 학습한다. 학습의 기준 자세·키프레임·BN/dropout 규칙은 세 조건에서 유지한다. 시험 자료와 원고 주표는 바꾸지 않는다.\n\n"
            + "\n".join(table) + "\n\n"
            "10/20에포크 중 두 운영점의 검증 최종 AP 평균으로 선택하며 같은 에포크 비교도 저장한다. 단일 시드 결과로 통계적 유의성을 주장하지 않는다. CPU 참조에서 q+15 절단을 검사하고 GPU 오차와 표본 최종 판정의 일치 여부를 별도로 기록한다. 전체 스트림 실시간 배포 검증은 아니다. "
            "출처는 `results/pdv2_graph_alignment_20261015/summary.json`, `protocol.json`, `completion_validation.json`과 `s1/diff_online/refresh.jsonl`이다.\n")
    wiki = REPO / "docs/학습과 선택.md"
    content = wiki.read_text()
    title = "## 유사 간선 학습·추론 정합 실험"
    if title in content:
        head, tail = content.split(title, 1)
        end = tail.find("\n## ")
        content = head + text + (tail[end:] if end >= 0 else "")
    else:
        content += "\n" + text
    wiki.write_text(content)
    diary = REPO / "docs/RAW/실험일지/2026-10-15_유사-간선-학습-추론-정합.md"
    diary.write_text("# 유사 간선 학습·추론 정합\n\n"
                     "## 왜\n\n학습과 온라인 평가의 유사 간선 선택·속성 차이가 문맥의 약한 기여를 설명하는지 확인한다.\n\n"
                     "## 설정\n\n구조·손실·표본·업데이트 수를 유지하고 온라인 선택 규칙을 매 업데이트 적용한다. 기존 두 대조군을 재사용한다. 시험셋은 쓰지 않는다.\n\n"
                     "## 결과\n\n" + "\n".join(table) + "\n\n"
                     "## 결론·다음\n\n단일 시드의 최종 AP·최대 F1·R@100P와 같은 에포크 차이를 함께 해석한다. "
                     "유사 간선 정합 외에 기준 자세/키프레임과 KISS-ICP 차이, BN/dropout 학습·추론 차이는 유지된다. "
                     f"검사 상태: {validation['status']}. 상세 근거: `results/pdv2_graph_alignment_20261015/`.\n")
    subprocess.run([sys.executable, "docs/도구/check_wiki.py", "docs"], cwd=REPO, check=True)


def main() -> None:
    """세 조건의 원장과 새 학습 경로를 독립적으로 확인한다."""
    import torch
    from _pdv2_pair_evaluate import ArrayCache
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_attention_neighbors_verify_reference import cpu_embedding, decision
    exp.protocol(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = exp.read(exp.OUT / "summary.json")
    reference = exp.arm_root("diff_full")
    init = exp.read(reference / "initial_model.json")
    assert init["n_parameters"] == 167561
    samples = exp.read(reference / "mining_calls.json")
    graph_ref = exp.read(reference / "graph_counts.json")
    count = rows = 0
    for arm in exp.ARMS:
        root = exp.arm_root(arm)
        assert exp.read(root / "initial_model.json") == init
        assert exp.read(root / "mining_calls.json") == samples
        p = exp.read(root / "training_progress.json")
        assert (p["epoch"], p["optimizer_updates"]) == (20, 900)
        g = exp.read(root / "graph_counts.json")
        for key in ("nodes", "temporal", "temporal_index_sha256", "temporal_attr_sha256"):
            assert g[key] == graph_ref[key]
        chosen = summary["arms"][arm]
        assert chosen["joint_AP"] == max(c["joint_AP"] for c in chosen["candidates"])
        for epoch in exp.EPOCHS:
            expected = next(c for c in chosen["candidates"] if c["epoch"] == epoch)
            assert expected["ops"] == exp.base.aggregate(1, arm, epoch, "val")
            for sensor, seq in exp.base.VAL:
                result = exp.read(exp.folder(1, arm, epoch) / f"val_{sensor}_{seq}.json")
                selection = exp.read(exp.base.BASE / "s1/confidence_selection.json")["ops"]
                for op in exp.base.OPS:
                    record = result["ops"][op]
                    path = REPO / record["ledger"]
                    assert sha256(path) == record["sha256"]
                    arrays, _ = exp.base.read_ledger(path)
                    ref, _ = exp.base.read_ledger(exp.base.BASE / f"s1/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
                    for key in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(arrays[key], ref[key])
                    assert np.all(arrays["j_star"] <= arrays["query_idx"] - 30)
                    assert np.all(arrays["j_star"] >= 0)
                    s = selection[op]
                    assert exp.base.sequence_metrics(arrays, s["decision"], s["tau_star"]) == record["metrics"]
                    count += 1
                    rows += len(arrays["query_idx"])
    root = exp.arm_root(exp.NEW)
    refresh = [json.loads(line) for line in (root / "refresh.jsonl").read_text().splitlines()]
    assert len(refresh) == 900
    assert [r["optimizer_updates_before"] for r in refresh] == list(range(900))
    assert all(r["refresh"] == i + 1 and r["epoch"] == i // 45 + 1 for i, r in enumerate(refresh))
    assert all(r["temporal"] == graph_ref["temporal"] for r in refresh)
    assert exp.read(root / "probe_state_audit.json")["weights_and_bn_buffers_unchanged"]
    assert exp.read(root / "first_graphs.json")["same_temporal_edges_and_attributes"]
    arm = exp.NEW
    epoch = summary["arms"][arm]["epoch"]
    bundle = exp.attention.load_bundle(arm, epoch)
    cpu_bundle = list(copy.deepcopy(bundle))
    cpu_bundle[1].cpu().eval()
    cpu_bundle[2].cpu().eval()
    prefixes = []
    for sequence in ("kitti/09", "mulran/Sejong01"):
        sensor, seq, paths, _ = exp.base.inputs("val", sequence)
        with np.load(paths["cache"]) as z:
            cache = ArrayCache({k: z[k] for k in z.files})
        with np.load(paths["layouts"]) as z:
            layout = z["nsd_layouts"].astype(np.float32)
        raw = cache["descriptors"].astype(np.float32)
        with np.load(exp.folder(1, arm, epoch) / f"keys_val_{sensor}_{seq}.npz") as z:
            full = z["embedding"]
        cpu_full = cpu_embedding(cache, sensor, raw, layout, cpu_bundle, arm)
        with np.load(paths["spec"]) as z:
            banks = {op: exp.base.bank_for(op, z) for op in exp.base.OPS}
        for q in (300, 900):
            n = q + 16
            cut, _ = exp.attention.embedding(TruncatedCache(cache, n), sensor, raw[:n], layout[:n], bundle, arm)
            cpu_cut = cpu_embedding(TruncatedCache(cache, n), sensor, raw[:n], layout[:n], cpu_bundle, arm)
            cpu_error = float(np.abs(cpu_full[:q + 1] - cpu_cut[:q + 1]).max())
            assert cpu_error < 2e-6, (sequence, q, cpu_error)
            checks = {}
            for op, bank in banks.items():
                a, ca = decision(cpu_full, bank, q, selection[op])
                b, cb = decision(cpu_cut, bank, q, selection[op])
                assert a == b and np.array_equal(ca, cb)
                ga, gca = decision(full, bank, q, selection[op])
                gb, gcb = decision(cut, bank, q, selection[op])
                checks[op] = {"cpu_decision_equal": True, "gpu_decision_equal": ga == gb,
                              "gpu_candidate_set_equal": bool(np.array_equal(gca, gcb))}
            prefixes.append({"sequence": sequence, "q": q, "cpu_max_error": cpu_error,
                             "gpu_max_error": float(np.abs(full[:q + 1] - cut[:q + 1]).max()),
                             "checks": checks})
    assert not list(exp.OUT.glob("s*/*/epoch*/test_*"))
    numerical_limit = any(p["gpu_max_error"] >= 2e-6 or any(
        not c["gpu_decision_equal"] or not c["gpu_candidate_set_equal"] for c in p["checks"].values()) for p in prefixes)
    validation = {"status": "passed_with_gpu_numerical_limit" if numerical_limit else "passed",
                  "checked_ledgers": count, "total_rows": rows, "same_initial_weights": True,
                  "same_triplets_and_budget": True, "same_temporal_edges_and_attributes": True,
                  "fresh_graph_every_update": True, "graph_refreshes": 900,
                  "prefix_checks": prefixes, "test_used": False,
                  "protocol_sha256": sha256(exp.OUT / "protocol.json"),
                  "summary_sha256": sha256(exp.OUT / "summary.json")}
    write_json(exp.OUT / "completion_validation.json", validation)
    document(summary, validation)
    print("VERIFIED", validation["status"], count, rows, flush=True)


if __name__ == "__main__":
    main()
