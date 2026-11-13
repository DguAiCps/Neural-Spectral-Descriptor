#!/usr/bin/env python3
"""검색 마진 보호의 학습 효과와 검증 장소별 개선을 확인한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import _pdv2_diffattn_margin as experiment
from retrieval.teacher_margin import triplet_margin


def main() -> None:
    """최종 선택을 변경하지 않고 학습·검증 진단을 별도로 저장한다."""
    run = experiment.run
    run.configure()
    spec = experiment.activate()
    summary = run.READ(experiment.OUT / "summary.json")
    fit = run.calibration()
    training = {}
    for seq in spec["train"]:
        folder = experiment.OUT / "pools" / seq.replace("/", "_")
        data = run.ARR(folder / "pool.npz")
        graph = torch.load(folder / "graph.pt", map_location="cuda", weights_only=False)
        ids = torch.from_numpy(data["triplets"]).cuda()
        positive, correct = data["positive"][data["eligible"]], data["correct"][data["eligible"]]
        plan = {k: torch.from_numpy(data["plan_"+k]).cuda() for k in ("query", "candidate", "weight")}
        baseline = run.predict_score(data["spectral"][data["eligible"], None], fit["spectral"])
        entries = {}
        for arm in ("frozen", *experiment.ARMS):
            net = run.model(None if arm == "frozen" else experiment.OUT / arm / "epoch4.pth")
            with torch.no_grad():
                key = F.normalize(net(graph), dim=1)
                margin = triplet_margin(key, ids).cpu().numpy()
                context = torch.cat([run.context_score(key, {k: v[i:i+256] for k,v in plan.items()})
                                     for i in range(0, len(positive), 256)]).cpu().numpy()
            if arm == "frozen":
                reference = margin.copy()
            protected = reference > 0
            loss = np.maximum(reference-margin, 0)
            logits = baseline + (context-fit["context_mean"])/fit["context_scale"]
            bce = np.logaddexp(0, logits)-correct*logits
            entries[arm] = dict(triplets=len(margin), protected=int(protected.sum()),
                protected_margin_decreased=int(((margin < reference-1e-5)&protected).sum()),
                protected_margin_flipped=int(((margin <= 0)&protected).sum()),
                protected_mean_loss=float(loss[protected].mean()) if protected.any() else None,
                positive_bce=float(bce[positive].mean()) if positive.any() else None,
                correct_context=float(context[correct].mean()) if correct.any() else None)
            del key, net
        training[seq] = entries
        del graph
        torch.cuda.empty_cache()
    deltas, intervals, winners = {}, {}, {}
    for arm in experiment.ARMS:
        values, frozen = summary["results"][arm], summary["results"]["frozen"]
        deltas[arm] = {f: {m: 100*(values["families"][f][m]-frozen["families"][f][m])
                             for m in run.base.METRICS} for f in values["families"]}
        family_seqs = {}
        for seq in spec["val"]:
            if values["sequences"][seq]["n_pos"]:
                family = "kitti" if seq.startswith("kitti") else seq.split("/")[0]
                family_seqs.setdefault(family, []).append(seq)
        rng = np.random.default_rng(20261104)
        sampled = []
        for family, sequences in family_seqs.items():
            matrix = np.array([[100*(values["sequences"][s][m]-frozen["sequences"][s][m])
                                for m in run.base.METRICS] for s in sequences])
            draw = rng.integers(0, len(sequences), size=(10000, len(sequences)))
            sampled.append(matrix[draw].mean(1))
        bootstrap = np.stack(sampled).mean(0)
        intervals[arm] = {m: np.quantile(bootstrap[:, i], [.025, .975]).tolist()
                          for i,m in enumerate(run.base.METRICS)}
        counts = dict(queries=0, same_winner=0, correct_gained=0, correct_lost=0)
        for seq in spec["val"]:
            name = seq.replace("/", "_")+".npz"
            a = run.ARR(experiment.OUT / arm / "epoch4/val" / name)
            b = run.ARR(experiment.PRIOR / "initial/epoch0/val" / name)
            np.testing.assert_array_equal(a["query_idx"], b["query_idx"])
            pa = (a["min_past_gt_dist"]<=5)&(a["top1_geo_dist"]<=5)
            pb = (b["min_past_gt_dist"]<=5)&(b["top1_geo_dist"]<=5)
            counts["queries"] += len(pa)
            counts["same_winner"] += int((a["j_star"]==b["j_star"]).sum())
            counts["correct_gained"] += int((pa&~pb).sum())
            counts["correct_lost"] += int((pb&~pa).sum())
        winners[arm] = counts
    restoration = run.READ(ROOT / "results/pdv2_learning_plan_20261104/restoration.json")
    figure_checks = {p: run.HASH(ROOT / "docs/paper/body_v2.2" / p)==h
                     for p,h in restoration["figure_sha256"].items()}
    assert all(figure_checks.values())
    previous = run.READ(experiment.BALANCED / "summary.json")["all_results"]["positive_only__epoch4__lambda0.5"]
    reproduction = {m:100*(summary["results"]["full_control"]["macro"][m]-previous["macro"][m])
                    for m in run.base.METRICS}
    run.WRITE(experiment.OUT / "diagnostics.json", dict(training_fixed_pairs=training,
        validation_family_delta_pp=deltas, validation_winner_changes=winners,
        prior_control_reproduction_delta_pp=reproduction,
        conditional_sequence_bootstrap_95_pp=intervals,
        bootstrap_note="고정 초기시드와 반복 사용한 검증셋에 조건을 둔 탐색 진단. 계열별 양성 시퀀스2/3/3개를 짝지어 재표집한 구간. 장소 내 상관과 학습 시드 분산·선택 편향을 해결하지 않으며 독립 확증 구간이 아님.",
        protected_margin_note="학습의 초기 키 삼중쌍 마진만 진단. 실제 융합 승자 유지 보장 아님.",
        figures_unchanged=len(figure_checks), test_used=False,
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in (
            Path(__file__), experiment.OUT / "summary.json", experiment.OUT / "runtime_amendments.json")}))
    print("DIAGNOSED", deltas, winners, "FIGURES", len(figure_checks), flush=True)


if __name__ == "__main__":
    main()
