#!/usr/bin/env python3
"""중심 보존의 학습 예산·최종 지표·후보/순위 오류를 독립 검산한다."""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_center_preservation as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json


def aggregate(rows: dict) -> dict:
    """평가 코드와 별도로 시퀀스→계열 평균을 계산한다."""
    families = {}
    for family in ("kitti", "helipr", "mulran"):
        selected = [v for s, v in rows.items() if s.startswith(family) and v["n_pos"]]
        families[family] = {m: sum(v[m] for v in selected)/len(selected) for m in run.base.exp.METRICS}
    return {"families": families,
            "macro": {m: sum(v[m] for v in families.values())/3 for m in run.base.exp.METRICS}}


def counts(row: dict, coverage: dict, tau: float) -> dict:
    """최종 정답·후보 누락·고정 임계값 수락을 실제 질의 개수로 센다."""
    pos = row["min_past_gt_dist"] <= 5
    neg = row["min_past_gt_dist"] > 25
    correct = pos & (row["top1_geo_dist"] <= 5)
    hit = coverage["union_hit"]
    accept = row[run.METHOD] >= tau
    assert not np.any(correct & ~hit)
    return {"queries": len(pos), "positive": int(pos.sum()), "negative": int(neg.sum()),
        "correct": int(correct.sum()), "key_hit": int((pos & coverage["key_hit"]).sum()),
        "aux_hit": int((pos & coverage["aux_hit"]).sum()), "union_hit": int((pos & hit).sum()),
        "all_positives_absent": int((pos & ~hit).sum()),
        "wrong_with_positive_in_pool": int((pos & hit & ~correct).sum()),
        "true_accept": int((correct & accept).sum()),
        "wrong_positive_accept": int((pos & ~correct & accept).sum()),
        "negative_false_accept": int((neg & accept).sum())}


def main() -> None:
    """54개 전체 원장을 확인하고 선택된 모델의 오류 손익을 집계한다."""
    spec = run.protocol(True)
    summary = run.read(run.OUT/"summary.json")
    assert summary["protocol_sha256"] == sha256(run.OUT/"protocol.json")
    sources = {**spec["runtime"], **spec["inputs"]}
    reference_initial = run.read(run.previous.OUT/"s1/diff_tanh/initial_model.json")
    reference_orders = run.read(run.previous.OUT/"s1/diff_tanh/training_complete.json")["orders"]
    tau = run.base.fixed.choose_tau(1)["tau"]
    ledger_count = row_count = metric_checks = scalar_checks = prefix_checks = 0
    max_metric_error = max_scalar_error = max_prefix_error = max_zero_key_error = 0.
    rows, coverage_rows, details = {}, {}, {}
    for arm, saved in summary["per_seed"]["1"].items():
        folder = run.base.root(1, arm)
        assert run.read(folder/"initial_model.json") == reference_initial
        complete = run.read(folder/"training_complete.json")
        assert complete["updates"] == 900 and complete["orders"] == reference_orders
        progress = run.read(folder/"training_progress.json")["records"]
        assert [x["updates"] for x in progress] == list(range(45, 901, 45))
        assert all(np.isfinite(x["loss"]) for x in progress)
        for p in (folder/"initial_model.json", folder/"training_complete.json", folder/"training_progress.json"):
            sources[str(p.relative_to(REPO))] = sha256(p)
        for name, h in complete["checkpoints"].items():
            p = folder/"checkpoints"/name
            assert sha256(p) == h
            state = torch.load(p, map_location="cpu", weights_only=False)
            epoch = state["epoch"]+1
            assert epoch in (10, 20)
            assert all(int(x["step"].item()) == 45*epoch for x in state["optimizer_state_dict"]["state"].values())
            assert all(torch.isfinite(v).all() for v in state["model_state_dict"].values())
            sources[str(p.relative_to(REPO))] = h
            if arm == "center_zero":
                assert h == sha256(run.previous.OUT/"s1/diff_tanh/checkpoints"/name)
        rows[arm], coverage_rows[arm], details[arm] = {}, {}, {}
        collected = []
        for epoch in (10, 20):
            metrics = {m: {} for m in ("raw", run.METHOD)}
            for sequence in spec["val"]:
                _, _, paths, _ = run.base.exp.old.base.inputs("val", sequence)
                cache = run.base.exp.arrays(paths["cache"])
                path = folder/f"epoch{epoch}"/(sequence.replace("/", "_")+".npz")
                info = run.read(path.with_suffix(".json"))
                kp, cp = path.with_name("keys_"+path.name), path.with_name("coverage_"+path.name)
                for p, expected in ((path, info["sha256"]), (kp, info["keys_sha256"]), (cp, run.read(cp.with_suffix(".json"))["sha256"])):
                    assert sha256(p) == expected
                    sources[str(p.relative_to(REPO))] = expected
                for p in (path.with_suffix(".json"), cp.with_suffix(".json")):
                    sources[str(p.relative_to(REPO))] = sha256(p)
                row, keys, coverage = (run.base.exp.arrays(p) for p in (path, kp, cp))
                q, j, key = row["query_idx"], row["j_star"], keys["embedding"]
                np.testing.assert_array_equal(q, np.arange(30, len(key)-15))
                np.testing.assert_array_equal(keys["scan_ids"], cache["scan_ids"])
                reference = run.base.exp.arrays(run.previous.OUT/f"s1/diff_tanh/epoch{epoch}"/path.name)
                for field in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(row[field], reference[field])
                if arm == "center_zero":
                    for field in row:
                        np.testing.assert_array_equal(row[field], reference[field])
                    expected = run.base.exp.arrays(run.previous.OUT/f"s1/diff_tanh/epoch{epoch}"/kp.name)["embedding"]
                    max_zero_key_error = max(max_zero_key_error, float(abs(key-expected).max()))
                    np.testing.assert_allclose(key, expected, atol=4e-6, rtol=4e-5)
                for field in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist"):
                    np.testing.assert_array_equal(row[field], coverage[field])
                np.testing.assert_array_equal(coverage["union_hit"], coverage["key_hit"] | coverage["aux_hit"])
                assert bool(((coverage["n_candidates"] >= 1) & (coverage["n_candidates"] <= 200)).all())
                xyz = cache["poses_gt"][:, :3, 3]
                np.testing.assert_allclose(np.linalg.norm(xyz[q]-xyz[j], axis=1), row["top1_geo_dist"], atol=1e-10, rtol=0)
                assert bool((j <= q-30).all() and (j >= 0).all())
                assert key.shape == (len(xyz), 416) and np.isfinite(key).all()
                np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1., atol=2e-6)
                for method in metrics:
                    measured = independent_metrics(row, row[method])
                    for name, value in measured.items():
                        expected = info["metrics"][method][name]
                        if value is None:
                            assert expected is None
                        else:
                            max_metric_error = max(max_metric_error, abs(value-expected))
                            np.testing.assert_allclose(value, expected, atol=5e-13, rtol=0)
                    metrics[method][sequence] = measured
                    metric_checks += 1
                with np.load(paths["spec"]) as z:
                    bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
                positions = cache["poses"][:, :3, 3]
                travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
                conf = spec["confidence"]["1"]
                for ix in (0, len(q)//3, 2*len(q)//3, len(q)-1):
                    t = int(q[ix])
                    a = run.base.prefix_topk(key, t, t-29, 100)
                    b = run.base.prefix_topk(bank[0], t, t-29, 100)
                    ids = np.unique(np.r_[a, b])
                    channels = (key[ids]@key[t], bank[0][ids]@bank[0][t],
                        run.base.align_scores(bank[1], bank[2], t, ids), run.base.column_align_scores(bank[3], t, ids))
                    fused = sum(float(w)*run.base.minmax(v) for w, v in zip(conf["tuple"], channels))
                    winner = int(ids[np.argsort(-fused, kind="stable")[0]])
                    assert winner == j[ix] and len(ids) == coverage["n_candidates"][ix]
                    for name, candidates in (("key_hit", a), ("aux_hit", b), ("union_hit", ids)):
                        assert bool((np.linalg.norm(xyz[candidates]-xyz[t], axis=1) <= 5).any()) == coverage[name][ix]
                    score = scalar_score(bank, travel, t, winner, conf["decision"], float(row["raw"][ix]))
                    max_scalar_error = max(max_scalar_error, abs(score-row[run.METHOD][ix]))
                    assert abs(score-row[run.METHOD][ix]) < 1e-12
                    scalar_checks += 1
                for check in info["prefixes"]:
                    assert check["winner_equal"] and check["score_error"] == 0
                    max_prefix_error = max(max_prefix_error, check["key_max_error"])
                    prefix_checks += 1
                if epoch == saved["selected"]["epoch"]:
                    rows[arm][sequence], coverage_rows[arm][sequence] = row, coverage
                    details[arm][sequence] = counts(row, coverage, tau)
                ledger_count += 1
                row_count += len(q)
            combined = {m: aggregate(v) for m, v in metrics.items()}
            expected = next(x for x in saved["candidates"] if x["epoch"] == epoch)
            for m in metrics:
                for key in run.base.exp.METRICS:
                    np.testing.assert_allclose(combined[m]["macro"][key], expected[m]["macro"][key], atol=5e-13, rtol=0)
            collected.append({"epoch": epoch, **combined})
        selected = max(collected, key=lambda x: (round(x[run.METHOD]["macro"]["AP"], 12), -x["epoch"]))
        assert selected["epoch"] == saved["selected"]["epoch"]
    totals = {arm: {k: sum(r[k] for r in records.values()) for k in next(iter(records.values()))} for arm, records in details.items()}
    transitions = {}
    for arm in run.CFG["train_arms"]:
        transitions[arm] = {}
        for ref_name in ("center_zero", "standard_tanh"):
            records = {}
            for seq in spec["val"]:
                r = rows[arm][seq]
                if ref_name == "center_zero":
                    ref = rows[ref_name][seq]
                else:
                    epoch = summary["references"][ref_name]["selected"]["epoch"]
                    ref = run.base.exp.arrays(run.previous.OUT/f"s1/{ref_name}/epoch{epoch}"/(seq.replace("/", "_")+".npz"))
                pos = r["min_past_gt_dist"] <= 5
                correct, old_correct = (pos & (v["top1_geo_dist"] <= 5) for v in (r, ref))
                records[seq] = {"gained": int((correct & ~old_correct).sum()), "lost": int((~correct & old_correct).sum()),
                    "directions": {str(g): {"positive": int((pos & (r["group"] == g)).sum()),
                        "gained": int((correct & ~old_correct & (r["group"] == g)).sum()),
                        "lost": int((~correct & old_correct & (r["group"] == g)).sum())} for g in (0, 1, 2)}}
            transitions[arm][ref_name] = {"per_sequence": records,
                "gained": sum(x["gained"] for x in records.values()), "lost": sum(x["lost"] for x in records.values())}
    write_json(run.OUT/"diagnostics.json", {"per_sequence": details, "totals": totals,
        "transitions": transitions, "fixed_tau": tau, "test_used": False,
        "scope": "공통양성10036의실제후보누락·정답개수,계열평균AP와다름. 수락개수는기존고정tau이며모델별새최대F1임계값이아님."})
    table = []
    for arm, data in {**summary["per_seed"]["1"], **summary["references"]}.items():
        selected = data["selected"]
        for scope, metrics in (("macro", selected[run.METHOD]["macro"]), *selected[run.METHOD]["families"].items()):
            table.append({"arm": arm, "epoch": selected["epoch"], "scope": scope,
                          **{m: 100*v for m, v in metrics.items()}})
    with (run.OUT/"final_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for p in (Path(__file__), run.OUT/"protocol.json", run.OUT/"summary.json", run.OUT/"diagnostics.json",
              run.OUT/"final_metrics.csv", run.OUT/"amp_smoke.json", REPO/"scripts/_pdv2_context_tail_finish.py",
              REPO/"scripts/_pdv2_temporal_test_control_finish.py"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT/"completion_validation.json", {"created_unix": time.time(), "sources": sources,
        "ledger_count": ledger_count, "rows": row_count, "metric_sets": metric_checks,
        "scalar_candidate_checks": scalar_checks, "prefix_checks": prefix_checks,
        "max_metric_error": max_metric_error, "max_scalar_error": max_scalar_error,
        "max_prefix_key_error": max_prefix_error, "max_zero_replay_key_error": max_zero_key_error,
        "initial_tensors_identical": True, "training_triplets_and_orders_identical": True,
        "updates_each": 900, "zero_training_reused": True, "zero_ledger_replay_exact": True, "test_used": False})
    print("VERIFIED", ledger_count, row_count, metric_checks, scalar_checks, prefix_checks, flush=True)
    print("TOTALS", totals, flush=True)
    print("TRANSITIONS", {a: {b: {k: v for k,v in x.items() if k != "per_sequence"} for b,x in records.items()} for a,records in transitions.items()}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
