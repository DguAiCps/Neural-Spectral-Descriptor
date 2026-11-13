#!/usr/bin/env python3
"""오답 제외 대조의 실제 학습 순서와 최종 판정 원장을 독립 검산한다."""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_negative_exclusion as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json


def aggregate(rows: dict) -> dict:
    """시퀀스→세 계열 평균을 독립 구현으로 재계산한다."""
    families = {}
    for family in ("kitti", "helipr", "mulran"):
        values = [r for s, r in rows.items() if s.startswith(family) and r["n_pos"]]
        families[family] = {m: sum(r[m] for r in values)/len(values) for m in run.base.exp.METRICS}
    return {"families": families,
            "macro": {m: sum(r[m] for r in families.values())/3 for m in run.base.exp.METRICS}}


def counts(row: dict, coverage: dict, tau: float) -> dict:
    """같은 양성 분모의 후보 누락과 실제 판정 수를 센다."""
    pos, neg = row["min_past_gt_dist"] <= 5, row["min_past_gt_dist"] > 25
    correct = pos & (row["top1_geo_dist"] <= 5)
    accept = row[run.METHOD] >= tau
    values = {"queries": len(pos), "positive": int(pos.sum()), "negative": int(neg.sum()),
        "correct": int(correct.sum()), "true_accept": int((correct & accept).sum()),
        "wrong_positive_accept": int((pos & ~correct & accept).sum()),
        "negative_false_accept": int((neg & accept).sum())}
    if coverage:
        hit = coverage["union_hit"]
        assert not np.any(correct & ~hit)
        values.update({"key_hit": int((pos & coverage["key_hit"]).sum()),
            "aux_hit": int((pos & coverage["aux_hit"]).sum()), "union_hit": int((pos & hit).sum()),
            "all_positives_absent": int((pos & ~hit).sum()),
            "wrong_with_positive_in_pool": int((pos & hit & ~correct).sum())})
    return values


def main() -> None:
    """학습·원장·최종 점수를 검사하고 조건별 손익을 보존한다."""
    spec, summary = run.protocol(True), run.read(run.OUT/"summary.json")
    assert summary["protocol_sha256"] == sha256(run.OUT/"protocol.json")
    sources = {**spec["runtime"], **spec["inputs"]}
    initial = run.read(run.reference_folder("center_zero")/"initial_model.json")
    orders = run.read(run.reference_folder("center_zero")/"training_complete.json")["orders"]
    tau = run.base.fixed.choose_tau(1)["tau"]
    totals, details, transitions, all_rows, loss_totals = {}, {}, {}, {}, {}
    ledgers = rows_count = metric_sets = scalar_checks = prefix_checks = reference_checks = 0
    max_metric_error = max_scalar_error = max_prefix_error = 0.
    for arm, data in summary["per_seed"]["1"].items():
        folder = run.base.root(1, arm)
        assert run.read(folder/"initial_model.json") == initial
        complete = run.read(folder/"training_complete.json")
        assert complete["updates"] == 900 and complete["orders"] == orders
        progress = run.read(folder/"training_progress.json")["records"]
        assert [r["updates"] for r in progress] == list(range(45, 901, 45))
        assert all(np.isfinite(r["loss"]) for r in progress)
        loss_audit = run.read(folder/"loss_batch_audit.json")
        assert len(loss_audit["epochs"]) == 20
        batches = []
        for epoch, (record, order) in enumerate(zip(loss_audit["epochs"], orders), 1):
            assert record["epoch"] == epoch and record["triplets"] == order["triplets"]
            assert len(record["batches"]) == 179 and sum(r["rows"] for r in record["batches"]) == 183249
            batches.extend(record["batches"])
        loss_totals[arm] = {k: sum(r[k] for r in batches) for k in batches[0]}
        assert loss_totals[arm]["zero_negative_rows"] == 0
        if arm == "diff_mask5":
            assert loss_totals[arm]["hard_excluded"] == 0
        for filename in ("initial_model.json", "training_complete.json", "training_progress.json", "loss_batch_audit.json"):
            p = folder/filename
            sources[str(p.relative_to(REPO))] = sha256(p)
        for filename, expected in complete["checkpoints"].items():
            p = folder/"checkpoints"/filename
            assert sha256(p) == expected
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            epoch = ckpt["epoch"]+1
            assert epoch in (10, 20)
            assert all(int(r["step"].item()) == epoch*45 for r in ckpt["optimizer_state_dict"]["state"].values())
            assert all(torch.isfinite(v).all() for v in ckpt["model_state_dict"].values())
            sources[str(p.relative_to(REPO))] = expected
        all_rows[arm], details[arm] = {}, {}
        candidates = []
        for epoch in (10, 20):
            metric_rows = {m: {} for m in ("raw", run.METHOD)}
            for sequence in spec["val"]:
                _, _, paths, _ = run.base.exp.old.base.inputs("val", sequence)
                cache = run.base.exp.arrays(paths["cache"])
                path = folder/f"epoch{epoch}"/(sequence.replace("/", "_")+".npz")
                info = run.read(path.with_suffix(".json"))
                kp, cp = path.with_name("keys_"+path.name), path.with_name("coverage_"+path.name)
                for p, h in ((path, info["sha256"]), (kp, info["keys_sha256"]),
                             (cp, run.read(cp.with_suffix(".json"))["sha256"])):
                    assert sha256(p) == h
                    sources[str(p.relative_to(REPO))] = h
                for p in (path.with_suffix(".json"), cp.with_suffix(".json")):
                    sources[str(p.relative_to(REPO))] = sha256(p)
                row, key_data, coverage = (run.base.exp.arrays(p) for p in (path, kp, cp))
                q, j, key = row["query_idx"], row["j_star"], key_data["embedding"]
                np.testing.assert_array_equal(q, np.arange(30, len(key)-15))
                np.testing.assert_array_equal(key_data["scan_ids"], cache["scan_ids"])
                ref = run.base.exp.arrays(run.reference_folder("center_zero")/f"epoch{epoch}"/path.name)
                for field in ("query_idx", "group", "min_past_gt_dist"):
                    np.testing.assert_array_equal(row[field], ref[field])
                for field in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist"):
                    np.testing.assert_array_equal(row[field], coverage[field])
                np.testing.assert_array_equal(coverage["union_hit"], coverage["key_hit"] | coverage["aux_hit"])
                xyz = cache["poses_gt"][:, :3, 3]
                np.testing.assert_allclose(np.linalg.norm(xyz[q]-xyz[j], axis=1), row["top1_geo_dist"], atol=1e-10, rtol=0)
                assert bool(((j >= 0) & (j <= q-30)).all())
                assert key.shape == (len(xyz), 416) and np.isfinite(key).all()
                np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1., atol=2e-6)
                for method in metric_rows:
                    measured = independent_metrics(row, row[method])
                    for name, value in measured.items():
                        expected = info["metrics"][method][name]
                        if value is None:
                            assert expected is None
                        else:
                            max_metric_error = max(max_metric_error, abs(value-expected))
                            np.testing.assert_allclose(value, expected, atol=5e-13, rtol=0)
                    metric_rows[method][sequence] = measured
                    metric_sets += 1
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
                    for name, candidates_ids in (("key_hit", a), ("aux_hit", b), ("union_hit", ids)):
                        assert bool((np.linalg.norm(xyz[candidates_ids]-xyz[t], axis=1) <= 5).any()) == coverage[name][ix]
                    score = scalar_score(bank, travel, t, winner, conf["decision"], float(row["raw"][ix]))
                    max_scalar_error = max(max_scalar_error, abs(score-row[run.METHOD][ix]))
                    assert abs(score-row[run.METHOD][ix]) < 1e-12
                    scalar_checks += 1
                for check in info["prefixes"]:
                    assert check["winner_equal"] and check["score_error"] == 0
                    max_prefix_error = max(max_prefix_error, check["key_max_error"])
                    prefix_checks += 1
                if epoch == data["selected"]["epoch"]:
                    all_rows[arm][sequence] = row
                    details[arm][sequence] = counts(row, coverage, tau)
                ledgers += 1
                rows_count += len(q)
            combined = {method: aggregate(v) for method, v in metric_rows.items()}
            expected = next(c for c in data["candidates"] if c["epoch"] == epoch)
            for method in combined:
                for scope in ("macro",):
                    for metric in run.base.exp.METRICS:
                        np.testing.assert_allclose(combined[method][scope][metric], expected[method][scope][metric], atol=5e-13)
            candidates.append({"epoch": epoch, **combined})
        chosen = max(candidates, key=lambda x: (round(x[run.METHOD]["macro"]["AP"], 12), -x["epoch"]))
        assert chosen["epoch"] == data["selected"]["epoch"]
    # 동결된 기존 대조의 선택 지표도 실제 원장에서 독립 재집계한다.
    for name, saved in summary["references"].items():
        epoch = saved["selected"]["epoch"]
        all_rows[name], details[name], metrics = {}, {}, {}
        for sequence in spec["val"]:
            path = run.reference_folder(name)/f"epoch{epoch}"/(sequence.replace("/", "_")+".npz")
            row = run.base.exp.arrays(path)
            cp = path.with_name("coverage_"+path.name)
            coverage = run.base.exp.arrays(cp) if cp.exists() else {}
            all_rows[name][sequence] = row
            details[name][sequence] = counts(row, coverage, tau)
            metrics[sequence] = independent_metrics(row, row[run.METHOD])
            reference_checks += 1
        values = aggregate(metrics)
        for m in run.base.exp.METRICS:
            np.testing.assert_allclose(values["macro"][m], saved["selected"][run.METHOD]["macro"][m], atol=5e-13)
    for arm, records in details.items():
        totals[arm] = {k: sum(r[k] for r in records.values()) for k in next(iter(records.values()))}
    for arm in run.ARMS:
        ref_name = run.CFG["models"][arm]
        record = {}
        for sequence in spec["val"]:
            row, ref = all_rows[arm][sequence], all_rows[ref_name][sequence]
            pos = row["min_past_gt_dist"] <= 5
            correct, old_correct = (pos & (r["top1_geo_dist"] <= 5) for r in (row, ref))
            record[sequence] = {"gained": int((correct & ~old_correct).sum()),
                "lost": int((~correct & old_correct).sum()),
                "directions": {str(g): {"positive": int((pos & (row["group"] == g)).sum()),
                    "gained": int((correct & ~old_correct & (row["group"] == g)).sum()),
                    "lost": int((~correct & old_correct & (row["group"] == g)).sum())} for g in (0, 1, 2)}}
        transitions[arm] = {"reference": ref_name, "per_sequence": record,
            "gained": sum(v["gained"] for v in record.values()), "lost": sum(v["lost"] for v in record.values())}
    for arm in ("full_mask25", "standard_mask25"):
        assert loss_totals[arm] == loss_totals["diff_mask25"]
    write_json(run.OUT/"diagnostics.json", {"totals": totals, "per_sequence": details,
        "transitions": transitions, "loss_totals": loss_totals, "fixed_tau": tau, "test_used": False})
    table = []
    for arm, data in {**summary["per_seed"]["1"], **summary["references"]}.items():
        s = data["selected"]
        for scope, metrics in (("macro", s[run.METHOD]["macro"]), *s[run.METHOD]["families"].items()):
            table.append({"arm": arm, "epoch": s["epoch"], "scope": scope, **{m: 100*v for m, v in metrics.items()}})
    with (run.OUT/"final_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for p in (Path(__file__), run.OUT/"protocol.json", run.OUT/"summary.json", run.OUT/"diagnostics.json",
              run.OUT/"final_metrics.csv", run.OUT/"training_diagnosis.json",
              REPO/"scripts/_pdv2_context_tail_finish.py", REPO/"scripts/_pdv2_temporal_test_control_finish.py"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT/"completion_validation.json", {"created_unix": time.time(), "sources": sources,
        "status": "passed", "ledgers": ledgers, "rows": rows_count, "metric_sets": metric_sets,
        "reference_selected_ledgers": reference_checks, "candidate_scalar_checks": scalar_checks,
        "prefix_checks": prefix_checks, "max_metric_error": max_metric_error,
        "max_scalar_error": max_scalar_error, "max_prefix_key_error": max_prefix_error,
        "initial_tensors_identical": True, "training_triplets_and_orders_identical": True,
        "loss_calls": 4*20*179, "updates_each": 900, "test_used": False})
    print("VERIFIED", ledgers, rows_count, metric_sets, scalar_checks, prefix_checks, flush=True)
    print("TOTALS", totals, flush=True)
    print("LOSS", loss_totals, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
