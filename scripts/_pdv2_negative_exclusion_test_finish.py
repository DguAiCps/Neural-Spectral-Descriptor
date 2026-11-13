#!/usr/bin/env python3
"""고정 시험의 최종 지표·공통 질의·검색·점수를 독립 검산한다."""
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_negative_exclusion_test as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json


def city_macro(rows: dict, cities: dict) -> dict:
    """평가 구현과 별도로 시퀀스와 도시의 동일 비중 평균을 구한다."""
    groups = {}
    for city, sequences in cities.items():
        valid = [rows[s] for s in sequences if rows[s]["n_pos"] > 0]
        groups[city] = {m: sum(r[m] for r in valid)/len(valid) for m in run.base.exp.METRICS}
    return {"cities": groups, "macro": {m: sum(r[m] for r in groups.values())/len(groups) for m in run.base.exp.METRICS}}


def main() -> None:
    """새 모델 두 개와 보존 대조의 시험 원장을 확인해 결과를 보존한다."""
    spec, summary = run.freeze(True), run.read(run.OUT/"summary.json")
    assert summary["freeze_sha256"] == sha256(run.OUT/"freeze.json")
    sources = {**spec["runtime"], **spec["inputs"]}
    baseline_meta = run.read(REPO/"results/pdv2_main_official_baselines/baselines_common.json")["methods"]["bevplace"]
    assert baseline_meta["n_queries"] == 195726
    measured = {a: {m: {} for m in run.SCORES} for a in (*run.ARMS, "standard_seed1")}
    measured["bevplace_seed1"] = {run.METHOD: {}}
    per_sequence, transitions = {}, {}
    scalar_checks = prefixes = new_rows = metric_sets = reference_metric_sets = 0
    maximum_metric_error = maximum_key_error = maximum_scalar_error = 0.
    counts_by_arm = {a: [] for a in measured}
    for sensor, seq in run.fixed.TEST:
        sequence = sensor+"/"+seq
        _, _, paths, _ = run.base.exp.old.base.inputs("test", sequence)
        cache = run.base.exp.arrays(paths["cache"])
        xyz = cache["poses_gt"][:, :3, 3]
        position = cache["poses"][:, :3, 3]
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(position, axis=0), axis=1))]
        with np.load(paths["spec"]) as z:
            bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
        all_rows = {}
        per_sequence[sequence] = {}
        for arm in run.ARMS:
            path = run.OUT/"test"/arm/f"{sensor}_{seq}.npz"
            info = run.read(path.with_suffix(".json"))
            assert info["freeze_sha256"] == summary["freeze_sha256"]
            assert info["checkpoint_sha256"] == spec["models"][arm]["checkpoint_sha256"]
            kp, cp = path.with_name("keys_"+path.name), path.with_name("coverage_"+path.name)
            for p, expected in ((path, info["sha256"]), (kp, info["keys_sha256"]), (cp, info["coverage_sha256"])):
                assert sha256(p) == expected
                sources[str(p.relative_to(REPO))] = expected
            sources[str(path.with_suffix(".json").relative_to(REPO))] = sha256(path.with_suffix(".json"))
            row, keys, coverage = (run.base.exp.arrays(p) for p in (path, kp, cp))
            all_rows[arm] = row
            q, j, key = row["query_idx"], row["j_star"], keys["embedding"]
            np.testing.assert_array_equal(q, np.arange(30, len(xyz)-15))
            np.testing.assert_array_equal(keys["scan_ids"], cache["scan_ids"])
            assert key.shape == (len(xyz), 416) and np.isfinite(key).all()
            np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1., atol=2e-6, rtol=0)
            assert ((j >= 0) & (j <= q-30)).all()
            np.testing.assert_allclose(np.linalg.norm(xyz[q]-xyz[j], axis=1), row["top1_geo_dist"], atol=1e-10, rtol=0)
            for name in run.base.FIELDS:
                np.testing.assert_array_equal(row[name], coverage[name])
            np.testing.assert_array_equal(coverage["union_hit"], coverage["key_hit"] | coverage["aux_hit"])
            assert ((coverage["n_candidates"] > 0) & (coverage["n_candidates"] <= 200)).all()
            for method in run.SCORES:
                values = independent_metrics(row, row[method])
                for name, value in values.items():
                    expected = info["metrics"][method][name]
                    if value is None:
                        assert expected is None
                    else:
                        maximum_metric_error = max(maximum_metric_error, abs(value-expected))
                        np.testing.assert_allclose(value, expected, atol=5e-13, rtol=0)
                measured[arm][method][sequence] = values
                metric_sets += 1
            conf = spec["confidence"]
            for index in (0, len(q)//3, 2*len(q)//3, len(q)-1):
                t = int(q[index])
                a = run.base.prefix_topk(key, t, t-29, 100)
                b = run.base.prefix_topk(bank[0], t, t-29, 100)
                ids = np.unique(np.r_[a, b])
                channels = (key[ids]@key[t], bank[0][ids]@bank[0][t],
                    run.base.align_scores(bank[1], bank[2], t, ids), run.base.column_align_scores(bank[3], t, ids))
                fused = sum(float(w)*run.base.minmax(v) for w, v in zip(conf["tuple"], channels))
                best = int(np.argsort(-fused, kind="stable")[0])
                winner = int(ids[best])
                assert winner == int(j[index]) and len(ids) == coverage["n_candidates"][index]
                raw = float(channels[2 if conf["decision"] == "c_spec" else 3][best])
                np.testing.assert_allclose(raw, row["raw"][index], atol=2e-7, rtol=0)
                score = scalar_score(bank, travel, t, winner, conf["decision"], raw)
                maximum_scalar_error = max(maximum_scalar_error, abs(score-row[run.METHOD][index]))
                np.testing.assert_allclose(score, row[run.METHOD][index], atol=3e-7, rtol=0)
                np.testing.assert_allclose(np.linalg.norm(xyz[:t-29]-xyz[t], axis=1).min(), row["min_past_gt_dist"][index], atol=1e-10, rtol=0)
                for name, ids_probe in (("key_hit", a), ("aux_hit", b), ("union_hit", ids)):
                    assert bool((np.linalg.norm(xyz[ids_probe]-xyz[t], axis=1) <= 5).any()) == coverage[name][index]
                scalar_checks += 1
            for item in info["prefixes"]:
                assert item["winner_equal"] and item["score_error"] == 0
                maximum_key_error = max(maximum_key_error, item["key_max_error"])
                prefixes += 1
            positive = row["min_past_gt_dist"] <= 5
            correct = positive & (row["top1_geo_dist"] <= 5)
            assert not np.any(correct & ~coverage["union_hit"])
            per_sequence[sequence][arm] = {"queries": len(q), "positive": int(positive.sum()),
                "correct": int(correct.sum()), "key_hit": int((positive & coverage["key_hit"]).sum()),
                "union_hit": int((positive & coverage["union_hit"]).sum()),
                "all_positives_absent": int((positive & ~coverage["union_hit"]).sum()),
                "wrong_with_positive_in_pool": int((positive & coverage["union_hit"] & ~correct).sum())}
            counts_by_arm[arm].append(len(q))
            new_rows += len(q)
        q = all_rows["diff_mask25"]["query_idx"]
        standard_path = run.fixed.OUT/"test/s1"/f"{sensor}_{seq}.npz"
        standard = run.base.exp.arrays(standard_path)
        all_rows["standard_seed1"] = standard
        info = run.read(run.fixed.OUT/"test"/f"{sensor}_{seq}.json")["results"]["1"]
        assert sha256(standard_path) == info["sha256"]
        for method in run.SCORES:
            measured["standard_seed1"][method][sequence] = independent_metrics(standard, standard[method])
            for name, value in measured["standard_seed1"][method][sequence].items():
                expected = info["metrics"][method][name]
                if value is None:
                    assert expected is None
                else:
                    np.testing.assert_allclose(value, expected, atol=5e-13, rtol=0)
            reference_metric_sets += 1
        b = run.base.exp.arrays(REPO/f"results/pdv2_bevplace_official/odom/dump_bevplace_{sensor}_{seq}.npz")
        ix = np.searchsorted(b["query_idx"], q)
        bev = {k: b[k][ix] for k in ("query_idx", "min_past_gt_dist", "top1_geo_dist", "j_star", baseline_meta["score"])}
        all_rows["bevplace_seed1"] = bev
        measured["bevplace_seed1"][run.METHOD][sequence] = independent_metrics(bev, bev[baseline_meta["score"]])
        reference_metric_sets += 1
        for arm, row in all_rows.items():
            for name in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(row[name], all_rows["diff_mask25"][name])
            if arm not in run.ARMS:
                counts_by_arm[arm].append(len(row["query_idx"]))
        a, b = all_rows["diff_baseline"], all_rows["diff_mask25"]
        np.testing.assert_array_equal(a["group"], b["group"])
        pos = a["min_past_gt_dist"] <= 5
        ca, cb = (pos & (x["top1_geo_dist"] <= 5) for x in (a, b))
        transitions[sequence] = {"gained": int((~ca & cb).sum()), "lost": int((ca & ~cb).sum()),
            "directions": {str(g): {"positive": int((pos & (a["group"] == g)).sum()),
                "gained": int((~ca & cb & (a["group"] == g)).sum()),
                "lost": int((ca & ~cb & (a["group"] == g)).sum())} for g in (0, 1, 2)}}
    assert all(sum(v) == 195726 for v in counts_by_arm.values())
    aggregate = {a: {m: city_macro(rows, spec["cities"]) for m, rows in methods.items()} for a, methods in measured.items()}
    for arm in run.ARMS:
        for method in run.SCORES:
            for scope in ("macro",):
                for name in run.base.exp.METRICS:
                    np.testing.assert_allclose(aggregate[arm][method][scope][name], summary["aggregate"][arm][method][scope][name], atol=1e-13, rtol=0)
            for city, metrics in aggregate[arm][method]["cities"].items():
                for name, value in metrics.items():
                    np.testing.assert_allclose(value, summary["aggregate"][arm][method]["cities"][city][name], atol=1e-13, rtol=0)
    for arm, expected in (("standard_seed1", summary["references"]["standard_seed1"]["macro"]),
                          ("bevplace_seed1", summary["references"]["bevplace_seed1"])):
        for name, value in aggregate[arm][run.METHOD]["macro"].items():
            np.testing.assert_allclose(value, expected[name], atol=5e-13, rtol=0)
    for arm in run.ARMS:
        assert [seq for seq, row in measured[arm][run.METHOD].items() if not row["n_pos"]] == ["kitti/07"]
    totals = {arm: {k: sum(r[arm][k] for r in per_sequence.values()) for k in next(iter(per_sequence.values()))[arm]}
              for arm in run.ARMS}
    write_json(run.OUT/"diagnostics.json", {"totals": totals, "per_sequence": per_sequence,
        "transitions": transitions, "gained": sum(r["gained"] for r in transitions.values()),
        "lost": sum(r["lost"] for r in transitions.values()), "test_used": True})
    comparison = []
    for arm, methods in aggregate.items():
        final = methods[run.METHOD]
        for scope, metrics in (("macro", final["macro"]), *final["cities"].items()):
            comparison.append({"model": arm, "seed": 1, "scope": scope,
                               **{name: 100*v for name, v in metrics.items()}})
    with (run.OUT/"final_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    replay_key_error = 0.
    for arm in run.ARMS:
        for seq in run.REPLAY:
            p = run.OUT/"val"/arm/(seq.replace("/", "_")+".json")
            info = run.read(p)
            assert info["validation_ledger_exact"]
            replay_key_error = max(replay_key_error, info["validation_key_max_error"])
            for path in (p, p.with_suffix(".npz"), p.with_name("keys_"+p.stem+".npz"), p.with_name("coverage_"+p.stem+".npz")):
                sources[str(path.relative_to(REPO))] = sha256(path)
    for p in (Path(__file__), REPO/"scripts/_pdv2_context_tail_finish.py", REPO/"scripts/_pdv2_temporal_test_control_finish.py",
              run.OUT/"summary.json", run.OUT/"freeze.json", run.OUT/"diagnostics.json", run.OUT/"final_metrics.csv",
              run.OUT/"validation_replay.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT/"completion_validation.json", {"status": "passed", "created_unix": time.time(), "sources": sources,
        "new_ledgers": 92, "new_rows": new_rows, "metric_sets": metric_sets,
        "reference_metric_sets": reference_metric_sets, "candidate_scalar_checks": scalar_checks,
        "prefix_checks": prefixes, "max_metric_error": maximum_metric_error,
        "max_scalar_score_error": maximum_scalar_error, "max_prefix_key_error": maximum_key_error,
        "validation_replay_max_key_error": replay_key_error, "validation_replay_ledgers": 6,
        "common_queries_each": 195726, "positive_sequences": 45, "cities": 6, "seed": 1,
        "test_used": True, "test_used_for_selection": False})
    print("VERIFIED", new_rows, metric_sets, reference_metric_sets, scalar_checks, prefixes, flush=True)
    print("FINAL", {a: m[run.METHOD]["macro"] for a, m in aggregate.items()}, flush=True)
    print("TOTALS", totals, flush=True)


if __name__ == "__main__":
    run.torch.set_num_threads(1)
    run.torch.backends.cuda.matmul.allow_tf32 = False
    run.torch.backends.cudnn.allow_tf32 = False
    main()
