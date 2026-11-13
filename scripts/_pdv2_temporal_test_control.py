#!/usr/bin/env python3
"""선택된 간선 없는 NSD 대조군을 고정 시험에 적용한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_sequence_fixed_test as fixed
import _pdv2_spectral_evidence as exp
import _pdv2_temporal_capacity as capacity
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from _pr_eval_lib import classify_queries, operating_point
from _causal_pr_dump import prefix_topk, minmax, align_scores, column_align_scores

OUT = REPO / "results/pdv2_temporal_test_control_20261020"
LOG = REPO / "logs/temporal_test_control_20261020"
METHOD = fixed.METHOD
SCORES = ("raw", METHOD)
REPLAY = ("kitti360/0004", "helipr/Roundabout02")


def freeze(check_inputs: bool = False) -> dict:
    """동결 코드·체크포인트·참조 결과를 확인한다."""
    spec = exp.read(OUT / "freeze.json")
    for field in ("runtime_sha256", "sources_sha256") if check_inputs else ("runtime_sha256",):
        for name, expected in spec[field].items():
            assert sha256(REPO / name) == expected, name
    for model in spec["models"].values():
        for row in model.values():
            assert sha256(REPO / row["checkpoint"]) == row["checkpoint_sha256"]
    return spec


def prepare() -> dict:
    """새 시험 결과를 계산하기 전에 비교와 기존 선택을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "freeze.json").exists():
        return freeze(True)
    reference = fixed.check_freeze()
    original = exp.protocol(True)
    cap = capacity.protocol(True)
    assert exp.read(capacity.OUT / "completion_validation.json")["status"] == "passed"
    runtime = {**reference["runtime_sha256"], **cap["runtime_sha256"]}
    runtime[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    sources = dict(original["input_sha256"])
    models = {"standard": reference["seeds"], "capacity_point": {}}
    for seed in (1, 2, 3):
        chosen = original["selected"]["capacity_point"][str(seed)]
        assert chosen["epoch"] == 20
        path = capacity.checkpoint_dir(seed, "point_original") / "checkpoint_epoch_20.pth"
        complete = exp.read(capacity.OUT / f"s{seed}/point_original/training_complete.json")
        assert sha256(path) == complete["checkpoints"][path.name]
        shared = reference["seeds"][str(seed)]
        models["capacity_point"][str(seed)] = {
            "epoch": 20, "checkpoint": str(path.relative_to(REPO)),
            "checkpoint_sha256": sha256(path), "confidence": shared["confidence"],
            "new_tau": shared["new_tau"], "validation_folder": chosen["folder"],
        }
        for sensor, seq in exp.old.base.VAL:
            p = fixed.VAL / f"s{seed}/capacity_point_{sensor}_{seq}.npz"
            sources[str(p.relative_to(REPO))] = sha256(p)
    for p in (fixed.OUT / "freeze.json", fixed.OUT / "summary.json",
              fixed.OUT / "completion_validation.json", capacity.OUT / "protocol.json",
              capacity.OUT / "summary.json", fixed.VAL / "summary.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for sensor, seq in fixed.TEST:
        sequence = sensor + "/" + seq
        _, _, paths, _ = exp.old.base.inputs("test", sequence)
        for p in paths.values():
            sources[str(p.relative_to(REPO))] = sha256(p)
        p = fixed.OUT / "test" / f"{sensor}_{seq}.json"
        record = exp.read(p)
        sources[str(p.relative_to(REPO))] = sha256(p)
        for row in record["results"].values():
            p = REPO / row["ledger"]
            assert sha256(p) == row["sha256"]
            sources[str(p.relative_to(REPO))] = row["sha256"]
    value = {
        "created_unix": time.time(), "models": models, "method": METHOD,
        "scores": list(SCORES), "test_roster": [s + "/" + q for s, q in fixed.TEST],
        "replay_roster": list(REPLAY), "runtime_sha256": runtime,
        "sources_sha256": sources, "test_used_for_selection": False,
        "scope": "사용자가 요청한 시험 대조 추가. 기존 시간 모델의 시험 결과를 본 이후의 분석이며 사전 등록 확증으로 주장하지 않는다. 검증에서 이미 선택한 용량 대조 20에포크·3시드와 판정식을 고정한다. 재학습·새 후보·시험 기반 선택·원고 자동 교체 없음.",
        "comparison": "같은 KISS 입력·q+15·j<=q-30·100+100·시드별 융합과 판정 채널. 각 모델의 후보와 승자는 다시 계산한다. 195726질의·양성45시퀀스→6도시→3시드. 현재 시간 모델의 보존 원장을 참조한다.",
        "control": "간선 없는 투영167705파라미터 대 시간 모델167561. 학습 표본·순서·예산과 입력/출력/게이트/정규화를 맞췄으나 연산 함수 형태와 선택 에포크는 같지 않다. 최종 스펙트럼 시간 판정은 양쪽 모두 유지하며 모든 시간 문맥 제거 대조가 아니다.",
        "fixed_threshold": "배포 운영점의 부가 비교는 기존 시간 모델의 검증 선택 임계값을 두 모델에 공통 적용한다. AP/최대F1/R100은 임계값 곡선 지표다.",
    }
    write_json(OUT / "freeze.json", value)
    print("FROZEN", len(runtime), len(sources), flush=True)
    return value


def worker(sequence: str, role: str, model: str) -> None:
    """선택된 모델의 검색 키·최종 원장을 별도 경로에 저장한다."""
    import torch
    from _pdv2_pair_evaluate import ArrayCache
    spec = freeze()
    destination = OUT / role / model
    target = destination / (sequence.replace("/", "_") + ".json")
    if target.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq, paths, min_past = exp.old.base.inputs(role, sequence)
    cache = ArrayCache(exp.arrays(paths["cache"]))
    raw = cache["descriptors"].astype(np.float32)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = exp.old.base.bank_for(exp.OP, z)
    arm = capacity.configure() if model == "capacity_point" else "standard_original"
    records = {}
    for seed in (1, 2, 3):
        f = spec["models"][model][str(seed)]
        cfg, net = exp.old.load_model(seed, arm, f["epoch"])
        tick = time.perf_counter()
        kind = "point" if model == "capacity_point" else "temporal"
        emb, graph = exp.old.old.embedding(cache, sensor, raw, cfg, net, kind)
        assert emb.shape == (len(raw), 416) and np.isfinite(emb).all()
        if kind == "point":
            assert graph["edges"] == 0
        key_seconds = time.perf_counter() - tick
        conf = f["confidence"]
        tick = time.perf_counter()
        a = evaluate_keys(emb, bank, cache["poses_gt"][:, :3, 3], min_past,
                          {seed: tuple(float(w) for w in conf["tuple"])})[seed]
        keep = a["query_idx"] < len(raw) - 15
        a = {k: v[keep] for k, v in a.items()}
        retrieval_seconds = time.perf_counter() - tick
        ref_path = (fixed.OUT / "test" / f"s{seed}/{sensor}_{seq}.npz" if role == "test"
                    else fixed.VAL / f"s{seed}/{model}_{sensor}_{seq}.npz")
        ref = exp.arrays(ref_path)
        for key in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(a[key], ref[key])
        a["group"] = ref["group"]
        q, j = a["query_idx"], a["j_star"]
        assert (j <= q - 30).all()
        tick = time.perf_counter()
        chunks = [fixed.frozen_scores(bank, travel, q[start:start+256], j[start:start+256],
                  conf["decision"], a[conf["decision"]][start:start+256])
                  for start in range(0, len(q), 256)]
        scores = {key: np.concatenate([c[key] for c in chunks]) for key in SCORES}
        score_seconds = time.perf_counter() - tick
        if role == "val":
            for key in ("query_idx", "j_star", "top1_geo_dist", *SCORES):
                np.testing.assert_array_equal(scores[key] if key in SCORES else a[key], ref[key])
            if kind == "point":
                old = REPO / f["validation_folder"] / f"keys_val_{sensor}_{seq}.npz"
                np.testing.assert_array_equal(emb, exp.arrays(old)["embedding"])
        prefixes = []
        for i in (len(q) // 3, 2 * len(q) // 3):
            end = int(q[i] + 16)
            small = fixed.frozen_scores(exp.truncate_bank(bank, end), travel[:end],
                                        q[i:i+1], j[i:i+1], conf["decision"],
                                        a[conf["decision"]][i:i+1])
            for key in SCORES:
                np.testing.assert_array_equal(small[key], scores[key][i:i+1])
            error, prefix_winner = None, None
            if kind == "point":
                prefix, _ = exp.old.old.embedding(cache, sensor, raw[:end], cfg, net, "point")
                assert prefix.shape == emb[:end].shape and np.isfinite(prefix).all()
                error = float(np.max(np.abs(prefix - emb[:end])))
                t = int(q[i])
                hkey, hfft, hnorm, column = bank
                ids = np.unique(np.r_[prefix_topk(prefix, t, t - 29, 100),
                                      prefix_topk(hkey, t, t - 29, 100)])
                channels = (prefix[ids] @ prefix[t], hkey[ids] @ hkey[t],
                            align_scores(hfft, hnorm, t, ids), column_align_scores(column, t, ids))
                fused = sum(float(w) * minmax(v) for w, v in zip(conf["tuple"], channels))
                prefix_winner = int(ids[np.argsort(-fused, kind="stable")[0]])
            prefixes.append({"q": int(q[i]), "score_max_error": 0., "key_max_error": error,
                             "prefix_winner": prefix_winner, "full_winner": int(j[i]),
                             "winner_equal": None if prefix_winner is None else prefix_winner == int(j[i])})
        folder = destination / f"s{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{sensor}_{seq}.npz"
        np.savez_compressed(path, **{key: a[key] for key in
                            ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist", "group")},
                            **scores)
        keypath = folder / f"keys_{sensor}_{seq}.npz"
        np.savez_compressed(keypath, embedding=emb, scan_ids=cache["scan_ids"])
        records[str(seed)] = {
            "ledger": str(path.relative_to(REPO)), "sha256": sha256(path),
            "keys": str(keypath.relative_to(REPO)), "keys_sha256": sha256(keypath),
            "checkpoint_sha256": f["checkpoint_sha256"], "graph": graph,
            "metrics": {key: exp.metrics(scores[key], a) for key in SCORES},
            "fixed_tau": operating_point(scores[METHOD], a["top1_geo_dist"],
                                         classify_queries(a["min_past_gt_dist"]), f["new_tau"]["tau"]),
            "prefix_checks": prefixes, "timing_s": {
                "keys": key_seconds, "retrieval": retrieval_seconds, "scores": score_seconds},
        }
        del net, emb
        torch.cuda.empty_cache()
    write_json(target, {"sequence": sequence, "role": role, "model": model, "results": records,
                       "freeze_sha256": sha256(OUT / "freeze.json"),
                       "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()}})


def report() -> None:
    """같은 도시·시드의 최종 차이와 승자 변화를 계산한다."""
    spec = freeze(True)
    all_records = {
        model: {s + "/" + q: exp.read(
            (fixed.OUT / "test" if model == "standard" else OUT / "test" / model)
            / f"{s}_{q}.json")["results"] for s, q in fixed.TEST}
        for model in ("standard", "capacity_point")
    }
    per_seed, means, details, transitions = {}, {}, {}, {}
    for model, records in all_records.items():
        per_seed[model], details[model] = {}, {}
        for seed in ("1", "2", "3"):
            details[model][seed] = {}
            for sequence, record in records.items():
                row = record[seed]
                p = REPO / row["ledger"]
                assert sha256(p) == row["sha256"]
                a = exp.arrays(p)
                for method in SCORES:
                    assert exp.metrics(a[method], a) == row["metrics"][method]
                details[model][seed][sequence] = row["metrics"]
            per_seed[model][seed] = {}
            for method in SCORES:
                cities = {}
                for city, members in fixed.CITY.items():
                    chosen = [records[seq][seed]["metrics"][method] for seq in members
                              if records[seq][seed]["metrics"][method]["n_pos"]]
                    cities[city] = {m: float(np.mean([r[m] for r in chosen])) for m in exp.METRICS}
                per_seed[model][seed][method] = {
                    "cities": cities, "macro": {m: float(np.mean([r[m] for r in cities.values()]))
                                              for m in exp.METRICS}}
        means[model] = {
            method: {m: {"mean": float(np.mean([per_seed[model][s][method]["macro"][m] for s in ("1", "2", "3")])),
                         "std": float(np.std([per_seed[model][s][method]["macro"][m] for s in ("1", "2", "3")], ddof=1))}
                     for m in exp.METRICS} for method in SCORES}
    reference = exp.read(fixed.OUT / "summary.json")
    for method in SCORES:
        assert means["standard"][method] == reference["mean_std"][method]
    delta = {
        method: {m: [100 * (per_seed["standard"][s][method]["macro"][m]
                            - per_seed["capacity_point"][s][method]["macro"][m]) for s in ("1", "2", "3")]
                 for m in exp.METRICS} for method in SCORES}
    cities = {
        city: {m: float(np.mean([100 * (per_seed["standard"][s][METHOD]["cities"][city][m]
                                       - per_seed["capacity_point"][s][METHOD]["cities"][city][m])
                                 for s in ("1", "2", "3")])) for m in exp.METRICS}
        for city in fixed.CITY}
    for seed in ("1", "2", "3"):
        transitions[seed] = {}
        for sequence in spec["test_roster"]:
            a, b = [exp.arrays(REPO / all_records[model][sequence][seed]["ledger"])
                    for model in ("capacity_point", "standard")]
            for key in ("query_idx", "min_past_gt_dist", "group"):
                np.testing.assert_array_equal(a[key], b[key])
            positive = a["min_past_gt_dist"] <= 5
            ca, cb = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
            transitions[seed][sequence] = {
                "queries": len(positive), "winner_changed": int((a["j_star"] != b["j_star"]).sum()),
                "correct_gained": int((positive & ~ca & cb).sum()),
                "correct_lost": int((positive & ca & ~cb).sum()),
            }
        assert sum(v["queries"] for v in transitions[seed].values()) == 195726
    write_json(OUT / "summary.json", {
        "per_seed": per_seed, "mean_std": means, "per_sequence": details,
        "temporal_minus_point_pp": delta, "city_temporal_minus_point_pp": cities,
        "transitions": transitions, "test_used": True, "test_used_for_selection": False,
        "freeze_sha256": sha256(OUT / "freeze.json"), "scope": spec["scope"],
    })
    print("FINAL_TEST", {m: means[m][METHOD] for m in means}, "DELTA_PP", delta[METHOD], flush=True)


def run() -> None:
    """검증 재현 후 고정 대조군의 시험 원장을 생성한다."""
    prepare()
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    def invoke(sequence: str, role: str, model: str) -> None:
        path = LOG / f"{role}_{model}_{sequence.replace('/', '_')}.log"
        with path.open("a") as handle:
            p = subprocess.run([sys.executable, str(Path(__file__)), "worker",
                                "--sequence", sequence, "--role", role, "--model", model],
                               stdout=handle, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(path.read_text()[-6000:])
    for model in ("standard", "capacity_point"):
        for sequence in REPLAY:
            invoke(sequence, "val", model)
            print("REPLAY", model, sequence, flush=True)
    write_json(OUT / "validation_replay.json", {"status": "passed", "sequences": list(REPLAY),
               "models": ["standard", "capacity_point"], "seeds": [1, 2, 3],
               "queries_winners_scores_exact": True, "point_keys_exact": True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(invoke, s + "/" + q, "test", "capacity_point"): s + "/" + q
                for s, q in fixed.TEST}
        for done, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            future.result()
            write_json(OUT / "status.json", {"status": "running", "done": done, "total": 46,
                       "elapsed_s": time.time() - started, "test_used": True})
            print("TEST", done, 46, jobs[future], flush=True)
    report()
    write_json(OUT / "status.json", {"status": "evaluated_pending_audit",
               "runtime_s": time.time() - started, "test_used": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "report", "run"))
    parser.add_argument("--sequence")
    parser.add_argument("--role", choices=("val", "test"), default="test")
    parser.add_argument("--model", choices=("standard", "capacity_point"), default="capacity_point")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.sequence, args.role, args.model)
    else:
        globals()[args.action]()
