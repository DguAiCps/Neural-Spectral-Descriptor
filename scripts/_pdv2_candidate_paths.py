#!/usr/bin/env python3
"""고정된 NSD 두 모델에서 후보 대응의 시간 제약을 최종 판정으로 비교한다."""
import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
import _pdv2_sequence_fixed_test as fixed
from _pdv2_context_accept import macro
from _pdv2_no_context_ablation import sha256, write_json
from _causal_pr_dump import prefix_topk
from _xs_eval_lib import align_scores, column_align_scores, minmax
from retrieval.candidate_paths import select_paths

OUT = REPO / "results/pdv2_candidate_paths_20261020"
CONFIG = REPO / "configs/pdv2_candidate_paths.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
FIELDS = ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")


def protocol(check_inputs: bool = False) -> dict:
    """실행 전 고정한 코드와 입력을 확인한다."""
    spec = exp.read(OUT / "protocol.json")
    for name in (("runtime", "inputs") if check_inputs else ("runtime",)):
        for p, h in spec[name].items():
            assert sha256(REPO / p) == h, p
    return spec


def prepare() -> None:
    """결과 관측 전에 선택 규칙·범위·기존 체크포인트를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    prior = exp.protocol(True)
    runtime = dict(prior["runtime_sha256"])
    for p in (Path(__file__), CONFIG, REPO / "src/retrieval/candidate_paths.py",
              REPO / "tests/retrieval/test_candidate_paths.py",
              REPO / "scripts/_pdv2_sequence_fixed_test.py",
              REPO / "scripts/_pdv2_context_accept.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    inputs = dict(prior["input_sha256"])
    for seed in (1, 2, 3):
        for model in CFG["models"]:
            for sequence in prior["val"]:
                p = fixed.VAL / f"s{seed}/{model}_{sequence.replace('/', '_')}.npz"
                inputs[str(p.relative_to(REPO))] = sha256(p)
            selection = prior["selected"][model][str(seed)]
            epoch = selection["epoch"]
            ck = (exp.old.checkpoint_dir(seed, "standard_original") if model == "standard"
                  else REPO / f"checkpoints/pdv2_temporal_capacity_s{seed}_point_original")
            p = ck / f"checkpoint_epoch_{epoch}.pth"
            inputs[str(p.relative_to(REPO))] = sha256(p)
    p = exp.OUT / "protocol.json"
    inputs[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "selected": prior["selected"],
        "val": prior["val"], "runtime": runtime, "inputs": inputs, "test_used": False,
        "scope": "이미 검증으로 선택한 표준/용량투영 모델을 고정한다. 재학습/새 체크포인트 선택 없음. 검증9·99236질의,1시드 예비. 신경망 모듈 추가 없음.",
        "candidate": "프레임별 j<=t-30인 100+100 합집합과 기존 융합 유지. 융합 상위20개만 경로 상태로 쓰며 기준 승자는 반드시 포함. GT는 출력의 사후 정답/후보 상한 진단에만 사용.",
        "operator": "q±5의11노드와 인접 시간 간선 사슬. u=기존융합점수/가중치합. null u=.5. 정/역방향 전체창 가설 각각 최대합 DP. 전이비용=.25*min(|signed ds_map-ds_query|/max(ds_query,.8),3). signed ds_map<-.8 또는 >3ds_query+3.2이면 금지. null 전환.15, null->null0. 누적거리만 사용하며 전역위치·GT회전·정답 없음. 이전창 경로를 재사용하지 않음.",
        "controls": "baseline=기존승자; unary_null=시간제약없이 최고융합/가중합<.5이면거절; joint_forced=이웃null허용·중심후보강제; joint_null=중심도null허용(주조건). 두모델에동일적용. 원래±5다중간선 전체를 최적화하는 대신 인접간선사슬을 사용.",
        "evaluation": "모든 선택 후보의 기존 arc_unique_b100을 다시 계산. DP목적함수로 수락점수를 대체하지 않음. null은j=-1·score/distance NaN으로 영구 미수락, 양성분모유지. q+5 키는 최대q+15를 봄. 기존원장과기준완전재현,입력접두와수작업점수대조.",
        "promotion": "standard/joint_null이 같은standard의baseline,unary_null 각각대비 AP+.1%p/R100손실<=.1%p/계열AP손실<=.25%p이면 두모델·전조건시드2,3. 다른조건사후주조건승격없음. 시험과원고교체없음.",
        "limitations": "다중프레임 후보 추론의 첫 고정설계. 20후보절단·거리비용·null값은 검증 탐색없이 설계값으로 고정. 계산량은 기존최종판정보다증가하며 배포속도동일주장없음. 신규성/독립시험확증아님.",
    })
    print("PREPARED", len(runtime), len(inputs), flush=True)


def candidates(key: np.ndarray, bank: tuple, weights: list, channel: str,
               q: int) -> tuple:
    """기존 두 검색과 융합으로 순위 및 원시 판정 점수를 반환한다."""
    ids = np.unique(np.r_[prefix_topk(key, q, q-29, 100),
                          prefix_topk(bank[0], q, q-29, 100)])
    a = align_scores(bank[1], bank[2], q, ids)
    c = column_align_scores(bank[3], q, ids)
    parts = [key[ids] @ key[q], bank[0][ids] @ bank[0][q], a, c]
    fused = sum(float(w)*minmax(v) for w, v in zip(weights, parts))
    order = np.argsort(-fused, kind="stable")
    raw = a if channel == "c_spec" else c
    return ids[order], (fused[order]/sum(weights)).astype(np.float64), raw[order]


def bank_candidates(key: np.ndarray, bank: tuple, conf: dict,
                    positions: np.ndarray) -> dict:
    """전체 후보를 같은 규칙으로 평가하고 최적화용 상위 후보를 저장한다."""
    n, k = len(key), CFG["candidate_top_k"]
    ids = np.full((n, k), -1, np.int64)
    unary = np.full((n, k), -np.inf)
    raw = np.full((n, k), np.nan)
    union_hit, shortlist_hit = np.zeros(n, bool), np.zeros(n, bool)
    counts = np.zeros(n, np.int64)
    for q in range(30, n-10):
        jj, uu, rr = candidates(key, bank, conf["tuple"], conf["decision"], q)
        m = min(k, len(jj))
        ids[q, :m], unary[q, :m], raw[q, :m] = jj[:m], uu[:m], rr[:m]
        # 다음 값은 최적화에 전달하지 않는 사후 후보 상한 진단이다.
        dist = np.linalg.norm(positions[jj]-positions[q], axis=1)
        union_hit[q], shortlist_hit[q], counts[q] = (dist <= 5).any(), (dist[:m] <= 5).any(), len(jj)
    return {"indices": ids, "unary": unary, "raw": raw, "union_hit": union_hit,
            "shortlist_hit": shortlist_hit, "candidate_count": counts}


def path_selection(bank: dict, travel: np.ndarray, queries: np.ndarray,
                   device: str = "cuda") -> dict:
    """동결된 설정으로 중심 후보 선택을 계산한다."""
    return select_paths(bank["indices"], bank["unary"], travel, queries,
                        radius=CFG["query_radius"], weight=CFG["transition_weight"],
                        null_unary=CFG["null_unary"], null_cost=CFG["null_transition_cost"],
                        device=device, batch_size=CFG["batch_size"])


def final_score(bank: tuple, travel: np.ndarray, queries: np.ndarray,
                winners: np.ndarray, raw: np.ndarray, channel: str) -> np.ndarray:
    """거절은 영구 미수락으로 두고 나머지의 기존 최종 판정을 계산한다."""
    score = np.full(len(queries), np.nan)
    valid = np.flatnonzero(winners >= 0)
    for start in range(0, len(valid), 256):
        rows = valid[start:start+256]
        score[rows] = fixed.frozen_scores(bank, travel, queries[rows], winners[rows],
                                        channel, raw[rows])[CFG["final_score"]]
    return score


def worker(seed: int, sequence: str) -> None:
    """같은 저장 키의 후보를 재생성하고 두 모델의 네 대조를 평가한다."""
    spec = protocol()
    folder = OUT / f"s{seed}"
    folder.mkdir(exist_ok=True)
    name = sequence.replace("/", "_")
    marker = folder / (name + ".json")
    if marker.exists():
        return
    sensor, seq, paths, _ = exp.old.base.inputs("val", sequence)
    cache = exp.arrays(paths["cache"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = exp.old.base.bank_for(exp.OP, z)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    gt = cache["poses_gt"][:, :3, 3]
    conf = exp.read(exp.old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][exp.OP]
    results = {}
    for model in CFG["models"]:
        started = time.perf_counter()
        source = REPO / spec["selected"][model][str(seed)]["folder"]
        keys = exp.arrays(source / f"keys_val_{sensor}_{seq}.npz")
        np.testing.assert_array_equal(keys["scan_ids"], cache["scan_ids"])
        key = keys["embedding"]
        reference = exp.arrays(fixed.VAL / f"s{seed}/{model}_{sensor}_{seq}.npz")
        q = reference["query_idx"]
        assert np.array_equal(q, np.arange(30, len(key)-15))
        cb = bank_candidates(key, bank, conf, gt)
        candidate_seconds = time.perf_counter()-started
        np.testing.assert_array_equal(cb["indices"][q, 0], reference["j_star"])
        np.testing.assert_array_equal(cb["raw"][q, 0], reference["raw"])
        tick = time.perf_counter()
        selected = path_selection(cb, travel, q)
        selection_seconds = time.perf_counter()-tick
        selections = {
            "baseline": cb["indices"][q, 0],
            "unary_null": np.where(cb["unary"][q, 0] >= CFG["null_unary"], cb["indices"][q, 0], -1),
            "joint_forced": selected["forced"], "joint_null": selected["joint"],
        }
        cp = folder / f"candidates_{model}_{name}.npz"
        np.savez_compressed(cp, **cb)
        records = {}
        tick = time.perf_counter()
        for arm, winners in selections.items():
            valid = winners >= 0
            assert np.all(winners[valid] <= q[valid]-30)
            assert np.all((winners[valid, None] == cb["indices"][q[valid]]).any(1))
            slot = (cb["indices"][q] == winners[:, None]).argmax(1)
            raw = np.where(valid, cb["raw"][q, slot], np.nan)
            scores = final_score(bank, travel, q, winners, raw, conf["decision"])
            dist = np.full(len(q), np.nan)
            dist[valid] = np.linalg.norm(gt[q[valid]]-gt[winners[valid]], axis=1)
            values = {**{k: reference[k] for k in ("query_idx", "min_past_gt_dist", "group")},
                      "j_star": winners, "top1_geo_dist": dist, "score": scores,
                      "raw": raw, "path_margin": selected["margin"], "direction": selected["direction"]}
            if arm == "baseline":
                np.testing.assert_array_equal(dist, reference["top1_geo_dist"])
                np.testing.assert_array_equal(scores, reference[CFG["final_score"]])
            target = folder / f"{model}_{arm}_{name}.npz"
            np.savez_compressed(target, **values)
            records[arm] = {"metrics": exp.metrics(scores, values), "sha256": sha256(target),
                            "null_count": int((~valid).sum()), "winner_changes": int((winners != reference["j_star"]).sum())}
        scoring_seconds = time.perf_counter()-tick
        prefixes = []
        for index in (len(q)//3, 2*len(q)//3):
            t, end = int(q[index]), int(q[index]+16)
            small = {k: v[:end] for k, v in cb.items()}
            small_selected = path_selection(small, travel[:end], q[index:index+1])
            for k in selected:
                np.testing.assert_array_equal(small_selected[k], selected[k][index:index+1])
            for qt in range(t-5, t+6):
                jj, uu, rr = candidates(key[:end], exp.truncate_bank(bank, end),
                                       conf["tuple"], conf["decision"], qt)
                m = min(CFG["candidate_top_k"], len(jj))
                np.testing.assert_array_equal(jj[:m], cb["indices"][qt, :m])
                np.testing.assert_array_equal(uu[:m], cb["unary"][qt, :m])
            prefixes.append({"q": t, "end": end, "candidate_frames": 11, "selection_equal": True})
        results[model] = {"arms": records, "candidate_sha256": sha256(cp), "prefixes": prefixes,
                          "timing_s": {"candidates": candidate_seconds, "selection": selection_seconds,
                                       "four_final_scores": scoring_seconds}}
        print("DONE", seed, sequence, model, round(time.perf_counter()-started, 2), flush=True)
    write_json(marker, {"seed": seed, "sequence": sequence, "models": results, "test_used": False})


def report(seeds: list) -> dict:
    """모든 최종 지표를 집계하고 사전에 정한 주조건의 확장만 판단한다."""
    spec = protocol()
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = {}
        records = {s: exp.read(OUT / f"s{seed}/{s.replace('/', '_')}.json") for s in spec["val"]}
        for model in CFG["models"]:
            per_seed[str(seed)][model] = {
                arm: macro({s: r["models"][model]["arms"][arm]["metrics"] for s, r in records.items()})
                for arm in CFG["arms"]}
    chosen = per_seed["1"][CFG["primary_model"]][CFG["primary_arm"]]
    checks = {}
    for arm in CFG["promotion"]["comparators"]:
        base = per_seed["1"][CFG["primary_model"]][arm]
        checks[arm] = {
            "AP": chosen["macro"]["AP"] >= base["macro"]["AP"]+.001,
            "R100": chosen["macro"]["R@100P"] >= base["macro"]["R@100P"]-.001,
            "family_AP": all(chosen["families"][f]["AP"] >= base["families"][f]["AP"]-.0025
                             for f in base["families"])}
    value = {"per_seed": per_seed, "promotion_checks": checks,
             "pilot_eligible": all(all(x.values()) for x in checks.values()),
             "test_used": False, "protocol_sha256": sha256(OUT / "protocol.json")}
    write_json(OUT / "summary.json", value)
    print("REPORT", value, flush=True)
    return value


def main() -> None:
    """동결된 예비를 끝내고 조건을 만족할 때만 두 추가 시드를 실행한다."""
    import os
    import subprocess
    prepare()
    started = time.perf_counter()
    log = REPO / "logs/candidate_paths_20261020"
    log.mkdir(parents=True, exist_ok=True)
    spec = protocol()
    seeds = []
    try:
        for seed in (1, 2, 3):
            if seed > 1 and not exp.read(OUT / "summary.json")["pilot_eligible"]:
                break
            def launch(sequence):
                with (log / f"s{seed}_{sequence.replace('/', '_')}.log").open("a") as stream:
                    result = subprocess.run([sys.executable, "-u", __file__, "worker",
                                             "--seed", str(seed), "--sequence", sequence],
                                            stdout=stream, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, OPENBLAS_NUM_THREADS="1",
                                                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                if result.returncode:
                    raise RuntimeError(sequence + ": worker failed; inspect its log")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(launch, s) for s in spec["val"]]
                for done, f in enumerate(concurrent.futures.as_completed(futures), 1):
                    f.result()
                    write_json(OUT / "status.json", {"status": "running", "seed": seed,
                               "done": done, "total": len(futures), "elapsed_s": time.perf_counter()-started})
                    print("SEQUENCE", seed, done, len(futures), flush=True)
            seeds.append(seed)
            report(seeds)
        write_json(OUT / "status.json", {"status": "evaluation_complete", "elapsed_s": time.perf_counter()-started,
                                        "seeds": seeds, "test_used": False})
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "error": str(error), "test_used": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "prepare", "worker"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == "run":
        main()
    elif args.action == "prepare":
        prepare()
    else:
        worker(args.seed, args.sequence)
