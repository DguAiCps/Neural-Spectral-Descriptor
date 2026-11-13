#!/usr/bin/env python3
"""현재 NSD의 표현을 고정하고 후보 선택의 작은 변경을 검증한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_whitening as white
from _causal_pr_dump import prefix_topk
from _pdv2_context_accept import macro
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json
from _xs_eval_lib import align_scores, column_align_scores, minmax, variant_inputs
from retrieval.spectral_evidence import correlation_profiles

OUT = REPO / "results/pdv2_minimal_rerank_20261101"
LOG = REPO / "logs/minimal_rerank_20261101"
ARMS = ("baseline", "white_fusion", "temporal_top2", "temporal_top5",
        "local5", "temporal_top5_cap", "local5_cap")
FIELDS = white.prior.BASE.FIELDS
METRICS = white.prior.METRICS


def read(path: Path) -> dict:
    """결과 JSON을 읽는다."""
    return json.loads(path.read_text())


def arrays(path: Path) -> dict:
    """압축 원장을 메모리로 읽는다."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def source_folder(seed: int, role: str, spec: dict) -> Path:
    """현재 선택 모델의 보존 키와 원장 경로다."""
    suffix = f"epoch{spec['models'][str(seed)]}" if role == "val" else "test"
    return white.prior.root(seed, "control") / suffix


def reference_path(seed: int, role: str, name: str) -> Path:
    """25% 보정을 포함한 현재 주모델의 최종 원장을 지정한다."""
    part = "blend/quarter" if role == "val" else "fixed_test"
    return white.OUT / part / f"s{seed}" / name


def prepare() -> dict:
    """새 검증 결과를 보기 전에 작은 후보군과 승격 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = read(target)
        for p, h in spec["sources"].items():
            assert sha256(REPO / p) == h, p
        return spec
    old = read(white.OUT / "protocol.json")
    paths = {Path(__file__), Path(white.__file__),
             REPO / "scripts/_pdv2_sequence_fixed_test.py",
             REPO / "src/retrieval/sequence_evidence.py",
             REPO / "src/retrieval/spectral_evidence.py",
             REPO / "scripts/_xs_eval_lib.py", REPO / "scripts/_causal_pr_dump.py",
             white.OUT / "statistics.npz", white.OUT / "protocol.json",
             white.OUT / "blend/summary.json"}
    for seed in old["seeds"]:
        for seq in old["val"]:
            name = seq.replace("/", "_") + ".npz"
            folder = source_folder(seed, "val", old)
            paths.update((folder / name, folder / ("keys_" + name),
                          folder / ("coverage_" + name), reference_path(seed, "val", name)))
    spec = {"created_unix": time.time(), "val": old["val"], "seeds": old["seeds"],
            "models": old["models"], "confidence": old["confidence"], "arms": ARMS,
            "scope": "현재288D+DiffAttnConv128D와학습통계고정. 재학습0,새학습파라미터0. 검증9x3시드. q+15, j<=q-30. 시험결과로선택하지않는다.",
            "white_fusion": "100+100검색은그대로. 네융합채널중정렬두채널을.75원본+.25상관보정으로교체한뒤기존minmax/가중치로승자선택. 최종점수는현재75:25시간점수.",
            "temporal": "기존융합상위2또는5개중현재75:25최종시간점수최대후보선택. 동률은기존융합순위. top5_cap은같은승자의수락점수를min(새점수,기존승자점수)로제한해다중비교오수락을대조.",
            "local": "기존승자와j-2,j-1,j+1,j+2중유효과거후보에서같은최종시간점수최대. 동률은기존승자우선. local5_cap은같은상한적용. 새원시스캔/관측지연증가없음;추가후보계산비용있음.",
            "selection": "평균AP+.1%p이상,모든시드AP개선,평균R1+.1%p이상,평균R100손실<=.1%p,계열평균AP손실<=.25%p를모두통과한조건중평균AP최대. 동률은ARMS순서. 최대한조건만설정동결후시험46x3실행. 미달이면현재유지. 원고자동교체없음.",
            "checks": "기준승자전수동일,현재최종점수오차<=1e-7,기준지표오차<=1e-10,독립지표재계산,과거후보제약,q+15입력절단동일성. GT는평가/손익집계만사용.",
            "test_exposure": "이시험분할은이전개발노출이있다. 새미노출확증으로표현하지않는다.",
            "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}}
    write_json(target, spec)
    return spec


def candidate_row(key: np.ndarray, bank: tuple, corrected: tuple,
                  conf: dict, q: int) -> tuple:
    """현재 후보 상위5개·보정 융합 승자·기존 승자 인근4개를 만든다."""
    ids = np.unique(np.r_[prefix_topk(key, q, q - 29, 100),
                          prefix_topk(bank[0], q, q - 29, 100)])
    a = align_scores(bank[1], bank[2], q, ids)
    c = column_align_scores(bank[3], q, ids)
    wa = align_scores(corrected[1], corrected[2], q, ids)
    wc = column_align_scores(corrected[3], q, ids)
    w = conf["tuple"]
    common = w[0] * minmax(key[ids] @ key[q]) + w[1] * minmax(bank[0][ids] @ bank[0][q])
    fusion = common + w[2] * minmax(a) + w[3] * minmax(c)
    amended = common + w[2] * minmax(.75 * a + .25 * wa) + w[3] * minmax(.75 * c + .25 * wc)
    order = np.argsort(-fusion, kind="stable")
    top = ids[order[:5]]
    top = np.pad(top, (0, 5 - len(top)), constant_values=top[0])
    white_best = ids[np.argsort(-amended, kind="stable")[0]]
    local = top[0] + np.array([-2, -1, 1, 2])
    local = np.where((local >= 0) & (local <= q - 30), local, top[0])
    return np.r_[top, white_best, local], int(len(ids))


def scores_for(bank: tuple, corrected: tuple, travel: np.ndarray,
               q: np.ndarray, j: np.ndarray, channel: str) -> np.ndarray:
    """같은 최종 시간 점수를 임의의 적격 후보에 계산한다."""
    pieces = []
    for start in range(0, len(q), 256):
        sl = slice(start, start + 256)
        values = []
        for index, item in enumerate((bank, corrected)):
            raw = correlation_profiles(item, q[sl], j[sl], channel).max(1)
            # 기존 승자 원장의 원본 점수는 float64 배열에 저장된다.
            if index == 0:
                raw = raw.astype(np.float64)
            values.append(white.prior.BASE.fixed.frozen_scores(
                item, travel, q[sl], j[sl], channel, raw)[white.METHOD])
        pieces.append(.75 * values[0] + .25 * values[1])
    return np.concatenate(pieces)


def chosen_columns(value: np.ndarray) -> dict:
    """GT와 무관하게 사전 고정한 후보 선택식만 적용한다."""
    local_columns = np.array([0, 6, 7, 8, 9])
    choices = {"baseline": np.zeros(len(value), np.int64),
               "white_fusion": np.full(len(value), 5, np.int64),
               "temporal_top2": value[:, :2].argmax(1),
               "temporal_top5": value[:, :5].argmax(1),
               "local5": local_columns[value[:, local_columns].argmax(1)]}
    choices["temporal_top5_cap"] = choices["temporal_top5"]
    choices["local5_cap"] = choices["local5"]
    return choices


def worker(seed: int, sequence: str, role: str = "val", limit: int = 0) -> None:
    """표현을 재사용하며 모든 후보와 최종 원장을 독립 폴더에 보존한다."""
    spec = read(OUT / "protocol.json")
    assert sha256(Path(__file__)) == spec["sources"][str(Path(__file__).relative_to(REPO))]
    arms = ARMS if role == "val" else ("baseline", read(OUT / "test_freeze.json")["selected"])
    name = sequence.replace("/", "_")
    folder = OUT / role / f"s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name + ("_smoke" if limit else "") + ".json")
    if target.exists():
        assert read(target)["status"] == "passed"
        return
    root = source_folder(seed, role, spec)
    ref = arrays(reference_path(seed, role, name + ".npz"))
    keys_path = root / ("keys_" + name + ".npz")
    key = arrays(keys_path)["embedding"]
    _, _, paths, _ = white.prior.BASE.exp.old.base.inputs(role, sequence)
    cache = arrays(paths["cache"])
    x = white.coefficients(paths["spec"])
    bank = variant_inputs(x, 8)
    corrected = variant_inputs(white.transform(x, arrays(white.OUT / "statistics.npz")["full"]), 8)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    positions = cache["poses_gt"][:, :3, 3]
    conf = spec["confidence"][str(seed)]
    if limit:
        ix = np.linspace(0, len(ref["query_idx"]) - 1, limit).astype(int)
        ref = {k: v[ix] for k, v in ref.items()}
    q = ref["query_idx"]
    tick = time.perf_counter()
    candidate_cache = folder / (name + ("_smoke" if limit else "") + "_candidates.npz")
    if candidate_cache.exists():
        cached = arrays(candidate_cache)
        np.testing.assert_array_equal(cached["query_idx"], q)
        ids, count = cached["indices"], cached["count"]
    else:
        ids = np.empty((len(q), 10), np.int64)
        count = np.empty(len(q), np.int64)
        for i, t in enumerate(q):
            ids[i], count[i] = candidate_row(key, bank, corrected, conf, int(t))
            if i and i % 4000 == 0:
                print("CANDIDATES", seed, sequence, i, len(q), flush=True)
        np.savez_compressed(candidate_cache, query_idx=q, indices=ids, count=count)
    candidate_seconds = time.perf_counter() - tick
    np.testing.assert_array_equal(ids[:, 0], ref["j_star"])
    assert (ids >= 0).all() and (ids <= q[:, None] - 30).all()
    tick = time.perf_counter()
    unique, inverse = np.unique((q[:, None] * len(key) + ids).ravel(), return_inverse=True)
    values = scores_for(bank, corrected, travel, unique // len(key), unique % len(key), conf["decision"])
    value = values[inverse].reshape(ids.shape)
    score_seconds = time.perf_counter() - tick
    error = float(abs(value[:, 0] - ref["score"]).max())
    np.testing.assert_allclose(value[:, 0], ref["score"], atol=1e-7, rtol=0)
    choices = chosen_columns(value)
    rows, changes = {}, {}
    for arm in arms:
        column = choices[arm]
        winner = ids[np.arange(len(q)), column]
        score = value[np.arange(len(q)), column]
        if arm.endswith("_cap"):
            score = np.minimum(score, value[:, 0])
        row = {k: ref[k] for k in FIELDS}
        row["j_star"] = winner
        row["top1_geo_dist"] = np.linalg.norm(positions[q] - positions[winner], axis=1)
        row["score"] = score
        metrics = independent_metrics(row, score)
        other = white.prior.BASE.exp.metrics(score, row)
        for m in METRICS:
            if metrics[m] is not None and metrics["n_pos"]:
                np.testing.assert_allclose(metrics[m], other[m], atol=1e-12, rtol=0)
        if arm == "baseline":
            expected = independent_metrics(ref, ref["score"])
            for m in METRICS:
                if metrics[m] is not None:
                    np.testing.assert_allclose(metrics[m], expected[m], atol=1e-10, rtol=0)
        positive = ref["min_past_gt_dist"] <= 5
        old_ok = positive & (ref["top1_geo_dist"] <= 5)
        new_ok = positive & (row["top1_geo_dist"] <= 5)
        changes[arm] = {"winner_changed": int((winner != ref["j_star"]).sum()),
                        "gained_correct": int((new_ok & ~old_ok).sum()),
                        "lost_correct": int((old_ok & ~new_ok).sum())}
        p = folder / f"{name}{'_smoke' if limit else ''}_{arm}.npz"
        np.savez_compressed(p, **row)
        rows[arm] = {"metrics": metrics, "path": str(p.relative_to(REPO)), "sha256": sha256(p)}
    prefixes = []
    for i in (len(q) // 3, 2 * len(q) // 3):
        end, t = int(q[i]) + 16, int(q[i])
        short = white.prior.BASE.exp.truncate_bank(bank, end)
        sw = white.prior.BASE.exp.truncate_bank(corrected, end)
        jj, _ = candidate_row(key[:end], short, sw, conf, t)
        np.testing.assert_array_equal(jj, ids[i])
        vv = scores_for(short, sw, travel[:end], np.full(10, t), jj, conf["decision"])
        np.testing.assert_array_equal(vv, value[i])
        prefixes.append({"query": t, "all_candidate_choices_and_scores_equal": True})
    write_json(target, {"status": "passed", "seed": seed, "sequence": sequence,
                       "role": role, "rows": len(q), "arms": rows, "changes": changes,
                       "baseline_score_max_error": error, "prefix_checks": prefixes,
                       "timing_s": {"candidate_construction": candidate_seconds,
                                    "all_candidate_final_scores": score_seconds},
                       "candidate_sha256": sha256(candidate_cache),
                       "inputs": {str(p.relative_to(REPO)): sha256(p) for p in
                                  (keys_path, reference_path(seed, role, name + ".npz"), *paths.values())}})
    print("DONE", seed, sequence, role, round(candidate_seconds + score_seconds, 1), flush=True)


def finish() -> None:
    """전체27원장의 최종 지표와 고정한 승격 조건을 검산한다."""
    spec = read(OUT / "protocol.json")
    per_seed, stats, decisions = {}, {}, {}
    total, checks = 0, []
    for arm in ARMS:
        per_seed[arm] = {}
        for seed in spec["seeds"]:
            rows = {}
            for seq in spec["val"]:
                info = read(OUT / "val" / f"s{seed}" / (seq.replace("/", "_") + ".json"))
                rec = info["arms"][arm]
                p = REPO / rec["path"]
                assert sha256(p) == rec["sha256"]
                row = arrays(p)
                metrics = independent_metrics(row, row["score"])
                assert metrics == rec["metrics"]
                rows[seq] = metrics
                total += len(row["query_idx"])
                checks.append(rec["sha256"])
            per_seed[arm][str(seed)] = {**macro(rows), "sequences": rows}
        stats[arm] = {m: {"mean": float(np.mean([r["macro"][m] for r in per_seed[arm].values()])),
                         "std": float(np.std([r["macro"][m] for r in per_seed[arm].values()], ddof=1))}
                      for m in METRICS}
    for arm in ARMS[1:]:
        delta = {m: 100 * (stats[arm][m]["mean"] - stats["baseline"][m]["mean"]) for m in METRICS}
        seed_ap = {str(s): 100 * (per_seed[arm][str(s)]["macro"]["AP"] - per_seed["baseline"][str(s)]["macro"]["AP"]) for s in spec["seeds"]}
        family_ap = {f: 100 * float(np.mean([per_seed[arm][str(s)]["families"][f]["AP"] - per_seed["baseline"][str(s)]["families"][f]["AP"] for s in spec["seeds"]])) for f in ("kitti", "helipr", "mulran")}
        passed = {"mean_AP": delta["AP"] >= .1, "all_seeds_AP": min(seed_ap.values()) > 0,
                  "mean_R1": delta["R@1"] >= .1, "mean_R100": delta["R@100P"] >= -.1,
                  "family_AP": min(family_ap.values()) >= -.25}
        decisions[arm] = {"delta_pp": delta, "seed_AP_delta_pp": seed_ap,
                          "family_AP_delta_pp": family_ap, "checks": passed,
                          "eligible": all(passed.values())}
    eligible = [a for a in ARMS[1:] if decisions[a]["eligible"]]
    selected = max(eligible, key=lambda a: stats[a]["AP"]["mean"]) if eligible else None
    write_json(OUT / "summary.json", {"stats": stats, "per_seed": per_seed,
                                      "decisions": decisions, "selected": selected,
                                      "test_used": False})
    write_json(OUT / "completion_validation.json", {"status": "passed", "ledgers": len(checks),
                                                     "rows": total, "test_used": False,
                                                     "summary_sha256": sha256(OUT / "summary.json")})
    print("SUMMARY", stats, decisions, "SELECTED", selected, flush=True)


def run() -> None:
    """독립 세 작업으로 검증 시퀀스들을 실행한다."""
    spec = prepare()
    LOG.mkdir(parents=True, exist_ok=True)
    def invoke(item: tuple) -> None:
        seed, seq = item
        with (LOG / f"s{seed}_{seq.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", "--seed", str(seed),
                            "--sequence", seq], stdout=log, stderr=subprocess.STDOUT, check=True)
    jobs = [(s, q) for s in spec["seeds"] for q in spec["val"]]
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(invoke, j): j for j in jobs}
        for n, future in enumerate(concurrent.futures.as_completed(pending), 1):
            future.result()
            write_json(OUT / "status.json", {"status": "running", "done": n, "total": len(jobs),
                                              "elapsed_s": time.time() - start})
            print("PROGRESS", n, len(jobs), pending[future], flush=True)
    finish()
    write_json(OUT / "status.json", {"status": "complete", "elapsed_s": time.time() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "run", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="kitti360/0004")
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.seed, args.sequence, args.role, args.limit)
    else:
        globals()[args.action]()
