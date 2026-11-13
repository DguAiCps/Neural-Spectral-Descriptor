#!/usr/bin/env python3
"""채택한 판정·융합을 공통 적용한 표4 대조를 최종 원장으로 측정한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import _pdv2_spectral_power as power
import _pdv2_table4_current as previous

base, cal, white = power.base, power.cal, power.base.white
REPO = power.REPO
OUT = power.OUT / "adoption/table4"
ARMS = ("NSD", "without_correlation_correction", "without_amplitude_compression",
        "graph_free", "standard", "key_only", "aux_only", "no_alignment")


def prepare() -> dict:
    """새 대조 결과를 보기 전에 동일 체크포인트와 모든 처리 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    pp = OUT / "protocol.json"
    if pp.exists():
        spec = power.read(pp)
        for p, h in spec["sources"].items():
            assert power.sha256(REPO / p) == h, p
        return spec
    main = power.read(power.OUT / "test_freeze.json")
    choices = {a: power.read(previous.OUT / f"freeze_{a}.json")["models"] for a in previous.LEARNED}
    paths = [Path(__file__), Path(previous.__file__), Path(power.__file__),
             power.OUT / "test_freeze.json", power.OUT / "protocol.json", power.OUT / "table2_summary.json",
             previous.OUT / "protocol.json", white.OUT / "statistics.npz"]
    for arm in previous.LEARNED:
        paths.append(previous.OUT / f"freeze_{arm}.json")
        paths.extend(REPO / r["checkpoint"] for r in choices[arm].values())
    spec = {"selected": main["selected"], "test": main["test"], "cities": main["cities"],
            "seeds": main["seeds"], "arms": ARMS, "models": choices,
            "selection": "주모델과 두 학습 대조는 기존 검증 선택 체크포인트를 그대로 사용. 동일 학습과 보정 전 최종 AP 선택 규칙. 새 시험에서 선택하지 않는다.",
            "common": "기존 후보/키/보정 행렬 유지; 보조 키 융합 가중치 1/2, 판정 양 채널의 스케일 보정 평균, 원본/보정 계수 각각 크기제곱근.",
            "controls": "보정 제거는 변환 원본만; 크기 압축 제거는 p=1만; 나머지 다섯 구조 대조는 새 공통 판정/융합 적용. no_alignment는 후보·중심·시간 이웃 모든 스펙트럼에서 0-shift, 양 채널 평균 고정. 새로운 대조별 조정 없음.",
            "scope": "기존 노출 시험의 구조 대조. 최종195726질의/시드,45양성시퀀스→6도시→3시드. NSD 행은 표2와 완전 일치 검산.",
            "test_used_for_selection": False, "created_unix": time.time(),
            "sources": {str(p.relative_to(REPO)): power.sha256(p) for p in paths}}
    power.write_json(pp, spec)
    return spec


def scores(bank: tuple, corrected: tuple, travel: np.ndarray, q: np.ndarray,
           j: np.ndarray, zero_shift: bool = False) -> dict:
    """두 채널의 원본·보정 시간 점수를 모두 같은 단위로 합친다."""
    channels = {}
    for ch in ("c_spec", "c_col"):
        values = []
        for b in (bank, corrected):
            parts = []
            for start in range(0, len(q), 256):
                sl = slice(start, start + 256)
                profile = previous.correlation_profiles(b, q[sl], j[sl], ch)
                # 원본 raw는 현재 원장의 float64 저장 규약과 일치한다.
                raw = profile[:, 0] if zero_shift else profile.max(1)
                if b is bank:
                    raw = raw.astype(np.float64)
                parts.append(previous.temporal_score(b, travel, q[sl], j[sl], ch, raw, zero_shift))
            values.append(np.concatenate(parts))
        channels[ch] = values
    original = .5 * (30 * channels["c_spec"][0] + channels["c_col"][0])
    corrected_score = .5 * (30 * channels["c_spec"][1] + channels["c_col"][1])
    # 두 채널을 각각 75:25로 혼합한 뒤 평균하는 주모델의 산술 순서를 보존한다.
    mixed = cal.confidence(.75 * channels["c_spec"][0] + .25 * channels["c_spec"][1],
                           .75 * channels["c_col"][0] + .25 * channels["c_col"][1], "equal", "c_col")
    return {"score": mixed, "original": original, "corrected": corrected_score}


def worker(seed: int, sequence: str) -> None:
    """한 시드·시퀀스에서 대조 여덟 개를 최종 점수까지 생성한다."""
    spec = power.read(OUT / "protocol.json")
    assert power.sha256(Path(__file__)) == spec["sources"][str(Path(__file__).relative_to(REPO))]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    name = sequence.replace("/", "_")
    folder = OUT / f"s{seed}"
    folder.mkdir(exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        assert power.read(target)["status"] == "passed"
        return
    tick = time.time()
    cfg = power.read(power.OUT / "protocol.json")
    refpath = power.OUT / "test" / f"s{seed}" / f"{name}_{spec['selected']}.npz"
    ref = power.arrays(refpath)
    _, _, paths, min_past = white.prior.BASE.exp.old.base.inputs("test", sequence)
    cache = power.arrays(paths["cache"])
    keypath = base.source_folder(seed, "test", cfg) / ("keys_" + name + ".npz")
    keys = power.arrays(keypath)
    np.testing.assert_array_equal(keys["scan_ids"], cache["scan_ids"])
    x = white.coefficients(paths["spec"])
    wx = white.transform(x, power.arrays(white.OUT / "statistics.npz")["full"])
    bank, corrected = base.variant_inputs(x, 8), base.variant_inputs(wx, 8)
    nb, nc = base.variant_inputs(power.transform(x, .5), 8), base.variant_inputs(power.transform(wx, .5), 8)
    xyz, pos = cache["poses"][:, :3, 3], cache["poses_gt"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    q = ref["query_idx"]
    weights = cal.fusion_weights(cfg["confidence"][str(seed)])[cal.FUSIONS.index("aux_half")]
    retrieved = previous.retrieve(keys["embedding"], bank, pos, min_past, weights, q,
                                 ("NSD", *previous.RETRIEVAL))
    for arm in previous.LEARNED:
        choice = spec["models"][arm][str(seed)]
        checkpoint = REPO / choice["checkpoint"]
        assert power.sha256(checkpoint) == choice["sha256"]
        net = previous.model(arm, seed).cuda().eval()
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(previous.source.OUT / "remetric_octave288.npy")).float().cuda(), requires_grad=False)
        key = previous.source.make_keys(net, cache["descriptors"].astype(np.float32), cache["poses"])
        retrieved.update(previous.retrieve(key, bank, pos, min_past, weights, q, (arm,)))
        # 기존 동결 구조의 접두 키 일치 검사를 새 실행에서도 확인한다.
        at = len(q) // 2
        end = int(q[at]) + 16
        small = previous.source.make_keys(net, cache["descriptors"].astype(np.float32), cache["poses"], end)
        np.testing.assert_allclose(small[:q[at] + 1], key[:q[at] + 1], atol=4e-6, rtol=4e-5)
        del net, key, small
        torch.cuda.empty_cache()
    result = {}
    for arm, (row, candidate_hash) in retrieved.items():
        row["group"] = ref["group"]
        np.testing.assert_array_equal(row["min_past_gt_dist"], ref["min_past_gt_dist"])
        endpoints = scores(nb, nc, travel, q, row["j_star"], arm == "no_alignment")
        if arm == "NSD":
            np.testing.assert_array_equal(row["j_star"], ref["j_star"])
            np.testing.assert_array_equal(endpoints["score"], ref["score"])
        variants = {arm: endpoints["score"]}
        if arm == "NSD":
            variants["without_correlation_correction"] = endpoints["original"]
            variants["without_amplitude_compression"] = scores(bank, corrected, travel, q, row["j_star"])["score"]
        prefixes = []
        for at in (len(q) // 3, 2 * len(q) // 3):
            end = int(q[at]) + 16
            b, c = (previous.source.base.exp.truncate_bank(v, end) for v in (nb, nc))
            partial = scores(b, c, travel[:end], q[at:at+1], row["j_star"][at:at+1], arm == "no_alignment")
            np.testing.assert_array_equal(partial["score"], endpoints["score"][at:at+1])
            prefixes.append({"query": int(q[at]), "score_exact": True})
        for label, value in variants.items():
            metric = base.independent_metrics(row, value)
            independent = white.prior.BASE.exp.metrics(value, row)
            for m in base.METRICS:
                if metric["n_pos"]:
                    np.testing.assert_allclose(metric[m], independent[m], atol=1e-12, rtol=0)
            ledger = folder / f"{name}_{label}.npz"
            np.savez_compressed(ledger, **{k: row[k] for k in base.FIELDS}, score=value)
            result[label] = {"path": str(ledger.relative_to(REPO)), "sha256": power.sha256(ledger),
                             "metrics": metric, "candidate_sha256": candidate_hash, "prefix_checks": prefixes}
    power.write_json(target, {"status": "passed", "rows": len(q), "arms": result,
        "main_matches_table2": True, "elapsed_s": time.time() - tick,
        "inputs": {str(p.relative_to(REPO)): power.sha256(p) for p in (refpath, keypath, *paths.values())}})
    print("DONE", seed, sequence, round(time.time() - tick, 1), flush=True)


def finish() -> None:
    """최종 원장을 다시 읽어 표2와 동일한 평균·표준편차를 구한다."""
    spec = prepare()
    means, controls, records = {}, {}, []
    main = power.read(power.OUT / "table2_summary.json")
    for arm in ARMS:
        seeds = {}
        for seed in spec["seeds"]:
            seqs, count = {}, 0
            for seq in spec["test"]:
                entry = power.read(OUT / f"s{seed}" / (seq.replace("/", "_") + ".json"))["arms"][arm]
                p = REPO / entry["path"]
                assert power.sha256(p) == entry["sha256"]
                row = power.arrays(p)
                seqs[seq] = base.independent_metrics(row, row["score"])
                assert seqs[seq] == entry["metrics"]
                count += len(row["query_idx"])
                records.append({"path": entry["path"], "sha256": entry["sha256"]})
            assert count == 195726
            seeds[str(seed)] = {**previous.macro(seqs, spec["cities"]), "sequences": seqs, "query_count": count}
        means[arm] = {m: {"mean": float(np.mean([v["macro"][m] for v in seeds.values()])),
                         "std": float(np.std([v["macro"][m] for v in seeds.values()], ddof=1))} for m in base.METRICS}
        controls[arm] = {"per_seed": seeds, "mean_std": means[arm]}
    assert means["NSD"] == main["mean_std"][spec["selected"]]
    power.write_json(OUT / "summary.json", {"mean_std": means, "controls": controls,
        "checks_passed": True, "test_used_for_selection": False, "protocol_sha256": power.sha256(OUT / "protocol.json")})
    power.write_json(OUT / "completion_validation.json", {"status": "passed", "ledgers": len(records),
        "records": records, "summary_sha256": power.sha256(OUT / "summary.json"), "main_table2_exact": True})
    print("COMPLETE_TABLE4", means, flush=True)


def run() -> None:
    """고정한 조건을 세 독립 작업 프로세스로 실행한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    jobs = [(s, q) for s in spec["seeds"] for q in spec["test"]]
    tick = time.time()
    def invoke(item: tuple) -> None:
        seed, seq = item
        with (logs / f"s{seed}_{seq.replace('/', '_')}.log").open("w") as f:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", "--seed", str(seed), "--sequence", seq],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(invoke, item): item for item in jobs}
        for n, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            power.write_json(OUT / "status.json", {"status": "running", "done": n, "total": len(jobs), "elapsed_s": time.time() - tick})
            print("PROGRESS", n, len(jobs), futures[f], flush=True)
    finish()
    power.write_json(OUT / "status.json", {"status": "complete", "done": len(jobs), "elapsed_s": time.time() - tick})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "run", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="kitti/08")
    args = parser.parse_args()
    worker(args.seed, args.sequence) if args.action == "worker" else globals()[args.action]()
