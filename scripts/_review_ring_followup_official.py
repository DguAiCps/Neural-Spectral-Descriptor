#!/usr/bin/env python3
"""공식 RING evaluation/evaluate.py의 evaluate()를 그대로 실행하고 기존 어댑터 결과와 대조한다.

서버 호스트에서 조건마다 GPU0 락을 잡고 실행한다.
    flock -x /tmp/nsd_native_reproduction_gpu0.lock docker exec -e CUDA_VISIBLE_DEVICES=0 \
        -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e LOKY_MAX_CPU_COUNT=4 \
        -w /workspace/Neural-Spectral-Codec ring_baseline \
        python3 scripts/_review_ring_followup_official.py run --split all --bev feat --impl maxfix
    ... summarize

공식 함수는 고치지 않는다. 바꾸는 것은 컨테이너에 없는 pygicp(ICP 정밀화, 장소 검색과 무관)와
seaborn(누적분포 그림)의 대체, compute_PR_pairs 입력 pair_dists를 저장하는 관찰 훅뿐이다.
평가 pickle은 공식 생성기 출력 evaluation_sets.json(독립 재구성과 비트 일치, sampling/report.json)의
시각·위치·자세를 그대로 담고 경로만 서버 원시 사본으로 둔다.
--impl maxfix는 기존 진단과 같은 결정적 격자 최댓값(scripts/_review_native_ring_maxfix.py)을 끼운다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/official_eval"
NATIVE = REPO / "results/pdv2_review_native_20261105"
OFFICIAL = REPO / "_external/RING"
sys.path.insert(0, str(REPO / "scripts"))
RADII = [5.0, 10.0, 15.0, 20.0]
PAPER_TABLE_VI_120 = {"R@1": 0.7321, "TE_m_q50_75_95": [0.56, 0.77, 1.31], "RE_deg_q50_75_95": [1.25, 2.14, 4.07]}


def digest(path: Path) -> str:
    """파일의 SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    """JSON을 원자적으로 저장한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def adapter_paths(split: str, bev: str, impl: str) -> tuple[Path, dict]:
    """같은 조건의 기존 어댑터 pair_dists와 요약 지표."""
    variant = "ring" if bev == "occ" else "ringpp"
    base = NATIVE / "maxfix" if impl == "maxfix" else NATIVE
    summary = json.loads((base / "summary.json").read_text())["variants"][f"{split}_{variant}"]
    return base / f"pair_dists_{split}_{variant}.npy", summary


def build_pickle(split: str, path: Path) -> None:
    """공식 EvaluationSet.save와 같은 [query_l, map_l] 튜플 목록을 쓴다."""
    sets = json.loads((NATIVE / "evaluation_sets.json").read_text())[split]

    def tup(e):
        return (np.int64(e["timestamp"]), str(NATIVE / "native_data" / e["raw_path"]),
                np.asarray(e["xy"], np.float32), np.asarray(e["pose"], np.float32))

    with path.open("wb") as stream:
        pickle.dump([[tup(e) for e in sets["query"]], [tup(e) for e in sets["map"]]], stream)


def run(split: str, bev: str, impl: str) -> None:
    """한 조건의 공식 evaluate()를 실행하고 원 출력과 대조 결과를 저장한다."""
    import torch
    import _review_native_ring as native

    assert not (bev == "occ" and impl == "maxfix")
    name = f"{split}_{bev}_{impl}"
    cond = OUT / name
    cond.mkdir(parents=True, exist_ok=True)
    pkl = cond / f"eval_{split}.pickle"
    build_pickle(split, pkl)
    for mod in ("pygicp", "seaborn"):           # 컨테이너에 없는 ICP·그림 의존성
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            stub.__spec__ = importlib.machinery.ModuleSpec(mod, None)
            sys.modules[mod] = stub
    native.imports()
    import evaluation.evaluate as ev
    import utils.core as core

    if impl == "maxfix":
        import _review_native_ring_maxfix as maxfix
        maxfix.install_maxfix()
        ev.generate_bev_feats = core.generate_bev_feats
    ev.fast_gicp = lambda source, target, max_correspondence_distance=1.0, init_pose=np.eye(4): (
        float("nan"), np.asarray(init_pose, dtype=np.float64))
    ev.plot_cdf = lambda *a, **k: None
    original_pr = ev.compute_PR_pairs
    captured = {}

    def hook(pair_dists, query_positions, map_positions, thresholds, save_path, revisit_threshold=10.0):
        if "pair_dists" not in captured:
            captured["pair_dists"] = np.array(pair_dists, copy=True)
            np.save(cond / "pair_dists.npy", captured["pair_dists"])
        return original_pr(pair_dists, query_positions, map_positions, thresholds, save_path, revisit_threshold)

    ev.compute_PR_pairs = hook
    torch.set_num_threads(4)
    start = time.time()
    save_json(cond / "status.json", {"stage": "evaluate_running", "pid": os.getpid(), "started": start})
    cwd = os.getcwd()
    os.chdir(cond)
    try:
        ev.evaluate("nclt", str(pkl), RADII, num_k=25, bev_type=bev)
    finally:
        os.chdir(cwd)
    elapsed = time.time() - start
    folder = cond / "results/nclt" / pkl.stem / ("revisit" + "".join(f"_{r}" for r in RADII))
    load = lambda n: np.loadtxt(folder / f"{n}_{bev}.txt")
    recalls_k, rot, trans = load("recalls_k"), load("rot_errors"), load("trans_errors")
    sets = json.loads((NATIVE / "evaluation_sets.json").read_text())[split]
    qxy = np.array([e["xy"] for e in sets["query"]], np.float64)
    mxy = np.array([e["xy"] for e in sets["map"]], np.float64)
    geo = np.linalg.norm(qxy[:, None] - mxy[None], axis=-1)
    pairs = captured["pair_dists"]
    top1 = pairs.argmin(axis=1)
    adapter_path, adapter_summary = adapter_paths(split, bev, impl)
    adapter = np.load(adapter_path)
    rows = {}
    for j, r in enumerate(RADII):
        correct = geo[np.arange(len(geo)), top1] <= r
        positive = (geo <= r).any(axis=1)
        te, re_ = trans[j][correct], rot[j][correct]
        rows[str(r)] = {
            "official_printed_R@1": float(recalls_k[j, 0]), "official_printed_R@5": float(recalls_k[j, 4]),
            "official_printed_R@25": float(recalls_k[j, 24]),
            "recomputed_R@1_from_pair_dists": float(correct.mean()),
            "positive_only_R@1": float(correct[positive].mean()), "positive_queries": int(positive.sum()),
            "adapter_official_R@1": float(adapter_summary[str(r)]["official_R@1"]),
            "TE_m_q50_75_95_correct": [float(v) for v in np.quantile(te, [.5, .75, .95])] if len(te) else None,
            "RE_deg_q50_75_95_correct": [float(v) for v in np.quantile(re_, [.5, .75, .95])] if len(re_) else None,
            "TE_m_q50_75_95_official_all_queries": [float(v) for v in np.quantile(trans[j], [.5, .75, .95])],
            "RE_deg_q50_75_95_official_all_queries": [float(v) for v in np.quantile(rot[j], [.5, .75, .95])],
        }
    adapter_top1 = adapter.argmin(axis=1)
    report = {
        "condition": name, "split": split, "bev_type": bev, "bev_impl": impl,
        "queries": int(pairs.shape[0]), "maps": int(pairs.shape[1]),
        "elapsed_s": elapsed, "gpu": torch.cuda.get_device_name(0),
        "replaced": {"fast_gicp": "pygicp 없음; 초기 자세를 그대로 반환(ICP 지표는 보고하지 않음)",
                     "plot_cdf": "seaborn 없음; 그림만 생략",
                     "compute_PR_pairs": "입력 pair_dists 저장 후 공식 함수 호출(관찰만)",
                     "generate_bev_feats": ("결정적 격자 max(_review_native_ring_maxfix.py)" if impl == "maxfix"
                                            else "공식 그대로")},
        "pickle_sha256": digest(pkl), "pair_dists_sha256": digest(cond / "pair_dists.npy"),
        "adapter_pair_dists": str(adapter_path.relative_to(REPO)), "adapter_pair_dists_sha256": digest(adapter_path),
        "vs_adapter": {"max_abs_pair_dist_diff": float(np.abs(pairs - adapter).max()),
                       "top1_agreement": float((top1 == adapter_top1).mean()),
                       "top1_disagreements": int((top1 != adapter_top1).sum())},
        "radii": rows,
        "paper_table_VI_RINGpp_120x120": PAPER_TABLE_VI_120 if bev == "feat" else None,
        "official_source_sha256": {p: digest(OFFICIAL / p) for p in
                                   ("evaluation/evaluate.py", "evaluation/plot_PR_curve.py", "utils/core.py",
                                    "utils/circorr2.py", "datasets/NCLTDataset.py", "utils/poses.py")},
        "script_sha256": digest(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(cond / "report.json", report)
    save_json(cond / "status.json", {"stage": "complete", "pid": os.getpid(), "elapsed_s": elapsed})
    print(json.dumps({k: report[k] for k in ("condition", "vs_adapter")} | {"10m": rows["10.0"]}), flush=True)


def summarize() -> None:
    """완료 조건을 한 표로 모은다."""
    rows = []
    for rep in sorted(OUT.glob("*/report.json")):
        d = json.loads(rep.read_text())
        r10 = d["radii"]["10.0"]
        rows.append({"condition": d["condition"], "queries": d["queries"], "maps": d["maps"],
                     "official_R@1_10m": r10["official_printed_R@1"],
                     "adapter_R@1_10m": r10["adapter_official_R@1"],
                     "positive_only_R@1_10m": r10["positive_only_R@1"],
                     "top1_agreement_vs_adapter": d["vs_adapter"]["top1_agreement"],
                     "max_abs_pair_dist_diff": d["vs_adapter"]["max_abs_pair_dist_diff"],
                     "TE_q_correct_10m": r10["TE_m_q50_75_95_correct"],
                     "RE_q_correct_10m": r10["RE_deg_q50_75_95_correct"]})
    save_json(OUT / "summary.json", {"rows": rows, "paper_table_VI_RINGpp_120x120": PAPER_TABLE_VI_120,
                                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps(rows, indent=1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "summarize"])
    parser.add_argument("--split", choices=["test", "all"])
    parser.add_argument("--bev", choices=["occ", "feat"])
    parser.add_argument("--impl", choices=["official", "maxfix"], default="official")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        run(args.split, args.bev, args.impl)
    else:
        summarize()
