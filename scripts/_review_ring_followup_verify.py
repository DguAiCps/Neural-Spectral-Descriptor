#!/usr/bin/env python3
"""RING 후속 진단의 독립 검산과 최종 보고서 작성.

서버 컨테이너 CPU 4스레드로 실행한다.
    docker exec -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
        -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_review_ring_followup_verify.py score
    ... official   (공식 evaluate() 재실행 결과 검산)

score: 고정 부분 질의(0, 25, 50, ..., 마지막)에서 FFT 없이 순환행렬 곱으로 θ축 상관을 직접 계산해
       공식 크기 합·부호 합 거리를 다시 만들고, R@1을 별도 구현으로 다시 센다.
official: 저장한 pair_dists로 R@1을 다시 세 공식 출력과 대조하고 어댑터 pair_dists와의 차이를 확인한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring"
NATIVE = REPO / "results/pdv2_review_native_20261105"
SCORE = ROOT / "score_c1"
OFFICIAL_EVAL = ROOT / "official_eval"
CACHES = {"ring": (NATIVE / "descriptor_cache", "ring"),
          "ringpp_released": (NATIVE / "descriptor_cache", "ringpp"),
          "ringpp_exactmax": (NATIVE / "maxfix/descriptor_cache", "ringpp")}


def digest(path: Path) -> str:
    """파일의 SHA256을 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    """JSON을 원자적으로 저장한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def tings(folder: Path, key: str, rows: list) -> np.ndarray:
    """(N, C, H, W) float64 TING을 읽고 채널마다 평균 0·편향 분산 1로 맞춘다(eps 1e-5)."""
    x = []
    for e in rows:
        with np.load(folder / (e["raw_path"].replace("/", "_").replace(".bin", ".npz"))) as z:
            x.append(z[key].astype(np.float64))
    x = np.stack(x)
    mu = x.mean(axis=(-2, -1), keepdims=True)
    sd = np.sqrt(((x - mu) ** 2).mean(axis=(-2, -1), keepdims=True) + 1e-5)
    return (x - mu) / sd


def direct_dists(q: np.ndarray, maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """corr[m,c,k,w] = Σ_n q[c,(n+k)%H,w]·m[c,n,w]/√H 를 행렬 곱으로 직접 계산한다."""
    c, h, w = q.shape
    shift = (np.arange(h)[None, :] + np.arange(h)[:, None]) % h        # [k, n] -> n+k
    abs_s = np.zeros((len(maps), h))
    sgn_s = np.zeros((len(maps), h))
    for ch in range(c):
        circ = q[ch][shift]                                              # (k, n, W)
        a = np.transpose(circ, (2, 0, 1))                                # (W, k, n)
        b = np.transpose(maps[:, ch], (2, 1, 0))                         # (W, n, M)
        corr = np.matmul(a, b) / np.sqrt(h)                              # (W, k, M)
        abs_s += np.sqrt(corr ** 2 + 1e-15).sum(axis=0).T
        sgn_s += corr.sum(axis=0).T
    scale = 0.15 * c * h * w
    return 1 - abs_s.max(axis=1) / scale, 1 - sgn_s.max(axis=1) / scale


def r1(pairs: np.ndarray, qxy: np.ndarray, mxy: np.ndarray, radius: float) -> tuple[float, float]:
    """전체 분모와 양성 전용 R@1을 별도로 센다."""
    geo = np.sqrt(((qxy[:, None, :] - mxy[None, :, :]) ** 2).sum(-1))
    hit = geo[np.arange(len(geo)), pairs.argmin(axis=1)] <= radius
    pos = (geo <= radius).any(axis=1)
    return float(hit.mean()), float(hit[pos].mean())


def verify_score() -> dict:
    """C1 결과의 독립 검산."""
    protocol = json.loads((SCORE / "protocol.json").read_text())
    summary = json.loads((SCORE / "summary.json").read_text())
    sets = json.loads((NATIVE / "evaluation_sets.json").read_text())
    checks = {"protocol_sha256_matches": summary["protocol_sha256"] == digest(SCORE / "protocol.json"),
              "verify_script_matches_frozen": protocol.get("verify_script_sha256") == digest(Path(__file__)),
              "evaluation_sets_unchanged": protocol["inputs"]["evaluation_sets_sha256"]
              == digest(NATIVE / "evaluation_sets.json"),
              "conditions": {}}
    for split in ("all", "test"):
        rows = sets[split]
        qxy = np.array([e["xy"] for e in rows["query"]], np.float64)
        mxy = np.array([e["xy"] for e in rows["map"]], np.float64)
        picks = sorted(set(range(0, len(rows["query"]), 25)) | {len(rows["query"]) - 1})
        for name, (folder, key) in CACHES.items():
            tag = f"{split}_{name}"
            maps = tings(folder, key, rows["map"])
            sub = tings(folder, key, [rows["query"][i] for i in picks])
            d_abs = np.load(SCORE / f"pair_dists_abs_{tag}.npy")
            d_sgn = np.load(SCORE / f"pair_dists_signed_{tag}.npy")
            err_abs, err_sgn = 0.0, 0.0
            for j, i in enumerate(picks):
                a, s = direct_dists(sub[j], maps)
                err_abs = max(err_abs, float(np.abs(a - d_abs[i]).max()))
                err_sgn = max(err_sgn, float(np.abs(s - d_sgn[i]).max()))
            got = summary["variants"][tag]
            metric_err = {}
            for r in (5.0, 10.0, 15.0, 20.0):
                for label, pairs in (("paper_signed_sum", d_sgn),
                                     ("official_abs_sum", np.load(Path(str(
                                         {"ring": NATIVE / "pair_dists_{s}_ring.npy",
                                          "ringpp_released": NATIVE / "pair_dists_{s}_ringpp.npy",
                                          "ringpp_exactmax": NATIVE / "maxfix/pair_dists_{s}_ringpp.npy"}[name]
                                     ).format(s=split))).astype(np.float64))):
                    a1, p1 = r1(pairs, qxy, mxy, r)
                    ref = got[label][str(r)]
                    metric_err[f"{label}_{r:g}m"] = max(abs(a1 - ref["official_R@1"]), abs(p1 - ref["positive_only_R@1"]))
            checks["conditions"][tag] = {
                "subset_queries": len(picks),
                "direct_vs_fft_max_abs_diff": {"abs_sum": err_abs, "signed_sum": err_sgn},
                "r1_recount_max_abs_err": max(metric_err.values()),
                "existing_recompute": got["official_recompute_vs_existing"],
            }
            print(tag, json.dumps(checks["conditions"][tag]), flush=True)
    worst = lambda k: max(v["direct_vs_fft_max_abs_diff"][k] for v in checks["conditions"].values())
    checks["passed"] = bool(checks["protocol_sha256_matches"] and checks["verify_script_matches_frozen"]
                            and checks["evaluation_sets_unchanged"]
                            and worst("abs_sum") < 1e-9 and worst("signed_sum") < 1e-9
                            and max(v["r1_recount_max_abs_err"] for v in checks["conditions"].values()) == 0.0
                            and all(v["existing_recompute"]["max_abs_diff"] < 1e-4
                                    for v in checks["conditions"].values()))
    checks["script_sha256"] = digest(Path(__file__))
    checks["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(SCORE / "verification.json", checks)
    return checks


def verify_official() -> dict:
    """공식 evaluate() 재실행 결과 검산."""
    sets = json.loads((NATIVE / "evaluation_sets.json").read_text())
    out = {"conditions": {}}
    for rep_path in sorted(OFFICIAL_EVAL.glob("*/report.json")):
        rep = json.loads(rep_path.read_text())
        cond = rep_path.parent
        split, bev = rep["split"], rep["bev_type"]
        pairs = np.load(cond / "pair_dists.npy")
        assert digest(cond / "pair_dists.npy") == rep["pair_dists_sha256"]
        qxy = np.array([e["xy"] for e in sets[split]["query"]], np.float64)
        mxy = np.array([e["xy"] for e in sets[split]["map"]], np.float64)
        folder = cond / "results/nclt" / f"eval_{split}" / "revisit_5.0_10.0_15.0_20.0"
        printed = np.loadtxt(folder / f"recalls_k_{bev}.txt")
        errs = {}
        for j, r in enumerate((5.0, 10.0, 15.0, 20.0)):
            a1, _ = r1(pairs, qxy, mxy, r)
            errs[f"{r:g}m"] = abs(a1 - float(printed[j, 0]))
        adapter = np.load(REPO / rep["adapter_pair_dists"])
        out["conditions"][rep["condition"]] = {
            "printed_vs_recount_R@1_max_abs_err": max(errs.values()),
            "adapter_sha256_matches": digest(REPO / rep["adapter_pair_dists"]) == rep["adapter_pair_dists_sha256"],
            "max_abs_pair_dist_diff_vs_adapter": float(np.abs(pairs - adapter).max()),
            "top1_agreement_vs_adapter": float((pairs.argmin(1) == adapter.argmin(1)).mean()),
        }
    out["passed"] = bool(out["conditions"]) and all(
        v["printed_vs_recount_R@1_max_abs_err"] < 1e-12 and v["adapter_sha256_matches"]
        for v in out["conditions"].values())
    out["deterministic_conditions_identical_to_adapter"] = all(
        v["top1_agreement_vs_adapter"] == 1.0 for k, v in out["conditions"].items()
        if k.endswith("occ_official") or k.endswith("feat_maxfix"))
    out["script_sha256"] = digest(Path(__file__))
    out["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(OFFICIAL_EVAL / "verification.json", out)
    print(json.dumps(out, indent=1), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["score", "official"])
    args = parser.parse_args()
    verify_score() if args.stage == "score" else verify_official()
