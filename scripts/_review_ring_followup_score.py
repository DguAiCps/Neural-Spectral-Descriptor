#!/usr/bin/env python3
"""RING/RING++ 점수 정의의 사전 고정 원인 대조(C1): 공식 Σ_ω|corr| 대 논문 식 12의 부호 있는 합.

서버 컨테이너 CPU 4스레드로 실행한다(GPU 불필요).
    docker exec -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
        -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_review_ring_followup_score.py freeze
    ... python3 scripts/_review_ring_followup_score.py run

공식 점수(_external/RING/utils/circorr2.py:61-88)는 InstanceNorm한 TING의 θ축 순환 상관을
ω열마다 복소 크기 sqrt(re²+im²+1e-15)로 바꾼 뒤 채널·ω에 대해 더하고, 최댓값을 0.15·C·H·W로 나눈다.
논문(arXiv 2210.05984 식 11-12, 19, 27-28)은 정규화한 TING의 θ·ω 곱을 부호 그대로 더한 상관의 최댓값이다.
C1은 크기 연산 하나만 없앤다. 서술자 캐시·정규화·척도·평가 집합·반경·지표 코드는 고정한다.
같은 실행에서 공식 점수도 독립 재계산해 기존 pair_dists와 대조한다(A3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/score_c1"
NATIVE = REPO / "results/pdv2_review_native_20261105"
OFFICIAL = REPO / "_external/RING"
VERIFY = REPO / "scripts/_review_ring_followup_verify.py"
RADII = [5.0, 10.0, 15.0, 20.0]
EPS_NORM = 1e-5          # nn.InstanceNorm2d 기본값
VARIANTS = {             # 이름: (캐시 폴더, npz 키, 기존 pair_dists 접두)
    "ring": (NATIVE / "descriptor_cache", "ring", NATIVE / "pair_dists_{split}_ring.npy"),
    "ringpp_released": (NATIVE / "descriptor_cache", "ringpp", NATIVE / "pair_dists_{split}_ringpp.npy"),
    "ringpp_exactmax": (NATIVE / "maxfix/descriptor_cache", "ringpp", NATIVE / "maxfix/pair_dists_{split}_ringpp.npy"),
}
PAPER = {"ring": 0.7098, "ringpp_released": 0.7321, "ringpp_exactmax": 0.7321}


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


def cache_manifest(folder: Path) -> str:
    """서술자 캐시 1,413개 파일 해시를 정렬해 다시 해시한다."""
    h = hashlib.sha256()
    for p in sorted(folder.glob("*.npz")):
        h.update(p.name.encode() + b" " + digest(p).encode() + b"\n")
    return h.hexdigest()


def instance_norm(x: np.ndarray) -> np.ndarray:
    """nn.InstanceNorm2d(affine=False): 채널마다 (H, W) 평균 0·편향 분산 1."""
    mu = x.mean(axis=(-2, -1), keepdims=True)
    var = x.var(axis=(-2, -1), keepdims=True)
    return (x - mu) / np.sqrt(var + EPS_NORM)


def load_stack(folder: Path, key: str, rows: list) -> np.ndarray:
    """평가 집합 순서대로 (N, C, H, W) float64 TING을 읽는다."""
    out = []
    for e in rows:
        with np.load(folder / (e["raw_path"].replace("/", "_").replace(".bin", ".npz"))) as z:
            out.append(z[key].astype(np.float64))
    return np.stack(out)


def scores(qn: np.ndarray, mf: np.ndarray, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """질의 1개 대 지도 전체의 (공식 크기 합, 부호 합) 거리. qn (C,H,W), mf (M,C,H,W) θ-FFT."""
    import scipy.fft as sfft

    c, h, w = qn.shape
    qf = sfft.fft(qn, axis=-2, norm="ortho", workers=4)
    d_abs = np.empty(len(mf))
    d_sgn = np.empty(len(mf))
    for b in range(0, len(mf), chunk):
        corr = sfft.ifft(qf[None] * np.conj(mf[b:b + chunk]), axis=-2, norm="ortho", workers=4)
        mag = np.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-15).sum(axis=(1, 3))     # (m, H)
        sgn = corr.real.sum(axis=(1, 3))
        d_abs[b:b + chunk] = 1 - mag.max(axis=-1) / (0.15 * c * h * w)
        d_sgn[b:b + chunk] = 1 - sgn.max(axis=-1) / (0.15 * c * h * w)
    return d_abs, d_sgn


def metrics(pairs: np.ndarray, qxy: np.ndarray, mxy: np.ndarray, pdf_prefix: Path) -> dict:
    """공식 R@k(query_radius ≤ r)·PR(공식 함수, 고정 임계값)과 보조 지표."""
    from evaluation.plot_PR_curve import compute_PR_pairs, compute_AUC, compute_AP

    geo = np.linalg.norm(qxy[:, None] - mxy[None], axis=-1)
    rank = np.argsort(pairs, axis=1)
    thresholds = np.concatenate([np.linspace(0, .35, 10), np.linspace(.35, .55, 80), np.linspace(.55, 1, 10)])
    dmin = pairs.min(axis=1)
    out = {}
    for r in RADII:
        hit = np.take_along_axis(geo, rank, axis=1) <= r
        positive = (geo <= r).any(axis=1)
        top_geo = geo[np.arange(len(geo)), rank[:, 0]]
        p, rec, f = compute_PR_pairs(pairs, qxy.astype(np.float32), mxy.astype(np.float32), thresholds,
                                     str(pdf_prefix) + f"_{r:g}m.pdf", r)
        correct = top_geo < r                     # 공식 PR의 TP 판정(엄격 부등호)
        best_f1 = 0.0
        for t in np.unique(dmin):                 # 임계값 격자 없는 공식 PR 정의의 최대 F1(보조)
            acc = dmin <= t
            tp = int((acc & correct).sum())
            fp = int((acc & ~correct).sum())
            fn = int((~acc & positive).sum())
            if tp:
                pr, rc = tp / (tp + fp), tp / (tp + fn)
                best_f1 = max(best_f1, 2 * pr * rc / (pr + rc))
        wrong = positive & (top_geo > r)
        out[str(r)] = {
            "queries": int(len(geo)), "positive_queries": int(positive.sum()),
            "official_R@1": float(hit[:, 0].mean()), "positive_only_R@1": float(hit[positive, 0].mean()),
            "official_R@5": float(hit[:, :5].any(axis=1).mean()),
            "official_R@25": float(hit[:, :25].any(axis=1).mean()),
            "correct_top1": int(hit[:, 0].sum()),
            "positive_wrong_r_to_20m": int((wrong & (top_geo <= 20)).sum()),
            "positive_wrong_over_20m": int((wrong & (top_geo > 20)).sum()),
            "official_grid_max_F1": float(f.max()), "official_grid_PR_AUC": float(compute_AUC(p, rec)),
            "official_grid_AP": float(compute_AP(p, rec)),
            "gridfree_official_PR_max_F1": float(best_f1),
            "min_dist_quantiles_5_50_95": [float(v) for v in np.quantile(dmin, [.05, .5, .95])],
        }
    return out


def protocol_body() -> dict:
    """결과 전에 고정하는 C1 명세."""
    return {
        "id": "C1_signed_correlation_sum",
        "hypothesis": "공식 circorr2가 ω열마다 상관의 크기를 취해 더하는 점이 논문 식 12·19·27-28의 부호 있는 상관과 달라 "
                      "R@1을 낮춘다.",
        "evidence": {
            "code": "_external/RING/utils/circorr2.py:72-86 (fft θ, ifft, sqrt(re²+im²+1e-15), sum c, sum w, max/0.15chw); "
                    "같은 연산이 _external/RINGSharp/glnet/models/localizer/circorr2.py:78-82에도 있다",
            "paper": "arXiv 2210.05984(results/pdv2_review_native_20261105/ringpp_paper.txt:315-360, 505-540) 식 11-12: C_i(kθ)=Σ_θj Σ_ωm M_Q(θj+kθ,ωm) M_i(θj,ωm); 식 19: 정규화 TING의 최대 상관; "
                     "식 27-28: 채널별 상관을 부호 그대로 더함. 크기 연산은 논문에 없다",
        },
        "changed": "ω열별 복소 크기 sqrt(re²+im²+1e-15)를 실수부 그대로 더하기로 바꾼다(그 밖의 연산 동일)",
        "unchanged": ["기존 서술자 캐시(공식 RING·공식 RING++·결정적 max RING++)", "InstanceNorm eps 1e-5",
                      "0.15·C·H·W 척도", "evaluation_sets.json의 지도·질의·자세", "반경 5/10/15/20m",
                      "공식 compute_PR_pairs·compute_AUC·compute_AP와 고정 임계값 격자"],
        "primary_endpoint": "전체 궤적(all) 10m official R@1과 양성 전용 R@1: RING, 결정적 max RING++",
        "secondary": ["test 영역", "공식(비원자 max) RING++ 캐시", "5/15/20m", "공식 격자 F1·AUC·AP",
                      "격자 없는 공식 PR 최대 F1", "R@5·R@25", "오검색 거리 분류"],
        "decision_rule": "R@1 변화를 부호·크기 그대로 보고한다. 원인으로 판정하려면 primary 두 방법 모두에서 논문 쪽으로 "
                         "증가해야 하며, 증가해도 논문 수치 재현으로 주장하지 않는다. 결과로 설정을 고르거나 주표·main 코드를 바꾸지 않는다.",
        "threshold_caveat": "부호 합은 거리 분포가 달라 공식 고정 임계값 격자가 F1·AUC에 불리할 수 있다. 격자 F1·AUC는 보조로만 쓴다.",
        "independent_check": "scripts/_review_ring_followup_verify.py: 고정 부분 질의의 순환행렬 직접 계산(FFT 없음)과 "
                             "별도 지표 구현으로 재검산",
    }


def freeze() -> None:
    """프로토콜을 결과 전에 저장한다. 이미 있으면 덮어쓰지 않는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    assert not path.exists(), "protocol already frozen"
    body = protocol_body()
    body.update({
        "inputs": {"evaluation_sets_sha256": digest(NATIVE / "evaluation_sets.json"),
                   "cache_manifest_sha256": {k: cache_manifest(v[0]) for k, v in VARIANTS.items()},
                   "existing_pair_dists_sha256": {f"{k}_{s}": digest(Path(str(v[2]).format(split=s)))
                                                  for k, v in VARIANTS.items() for s in ("test", "all")}},
        "official_source_sha256": {p: digest(OFFICIAL / p) for p in ("utils/circorr2.py", "utils/core.py",
                                                                     "evaluation/plot_PR_curve.py")},
        "script_sha256": digest(Path(__file__)),
        "verify_script_sha256": digest(VERIFY) if VERIFY.exists() else None,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    save_json(path, body)
    print(json.dumps({"frozen": str(path), "script_sha256": body["script_sha256"]}), flush=True)


def run() -> None:
    """고정 프로토콜을 확인하고 세 변형 × 두 영역의 두 점수를 계산한다."""
    import _review_native_ring as native

    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["script_sha256"] == digest(Path(__file__)), "script changed after freeze"
    native.imports()
    sets = json.loads((NATIVE / "evaluation_sets.json").read_text())
    assert digest(NATIVE / "evaluation_sets.json") == protocol["inputs"]["evaluation_sets_sha256"]
    import scipy.fft as sfft

    result = {"protocol_sha256": digest(OUT / "protocol.json"), "variants": {}}
    start = time.time()
    for split in ("all", "test"):
        rows = sets[split]
        qxy = np.array([e["xy"] for e in rows["query"]], np.float64)
        mxy = np.array([e["xy"] for e in rows["map"]], np.float64)
        for name, (folder, key, existing) in VARIANTS.items():
            tag = f"{split}_{name}"
            maps = instance_norm(load_stack(folder, key, rows["map"]))
            mf = sfft.fft(maps, axis=-2, norm="ortho", workers=4)
            queries = instance_norm(load_stack(folder, key, rows["query"]))
            d_abs = np.empty((len(queries), len(maps)))
            d_sgn = np.empty_like(d_abs)
            for q in range(len(queries)):
                d_abs[q], d_sgn[q] = scores(queries[q], mf)
                if q % 100 == 0:
                    save_json(OUT / "status.json", {"stage": "scores", "condition": tag, "finished": q + 1,
                                                    "total": len(queries), "elapsed_s": time.time() - start,
                                                    "pid": os.getpid()})
            np.save(OUT / f"pair_dists_abs_{tag}.npy", d_abs)
            np.save(OUT / f"pair_dists_signed_{tag}.npy", d_sgn)
            old = np.load(Path(str(existing).format(split=split)))
            result["variants"][tag] = {
                "official_recompute_vs_existing": {
                    "max_abs_diff": float(np.abs(d_abs - old).max()),
                    "top1_agreement": float((d_abs.argmin(1) == old.argmin(1)).mean()),
                    "top1_disagreements": int((d_abs.argmin(1) != old.argmin(1)).sum())},
                "official_abs_sum": metrics(old.astype(np.float64), qxy, mxy, OUT / f"pr_{tag}_abs"),
                "paper_signed_sum": metrics(d_sgn, qxy, mxy, OUT / f"pr_{tag}_signed"),
                "top1_changed_by_signed": int((d_sgn.argmin(1) != old.argmin(1)).sum()),
                "paper_R@1": PAPER[name],
            }
            r10 = result["variants"][tag]
            r10["delta_10m_R@1_pp"] = 100 * (r10["paper_signed_sum"]["10.0"]["official_R@1"]
                                             - r10["official_abs_sum"]["10.0"]["official_R@1"])
            r10["delta_10m_positive_only_R@1_pp"] = 100 * (r10["paper_signed_sum"]["10.0"]["positive_only_R@1"]
                                                           - r10["official_abs_sum"]["10.0"]["positive_only_R@1"])
            save_json(OUT / "summary.json", result)
            print(json.dumps({"condition": tag, "check": r10["official_recompute_vs_existing"],
                              "delta_10m_R@1_pp": r10["delta_10m_R@1_pp"]}), flush=True)
    result["elapsed_s"] = time.time() - start
    result["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(OUT / "summary.json", result)
    save_json(OUT / "status.json", {"stage": "complete", "elapsed_s": result["elapsed_s"], "pid": os.getpid()})


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "scripts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else run()
