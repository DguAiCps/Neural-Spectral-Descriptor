#!/usr/bin/env python3
"""실제 반대 방향 학습 쌍의 원 특징·문맥 구분력과 점수 차이를 분해한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_real_revisit as run
from _pdv2_real_revisit_report import unit, auxiliary_scores
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_revisit_observation_20261025"


def statistics(values: np.ndarray) -> dict:
    """유한한 값의 평균과 분포를 기록한다."""
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all()
    return {
        "n": len(a),
        "mean": float(a.mean()) if len(a) else None,
        "quantiles": np.quantile(a, [0.05, 0.5, 0.95]).tolist() if len(a) else None,
    }


def summarize(key: np.ndarray, pool: dict) -> tuple:
    """같은 후보 집합에서 원 특징과 문맥의 정답 우위를 분리한다."""
    q, p, n = pool["anchors"], pool["positives"], pool["negatives"]
    pv, nv = p >= 0, n >= 0
    raw, context = unit(key[:, :288]), unit(key[:, 288:])
    array = {}
    summary = {}
    for name, x in (("raw", raw), ("context", context), ("key", unit(key))):
        positive = np.einsum("nd,nkd->nk", x[q], x[p.clip(0)])
        negative = np.einsum("nd,nkd->nk", x[q], x[n.clip(0)])
        negative = np.where(nv, negative, -np.inf)
        margin = positive - negative.max(1)[:, None]
        per_query = np.where(pv, margin, 0).sum(1) / pv.sum(1)
        best = np.where(pv, margin, -np.inf).max(1)
        summary[name] = {
            "mean_positive_margin": statistics(per_query),
            "best_positive_margin": statistics(best),
            "any_positive_win": int((best > 0).sum()),
            "all_positive_win": int(np.all((margin > 0) | ~pv, axis=1).sum()),
            "positive_cosine": statistics(positive[pv]),
            "negative_cosine": statistics(negative[nv]),
        }
        array[name + "_positive"] = positive
        array[name + "_negative"] = negative
        array[name + "_mean_margin"] = per_query
        array[name + "_best_margin"] = best
    # 전체 키가 고른 같은 정답·오답 쌍에서 내적의 두 가산 부분을 나눈다.
    bp = p[np.arange(len(q)), np.where(pv, array["key_positive"], -np.inf).argmax(1)]
    bn = n[np.arange(len(q)), array["key_negative"].argmax(1)]
    values = key.astype(np.float64)
    raw_part = np.sum(values[q, :288] * (values[bp, :288] - values[bn, :288]), axis=1)
    context_part = np.sum(
        values[q, 288:] * (values[bp, 288:] - values[bn, 288:]), axis=1
    )
    direct = np.sum(values[q] * (values[bp] - values[bn]), axis=1)
    error = float(abs(raw_part + context_part - direct).max())
    assert error < 1e-12
    summary["fixed_key_best_pair"] = {
        "raw_margin": statistics(raw_part),
        "context_margin": statistics(context_part),
        "raw_good_context_bad": int(((raw_part > 0) & (context_part <= 0)).sum()),
        "raw_good_final_bad": int(((raw_part > 0) & (direct <= 0)).sum()),
        "raw_bad_context_good": int(((raw_part <= 0) & (context_part > 0)).sum()),
        "both_bad": int(((raw_part <= 0) & (context_part <= 0)).sum()),
        "additivity_error": error,
    }
    array.update(
        best_positive=bp,
        best_negative=bn,
        raw_contribution_margin=raw_part,
        context_contribution_margin=context_part,
        full_margin=direct,
    )
    return summary, array


def main() -> None:
    """학습 자료만으로 난이도·세션·거리별 진단을 고정하며 새 학습은 하지 않는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "summary.json").exists()
    spec = run.protocol(True)
    audit = run.base.read(run.OUT / "integrated_validation.json")
    sources = dict(audit["sources"])
    for name in (
        "scripts/_pdv2_revisit_observation.py",
        "scripts/_pdv2_real_revisit_report.py",
    ):
        sources[name] = sha256(REPO / name)
    write_json(
        OUT / "protocol.json",
        {
            "sources": sources,
            "posthoc": True,
            "test_used": False,
            "scope": "기존1477질의·8590반대방향정답·54838초기채굴오답의 학습 전용 진단. "
            "쉬운 무작위오답이나 온라인 전체 질의 모집단의 성능으로 해석하지 않음. 새 학습 없음.",
        },
    )
    pool = run.base.exp.arrays(run.OUT / "auxiliary_pool.npz")
    q, p, n = pool["anchors"], pool["positives"], pool["negatives"]
    posvalid = p >= 0
    distances = np.linalg.norm(
        pool["positions"][q, None] - pool["positions"][p.clip(0)], axis=2
    )
    group = pool["groups"]
    cross = group[q, None] != group[p.clip(0)]
    selections = run.base.read(run.OUT / "selection.json")["selected_epoch"]
    results = {}
    for arm in ["start"] + run.CFG["arms"]:
        if arm == "start":
            key = pool["initial_keys"]
        else:
            key = np.concatenate(
                [
                    run.base.exp.arrays(
                        run.OUT
                        / arm
                        / f"epoch{selections[arm]}"
                        / "train"
                        / ("keys_" + seq.replace("/", "_") + ".npz")
                    )["embedding"]
                    for seq in spec["train"]
                ]
            )
        summary, arrays = summarize(key, pool)
        summary["by_sequence"] = {
            seq: {
                name: statistics(arrays[name + "_best_margin"][group[q] == i])
                for name in ("raw", "context", "key")
            }
            for i, seq in enumerate(spec["train"])
            if np.any(group[q] == i)
        }
        summary["by_pair_subset"] = {}
        for label, mask in [
            ("within_session", posvalid & ~cross),
            ("cross_session", posvalid & cross),
            ("distance_le1", posvalid & (distances <= 1)),
            ("distance_1to3", posvalid & (distances > 1) & (distances <= 3)),
            ("distance_3to5", posvalid & (distances > 3) & (distances <= 5)),
        ]:
            summary["by_pair_subset"][label] = {
                name: statistics(
                    (
                        arrays[name + "_positive"]
                        - arrays[name + "_negative"].max(1)[:, None]
                    )[mask]
                )
                for name in ("raw", "context", "key")
            }
        if arm != "start":
            np.testing.assert_allclose(
                unit(key[:, :288]),
                unit(pool["initial_keys"][:, :288]),
                atol=4e-6,
                rtol=4e-5,
            )
        np.savez_compressed(
            OUT / (arm + ".npz"),
            anchors=q,
            positives=p,
            negatives=n,
            positive_valid=posvalid,
            distances=distances,
            cross_session=cross,
            **arrays,
        )
        results[arm] = summary
    gradients = run.base.exp.arrays(run.OUT / "calibration_gradients.npz")
    names = list(gradients)
    cosine = {
        a: {
            b: float(
                gradients[a]
                @ gradients[b]
                / np.linalg.norm(gradients[a])
                / np.linalg.norm(gradients[b])
            )
            for b in names
        }
        for a in names
    }
    result = {
        "results": results,
        "initial_gradient_cosine": cosine,
        "test_used": False,
        "new_updates": 0,
        "scope": "학습 보조 후보에 조건부인 진단. 각 양성점수·오답점수의 난이도는 채굴 정의에 의존한다. "
        "거리·세션 조건은 인과적 관측 겹침 측정이 아니다.",
    }
    write_json(OUT / "summary.json", result)
    for path in OUT.glob("*.npz"):
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path in (OUT / "protocol.json", OUT / "summary.json"):
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(
        OUT / "completion_validation.json",
        {
            "passed": True,
            "sources": sources,
            "anchors": len(q),
            "positive_pairs": int(posvalid.sum()),
            "negative_pairs": int((n >= 0).sum()),
            "test_used": False,
        },
    )
    print("OBSERVATION_DIAGNOSIS_COMPLETE", cosine, flush=True)
    for arm, row in results.items():
        print(
            arm,
            {name: row[name]["any_positive_win"] for name in ("raw", "context", "key")},
            row["fixed_key_best_pair"],
            flush=True,
        )


if __name__ == "__main__":
    main()
