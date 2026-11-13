#!/usr/bin/env python3
"""학습 전체의 적격 오답을 검색하여 재방문 특징 비교의 채굴 편향을 검사한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_real_revisit as run
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_revisit_observation import statistics

OUT = REPO / "results/pdv2_revisit_fullbank_20261025"
MATCHED = REPO / "results/pdv2_revisit_matched_20261025"


def normalize(x: np.ndarray) -> np.ndarray:
    """배정밀도로 정규화한다."""
    y = x.astype(np.float64)
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-15)


def analyze(x: np.ndarray, pool: dict) -> tuple:
    """같은 좌표의 25m 초과 오답을 빠짐없이 검색하고 독립 CPU 계산과 대조한다."""
    q, p, n = (pool[k] for k in ("anchors", "positives", "negatives"))
    pos, coord = pool["positions"], pool["coordinate"]
    x = normalize(x)
    gpu = torch.from_numpy(x).cuda()
    negative, negative_id, eligible_counts = [], [], []
    check_rows = set(np.linspace(0, len(q) - 1, 17, dtype=int).tolist())
    check_rows.update(np.flatnonzero(np.r_[True, coord[q][1:] != coord[q][:-1]]))
    errors = []
    for start in range(0, len(q), 32):
        anchors = q[start : start + 32]
        valid = (coord[anchors, None] == coord[None]) & (
            np.linalg.norm(pos[anchors, None] - pos[None], axis=2) > 25.0
        )
        score = gpu[torch.as_tensor(anchors, device="cuda")] @ gpu.T
        values, indices = score.masked_fill(
            ~torch.from_numpy(valid).cuda(), -torch.inf
        ).max(1)
        vals, ids = values.cpu().numpy(), indices.cpu().numpy()
        assert np.isfinite(vals).all() and valid[np.arange(len(ids)), ids].all()
        for local in range(len(anchors)):
            if start + local in check_rows:
                independent = x[anchors[local]] @ x[valid[local]].T
                errors.append(float(abs(independent.max() - vals[local])))
        negative.extend(vals)
        negative_id.extend(ids)
        eligible_counts.extend(valid.sum(1))
    negative = np.asarray(negative)
    pv, nv = p >= 0, n >= 0
    positive = np.einsum("nd,nkd->nk", x[q], x[p.clip(0)])
    frozen = np.where(nv, np.einsum("nd,nkd->nk", x[q], x[n.clip(0)]), -np.inf)
    margin = positive - negative[:, None]
    best = np.where(pv, margin, -np.inf).max(1)
    mean = np.where(pv, margin, 0).sum(1) / pv.sum(1)
    frozen_best = np.where(pv, positive - frozen.max(1)[:, None], -np.inf).max(1)
    gap = negative - frozen.max(1)
    assert gap.min() > -1e-12 and max(errors) < 1e-12
    summary = {
        "any_positive_win": int((best > 0).sum()),
        "robust_any_positive_win": int((best > 1e-6).sum()),
        "ambiguous_queries_1e6": int((abs(best) <= 1e-6).sum()),
        "all_positive_win": int(np.all((margin > 0) | ~pv, axis=1).sum()),
        "best_positive_margin": statistics(best),
        "mean_positive_margin": statistics(mean),
        "frozen_any_positive_win": int((frozen_best > 0).sum()),
        "fullbank_extra_hardness": statistics(gap),
        "harder_than_frozen_1e6": int((gap > 1e-6).sum()),
        "eligible_negatives": statistics(np.asarray(eligible_counts)),
        "independent_checks": len(errors),
        "independent_max_error": max(errors),
    }
    array = dict(
        positive=positive,
        negative=negative,
        negative_id=negative_id,
        best_margin=best,
        mean_margin=mean,
        frozen_best_margin=frozen_best,
    )
    return summary, array


def main() -> None:
    """다섯 고정 모델의 학습 진단을 실행하고 입력과 산출물 해시를 검증한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "protocol.json").exists()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {}
    for parent in (run.OUT, MATCHED):
        audit = run.base.read(parent / "integrated_validation.json")
        assert audit["passed"]
        sources.update(audit["sources"])
        sources[str((parent / "integrated_validation.json").relative_to(REPO))] = (
            sha256(parent / "integrated_validation.json")
        )
    for name in (
        "scripts/_pdv2_revisit_fullbank.py",
        "scripts/_pdv2_revisit_observation.py",
    ):
        sources[name] = sha256(REPO / name)
    write_json(
        OUT / "protocol.json",
        {
            "sources": sources,
            "test_used": False,
            "new_updates": 0,
            "scope": "학습1477질의·8590정답쌍을 고정하고 같은 좌표25m초과 전체 오답에서 "
            "각 특징별 최고 오답을 독립 검색. 순방향·역방향과 미래 학습 관측을 모두 포함하므로 "
            "온라인 최종 성능이 아닌 학습 보조 과제 진단이다.",
            "precision": "float64 exhaustive CUDA, independent numpy checks; near ties <=1e-6 separately",
        },
    )
    pool = run.base.exp.arrays(run.OUT / "auxiliary_pool.npz")
    q, p = pool["anchors"], pool["positives"]
    sequences = run.base.read(run.OUT / "protocol.json")["train"]
    arms = ["start", "clean_rank", "pair_key", "context_clipped", "context_matched"]
    results, raw_result = {}, None
    for arm in arms:
        if arm == "start":
            key = pool["initial_keys"]
        else:
            parent = MATCHED if arm == "context_matched" else run.OUT
            source_arm = "pair_context" if arm.startswith("context_") else arm
            epoch = run.base.read(parent / "selection.json")["selected_epoch"][
                source_arm
            ]
            key = np.concatenate(
                [
                    run.base.exp.arrays(
                        parent
                        / source_arm
                        / f"epoch{epoch}"
                        / "train"
                        / ("keys_" + seq.replace("/", "_") + ".npz")
                    )["embedding"]
                    for seq in sequences
                ]
            )
        arrays, summary = {}, {}
        for name, x in (("raw", key[:, :288]), ("context", key[:, 288:]), ("key", key)):
            if name == "raw" and raw_result is not None:
                np.testing.assert_allclose(
                    normalize(x),
                    normalize(pool["initial_keys"][:, :288]),
                    atol=4e-6,
                    rtol=4e-5,
                )
                row, data = raw_result
            else:
                row, data = analyze(x, pool)
                if name == "raw":
                    raw_result = row, data
            summary[name] = row
            arrays.update({name + "_" + k: v for k, v in data.items()})
            print(
                "FULLBANK",
                arm,
                name,
                row["any_positive_win"],
                row["harder_than_frozen_1e6"],
                flush=True,
            )
        raw_good = arrays["raw_best_margin"] > 0
        context_good = arrays["context_best_margin"] > 0
        summary["complementarity"] = {
            "raw_only": int((raw_good & ~context_good).sum()),
            "context_only": int((~raw_good & context_good).sum()),
            "both": int((raw_good & context_good).sum()),
            "neither": int((~raw_good & ~context_good).sum()),
        }
        summary["by_sequence"] = {}
        for i, seq in enumerate(sequences):
            mask = pool["groups"][q] == i
            if mask.any():
                summary["by_sequence"][seq] = {
                    name: {
                        "n": int(mask.sum()),
                        "wins": int((arrays[name + "_best_margin"][mask] > 0).sum()),
                        "margin": statistics(arrays[name + "_best_margin"][mask]),
                    }
                    for name in ("raw", "context", "key")
                }
        cross = pool["groups"][q, None] != pool["groups"][p.clip(0)]
        summary["by_pair_subset"] = {
            label: {
                name: statistics(
                    (arrays[name + "_positive"] - arrays[name + "_negative"][:, None])[
                        mask
                    ]
                )
                for name in ("raw", "context", "key")
            }
            for label, mask in (
                ("within_session", (p >= 0) & ~cross),
                ("cross_session", (p >= 0) & cross),
            )
        }
        if arm == "start":
            for name in ("context", "key"):
                assert summary[name]["fullbank_extra_hardness"]["quantiles"][-1] < 1e-6
                assert summary[name]["harder_than_frozen_1e6"] == 0
        np.savez_compressed(OUT / (arm + ".npz"), anchors=q, positives=p, **arrays)
        results[arm] = summary
    write_json(
        OUT / "summary.json", {"results": results, "test_used": False, "new_updates": 0}
    )
    for path in OUT.iterdir():
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(
        OUT / "completion_validation.json",
        {
            "passed": True,
            "sources": sources,
            "anchors": len(q),
            "bank_nodes": len(pool["groups"]),
            "positive_pairs": int((p >= 0).sum()),
            "test_used": False,
            "independent_checks": sum(
                row[name]["independent_checks"]
                for arm, row in results.items()
                for name in ("context", "key")
            )
            + results["start"]["raw"]["independent_checks"],
        },
    )
    print("FULLBANK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
