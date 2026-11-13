#!/usr/bin/env python3
"""문맥 수락 대조의 원장·동결 입력·독립 수치 계산을 확인한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_context_accept_fp32 as effective
run = effective.run
effective.configure()
from _pdv2_context_accept_lib import ARMS, labels
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_sequence_fixed_test import frozen_scores
from retrieval.sequence_evidence import neighbor_plans
from retrieval.spectral_evidence import correlation_profiles


def main() -> None:
    """별도 계산으로 정규화·가중합·일부 시간 평균·최종 지표를 검증한다."""
    spec = run.prepare()
    summary = run.exp.read(run.OUT / "summary.json")
    fits = run.exp.read(run.OUT / "fitted.json")
    for p, expected in summary["output_sha256"].items():
        assert sha256(REPO / p) == expected, p
    rows, prefix, references = 0, 0, []
    source_hashes = dict(spec["sources"])
    amendment_path = run.OUT / "fp32_amendment.json"
    amendment = run.exp.read(amendment_path)
    for p, expected in amendment["sources"].items():
        assert sha256(REPO/p) == expected
        source_hashes[p] = expected
    source_hashes[str(amendment_path.relative_to(REPO))] = sha256(amendment_path)
    gpu_prefix = 0
    for seed in (1, 2, 3):
        root = run.train_root(seed)
        info = run.exp.read(root/"keys_complete.json")
        gpu_prefix += len(info["prefix_checks"])
        for name, expected in info["outputs"].items():
            assert sha256(root/name) == expected
        source_hashes[str((root/"keys_complete.json").relative_to(REPO))] = sha256(root/"keys_complete.json")
    for seed in (1, 2, 3):
        fit_x, fit_y, fit_w = [], [], []
        for role in ("train", "val"):
            role_rows = 0
            for sequence in spec[role]:
                p = run.feature_file(role, seed, sequence)
                a = run.exp.arrays(p)
                metadata = run.exp.read(p.with_suffix(".json"))
                assert sha256(p) == metadata["sha256"]
                for source, expected in metadata["sources"].items():
                    assert sha256(REPO / source) == expected, source
                    source_hashes[source] = expected
                source_hashes[str(p.relative_to(REPO))] = sha256(p)
                outputs = run.exp.arrays(run.OUT / f"scores/{role}/s{seed}/{sequence.replace('/', '_')}.npz")
                for field in run.FIELDS:
                    np.testing.assert_array_equal(a[field], outputs[field])
                for method in run.METHODS:
                    current = run.exp.metrics(outputs[method], a)
                    assert current == summary["per_seed"][str(seed)][role][method]["per_sequence"][sequence]
                    assert np.isfinite(outputs[method]).all()
                prefix += len(metadata["prefix_queries"])
                rows += len(a["query_idx"])
                role_rows += len(a["query_idx"])
                eligible, y = labels(a)
                if role == "train":
                    fit_x.append({f: a[f][eligible].astype(np.float64) for f in run.FEATURES})
                    fit_y.append(y[eligible])
                    count = sum(run.family(s) == run.family(sequence) for s in spec["train"])
                    fit_w.append(np.full(int(eligible.sum()), 1/(2*count*eligible.sum())))
                # 각 원장에서 독립 행 계산 네 개를 검사한다.
                sensor, seq = sequence.split("/")
                if role == "train":
                    root = run.train_root(seed)
                    ids = np.flatnonzero(run.exp.arrays(root/"training_metadata.npz")["sequences"] == spec["train"].index(sequence))
                    key = np.load(root/"retrieval_keys.npy", mmap_mode="r")[ids]
                    cachepath, specpath = run.cache_path(sequence), run.spec_path(sequence)
                    original = run.exp.arrays(root/f"ledger_{sensor}_{seq}.npz")
                    raw = original[spec["seeds"][str(seed)]["confidence"]["decision"]]
                else:
                    folder = REPO/spec["selected"][str(seed)]["folder"]
                    key = run.exp.arrays(folder/f"keys_val_{sensor}_{seq}.npz")["embedding"]
                    _, _, paths, _ = run.exp.old.base.inputs("val", sequence)
                    cachepath, specpath = paths["cache"], paths["spec"]
                    original = run.exp.arrays(run.VAL_SOURCE/f"s{seed}/standard_{sensor}_{seq}.npz")
                    raw = original["raw"]
                xyz = run.exp.arrays(cachepath)["poses"][:, :3, 3]
                if role == "train":
                    xyz = xyz.astype(np.float64)
                travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                with np.load(specpath) as z:
                    bank = run.exp.old.base.bank_for(run.exp.OP, z)
                take = np.linspace(0, len(a["query_idx"])-1, 4, dtype=int)
                channel = spec["seeds"][str(seed)]["confidence"]["decision"]
                q, j = a["query_idx"][take], a["j_star"][take]
                expected = frozen_scores(bank, travel, q, j, channel, raw[take])[run.BASELINE]
                np.testing.assert_array_equal(expected, a["baseline"][take])
                qn, jn, valid = neighbor_plans(travel, q, j)["arc_unique"]
                beta = (2/3)*(11/10)
                max_error = 0.
                for r, index in enumerate(take):
                    direction = int(a["direction"][index])
                    js = jn[direction, r][valid[direction, r]]
                    qs = qn[direction, r][valid[direction, r]]
                    weights = np.array([1/np.count_nonzero(js == v) for v in js])
                    for name, section in (("d", slice(0, 288)), ("context", slice(288, None))):
                        def cosine(left: int, right: int) -> float:
                            u, v = key[left, section].astype(np.float64), key[right, section].astype(np.float64)
                            return float((u @ v)/(np.linalg.norm(u)*np.linalg.norm(v)))
                        center = cosine(q[r], j[r])
                        neighbor = np.array([cosine(left, right) for left, right in zip(qs, js)])
                        average = (center+weights @ neighbor)/(1+weights.sum())
                        ref = (1-beta)*center+beta*average
                        np.testing.assert_allclose(ref, a[name+"_arc"][index], atol=3e-7, rtol=0)
                        np.testing.assert_allclose(center, a[name+"_center"][index], atol=3e-7, rtol=0)
                        max_error = max(max_error, abs(ref-a[name+"_arc"][index]))
                references.append({"seed": seed, "role": role, "sequence": sequence, "rows": 4, "max_error": max_error})
            assert role_rows == (71946 if role == "train" else 99236)
        y, weights = np.concatenate(fit_y), np.concatenate(fit_w)
        weights /= weights.sum()
        for method, model in fits["models"][str(seed)].items():
            x = np.column_stack([np.concatenate([r[f] for r in fit_x]) for f in model["fields"]])
            mean = np.average(x, axis=0, weights=weights)
            scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
            np.testing.assert_allclose(mean, model["mean"], atol=1e-12)
            np.testing.assert_allclose(scale, model["scale"], atol=1e-12)
            z = (x-mean)/scale
            coefficient = np.asarray(model["coefficients"])
            score = z @ coefficient+model["intercept"]
            residual = weights*(expit(score)-y)
            grad = z.T @ residual+model["ridge"]*coefficient
            projected = np.where(coefficient <= 1e-6, np.minimum(grad, 0), grad)
            assert max(np.max(np.abs(projected)), abs(residual.sum())) < 2e-6
    for p in (Path(__file__), run.OUT/"summary.json", run.OUT/"fitted.json", run.OUT/"protocol.json"):
        source_hashes[str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT / "completion_validation.json", {
        "status": "passed", "rows": rows, "ledgers": 51, "prefix_checks": prefix,
        "gpu_key_prefix_checks": gpu_prefix, "tf32_enabled": False,
        "independent_reference_rows": len(references)*4, "references": references,
        "sources": source_hashes, "source_count": len(source_hashes),
        "model_unchanged": True, "winners_unchanged": True, "test_used": False,
        "calibration_checks": "학습평균/분산독립계산,양의가중합최적화KKT<2e-6,단변량보정AP동일"})
    print("VERIFIED", rows, len(references)*4, len(source_hashes), flush=True)


if __name__ == "__main__":
    main()
