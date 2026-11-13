#!/usr/bin/env python3
"""저장한 native 상관 행렬에서 공식 지표를 독립적으로 재검산한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def sha256(path: Path) -> str:
    """파일 해시를 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """네 상관 행렬·공식 PR 산식·R@1을 검산한다."""
    summary = json.loads((OUT / "summary.json").read_text())
    sets = json.loads((OUT / "evaluation_sets.json").read_text())
    assert summary["protocol_sha256"] == sha256(OUT / "protocol.json")
    thresholds = np.concatenate([np.linspace(0, .35, 10), np.linspace(.35, .55, 80), np.linspace(.55, 1, 10)])
    errors = {}
    for split, groups in sets.items():
        q = np.array([e["xy"] for e in groups["query"]])
        m = np.array([e["xy"] for e in groups["map"]])
        geo = np.sqrt(np.sum((q[:, None] - m[None]) ** 2, axis=2))
        for variant in ("ring", "ringpp"):
            key = f"{split}_{variant}"
            pairs = np.load(OUT / f"pair_dists_{key}.npy")
            assert pairs.shape == geo.shape
            assert pairs.dtype == np.float32 and np.isfinite(pairs).all()
            rank = np.argsort(pairs, axis=1)
            selected = np.argmin(pairs, axis=1)
            assert np.array_equal(rank[:, 0], selected)
            accept = pairs.min(axis=1)[:, None] < thresholds
            for radius in (5.0, 10.0, 15.0, 20.0):
                row = summary["variants"][key][str(radius)]
                good = geo[np.arange(len(q)), selected] < radius
                positive = (geo <= radius).any(axis=1)
                hits = (good[:, None] & accept).sum(axis=0)
                fp = (~good[:, None] & accept).sum(axis=0)
                misses = (positive[:, None] & ~accept).sum(axis=0)
                precision = hits / (hits + fp + 1e-10)
                recall = hits / (hits + misses + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                auc = np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) / 2)
                ap = np.sum(np.diff(recall) * precision[1:])
                check = {"official_R@1": float((geo[np.arange(len(q)), selected] <= radius).mean()),
                         "official_max_F1": float(f1.max()), "official_PR_AUC": float(auc),
                         "official_AP": float(ap)}
                errors[f"{key}_{radius:g}m"] = {k: abs(v - row[k]) for k, v in check.items()}
                assert max(errors[f"{key}_{radius:g}m"].values()) < 1e-12
    raw_hashes = json.loads((OUT / "raw_sha256.json").read_text())
    manifest = json.loads((OUT / "descriptor_manifest.json").read_text())
    assert len(raw_hashes) == len(manifest) == 1413
    for rel, expected in raw_hashes.items():
        key = rel.replace("/", "_").replace(".bin", ".npz")
        assert manifest[key]["raw_sha256"] == expected
        assert manifest[key]["finite"]
        assert manifest[key]["shapes"] == {"ring": [1, 120, 120], "ringpp": [6, 120, 120]}
    report = {"passed": True, "raw_input_hashes_matched": len(raw_hashes),
              "metric_checks": 64, "metric_absolute_errors": errors,
              "input_sha256": {p.name: sha256(p) for p in OUT.glob("pair_dists_*.npy")}}
    if OUT.name == "maxfix":
        same = {}
        for split in ("test", "all"):
            original = np.load(OUT.parent / f"pair_dists_{split}_ring.npy")
            diagnostic = np.load(OUT / f"pair_dists_{split}_ring.npy")
            same[split] = bool(np.array_equal(original, diagnostic))
            assert same[split], "RING 기준선이 maxfix 진단에서 달라짐"
        report["unchanged_RING_bitwise_equal"] = same
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metric_absolute_errors"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["official", "maxfix"], default="official")
    args = parser.parse_args()
    if args.variant == "maxfix":
        OUT = OUT / "maxfix"
    main()
