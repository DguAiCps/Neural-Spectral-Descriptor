#!/usr/bin/env python3
"""Collect hashes and protocol metadata for supplementary experiment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_summary(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    summary = {}
    if isinstance(payload, dict):
        for key in ("_meta", "_query_weighted_mean", "env", "command", "data_root",
                    "weights", "sequences"):
            if key in payload:
                summary[key] = payload[key]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/supplementary/run_provenance_manifest.json",
    )
    args = parser.parse_args()

    roots = [
        REPO / "results/supplementary",
        REPO / "results/deployed_ablation.json",
        REPO / "results/per_query_baselines_regen/bevplace",
        REPO / "results/baseline_latency_bench.json",
        REPO / "results/supplementary/fixed_budget_9seq_summary.md",
        REPO / "docs/SUPPLEMENTARY_EXPERIMENT_STATUS_20260730.md",
        REPO / "docs/SUPPLEMENTARY_PAIRED_ANALYSIS_PROTOCOL_20260730.md",
        REPO / "docs/SUPPLEMENTARY_EVIDENCE_PACKAGE_20260731.md",
        REPO / "docs/SUPPLEMENTARY_COMPETITIVE_EXPERIMENTS_20260731.md",
    ]
    artifacts: list[Path] = []
    for root in roots:
        if root.is_file():
            artifacts.append(root)
        elif root.is_dir():
            artifacts.extend(sorted(root.rglob("*.json")))

    output = args.output.resolve()
    records = []
    for path in artifacts:
        if path.resolve() == output:
            continue
        records.append({
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "summary": json_summary(path),
        })

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector_command": "python3 scripts/collect_supplementary_provenance.py",
        "artifact_count": len(records),
        "artifacts": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output} ({len(records)} artifacts)")


if __name__ == "__main__":
    main()
