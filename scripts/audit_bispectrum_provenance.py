#!/usr/bin/env python3
"""Record reproducible provenance for the historical bispectrum control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "training_kitti_phase_bispectrum_dualstream_fast.yaml"
SOURCES = [REPO / "src/gnn/phase_diff_conv.py", REPO / "src/gnn/model.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/supplementary/bispectrum_provenance.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    dual = cfg["gnn"]["dual_stream"]
    artifacts = sorted(str(p.relative_to(REPO)) for p in REPO.glob("results/**/*bispec*"))
    report = {
        "scope": "historical KITTI-only dual-stream control; not a deployed NSD component",
        "config": str(CONFIG.relative_to(REPO)),
        "config_sha256": sha256(CONFIG),
        "phase_source": cfg["encoding"]["phase_features"]["source"],
        "n_rows": dual["n_rows"],
        "n_freqs": dual["n_freqs"],
        "use_bispectrum": dual["use_bispectrum"],
        "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in SOURCES},
        "result_artifacts_found": artifacts,
        "numeric_claim_status": (
            "No dedicated result artifact/checkpoint was found; historical numeric values "
            "in EXPERIMENT_HANDOFF.md are provenance notes, not independently rerunnable evidence."
            if not artifacts else "Inspect the listed artifact(s) before quoting numeric results."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
