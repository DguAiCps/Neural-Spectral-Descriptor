#!/usr/bin/env python3
"""주표 원천(pdv2_baseline_seeds/summary.json)에서 RING++ 행만 정확max 46원장으로 바꾼 사본을 만든다.

공통 원천 파일은 고치지 않는다. 다른 방법의 항목은 바이트 단위 JSON 값 그대로 둔다.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_main_baselines_official import METRICS, aggregate, sha256  # noqa: E402

BASE = REPO / "results/pdv2_baseline_seeds/summary.json"
FOLDER = "results/pdv2_ringpp_maxfix_20261105/odom"
OUT = REPO / "results/pdv2_baseline_revision_20261107/ringpp/baseline_seeds_summary_ringpp_exactmax.json"


def main() -> None:
    base = json.loads(BASE.read_text())
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    entry = aggregate(FOLDER, "dump_ringpp", "score_corr", "pr_summary_ringpp_maxfix.json", "ringpp", track)
    verified = json.loads((REPO / "results/pdv2_ringpp_maxfix_20261105/summary_verified.json").read_text())["ringpp_maxfix"]["macro"]
    assert all(abs(entry["macro"][k] - verified[k]) < 1e-12 for k in METRICS)
    replaced = dict(base)
    replaced["methods"] = dict(base["methods"])
    replaced["methods"]["ringpp"] = {
        "score": "score_corr", "training_free": True, "seeds": ["-"],
        "per_seed": {"-": {"source": entry["source"], "macro": entry["macro"]}},
        "mean_std": {k: {"mean": entry["macro"][k], "std": None, "values": [entry["macro"][k]]} for k in METRICS},
        "implementation": "released point features + verified exact grid max (np.fmax.at); scripts/_review_native_ring_maxfix.py",
    }
    replaced["ringpp_replacement"] = {"base_summary": str(BASE.relative_to(REPO)), "base_summary_sha256": sha256(BASE),
                                      "previous_ringpp_macro": base["methods"]["ringpp"]["per_seed"]["-"]["macro"],
                                      "ringpp_inputs_sha256": sources, "source_sha256": sha256(Path(__file__))}
    unchanged = [m for m in base["methods"] if m != "ringpp" and replaced["methods"][m] == base["methods"][m]]
    assert len(unchanged) == len(base["methods"]) - 1
    OUT.write_text(json.dumps(replaced, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"ringpp_macro": entry["macro"], "unchanged_methods": unchanged}, ensure_ascii=False))


if __name__ == "__main__":
    main()
