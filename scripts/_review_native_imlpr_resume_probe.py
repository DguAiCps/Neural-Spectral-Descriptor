#!/usr/bin/env python3
"""보존 ImLPR 학습 재개 상태와 추론 후보의 동일성을 읽기 전용으로 검사한다."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _review_selection_full_catalog import tensor_hash
from _review_native_helios import sha, write


def main() -> None:
    import torch
    torch.set_num_threads(1)
    source = REPO / "results/pdv2_review_selection_full_20261105/retained_protocol.json"
    candidates = json.loads(source.read_text())["methods"]["imlpr"]
    rows = []
    for seed in (1, 2, 3):
        suffix = "" if seed == 1 else f"_s{seed}"
        path = REPO / f"_external/imlpr_official_pdv2/runs{suffix}/ImLPR_pdv2_official/resume.pth"
        state = torch.load(path, map_location="cpu", weights_only=False)
        digest, tensors, elements = tensor_hash({k.removeprefix("module."): v for k, v in state["model"].items()})
        matching = [p for p in candidates[str(seed)]["candidates"] if p["model_tensor_sha256"] == digest]
        steps = [float(x["step"]) for x in state["optimizer"]["state"].values() if "step" in x]
        row = {"seed": seed, "path": str(path.relative_to(REPO)), "sha256": sha(path),
               "epoch": int(state["epoch"]), "keys": sorted(state), "model_tensor_sha256": digest,
               "model_tensors": tensors, "model_elements": elements,
               "matching_retained_candidate_ids": [p["id"] for p in matching],
               "optimizer_lrs": [p["lr"] for p in state["optimizer"]["param_groups"]],
               "optimizer_steps_min_max": [min(steps), max(steps)], "scheduler": state["scheduler"],
               "rng_keys": sorted(state["rng"]), "best_recall": state["best_recall"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    write(REPO / "results/pdv2_review_native_20261105/imlpr/resume_state_audit.json", {"read_only": True, "retained_protocol_sha256": sha(source), "rows": rows})


if __name__ == "__main__":
    main()
