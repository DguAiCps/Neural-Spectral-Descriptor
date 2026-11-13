#!/usr/bin/env python3
"""시간 혼합의 이득을 두 정렬 채널에서 각각 확인한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.spectral_evidence import temporal_scores

OUT = REPO / "results/pdv2_nsd_channel_20261018"


def main() -> None:
    """고정된 두 채널×기존/25%혼합의 전체 교차 결과를 보고한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = exp.protocol(True)
    write_json(OUT / "protocol.json", {"script_sha256": sha256(Path(__file__)),
        "input_protocol_sha256": sha256(exp.OUT / "protocol.json"), "test_used": False,
        "scope": "후속시간25%혼합의이득을본뒤시드별채널선택과의상호작용확인. c_spec/c_col 각각raw/25%혼합, 전부공시. 새로운채널선택이나시험사용없음. 최종후보고정."})
    rows = {}
    for sensor, seq in exp.old.base.VAL:
        sequence = sensor+"/"+seq
        _, _, paths, _ = exp.old.base.inputs("val", sequence)
        cache = exp.arrays(paths["cache"])
        yaw = np.arctan2(cache["poses"][:, 1, 0], cache["poses"][:, 0, 0])
        with np.load(paths["spec"]) as z:
            bank = exp.old.base.bank_for(exp.OP, z)
        result = {}
        for seed in (1, 2, 3):
            result[str(seed)] = {}
            for model in spec["config"]["models"]:
                path = REPO / spec["selected"][model][str(seed)]["folder"] / f"val_{exp.OP}_{sensor}_{seq}.npz"
                a = exp.arrays(path)
                result[str(seed)][model] = {}
                for channel in ("c_spec", "c_col"):
                    chunks = []
                    for start in range(0, len(a["query_idx"]), 512):
                        sl = slice(start, start+512)
                        chunks.append(temporal_scores(bank, yaw, a["query_idx"][sl], a["j_star"][sl], channel)["temporal_independent"])
                    temporal = np.concatenate(chunks)
                    blend = .75*a[channel]+.25*temporal
                    result[str(seed)][model][channel] = {"raw": exp.metrics(a[channel], a), "blend25": exp.metrics(blend, a)}
                    out = OUT / f"s{seed}/{model}_{channel}_{sensor}_{seq}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out, **{k: a[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")}, raw=a[channel], blend25=blend)
        rows[sequence] = result
        write_json(OUT / f"{sensor}_{seq}.json", {"results": result, "test_used": False})
        print("CHANNEL", sequence, flush=True)
    means, values = {}, {}
    for seed in ("1", "2", "3"):
        values[seed] = {}
        for model in spec["config"]["models"]:
            values[seed][model] = {}
            for channel in ("c_spec", "c_col"):
                values[seed][model][channel] = {}
                for method in ("raw", "blend25"):
                    families = []
                    for fam in ("kitti", "helipr", "mulran"):
                        selected = [r[seed][model][channel][method] for seq, r in rows.items() if seq.startswith(fam) and r[seed][model][channel][method]["n_pos"]]
                        families.append({m: float(np.mean([r[m] for r in selected])) for m in exp.METRICS})
                    values[seed][model][channel][method] = {m: float(np.mean([r[m] for r in families])) for m in exp.METRICS}
    for model in spec["config"]["models"]:
        means[model] = {channel: {method: {m: float(np.mean([values[s][model][channel][method][m] for s in values])) for m in exp.METRICS} for method in ("raw", "blend25")} for channel in ("c_spec", "c_col")}
    write_json(OUT / "summary.json", {"per_seed": values, "mean": means, "test_used": False})
    write_json(OUT / "status.json", {"status": "complete", "test_used": False})
    print("CHANNEL_SUMMARY", means, flush=True)


if __name__ == "__main__":
    main()
