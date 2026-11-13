#!/usr/bin/env python3
"""새 native SeqOT 특징을 같은 NCLT 공통 키프레임에 놓아 기존 원장과 대조한다."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _review_native_helios import sha, write

ROOT = REPO / "results/pdv2_review_native_20261105/seqot"


def main() -> None:
    import torch
    from _disco_official_dump import ledger
    torch.set_num_threads(1)
    native = json.loads((ROOT / "protocol.json").read_text())
    out = ROOT / "common_descriptor_audit"
    out.mkdir(exist_ok=True)
    protocol = {"sequence": "nclt/2012-01-08", "models": ["v2_s1", "v2_s2", "v2_s3"],
                "frozen": "같은원시시각·기존공통키프레임·체크포인트·후보·정답·원CPU검색",
                "new_features": "native실행의새batch1전체stage1출력;공식GeM의같은20장창을CPU로실행",
                "cpu_GeM_check": "nativeDB에동시에포함된모든공통키프레임에서기존GPU공식GeM과최대차이1e-5이하",
                "purpose": "기존공통파이프라인과새native추론사이의수치차이가저성능을설명하는지확인.주표교체/시험설정선택안함"}
    if (out / "protocol.json").exists():
        assert json.loads((out / "protocol.json").read_text()) == protocol
    else:
        write(out / "protocol.json", protocol)
    spec = importlib.util.spec_from_file_location("official_seqot_gem_audit", REPO / "_external/SeqOT/modules/gem.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = []
    for seed in (1, 2, 3):
        model = f"v2_s{seed}"
        work = REPO / "_external/seqot_official_pdv2"
        if seed != 1:
            work /= f"s{seed}"
        cached_path = work / "centered/desc/test_odom/nclt_2012-01-08.npz"
        cached = np.load(cached_path)
        ck = native["models"][model]
        for stage in ("stage1", "stage2"):
            assert str(cached[f"{stage}_sha256"]) == ck["sha256"][stage]
        gem = module.GeM().eval()
        gem.load_state_dict(torch.load(REPO / ck["stage2"], map_location="cpu", weights_only=False)["state_dict"], strict=True)
        source = ROOT / model / "sub/database"
        total = json.loads((source / "done.json").read_text())["total"]
        assert total == 28127
        sub = np.zeros((total + 1, 256), np.float32)
        seen = np.zeros(total, bool)
        for p in source.glob("shard_*.npz"):
            z = np.load(p)
            sub[z["indices"]] = z["descriptors"]
            seen[z["indices"]] = True
        assert seen.all()
        ids = cached["kf_scan_ids"]
        fresh = np.empty_like(cached["final"], dtype=np.float32)
        with torch.inference_mode():
            for row, scan in enumerate(ids):
                indices = np.arange(int(scan) - 10, int(scan) + 10)
                if (indices < 0).any() or (indices >= len(sub)).any():
                    indices = np.full(20, scan)
                fresh[row] = gem(torch.from_numpy(sub[indices][None])).reshape(256).numpy()
        reference = np.load(ROOT / model / "native/database_descriptors.npy")
        shared = ids % 10 == 0
        cpu_error = float(np.max(np.abs(fresh[shared] - reference[ids[shared] // 10])))
        assert cpu_error < 1e-5, cpu_error
        suffix = "" if seed == 1 else f"_s{seed}"
        old_path = REPO / f"results/pdv2_seqot_official_centered{suffix}/odom/dump_seqotl2_nclt_2012-01-08.npz"
        old = np.load(old_path)
        meta = json.loads(str(old["meta"]))
        assert meta["caches"]["n_raw"] == total
        poses = np.load(REPO / meta["caches"]["operating_cache"])["poses_gt"][:, :3, 3]
        tick = time.time()
        actual = ledger(fresh, poses)
        np.testing.assert_array_equal(actual["query_idx"], old["query_idx"])
        np.testing.assert_array_equal(actual["min_past_gt_dist"], old["min_past_gt_dist"])
        common = actual["query_idx"] < len(ids) - 15
        positive = common & (actual["min_past_gt_dist"] <= 5)
        changed = actual["j_star"] != old["j_star"]
        error = np.max(np.abs(fresh - cached["final"]), axis=1)
        row = {"model": model, "keyframes": len(ids), "shared_native_DB_keyframes": int(shared.sum()),
               "cpu_GeM_max_abs_difference": cpu_error,
               "new_vs_old_descriptor_max_abs_difference_quantiles": np.quantile(error, [0, .5, .95, 1]).tolist(),
               "common_queries": int(common.sum()), "common_positive_queries": int(positive.sum()),
               "changed_top1_common_queries": int((changed & common).sum()),
               "old_common_r1": float((old["top1_geo_dist"][positive] <= 5).mean()),
               "fresh_common_r1": float((actual["top1_geo_dist"][positive] <= 5).mean()),
               "score_max_abs_difference": float(np.max(np.abs(actual["score_disco"] - old["score_l2"]))),
               "same_stage1_stage2_hashes": True, "old_descriptor_sha256": sha(cached_path),
               "old_ledger_sha256": sha(old_path), "search_elapsed_s": time.time() - tick}
        np.savez_compressed(out / f"{model}_ledger.npz", **actual)
        reports.append(row)
        write(out / "report.json", {"rows": reports, "complete": len(reports) == 3})
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
