#!/usr/bin/env python3
"""로컬 입력 목록: 분할 v2 키프레임(scan_ids)과 원시 파일 대응·크기 (2026-11-07).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/fresco_official/inventory.py
-> results/pdv2_fresco_official_20261107/pipeline/inventory_local.json
파일을 읽지 않고 stat만 한다(전송량 추정). 원시 점은 열지 않는다.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FAMILY, REPO, RESULTS, ROLES, cache_path, key, raw_files,  # noqa: E402
                    sha256, write_json)

# 전송 페이로드 바이트/점(무손실 근거는 producer.py 머리 주석)
PAYLOAD_BPP = {"kitti_hdl64": 12, "mulran_os1": 12, "mcd_os1_transposed": 12,
               "nclt_oxford_hdl32": 6, "helipr_vlp16": 13}
RAW_BPP = {"kitti_hdl64": 16, "mulran_os1": 16, "mcd_os1_transposed": 16,
           "nclt_oxford_hdl32": 8, "helipr_vlp16": 22}


def main():
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "roles": {}}
    for role, (seqs, _, _) in ROLES.items():
        rows = {}
        for sensor, seq in seqs:
            cp = cache_path(role, sensor, seq)
            with np.load(cp) as z:
                sids = z["scan_ids"].astype(np.int64)
                has_gt = "poses_gt" in z.files
            files = raw_files(sensor, seq)
            assert sids.max() < len(files) and np.all(np.diff(sids) > 0), key(sensor, seq)
            sizes = np.array([os.stat(files[int(s)]).st_size for s in sids], dtype=np.int64)
            fam = FAMILY[sensor]
            npts = sizes // RAW_BPP[fam]
            rows[key(sensor, seq)] = {
                "family": fam, "n_keyframes": int(len(sids)), "n_raw_files": len(files),
                "scan_id_range": [int(sids[0]), int(sids[-1])],
                "cache": str(cp.relative_to(REPO)), "cache_sha256": sha256(cp),
                "cache_has_poses_gt": bool(has_gt),
                "raw_first": str(files[int(sids[0])]), "raw_bytes_keyframes": int(sizes.sum()),
                "points_mean": float(npts.mean()),
                "payload_bytes_est": int((npts * PAYLOAD_BPP[fam]).sum()),
            }
            print(role, key(sensor, seq), rows[key(sensor, seq)]["n_keyframes"],
                  f"{sizes.sum() / 2**30:.2f} GiB", flush=True)
        out["roles"][role] = {
            "sequences": rows,
            "n_keyframes": sum(r["n_keyframes"] for r in rows.values()),
            "raw_bytes_keyframes": sum(r["raw_bytes_keyframes"] for r in rows.values()),
            "payload_bytes_est": sum(r["payload_bytes_est"] for r in rows.values()),
        }
    write_json(RESULTS / "inventory_local.json", out)
    for role, r in out["roles"].items():
        print(role, r["n_keyframes"], f"raw {r['raw_bytes_keyframes'] / 2**30:.1f} GiB",
              f"payload {r['payload_bytes_est'] / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
