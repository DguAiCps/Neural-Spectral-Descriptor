#!/usr/bin/env python3
"""원시 점 구름에서 현재 NSD 캐시의 세 입력을 다시 만들어 확인한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_audit_nsd_encoding.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]

from _audit_current_nsd import local_path, sha256  # noqa: E402
from _bevclip_spec import _sketch  # noqa: E402
from _height_candidate_fusion import make_height_grids  # noqa: E402
from _nclt_fov_fix import fixed_config  # noqa: E402
from _pbev_branch_fusion import BP  # noqa: E402
from _pd_build_test_caches import (  # noqa: E402
    _effective_config, _make_desc_encoder, _make_loader,
)
from run_kitti_operating_point import (  # noqa: E402
    _load_config, _make_encoder, _project_nsd_layout,
)


def main() -> None:
    """KITTI·NCLT의 시작·중간·끝 키프레임을 결정적으로 검사한다."""
    result = {}
    for sensor, seq, role in (("kitti", "09", "val"), ("nclt", "2012-01-08", "test")):
        prefix = "valdump" if role == "val" else "dump"
        path = REPO / f"results/pdv2_c3_bevclip_s1/odom/{prefix}_pdv2_causal_pbev128_main_s1_{sensor}_{seq}.npz"
        with np.load(path) as ledger:
            meta = json.loads(str(ledger["meta"]))
        cache, layouts, spec = [np.load(local_path(meta["caches"][key]))
                                for key in ("cache", "layouts", "spec")]
        cfg = fixed_config() if sensor == "nclt" else _effective_config()
        encoder = _make_desc_encoder(cfg, sensor, "cuda")
        cylindrical_encoder = _make_encoder(_load_config(REPO / "configs/training_place_disjoint.yaml"), "cpu")
        loader = _make_loader(sensor, seq)
        samples = []
        for row in (0, len(cache["scan_ids"]) // 2, len(cache["scan_ids"]) - 1):
            sid = int(cache["scan_ids"][row])
            points = loader[sid]["points"]
            descriptor = encoder.encode_points(points).detach().cpu().numpy()
            cylinder = _project_nsd_layout(cylindrical_encoder, points, n_layout_sectors=60)
            bev, _ = BP.project(points, keep_intensity=False)
            clipped = _sketch(np.maximum(bev, 0))
            _, height, _, _ = make_height_grids(points)
            pairs = {"descriptor": (descriptor, cache["descriptors"][row]),
                     "cylinder": (cylinder, layouts["nsd_layouts"][row]),
                     "clipped_bev": (clipped, spec["sk_re"][row] + 1j * spec["sk_im"][row]),
                     "height": (height.astype(np.float16), spec["raw_height"][row])}
            errors = {}
            for name, (actual, expected) in pairs.items():
                diff = float(np.max(np.abs(actual.astype(np.complex128) - expected)))
                scale = max(float(np.max(np.abs(expected))), 1.)
                assert np.isfinite(actual).all()
                assert diff <= 1e-6 * scale, (sensor, seq, sid, name, diff, scale)
                errors[name] = {"shape": list(actual.shape), "max_abs_diff": diff,
                                "relative_to_max": diff / scale}
            samples.append({"keyframe": row, "scan_id": sid, "checks": errors})
        result[f"{sensor}/{seq}"] = {"samples": samples,
                                    "d_elevation_range": cfg["encoding"]["sensor_elevation_ranges"][sensor],
                                    "cylindrical_elevation_range": [-24.8, 2.]}
    result["script_sha256"] = sha256(Path(__file__))
    out = REPO / "results/nsd_structure_audit_20261013/encoding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
