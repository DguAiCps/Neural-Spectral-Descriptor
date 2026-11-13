#!/usr/bin/env python3
"""공식 RING NCLT 세션 간 평가를 별도 산출물로 재현한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"
OFFICIAL = REPO / "_external/RING"
SEQUENCES = ("2012-02-04", "2012-03-17")
sys.path.insert(0, str(REPO / "scripts"))
import _ring_official_lib as rol


def digest(path: Path) -> str:
    """파일의 SHA256을 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    """JSON을 원자적으로 저장한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def imports() -> None:
    """공식 절대 import의 이름 공간을 등록한다."""
    rol.import_ring()
    mod = types.ModuleType("evaluation")
    mod.__path__ = [str(OFFICIAL / "evaluation")]
    mod.__spec__ = importlib.machinery.ModuleSpec("evaluation", None, is_package=True)
    sys.modules["evaluation"] = mod


def prepare(data_root: Path) -> None:
    """공식 생성기로 명시된 두 평가 집합을 고정한다."""
    imports()
    from evaluation.generate_evaluation_sets import generate_evaluation_set

    layout = OUT / "official_layout"
    layout.mkdir(parents=True, exist_ok=True)
    for seq in SEQUENCES:
        (layout / seq / "ground_truth").mkdir(parents=True, exist_ok=True)
        (layout / seq / "velodyne_data").mkdir(parents=True, exist_ok=True)
        for dest, source in [
            (layout / seq / "ground_truth" / f"groundtruth_{seq}.csv", data_root / seq / f"groundtruth_{seq}.csv"),
            (layout / seq / "velodyne_data/velodyne_sync", data_root / seq / "velodyne_sync"),
        ]:
            if not dest.exists():
                dest.symlink_to(source)
    sets = {}
    raw_paths = set()
    for split in ("test", "all"):
        ev = generate_evaluation_set("nclt", str(layout), *SEQUENCES, split=split,
                                     map_sampling_distance=20.0,
                                     query_sampling_distance=5.0, dist_threshold=20.0)
        rows = {}
        for kind, elems in [("map", ev.map_set), ("query", ev.query_set)]:
            rows[kind] = []
            for e in elems:
                seq = Path(e.rel_scan_filepath).parts[-4]
                rel = f"{seq}/velodyne_sync/{Path(e.rel_scan_filepath).name}"
                raw_paths.add(rel)
                rows[kind].append({"raw_path": rel, "timestamp": int(e.timestamp),
                                   "xy": e.position.tolist(), "pose": e.pose.tolist()})
        sets[split] = rows
    source_files = ["datasets/NCLTDataset.py", "evaluation/generate_evaluation_sets.py",
                    "evaluation/evaluate.py", "evaluation/plot_PR_curve.py", "utils/core.py",
                    "utils/config.py", "utils/tools.py", "utils/circorr2.py"]
    protocol = {
        "purpose": "RING/RING++ 저자 공개 NCLT 세션 간 장소 검색 재현; 주표 교체·설정 선택에 사용하지 않음",
        "official_url": "https://github.com/lus6-Jenny/RING",
        "paper_url": "https://arxiv.org/abs/2210.05984",
        "official_commit": "c79c58584ad841629506b5873ed1877c9c278ce5",
        "map_sequence": SEQUENCES[0], "query_sequence": SEQUENCES[1],
        "map_sampling_m": 20.0, "query_sampling_m": 5.0,
        "query_filter_max_distance_m": 20.0,
        "revisit_thresholds_m": [5.0, 10.0, 15.0, 20.0],
        "num_k": 25, "batch_size": 16,
        "regions": {"test": "공식 README 및 생성기 기본 test 영역", "all": "전체 궤적; 공식 생성기 split=all"},
        "frozen_before_results": True, "training": "학습 없음; 체크포인트·가중치·장소 오염 없음",
        "scope_limit": "공식 장소 검색 부분만 실행; 자세 추정·ICP 미실행. 논문 표 II의 정확한 영역·정답반경이 명시적으로 확인되지 않아 숫자 일치 재현으로 주장하지 않음",
        "paper_table_II_nclt_reference_only": {"ring": {"R@1": .7098, "F1": .7658, "AUC": .8246},
                                                "ringpp": {"R@1": .7321, "F1": .7890, "AUC": .8374}},
        "source_sha256": {p: digest(OFFICIAL / p) for p in source_files},
        "pose_sha256": {s: digest(data_root / s / f"groundtruth_{s}.csv") for s in SEQUENCES},
        "counts": {s: {k: len(v) for k, v in rows.items()} for s, rows in sets.items()},
        "unique_raw_scans": len(raw_paths), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(OUT / "protocol.json", protocol)
    save_json(OUT / "evaluation_sets.json", sets)
    (OUT / "raw_files.txt").write_text("\n".join(sorted(raw_paths)) + "\n")
    print(json.dumps(protocol, indent=2, ensure_ascii=False), flush=True)


def run(max_frames: int = 0, shard_index: int = 0, shards: int = 1) -> None:
    """공식 입력·표현·전체 지도 상관을 실행하고 재개 가능한 결과를 쓴다."""
    import torch
    imports()
    core = rol.import_ring_core()
    from datasets.NCLTDataset import NCLTPointCloudLoader
    from evaluation.plot_PR_curve import compute_PR_pairs, compute_AUC, compute_AP

    torch.set_num_threads(4)
    sets = json.loads((OUT / "evaluation_sets.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert digest(OFFICIAL / name) == expected, name
    loader = NCLTPointCloudLoader()
    raw_paths = sorted({e["raw_path"] for rows in sets.values() for group in rows.values() for e in group})
    cache = OUT / "descriptor_cache"
    cache.mkdir(exist_ok=True)
    start = time.time()
    generated = 0
    for i, rel in enumerate(raw_paths):
        if i % shards != shard_index:
            continue
        key = rel.replace("/", "_").replace(".bin", ".npz")
        dest = cache / key
        if dest.exists():
            continue
        raw = OUT / "native_data" / rel
        pc = loader.read_pc(str(raw))[:, :3]
        pc_norm = loader.normalize_pc(pc)
        assert len(pc_norm) >= 30, (rel, len(pc_norm))
        with torch.no_grad():
            occ = core.generate_bev_occ(pc_norm)
            feat = core.generate_bev_feats(pc_norm)
            rol.check_bev(occ.cpu().numpy(), pc_norm, feat.cpu().numpy())
            ring = core.generate_RING(occ)[1].cpu().numpy()
            ringpp = core.generate_RING(feat)[1].cpu().numpy()
        assert ring.dtype == np.float32 and ringpp.dtype == np.float32
        assert np.isfinite(ring).all() and np.isfinite(ringpp).all()
        temp = dest.with_name(dest.stem + ".tmp.npz")
        np.savez_compressed(temp, ring=ring, ringpp=ringpp, raw_sha256=digest(raw), n_points=len(pc_norm))
        temp.replace(dest)
        generated += 1
        if generated % 25 == 0 or generated == 1:
            status = {"stage": "descriptors", "finished": i + 1, "total": len(raw_paths),
                      "generated_this_run": generated, "elapsed_s": time.time() - start,
                      "mean_s_per_scan": (time.time() - start) / generated,
                      "gpu": torch.cuda.get_device_name(0), "pid": os.getpid()}
            save_json(OUT / f"status_gen{shard_index}.json", status)
            print(json.dumps(status), flush=True)
        if max_frames and generated >= max_frames:
            return
    if shards > 1:
        save_json(OUT / f"status_gen{shard_index}.json", {
            "stage": "descriptors_complete", "shard": shard_index, "shards": shards,
            "generated_this_run": generated, "elapsed_s": time.time() - start,
            "pid": os.getpid()})
        return
    result = {"protocol_sha256": digest(OUT / "protocol.json"), "variants": {},
              "gpu": torch.cuda.get_device_name(0), "raw_scans": len(raw_paths),
              "metric_note": "official_*는 원 코드 PR 분모; fixed_denominator_*는 양성 질의 고정 분모"}
    for split, rows in sets.items():
        map_xy = np.array([e["xy"] for e in rows["map"]])
        query_xy = np.array([e["xy"] for e in rows["query"]])
        geo = np.linalg.norm(query_xy[:, None] - map_xy[None], axis=-1)
        for variant in ("ring", "ringpp"):
            key = f"{split}_{variant}"
            def load(row):
                name = row["raw_path"].replace("/", "_").replace(".bin", ".npz")
                with np.load(cache / name) as z:
                    return z[variant]
            maps = torch.from_numpy(np.stack([load(e) for e in rows["map"]])).cuda()
            pair_path = OUT / f"pair_dists_{key}.npy"
            if pair_path.exists():
                pairs = np.load(pair_path)
            else:
                pairs = np.empty((len(rows["query"]), len(rows["map"])), dtype=np.float32)
                for q, row in enumerate(rows["query"]):
                    query = torch.from_numpy(load(row)).cuda()
                    with torch.no_grad():
                        for begin in range(0, len(maps), 16):
                            batch = maps[begin:begin + 16]
                            dist, _ = core.batch_circorr(query.repeat(len(batch), 1, 1, 1), batch)
                            pairs[q, begin:begin + len(batch)] = dist
                    if q % 100 == 0:
                        status = {"stage": "retrieval", "condition": key, "finished": q + 1,
                                  "total": len(rows["query"]), "elapsed_s": time.time() - start,
                                  "pid": os.getpid()}
                        save_json(OUT / "status.json", status)
                        print(json.dumps(status), flush=True)
                assert np.isfinite(pairs).all()
                np.save(pair_path, pairs)
            rank = np.argsort(pairs, axis=1)
            thresholds = np.concatenate([np.linspace(0, .35, 10), np.linspace(.35, .55, 80), np.linspace(.55, 1, 10)])
            stats = {}
            for radius in protocol["revisit_thresholds_m"]:
                p, r, f = compute_PR_pairs(pairs, query_xy, map_xy, thresholds,
                                           str(OUT / f"pr_{key}_{radius:g}m.pdf"), radius)
                hit_rank = np.take_along_axis(geo, rank, axis=1) <= radius
                positive = (geo <= radius).any(axis=1)
                selected_correct = geo[np.arange(len(geo)), rank[:, 0]] < radius
                accept = pairs.min(axis=1)[:, None] < thresholds
                tp = (accept & selected_correct[:, None]).sum(axis=0)
                fp = (accept & ~selected_correct[:, None]).sum(axis=0)
                fixed_r = tp / max(int(positive.sum()), 1)
                fixed_p = tp / np.maximum(tp + fp, 1)
                fixed_f = 2 * fixed_p * fixed_r / np.maximum(fixed_p + fixed_r, 1e-10)
                stats[str(radius)] = {"positive_queries": int(positive.sum()), "queries": len(geo),
                    "official_R@1": float(hit_rank[:, 0].mean()),
                    "positive_only_R@1": float(hit_rank[positive, 0].mean()),
                    "official_R@25": float(hit_rank[:, :25].any(axis=1).mean()),
                    "official_max_F1": float(f.max()), "official_PR_AUC": float(compute_AUC(p, r)),
                    "official_AP": float(compute_AP(p, r)),
                    "fixed_denominator_max_F1": float(fixed_f.max()),
                    "fixed_denominator_PR_AUC": float(compute_AUC(fixed_p, fixed_r))}
            result["variants"][key] = stats
            save_json(OUT / "summary.json", result)
            del maps
    save_json(OUT / "status.json", {"stage": "complete", "elapsed_s": time.time() - start, "pid": os.getpid()})
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/nclt"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args.data_root)
    else:
        assert 0 <= args.shard_index < args.shards
        run(args.max_frames, args.shard_index, args.shards)
