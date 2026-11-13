#!/usr/bin/env python3
"""서버 ring_baseline에서 공개 BEVPlace++ KITTI 평가를 재현한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/pdv2_remaining_native_20261106/bevplace"
PAPER = {"00": 100.0, "02": 99.3, "05": 100.0, "06": 100.0, "08": 99.1}


def sha(path: Path) -> str:
    """원천 파일 해시를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, value: dict) -> None:
    """실행 상태와 결과를 원자적으로 저장한다."""
    path = OUT / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    """공개 입력·가중치·공식 추론과 공식 지표를 고정해 실행한다."""
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", default=["08", "05", "06", "00", "02"])
    parser.add_argument("--last-checkpoint-diagnostic", action="store_true")
    args = parser.parse_args()
    import torch
    import faiss
    from _bevplace_official_dump import _official_model
    if args.last_checkpoint_diagnostic:
        OUT = OUT.with_name("bevplace_last_diagnostic")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    ck = REPO / "_external/BEVPlace2/runs/Aug08_10-17-29/model_best.pth.tar"
    if args.last_checkpoint_diagnostic:
        ck = ck.with_name("checkpoint.pth.tar")
    dataset_root = OUT.parent / "author_data/datasets/KITTI"
    model_module, _, _ = _official_model(ck, 4, 16, "cuda")
    official = sys.modules["kitti_dataset"]
    rows = {}
    started = time.time()
    save("protocol.json", {
        "purpose": "BEVPlace++ 논문 표II의 공개 가중치·공개 KITTI 입력 재현",
        "checkpoint": str(ck.relative_to(REPO)), "checkpoint_sha256": sha(ck),
        "dataset_zip_sha256": sha(OUT.parent / "bevplace_author_dataset.zip"),
        "sequences": args.sequences, "paper_R1_percent": PAPER,
        "training": False, "selection": False, "hardware": torch.cuda.get_device_name(0),
        "official_sources": {n: sha(REPO / "_external/BEVPlace2" / n)
                             for n in ["main.py", "kitti_dataset.py", "REIN.py"]},
    })
    for seq in args.sequences:
        save("status.json", {"stage": "inference", "sequence": seq,
                             "completed": list(rows), "elapsed_s": time.time() - started})
        ds = official.InferDataset(seq, dataset_path=str(dataset_root) + "/", sample_inteval=1)
        path = OUT / f"descriptors_{seq}.npy"
        if path.exists():
            desc = np.load(path)
            assert len(desc) == len(ds)
        else:
            desc = model_module.infer(ds)
            assert np.isfinite(desc).all()
            np.save(path, desc)
        result = official.evaluateResults(seq, desc, None, ds)
        start = ds.db_split_index + 200
        index = faiss.IndexFlatL2(desc.shape[1])
        index.add(np.ascontiguousarray(desc[:ds.db_split_index], dtype=np.float32))
        distance, prediction = index.search(np.ascontiguousarray(desc[start:], dtype=np.float32), 1)
        xyz = ds.poses[:, [3, 7, 11]]
        nearest, selected = [], []
        for q, winner in zip(range(start, len(ds)), prediction[:, 0]):
            squared = ((xyz[:ds.db_split_index] - xyz[q]) ** 2).sum(axis=1)
            nearest.append(float(squared.min()))
            selected.append(float(squared[winner]))
        positive = np.array(nearest) < 25
        correct = np.array(selected) < 25
        independent = float(correct[positive].mean())
        assert abs(float(result) - independent) < 1e-8, (result, independent)
        np.savez_compressed(OUT / f"ledger_{seq}.npz", query_idx=np.arange(start, len(ds)),
                            winner=prediction[:, 0], distance=distance[:, 0],
                            nearest_squared=nearest, selected_squared=selected)
        rows[seq] = {"frames": len(ds), "database": ds.db_split_index,
                     "queries": len(positive), "positive_queries": int(positive.sum()),
                     "hits": int(correct[positive].sum()), "R1_percent": 100 * independent,
                     "paper_R1_percent": PAPER[seq], "delta_pp": 100 * independent - PAPER[seq],
                     "official_function_agrees": True, "poses_sha256": sha(dataset_root / "poses" / f"{seq}.txt"),
                     "descriptors_sha256": sha(path)}
        save("report.json", {"rows": rows, "complete": len(rows) == len(args.sequences),
                             "elapsed_s": time.time() - started})
        print(json.dumps({seq: rows[seq]}, ensure_ascii=False), flush=True)
    save("status.json", {"stage": "complete", "completed": list(rows),
                         "elapsed_s": time.time() - started})


if __name__ == "__main__":
    main()
