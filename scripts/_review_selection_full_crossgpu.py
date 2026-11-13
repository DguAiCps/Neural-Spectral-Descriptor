#!/usr/bin/env python3
"""같은 ImLPR 모델의 교차 GPU 수치·승자·지표를 검증 입력에서 대조한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]


def main(tag: str, sequence: str) -> None:
    """원래 FP32·batch16·CPU prefix 검색의 전체 검증 원장을 보존한다."""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import torch
    from _review_selection_bev import read, write, sha, metrics
    from _review_selection_full_eval import OUT, descriptors, positions, query_base
    from _causal_bevplace_pr import ledger
    torch.set_num_threads(4)
    home = OUT / "local_gpu_smoke" / "imlpr_crossgpu"
    home.mkdir(parents=True, exist_ok=True)
    row = read(OUT / "retained_protocol.json")["methods"]["imlpr"]["1"]["candidates"][0]
    assert sha(ROOT / row["path"]) == row["sha256"]
    sensor, seq = sequence.split("/")
    args = argparse.Namespace(method="imlpr", seed=1, candidate=row["id"], scope="retained", role="val", sequence=sequence)
    ids, pos = positions(sensor, seq, "val")
    tick = time.time()
    desc, _ = descriptors(args, row, ids, home)
    encode_s = time.time() - tick
    np.save(home / f"{tag}_descriptors.npy", desc)
    tick = time.time()
    arrays = ledger(desc, pos)
    ix = np.searchsorted(arrays["query_idx"], query_base(sensor, seq, "val")["query_idx"])
    arrays = {k: v[ix] for k, v in arrays.items()}
    search_s = time.time() - tick
    path = home / f"{tag}.npz"
    np.savez_compressed(path, **arrays)
    result = dict(tag=tag, completed_unix=time.time(), sequence=sequence, candidate=row,
                  gpu=torch.cuda.get_device_name(), n_frames=len(ids), n_queries=len(ix),
                  encode_s=encode_s, search_s=search_s, ledger_sha256=sha(path),
                  descriptor_sha256=sha(home / f"{tag}_descriptors.npy"),
                  metrics=metrics(dict(arrays, score_negl2=arrays["score_cos"])),
                  cuda_peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                  precision="original FP32 / batch16 / original CPU prefix cosine")
    write(home / f"{tag}.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sequence", default="kitti360/0004")
    args = parser.parse_args()
    main(args.tag, args.sequence)
