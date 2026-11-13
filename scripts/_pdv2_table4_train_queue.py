#!/usr/bin/env python3
"""표 4의 그래프 없는 투영 세 시드를 순서대로 학습한다."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_table4_20261014"
LOG = REPO / "logs/table4_20261014"


def main() -> None:
    """원래 학습 명령이 정상 종료한 체크포인트만 완료로 기록한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    spec = json.loads((OUT / "training_protocol.json").read_text())
    for path, digest in spec["sha256"].items():
        assert sha256(REPO / path) == digest, path
    for seed in (1, 2, 3):
        dest = REPO / f"checkpoints/pdv2_graph_free_s{seed}"
        marker = OUT / f"training_s{seed}_complete.json"
        if marker.exists():
            assert sha256(dest / "best_model.pth") == json.loads(marker.read_text())["checkpoint_sha256"]
            continue
        assert not dest.exists() or not any(dest.iterdir()), str(dest)
        command = [sys.executable, "-u", "scripts/_train_graph_free.py", "--config", spec["config"],
                   "--checkpoint-dir", str(dest), "--data-root", "/workspace/data", "--seed", str(seed),
                   *spec["mandatory_options"]]
        started = time.time()
        write_json(OUT / "training_status.json", {"status": "running", "seed": seed,
                                                  "started_unix": started, "command": command})
        with (LOG / f"train_graph_free_s{seed}.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        import torch
        blob = torch.load(dest / "best_model.pth", map_location="cpu", weights_only=False)
        state = blob["model_state_dict"]
        assert "gnn.convs.0.linear.weight" in state
        assert all(torch.isfinite(v).all() for v in state.values())
        write_json(marker, {"seed": seed, "started_unix": started, "finished_unix": time.time(),
                            "checkpoint_sha256": sha256(dest / "best_model.pth"),
                            "training_protocol_sha256": sha256(OUT / "training_protocol.json"),
                            "epoch": blob.get("epoch")})
        print("TRAIN_DONE", seed, flush=True)
    write_json(OUT / "training_status.json", {"status": "complete", "seeds": [1, 2, 3]})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "training_status.json", {"status": "failed", "error": repr(error)})
        raise
