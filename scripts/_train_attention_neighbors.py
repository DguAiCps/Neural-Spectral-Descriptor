#!/usr/bin/env python3
"""같은 초기화·표본·학습량에서 attention 값 전달과 이웃 종류를 비교한다."""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_attention_neighbors import OUT, SAMPLES
from _pdv2_no_context_ablation import sha256, write_json


def array_hash(array: np.ndarray) -> str:
    """자료형·모양·내용을 포함하는 배열 해시를 만든다."""
    array = np.asarray(array)
    h = hashlib.sha256(str((array.dtype, array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def main() -> None:
    """원래 학습기의 데이터·손실·최적화 경로를 유지한 채 실험을 실행한다."""
    import gnn.model as model_module
    from gnn.graph_free import create_graph_free_model
    from gnn.triplet_miner import TripletMiner
    from gnn.trainer import GNNTrainer
    from _pdv2_attention_neighbors import protocol, checkpoint_dir
    spec = protocol()
    index = sys.argv.index("--arm")
    arm = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    assert arm in spec["arms"]
    seed = int(sys.argv[sys.argv.index("--seed") + 1])
    assert "--encoder-preset" in sys.argv and "no_interdiff" in sys.argv
    assert "--use-gated-context" in sys.argv
    assert "--gate-initial-alpha" in sys.argv and "0.0625" in sys.argv
    assert "--resume-checkpoint" not in sys.argv
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = OUT / f"s{seed}" / arm
    folder.mkdir(parents=True, exist_ok=True)
    sample_dir = SAMPLES
    audit = json.loads((sample_dir / "audit.json").read_text())
    for name, expected in audit["sample_sha256"].items():
        assert sha256(sample_dir / name) == expected
    with np.load(sample_dir / "training_inputs.npz") as z:
        reference_inputs = {k: z[k] for k in z.files}
    condition = "original"
    calls = []

    def fixed_miner(self, descriptors, poses, n_triplets_per_anchor=1, sequence_ids=None, **kwargs):
        assert n_triplets_per_anchor == 3
        assert (self.positive_distance_max, self.negative_distance_min) == (5., 10.)
        assert self.mining_strategy == "hard" and self.metric == "l2"
        np.testing.assert_array_equal(descriptors, reference_inputs["descriptors"])
        np.testing.assert_array_equal(poses, reference_inputs["poses"])
        np.testing.assert_array_equal(sequence_ids, reference_inputs["sequences"])
        epoch = 1 + len(calls) * 5
        path = sample_dir / f"{condition}_epoch{epoch}.npz"
        with np.load(path) as z:
            trips = z["triplets"]
        calls.append({"epoch": epoch, "sha256": sha256(path), "count": len(trips)})
        write_json(folder / "mining_calls.json", {"calls": calls})
        return trips.tolist()

    TripletMiner.mine_triplets = fixed_miner
    create_original = model_module.create_spectral_gnn

    def create_model(**kwargs):
        assert kwargs["diffattn_value_source"] == arm.split("_")[0]
        net = create_original(**kwargs)
        hashes = {k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()}
        write_json(folder / "initial_model.json", {"tensor_sha256": hashes,
                   "n_parameters": sum(p.numel() for p in net.parameters())})
        return net

    model_module.create_spectral_gnn = create_model
    init_original = GNNTrainer.__init__
    save_original = GNNTrainer.save_checkpoint
    epoch_original = GNNTrainer.train_epoch

    def init_trainer(self, *args, **kwargs):
        init_original(self, *args, **kwargs)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
        self.matching_updates = 0

        def after_step(optimizer, args, kwargs):
            self.matching_updates += 1

        self.optimizer.register_step_post_hook(after_step)

    def train_epoch(self, graph, *args, **kwargs):
        if self.epoch == 0:
            write_json(folder / "training_graph.json", {key: array_hash(getattr(graph, key).cpu().numpy())
                       for key in ("x", "edge_index", "edge_attr", "edge_type")})
            mask = graph.edge_type == 0
            write_json(folder / "graph_counts.json", {"nodes": int(graph.num_nodes),
                       "temporal": int(mask.sum()), "similarity": int((~mask).sum()),
                       "temporal_index_sha256": array_hash(graph.edge_index[:, mask].cpu().numpy()),
                       "temporal_attr_sha256": array_hash(graph.edge_attr[mask].cpu().numpy())})
            if arm.endswith("temporal"):
                assert bool(mask.all())
        value = epoch_original(self, graph, *args, **kwargs)
        assert np.isfinite(value)
        assert self.matching_updates == (self.epoch + 1) * 45, self.matching_updates
        write_json(folder / "training_progress.json", {"epoch": self.epoch + 1,
                   "optimizer_updates": self.matching_updates, "loss": float(value)})
        return value

    def save_checkpoint(self, filename):
        assert self.checkpoint_dir == checkpoint_dir(seed, arm)
        save_original(self, filename)

    GNNTrainer.__init__ = init_trainer
    GNNTrainer.train_epoch = train_epoch
    GNNTrainer.save_checkpoint = save_checkpoint
    sys.argv[0] = str(REPO / "train_multi_dataset.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
    assert len(calls) == 4


if __name__ == "__main__":
    main()
