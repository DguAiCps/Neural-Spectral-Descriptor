#!/usr/bin/env python3
"""시간 전용 attention·방향 표본을 기존 손실과 최적화로 비교한다."""
from __future__ import annotations

import json
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_temporal_focus import OUT, SAMPLES, protocol, checkpoint_dir, arm_dir
from _pdv2_temporal3 import SAMPLES as ORIGINAL_SAMPLES
from _train_attention_neighbors import array_hash
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """학습 표본과 배치 순서를 고정하고 최적화 횟수를 검증한다."""
    import gnn.model as model_module
    from gnn.graph_free import create_graph_free_model
    from gnn.triplet_miner import TripletMiner
    from gnn.trainer import GNNTrainer
    spec = protocol()
    index = sys.argv.index("--arm")
    arm = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    seed = int(sys.argv[sys.argv.index("--seed") + 1])
    assert arm in spec["config"]["arms"] and seed in spec["config"]["seeds"]
    for option in ("--encoder-preset", "no_interdiff", "--use-gated-context",
                   "--gate-initial-alpha", "0.0625"):
        assert option in sys.argv
    assert "--resume-checkpoint" not in sys.argv
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = OUT / f"s{seed}/{arm}"
    folder.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ORIGINAL_SAMPLES / "audit.json").read_text())
    for name in ("training_inputs.npz", *(f"original_epoch{e}.npz" for e in (1, 6, 11, 16))):
        assert sha256(ORIGINAL_SAMPLES / name) == audit["sample_sha256"][name]
    with np.load(ORIGINAL_SAMPLES / "training_inputs.npz") as z:
        reference = {k: z[k] for k in z.files}
    calls, orders = [], []

    def fixed_miner(self, descriptors, poses, n_triplets_per_anchor=1, sequence_ids=None, **kwargs):
        assert n_triplets_per_anchor == 3
        assert (self.positive_distance_max, self.negative_distance_min) == (5., 10.)
        assert self.mining_strategy == "hard" and self.metric == "l2"
        for a, b in ((descriptors, reference["descriptors"]), (poses, reference["poses"]),
                     (sequence_ids, reference["sequences"])):
            np.testing.assert_array_equal(a, b)
        epoch = 1 + len(calls) * 5
        policy = arm.split("_")[1]
        path = SAMPLES / f"{policy}_epoch{epoch}.npz"
        with np.load(path) as z:
            trips = z["triplets"]
        assert trips.shape == (183249, 3)
        calls.append({"epoch": epoch, "sha256": sha256(path), "count": len(trips)})
        write_json(folder / "mining_calls.json", {"calls": calls})
        return trips.tolist()

    TripletMiner.mine_triplets = fixed_miner
    create_original = model_module.create_spectral_gnn

    def create_model(**kwargs):
        assert kwargs["diffattn_value_source"] == ("standard" if arm.startswith("standard") else "diff")
        net = (create_original(**kwargs) if not arm.startswith("point")
               else create_graph_free_model(**kwargs))
        write_json(folder / "initial_model.json", {
            "tensor_sha256": {k: array_hash(v.detach().cpu().numpy()) for k, v in net.state_dict().items()},
            "n_parameters": sum(p.numel() for p in net.parameters())})
        return net

    model_module.create_spectral_gnn = create_model
    init_original = GNNTrainer.__init__
    epoch_original = GNNTrainer.train_epoch
    save_original = GNNTrainer.save_checkpoint
    shuffle_original = np.random.shuffle

    def init_trainer(self, *args, **kwargs):
        init_original(self, *args, **kwargs)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
        self.focus_updates = 0

        def after_step(optimizer, args, kwargs):
            self.focus_updates += 1

        self.optimizer.register_step_post_hook(after_step)

    def train_epoch(self, graph, *args, **kwargs):
        assert bool((graph.edge_type == 0).all())
        if self.epoch == 0:
            write_json(folder / "training_graph.json", {
                "hashes": {k: array_hash(getattr(graph, k).cpu().numpy())
                           for k in ("x", "edge_index", "edge_attr", "edge_type")},
                "nodes": int(graph.num_nodes), "temporal": int(graph.edge_index.shape[1]),
                "similarity": 0})
        epoch_seed = seed * 100000 + self.epoch + 1
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed_all(epoch_seed)

        def audited_shuffle(array):
            state_before = np.random.get_state()
            shuffle_original(array)
            if isinstance(array, np.ndarray) and array.shape == (183249, 3):
                state_after = np.random.get_state()
                order = np.arange(len(array))
                np.random.set_state(state_before)
                shuffle_original(order)
                np.random.set_state(state_after)
                orders.append({"epoch": self.epoch + 1, "sha256": array_hash(array),
                               "permutation_sha256": array_hash(order)})

        np.random.shuffle = audited_shuffle
        try:
            value = epoch_original(self, graph, *args, **kwargs)
        finally:
            np.random.shuffle = shuffle_original
        assert np.isfinite(value)
        assert self.focus_updates == (self.epoch + 1) * 45
        assert len(orders) == self.epoch + 1
        write_json(folder / "batch_order.json", {"epochs": orders})
        write_json(folder / "training_progress.json", {"epoch": self.epoch + 1,
                   "optimizer_updates": self.focus_updates, "loss": float(value)})
        return value

    def save_checkpoint(self, filename):
        assert self.checkpoint_dir == checkpoint_dir(seed, arm)
        save_original(self, filename)

    GNNTrainer.__init__, GNNTrainer.train_epoch = init_trainer, train_epoch
    GNNTrainer.save_checkpoint = save_checkpoint
    sys.argv[0] = str(REPO / "train_multi_dataset.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
    assert len(calls) == 4 and len(orders) == 20


if __name__ == "__main__":
    main()
