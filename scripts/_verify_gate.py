#!/usr/bin/env python3
"""Dump the deployed residual-gate alpha and whether the checkpoint carries gate weights."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"

# 1) inspect raw checkpoint keys
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
gate_keys = {k: tuple(v.shape) for k, v in sd.items() if "gate" in k.lower()}
print("checkpoint gate.* keys:", gate_keys if gate_keys else "NONE")

# 2) build model exactly as the eval does and read alpha on KITTI 00
cfg = _apply_encoder_preset(yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
cfg["gnn"]["use_residual_gate"] = True
cfg["gnn"]["gate_initial_alpha"] = 0.0625
model = _make_model(cfg, CKPT, DEVICE)
core = model.gnn if hasattr(model, "gnn") else model
print("core.use_residual_gate =", getattr(core, "use_residual_gate", "?"), "| core.gate is None:", getattr(core, "gate", None) is None)
if getattr(core, "gate", None) is not None:
    print("gate final bias =", float(core.gate[-1].bias.item()),
          "-> sigmoid(bias) =", float(torch.sigmoid(core.gate[-1].bias).item()))
    print("gate final weight abs-sum =", float(core.gate[-1].weight.abs().sum().item()))

cache = np.load(REPO / "data/_verify_cache/kitti_operating_00_layout60.npz")
desc = cache["descriptors"].astype(np.float32); poses = cache["poses"]
graph = _build_eval_graph(
    keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=desc, cache=cache,
    config=cfg, device=DEVICE, temporal_edge_mode="bidirectional",
    temporal_direction_mode="none", similarity_min_k=0, phase_features=None, sensor_key="kitti")
with torch.no_grad():
    _ = model(graph.to(DEVICE))
core2 = model.gnn if hasattr(model, "gnn") else model
alpha = getattr(core2, "_last_alpha", None)
if alpha is not None:
    a = alpha.detach().cpu().numpy().ravel()
    print(f"deployed alpha: mean={a.mean():.5f} std={a.std():.5f} min={a.min():.5f} max={a.max():.5f}")
else:
    print("no _last_alpha recorded (gate not applied)")
