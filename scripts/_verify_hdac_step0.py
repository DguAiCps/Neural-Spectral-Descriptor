#!/usr/bin/env python3
"""Step-0 identity checks for the height-aware DAC experiment:
with W_H = 0, the mode-2 pipeline must be indistinguishable from the 288D
baseline -- same graph, same context, same 416D output."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
sys.path.insert(0, str(REPO / 'src'))

from keyframe.selector import Keyframe
from keyframe.graph_manager import build_graph_from_keyframes_batch
from gnn.model import create_spectral_gnn

rng = np.random.RandomState(0)
N = 120
desc528 = (rng.randn(N, 528).astype(np.float32)
           * np.concatenate([np.full(288, 50.0), np.full(240, 1.0)]).astype(np.float32))
poses = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
poses[:, 0, 3] = np.arange(N) * 2.0
poses[:, 1, 3] = rng.randn(N)


def kfs(desc):
    return [Keyframe(keyframe_id=i, scan_id=i, points=np.empty((0, 3), np.float32),
                     pose=poses[i], timestamp=float(i), descriptor=desc[i])
            for i in range(len(desc))]


def build(env):
    if env is None:
        os.environ.pop('NSD_HEIGHT_KEY', None)
        d = desc528[:, :288].copy()
    else:
        os.environ['NSD_HEIGHT_KEY'] = env
        d = desc528
    return build_graph_from_keyframes_batch(
        kfs(d), temporal_neighbors=10, device='cpu', poses=poses,
        descriptors=None, similarity_threshold=None)


def make_model(env):
    if env is None:
        os.environ.pop('NSD_HEIGHT_KEY', None)
    else:
        os.environ['NSD_HEIGHT_KEY'] = env
    torch.manual_seed(0)
    m = create_spectral_gnn(input_dim=288, hidden_dim=256, context_dim=128,
                            use_residual_gate=True, gate_initial_alpha=0.0625)
    ck = torch.load(REPO / 'checkpoints/800d_4sensor_20260511_161726/best_model.pth',
                    map_location='cpu', weights_only=False)
    sd = {k[4:] if k.startswith('gnn.') else k: v
          for k, v in ck['model_state_dict'].items()}
    tgt = m.state_dict()
    compat = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
    m.load_state_dict(compat, strict=False)
    m.eval()
    return m, len(compat), len(tgt)


gA = build('2')          # mode 2: x 288 + x_height 240
gB = build(None)         # baseline: x 288

print('CHECK graph: x dims', tuple(gA.x.shape), tuple(gB.x.shape),
      'x equal:', bool(torch.equal(gA.x, gB.x)))
print('CHECK edge_index identical:', bool(torch.equal(gA.edge_index, gB.edge_index)))
print('CHECK edge_attr identical:', bool(torch.allclose(gA.edge_attr, gB.edge_attr)))
print('CHECK x_height present:', hasattr(gA, 'x_height'), tuple(gA.x_height.shape))

mA, nA, tA = make_model('2')
mB, nB, tB = make_model(None)
print(f'CHECK load: mode2 {nA}/{tA} tensors, baseline {nB}/{tB}')
print('CHECK height_proj zero:', float(mA.height_proj.weight.abs().max()) == 0.0)

with torch.no_grad():
    oA = mA(gA)
    oB = mB(gB)
print('CHECK output dim:', tuple(oA.shape), tuple(oB.shape))
diff = float((oA - oB).abs().max())
print(f'CHECK output max|diff| = {diff:.3e}  ({"PASS" if diff < 1e-6 else "FAIL"})')

# nonzero W_H must change context but never the raw anchor block
with torch.no_grad():
    mA.height_proj.weight.normal_(0, 0.01)
    oA2 = mA(gA)
raw_diff = float((oA2[:, :288] - oA[:, :288]).abs().max())
ctx_diff = float((oA2[:, 288:] - oA[:, 288:]).abs().max())
print(f'CHECK W_H!=0: raw-block diff {raw_diff:.3e} (must be 0), '
      f'ctx diff {ctx_diff:.3e} (must be >0)')
print('STEP0_VERIFY_DONE')
