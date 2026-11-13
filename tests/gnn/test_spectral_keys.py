"""차원 대조에서 공통 초기값과 시간 의존 범위를 검사한다."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import _pdv2_spectral_keys as run


def test_common_initialization_and_input_dimension():
    """특징 차원 외의 학습 초기값을 같게 하고 역전파를 검사한다."""
    original = run.model("octave288", 1).state_dict()
    rng = np.random.default_rng(10)
    pose = np.repeat(np.eye(4)[None], 40, axis=0)
    pose[:, 0, 3] = np.arange(40) * 0.8
    for arm in run.ARMS:
        net = run.model(arm, 1)
        for name, value in net.state_dict().items():
            if name != "gnn.input_proj.weight" or arm != "trunc352":
                torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        features = rng.normal(size=(40, run.dimension(arm))).astype(np.float32)
        graph = run.base.make_graph(features, pose, np.zeros(40, np.int64))
        net.train()
        output = net(graph)
        assert output.shape == (40, run.dimension(arm) + 128)
        assert output.dtype == torch.float32 and torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()
        )
        assert net.gnn.input_proj.weight.grad.abs().sum() > 0
        net.eval()
        with torch.no_grad():
            full = net(graph)
            prefix = run.base.make_graph(
                features[:31], pose[:31], np.zeros(31, np.int64)
            )
            short = net(prefix)
        torch.testing.assert_close(short[:16], full[:16], atol=4e-6, rtol=4e-5)
