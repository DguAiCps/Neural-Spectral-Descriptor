"""시간 이웃 입력의 사용과 노드별 투영의 이웃 독립성을 확인한다."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_temporal3 import embedding
from _pdv2_pair_evaluate import ArrayCache
from _lag_branch_fusion import TruncatedCache, lag_pass1_graph
from gnn.model import create_spectral_gnn
from gnn.graph_free import create_graph_free_model
from keyframe.graph_manager import filter_edges_by_type


def fixture():
    """고정된 작은 주행과 descriptor를 만든다."""
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(50, 288)).astype(np.float32)
    poses = np.tile(np.eye(4), (50, 1, 1))
    poses[:, 0, 3] = np.arange(50)
    cache = ArrayCache({"poses": poses, "scan_ids": np.arange(50),
                        "keyframe_ids": np.arange(50), "timestamps": np.arange(50) * .1})
    cfg = {"keyframe": {"temporal_neighbors": 10, "graph": {"similarity_threshold": -1.}}}
    kwargs = dict(input_dim=288, hidden_dim=128, context_dim=128, n_layers=2,
                  n_heads=4, dropout=.1, use_residual_gate=True, gate_initial_alpha=.0625,
                  edge_encoder_config={"d_edge": 32, "n_edge_types": 2,
                                       "d_type_embed": 16, "d_rot_encode": 16, "dropout": .1})
    return raw, cache, cfg, kwargs


def test_point_independent_and_temporal_uses_neighbor():
    """같은 쿼리에서 이웃 특징 변화가 시간 모델에만 영향을 준다."""
    torch.set_num_threads(1)
    raw, cache, cfg, kwargs = fixture()
    changed = raw.copy()
    changed[22] *= -1
    for arm, factory in (("temporal", create_spectral_gnn), ("point", create_graph_free_model)):
        torch.manual_seed(7)
        net = factory(**kwargs).eval()
        first, counts = embedding(cache, "kitti", raw, cfg, net, arm, "cpu")
        second, _ = embedding(cache, "kitti", changed, cfg, net, arm, "cpu")
        assert first.shape == (50, 416) and first.dtype == np.float32
        assert np.isfinite(first).all() and counts["similarity"] == 0
        if arm == "point":
            np.testing.assert_array_equal(first[20], second[20])
        else:
            assert np.max(np.abs(first[20] - second[20])) > 1e-6
        cut, _ = embedding(TruncatedCache(cache, 36), "kitti", raw[:36], cfg, net, arm, "cpu")
        np.testing.assert_allclose(first[:21], cut[:21], atol=2e-6, rtol=0)


def test_disabled_similarity_keeps_reference_temporal_graph():
    """유사 간선 생성 생략이 기존 시간 간선·속성을 보존한다."""
    raw, cache, cfg, _ = fixture()
    reference = filter_edges_by_type(lag_pass1_graph(cache, raw, cfg, "cpu", "kitti"), "temporal_only")
    cfg["keyframe"]["graph"].update(similarity_threshold=2., similarity_max_k=0)
    actual = lag_pass1_graph(cache, raw, cfg, "cpu", "kitti")
    for field in ("x", "edge_index", "edge_attr", "edge_type"):
        torch.testing.assert_close(getattr(actual, field), getattr(reference, field), rtol=0, atol=0)
