"""실제 온라인 경로와 학습 간선 경로의 일치·상태 보존을 합성 자료로 검사한다."""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from _train_graph_alignment import OnlineGraphRefresh, select_online_edges
from _pdv2_pair_evaluate import ArrayCache
from _lag_branch_fusion import lag_pass1_graph, nsd_embedding_lag
from _causal_branch_fusion import EdgeMLP, N_FEAT
from build_and_train_edge_classifier import range_phase_flat
from _xs_eval_lib import normalize
from gnn.model import create_spectral_gnn


def fixture_data():
    """임계값을 넘는 유사 특징과 비상수 위상 특징을 만든다."""
    torch.manual_seed(31)
    rng = np.random.default_rng(31)
    n = 96
    raw = (1. + rng.normal(0, .03, (n, 288))).astype(np.float32)
    pose = np.tile(np.eye(4), (n, 1, 1))
    pose[:, 0, 3] = np.arange(n) * .8
    cache = ArrayCache(dict(descriptors=raw, poses=pose, timestamps=np.arange(n).astype(float),
                            keyframe_ids=np.arange(n), scan_ids=np.arange(n)))
    layout = rng.uniform(0, 50, (n, 16, 60)).astype(np.float32)
    cfg = {"keyframe": {"temporal_neighbors": 10, "graph": {"similarity_threshold": .993,
                        "similarity_max_k": 10, "similarity_exclude_temporal": True}}}
    net = create_spectral_gnn(input_dim=288, hidden_dim=128, context_dim=128,
                             n_layers=2, n_heads=4, dropout=.1, norm_type="batch_norm",
                             edge_encoder_config={"d_edge": 32, "n_edge_types": 2, "d_type_embed": 16,
                                                  "d_rot_encode": 16, "dropout": .1},
                             use_residual_gate=True, gate_initial_alpha=.0625)
    clf = EdgeMLP(N_FEAT).eval()
    blob = {"mu": np.zeros(14, np.float32), "sd": np.ones(14, np.float32)}
    rvec = np.linspace(.5, 1.5, 288, dtype=np.float32)
    return cache, layout, cfg, net, clf, blob, rvec


def test_refresh_matches_canonical_graph_and_preserves_state(tmp_path):
    """같은 가중치의 실제 두 통과와 간선/속성·둘째 출력을 비교한다."""
    torch.set_num_threads(1)
    cache, layout, cfg, net, clf, blob, rvec = fixture_data()
    raw = cache["descriptors"]
    reference = copy.deepcopy(net).eval()
    reference.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(rvec), requires_grad=False)
    captured = []
    hook = reference.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    wanted = nsd_embedding_lag(cache, "kitti", raw, layout, cfg, reference, clf, blob, rvec, "cpu")
    hook.remove()
    graph = lag_pass1_graph(cache, raw, cfg, "cpu", "kitti")
    first = graph.clone()
    record = {"sequence": "synthetic/train", "offset": 0, "cache": cache, "graph": first,
              "d_rm": normalize(raw * rvec[None]), "phase": range_phase_flat(layout)}
    net.train()
    refresh = OnlineGraphRefresh(graph, net, tmp_path, cfg, clf, blob, rvec, [record], workers=1)
    try:
        refresh.refresh(0, 0)
        assert net.training and net.gnn.key_scale is None
        assert torch.equal(graph.edge_index, captured[1].edge_index)
        torch.testing.assert_close(graph.edge_attr, captured[1].edge_attr, atol=1e-7, rtol=1e-7)
        assert torch.equal(graph.edge_type, captured[1].edge_type)
        net.eval()
        net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(rvec), requires_grad=False)
        with torch.no_grad():
            result = normalize(net(graph).numpy())
        np.testing.assert_allclose(result, wanted, atol=1e-7, rtol=1e-7)
        net.gnn.key_scale = None
        net.train()
        output = net(graph)
        assert output.shape == (96, 416) and output.dtype == torch.float32
        output[:, 288:].square().sum().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    finally:
        refresh.close()


def test_low_probability_keeps_top_k_and_pose_is_not_a_label_input():
    """확률 하한 없이 과거10개를 고르고 GT 위치를 선택에 쓰지 않는지 본다."""
    torch.set_num_threads(1)
    cache, layout, cfg, net, clf, blob, rvec = fixture_data()
    for p in clf.parameters():
        p.data.zero_()
    list(clf.parameters())[-1].data.fill_(-50.)
    net.eval()
    with torch.no_grad():
        emb = net(lag_pass1_graph(cache, cache["descriptors"], cfg, "cpu", "kitti")).numpy()
    d_rm, phase = normalize(cache["descriptors"] * rvec[None]), range_phase_flat(layout)
    idx, attr = select_online_edges(emb, d_rm, phase, cache["poses"], clf, blob, "cpu")
    altered = cache["poses"].copy()
    altered[:, :3, 3] = 1e6
    idx2, attr2 = select_online_edges(emb, d_rm, phase, altered, clf, blob, "cpu")
    np.testing.assert_array_equal(idx, idx2)
    np.testing.assert_array_equal(attr, attr2)
    assert (idx[0] <= idx[1] - 30).all()
    counts = np.bincount(idx[1], minlength=len(emb))
    np.testing.assert_array_equal(counts, np.minimum(10, np.maximum(0, np.arange(len(emb)) - 29)))
    assert (attr[:, 4] < 1e-10).all()
