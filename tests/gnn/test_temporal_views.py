"""시간 간선 변형의 접두 일관성·대칭·속성 정합성을 검증한다."""
import torch
from torch_geometric.data import Data

from src.gnn.temporal_views import temporal_view, temporal_view_mask


def graph(n=30):
    source, target = [], []
    for i in range(n):
        for d in range(-5, 6):
            if d and 0 <= i+d < n:
                source.append(i)
                target.append(i+d)
    edge = torch.tensor([source, target])
    return Data(x=torch.arange(n*3).reshape(n, 3).float(), edge_index=edge,
                edge_attr=torch.arange(edge.shape[1]*5).reshape(-1, 5).float(),
                edge_type=torch.zeros(edge.shape[1], dtype=torch.long))


def test_prefix_and_reverse_edges():
    full, prefix = graph(), graph(17)
    for mode in ('clean', 'drop', 'stride'):
        keep = temporal_view_mask(full.edge_index, mode, 37)
        reverse = temporal_view_mask(full.edge_index.flip(0), mode, 37)
        torch.testing.assert_close(keep, reverse)
        common = (full.edge_index < 17).all(0)
        torch.testing.assert_close(keep[common], temporal_view_mask(prefix.edge_index, mode, 37))
        assert keep.dtype == torch.bool


def test_edge_attributes_and_original_preserved():
    original = graph()
    saved = original.clone()
    view = temporal_view(original, 'drop', 15)
    keep = temporal_view_mask(original.edge_index, 'drop', 15)
    for name in ('x', 'edge_index', 'edge_attr', 'edge_type'):
        torch.testing.assert_close(getattr(original, name), getattr(saved, name))
    torch.testing.assert_close(view.x, original.x)
    torch.testing.assert_close(view.edge_attr, original.edge_attr[keep])
    distance = (original.edge_index[0] - original.edge_index[1]).abs()
    assert keep[distance == 1].all() and 0 < keep.sum() < len(keep)
    even = temporal_view(original, 'stride')
    assert set((even.edge_index[0] - even.edge_index[1]).abs().tolist()) == {2, 4}
    assert torch.isfinite(view.x).all() and torch.isfinite(view.edge_attr).all()


def test_mask_determinism_seed_changes_and_empty_edges():
    edges = graph(100).edge_index
    torch.testing.assert_close(temporal_view_mask(edges, 'drop', 9), temporal_view_mask(edges, 'drop', 9))
    assert not torch.equal(temporal_view_mask(edges, 'drop', 9), temporal_view_mask(edges, 'drop', 10))
    assert temporal_view_mask(torch.empty(2, 0, dtype=torch.long), 'drop').shape == (0,)
