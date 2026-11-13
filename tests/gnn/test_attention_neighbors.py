"""정보 전달 변경과 이웃 제거의 수학·자료 불변량을 검사한다."""
import torch
from torch_geometric.data import Data

from gnn.model import DiffAttnConv
from keyframe.graph_manager import filter_edges_by_type


def test_attention_weights_and_message_difference():
    """선형 K의 중심 빼기는 softmax에서 상쇄되고 값 전달만 달라진다."""
    torch.manual_seed(17)
    diff = DiffAttnConv(16, heads=4, edge_dim=8, dropout=0, value_source="diff").double().eval()
    standard = DiffAttnConv(16, heads=4, edge_dim=8, dropout=0, value_source="standard").double().eval()
    standard.load_state_dict(diff.state_dict())
    # 각 목적 노드에 이웃 둘씩, 고립 노드 없음.
    src = torch.tensor([1, 2, 0, 2, 0, 1, 1, 2])
    dst = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    edges = torch.stack([src, dst])
    x = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
    attrs = torch.randn(8, 8, dtype=torch.float64)
    a, (_, wa) = diff(x, edges, attrs, return_attention_weights=True)
    b, (_, wb) = standard(x, edges, attrs, return_attention_weights=True)
    torch.testing.assert_close(wa, wb, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(b - a, diff.W_v(x), atol=1e-12, rtol=1e-12)
    assert sum(p.numel() for p in diff.parameters()) == sum(p.numel() for p in standard.parameters())
    (a.square().mean() + b.square().mean()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_temporal_filter_preserves_inputs_and_edge_attributes():
    """유사 간선을 제거해도 시간 간선·자세 속성·노드 입력은 바뀌지 않는다."""
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    graph = Data(x=x.clone(), edge_index=torch.tensor([[0, 1, 0], [1, 2, 3]]),
                 edge_type=torch.tensor([0, 0, 1]), edge_attr=torch.arange(15).reshape(3, 5),
                 edge_pose_label=torch.tensor([1., 1., 0.]))
    out = filter_edges_by_type(graph.clone(), "temporal_only")
    torch.testing.assert_close(out.x, x)
    torch.testing.assert_close(out.edge_index, graph.edge_index[:, :2])
    torch.testing.assert_close(out.edge_attr, graph.edge_attr[:2])
    torch.testing.assert_close(out.edge_pose_label, graph.edge_pose_label[:2])
    assert out.edge_type.dtype == torch.long and bool((out.edge_type == 0).all())
    assert torch.isfinite(out.x).all()
