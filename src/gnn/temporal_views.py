"""기존 관측 범위 안에서 시간 간선만 가리는 학습용 입력 변형."""
import torch


def temporal_view_mask(edge_index: torch.Tensor, mode: str, seed: int = 0) -> torch.Tensor:
    """노드 특징·간선 속성을 바꾸지 않는 결정적 양방향 간선 마스크다.

    drop은 ±1을 보존하고 나머지 무방향 간선 쌍의 약 절반을 남긴다.
    stride는 기존 ±5 중 ±2·±4만 남긴다. 해시는 전체 노드 수에 의존하지
    않으므로 같은 시드의 접두 입력에서도 공통 간선의 선택이 같다.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('간선 배열은 [2,E]여야 한다')
    source, target = edge_index
    distance = (source - target).abs()
    if mode == 'clean':
        return torch.ones_like(source, dtype=torch.bool)
    if mode == 'stride':
        return (distance == 2) | (distance == 4)
    if mode != 'drop':
        raise ValueError('지원하지 않는 시간 입력 변형')
    low, high = torch.minimum(source, target), torch.maximum(source, target)
    code = ((low * 73856093) ^ (high * 19349663) ^ (int(seed) * 83492791)) & 0x7fffffff
    code = ((code ^ (code >> 13)) * 1274126177) & 0x7fffffff
    code = code ^ (code >> 16)
    return (distance == 1) | ((code & 1) == 0)


def temporal_view(graph, mode: str, seed: int = 0):
    """원 그래프를 보존하고 선택한 시간 간선과 대응 속성을 복사한다."""
    result = graph.clone()
    keep = temporal_view_mask(graph.edge_index, mode, seed)
    result.edge_index = graph.edge_index[:, keep]
    for name in ('edge_attr', 'edge_type'):
        value = getattr(graph, name, None)
        if value is not None:
            setattr(result, name, value[keep])
    return result
