"""장소 제외 그래프·학습 쌍과 고정 승자의 문맥 점수를 계산한다."""
from typing import Tuple

import numpy as np

from _pdv2_context_accept_lib import unit
from retrieval.sequence_evidence import neighbor_plans, path_means, repeated_candidate_weights


def restricted_triplets(triplets: np.ndarray, node_ids: np.ndarray,
                        total_nodes: int) -> np.ndarray:
    """세 노드가 모두 남은 삼중쌍만 골라 연속 노드 번호로 바꾼다."""
    mapping = np.full(total_nodes, -1, dtype=np.int64)
    mapping[node_ids] = np.arange(len(node_ids))
    local = mapping[triplets]
    return local[np.all(local >= 0, axis=1)]


def temporal_subgraph(graph, node_ids: np.ndarray):
    """제외 노드를 물리적으로 제거하고 시간 간선과 그 속성만 보존한다."""
    import torch
    from torch_geometric.data import Data
    mapping = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    mapping[torch.as_tensor(node_ids)] = torch.arange(len(node_ids))
    local = mapping[graph.edge_index]
    keep = (local >= 0).all(dim=0) & (graph.edge_type == 0)
    result = Data(x=graph.x[node_ids].clone(), edge_index=local[:, keep].clone(),
                  edge_attr=graph.edge_attr[keep].clone(), edge_type=graph.edge_type[keep].clone())
    return result


def context_on_fixed_winner(key: np.ndarray, path: np.ndarray, row: dict) -> dict:
    """저장된 스펙트럼 방향·대응과 승자를 유지하고 문맥만 다시 계산한다."""
    q, j = row['query_idx'], row['j_star']
    c = unit(key[:, 288:])
    qn, jn, valid = neighbor_plans(path, q, j)['arc_unique']
    center = np.einsum('nd,nd->n', c[q], c[j])
    values = np.zeros(qn.shape, dtype=np.float64)
    values[valid] = np.einsum('nd,nd->n', c[qn[valid]], c[jn[valid]])
    means = path_means(center, values, valid, repeated_candidate_weights(jn, valid))
    beta = (2./3.)*(11./10.)
    result = dict(row)
    result['context_center'] = center
    result['context_arc'] = (1-beta)*center+beta*means[row['direction'], np.arange(len(q))]
    assert np.isfinite(result['context_arc']).all()
    assert np.all(qn[valid]+10 <= np.broadcast_to(q[None, :, None]+15, qn.shape)[valid])
    assert np.all(jn[valid]+10 <= np.broadcast_to(q[None, :, None]+15, qn.shape)[valid])
    return result
