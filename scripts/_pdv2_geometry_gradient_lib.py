"""학습을 바꾸지 않고 손실 기울기와 Adam의 가중치 감쇠를 비교한다."""
from typing import Dict, List

import torch


def parameter_group(name: str) -> str:
    """기하·특징 분기의 학습 매개변수를 고정 묶음으로 구분한다."""
    if name.startswith('gnn.edge_encoder.'):
        return 'edge_encoder'
    for layer in (0, 1):
        prefix = f'gnn.convs.{layer}.'
        if name.startswith(prefix + 'bias_mlp.'):
            return f'bias{layer}_' + ('last' if '.2.' in name else 'hidden')
        if name.startswith(prefix):
            return f'content{layer}'
    return 'other'


def gradient_statistics(named_parameters: list, weight_decay: float) -> Dict[str, dict]:
    """현재 기울기와 coupled L2 항의 크기·방향을 독립 집계한다."""
    groups = {}
    for name, parameter in named_parameters:
        group = groups.setdefault(parameter_group(name), [])
        if parameter.grad is not None:
            group.append((parameter.detach().double().flatten(),
                          parameter.grad.detach().double().flatten()))
    result = {}
    for name, values in groups.items():
        if not values:
            result[name] = {'parameters_with_grad': 0}
            continue
        parameter = torch.cat([p for p, _ in values])
        gradient = torch.cat([g for _, g in values])
        decay = weight_decay * parameter
        pn, gn, dn = (float(t.norm()) for t in (parameter, gradient, decay))
        dot = float(torch.dot(gradient, decay))
        result[name] = {'parameters_with_grad': len(values), 'weight_norm': pn,
            'loss_gradient_norm': gn, 'decay_gradient_norm': dn,
            'loss_to_decay_norm_ratio': gn / dn if dn else None,
            'loss_decay_cosine': dot / (gn * dn) if gn and dn else None,
            'zero_loss_gradient_fraction': float((gradient == 0).double().mean()),
            'combined_gradient_norm': float((gradient + decay).norm())}
    return result


def observe_bias(net: torch.nn.Module, attributes: torch.Tensor) -> dict:
    """가중치·난수 상태를 바꾸지 않고 10이웃 기하 편향의 변동을 잰다."""
    modules = [net.gnn.edge_encoder, *[c.bias_mlp for c in net.gnn.convs]]
    modes = [[(m, m.training) for m in root.modules()] for root in modules]
    for module in modules:
        module.eval()
    result = {}
    try:
        with torch.no_grad():
            edge = net.gnn.edge_encoder(attributes, torch.zeros(
                len(attributes), device=attributes.device, dtype=torch.long))
            for i, conv in enumerate(net.gnn.convs):
                hidden = conv.bias_mlp[0](edge).relu()
                bias = conv.bias_mlp[-1](hidden).reshape(-1, 10, 4)
                centered = bias - bias.mean(1, keepdim=True)
                result[str(i)] = {'centered_bias_rms': float(centered.square().mean().sqrt()),
                    'hidden_active_fraction': float((hidden > 0).float().mean()),
                    'last_weight_norm': float(conv.bias_mlp[-1].weight.norm())}
    finally:
        for states in modes:
            for module, mode in states:
                module.training = mode
    return result


def optimizer_partition(net: torch.nn.Module, geometry_decay: float,
                        other_decay: float = 1e-5) -> List[dict]:
    """기하 분기만 별도 감쇠로 묶되 원래 매개변수 순서를 각 묶음에서 보존한다."""
    geometry, other = [], []
    for name, parameter in net.named_parameters():
        target = geometry if parameter_group(name) == 'edge_encoder' or parameter_group(name).startswith('bias') else other
        target.append(parameter)
    assert len(geometry) + len(other) == len(list(net.parameters()))
    return [{'params': geometry, 'weight_decay': geometry_decay},
            {'params': other, 'weight_decay': other_decay}]
