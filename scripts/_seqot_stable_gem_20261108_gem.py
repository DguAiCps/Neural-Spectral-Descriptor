"""Log-domain GeM: the same function as the official SeqOT GeM, stable in float32.

Official (modules/gem.py:17):
    F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1. / p)
           = (mean_k clamp(x_k, eps)^p)^(1/p)
           = exp((logsumexp_k(p * log clamp(x_k, eps)) - log n) / p)

The official float32 form underflows when a whole pooled column is clamped
(clamp(x, 1e-6)^p < 1.18e-38 for p > 6.32) and its backward overflows to
inf/NaN at p ~ 7.409.  The log-domain form has no such underflow.  The
objective, eps, p parameterisation, optimizer and data are unchanged; only
floating-point rounding differs (not bitwise equal to the official form).

NOT applied to any production code.  Proposal for root review only.
"""
import math

import torch


def gem_logdomain(x: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GeM over dim -2 with keepdim, matching the official avg_pool2d output shape."""
    lx = x.clamp(min=eps).log()
    return ((torch.logsumexp(p * lx, dim=-2, keepdim=True) - math.log(x.size(-2))) / p).exp()


def make_gem_logdomain(official_gem_cls):
    """Subclass of the official GeM whose forward uses gem_logdomain.

    Same parameter name (p) and state_dict keys, so official checkpoints load
    unchanged."""

    class GeMLogDomain(official_gem_cls):
        def gem(self, x, p=3, eps=1e-6):
            return gem_logdomain(x, p, eps)

    return GeMLogDomain
