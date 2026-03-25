"""Baseline LiDAR place recognition methods for comparative evaluation."""

from baselines.base import BaselineEncoder

REGISTRY = {}


def register(cls):
    # Instantiate to access property-based short_name
    inst = cls()
    REGISTRY[inst.short_name] = cls
    return cls


def get_available_methods():
    """Return dict of {short_name: class} for methods whose dependencies are met."""
    return {k: v for k, v in REGISTRY.items() if v().is_available()}


def get_method(name):
    """Get method class by short_name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]
