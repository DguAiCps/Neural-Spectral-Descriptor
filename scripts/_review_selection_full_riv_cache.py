#!/usr/bin/env python3
"""같은 원시 스캔의 읽기만 캐시하여 원래 누적 RIV를 보존한다."""
from __future__ import annotations

from collections import OrderedDict
import numpy as np


class ScanCache:
    """원래 로더 결과를 256 MiB 안에서 읽기 전용으로 재사용한다."""
    def __init__(self, loader, byte_limit: int = 256 * 2**20):
        self.loader = loader
        self.byte_limit = byte_limit
        self.items = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __getitem__(self, index):
        index = int(index)
        if index in self.items:
            self.hits += 1
            self.items.move_to_end(index)
            return self.items[index][0]
        self.misses += 1
        item = self.loader[index]
        size = 0
        for value in item.values():
            if isinstance(value, np.ndarray):
                size += value.nbytes
                value.flags.writeable = False
        self.items[index] = item, size
        self.bytes += size
        while self.bytes > self.byte_limit and len(self.items) > 1:
            _, (_, removed) = self.items.popitem(last=False)
            self.bytes -= removed
        return item
