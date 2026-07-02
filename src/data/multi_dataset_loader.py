"""Small multi-dataset wrapper used by legacy pipeline entry points."""

from __future__ import annotations

from typing import Dict, Iterable, List


class MultiDatasetLoader:
    """Concatenate several dataset loaders behind one indexable interface."""

    def __init__(self, datasets_config: Iterable[Dict], lazy_load: bool = True):
        self.loaders = []
        for cfg in datasets_config:
            dataset_type = cfg["type"]
            root = cfg["root"]
            for sequence in cfg["sequences"]:
                if dataset_type == "kitti":
                    from data.kitti_loader import KITTILoader
                    self.loaders.append(KITTILoader(root, sequence, lazy_load=lazy_load))
                elif dataset_type == "nclt":
                    from data.nclt_loader import NCLTLoader
                    self.loaders.append(NCLTLoader(root, sequence, lazy_load=lazy_load))
                elif dataset_type == "helipr":
                    from data.helipr_loader import HeLiPRLoader
                    self.loaders.append(HeLiPRLoader(root, sequence, lazy_load=lazy_load))
                elif dataset_type == "mulran":
                    from data.mulran_loader import MulRanLoader
                    self.loaders.append(MulRanLoader(root, sequence, lazy_load=lazy_load))
                else:
                    raise ValueError(f"Unknown dataset type: {dataset_type}")

        self._lengths: List[int] = [len(loader) for loader in self.loaders]
        self._offsets: List[int] = []
        offset = 0
        for length in self._lengths:
            self._offsets.append(offset)
            offset += length
        self._total = offset

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        for loader, offset, length in zip(self.loaders, self._offsets, self._lengths):
            if offset <= idx < offset + length:
                return loader[idx - offset]
        raise IndexError(idx)


def create_multi_dataset_loader(config: Dict, mode: str = "train", lazy_load: bool = True):
    datasets = config["data"]["datasets"][mode]
    return MultiDatasetLoader(datasets, lazy_load=lazy_load)
