"""
Multi-Dataset Loader for Training on Multiple LiDAR Datasets

Combines KITTI, NCLT, and other datasets into a unified interface.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path

from data.kitti_loader import KITTILoader
from data.nclt_loader import NCLTLoader


class MultiDatasetLoader:
    """
    Multi-dataset loader that combines multiple datasets

    Provides unified interface for training on KITTI + NCLT + others.
    """

    def __init__(self, datasets: List[dict], lazy_load: bool = True):
        """
        Initialize multi-dataset loader

        Args:
            datasets: List of dataset configurations, each dict containing:
                - 'type': 'kitti' or 'nclt'
                - 'root': path to dataset root
                - 'sequences': list of sequences/dates to load
                - 'weight': optional weight for sampling (default 1.0)
            lazy_load: If True, load point clouds on demand

        Example:
            datasets = [
                {
                    'type': 'kitti',
                    'root': '/data/kitti/dataset',
                    'sequences': ['00', '05', '08'],
                    'weight': 1.0
                },
                {
                    'type': 'nclt',
                    'root': '/data/nclt',
                    'sequences': ['2012-01-08', '2012-05-11'],
                    'weight': 1.0
                }
            ]
        """
        self.datasets = []
        self.dataset_info = []
        self.cumulative_lengths = [0]
        self.lazy_load = lazy_load

        # Load all datasets
        for ds_config in datasets:
            ds_type = ds_config['type']
            root = ds_config['root']
            sequences = ds_config['sequences']
            weight = ds_config.get('weight', 1.0)

            for seq in sequences:
                # Create loader for this sequence
                if ds_type == 'kitti':
                    loader = KITTILoader(root, seq, lazy_load=lazy_load)
                    info = {
                        'type': 'kitti',
                        'sequence': seq,
                        'weight': weight
                    }
                elif ds_type == 'nclt':
                    loader = NCLTLoader(root, seq, lazy_load=lazy_load)
                    info = {
                        'type': 'nclt',
                        'date': seq,
                        'weight': weight
                    }
                else:
                    raise ValueError(f"Unknown dataset type: {ds_type}")

                self.datasets.append(loader)
                self.dataset_info.append(info)
                self.cumulative_lengths.append(
                    self.cumulative_lengths[-1] + len(loader)
                )

        self.total_length = self.cumulative_lengths[-1]
        # Convert to numpy array for O(log n) binary search
        self.cumulative_lengths = np.array(self.cumulative_lengths)

        if self.total_length == 0:
            raise ValueError("No data loaded! Check dataset configurations.")

        print(f"Loaded {len(self.datasets)} sequences, {self.total_length} total frames")
        self._print_summary()

    def _print_summary(self):
        """Print summary of loaded datasets"""
        print("\nDataset Summary:")
        print("-" * 70)

        kitti_count = 0
        nclt_count = 0
        kitti_frames = 0
        nclt_frames = 0

        for i, (loader, info) in enumerate(zip(self.datasets, self.dataset_info)):
            num_frames = len(loader)
            if info['type'] == 'kitti':
                seq = info['sequence']
                print(f"  [{i:2d}] KITTI Seq {seq:>2s}: {num_frames:>6d} frames")
                kitti_count += 1
                kitti_frames += num_frames
            else:
                date = info['date']
                print(f"  [{i:2d}] NCLT  {date}: {num_frames:>6d} frames")
                nclt_count += 1
                nclt_frames += num_frames

        print("-" * 70)
        print(f"Total: {kitti_count} KITTI sequences ({kitti_frames} frames), "
              f"{nclt_count} NCLT sequences ({nclt_frames} frames)")
        print(f"Grand total: {self.total_length} frames")
        print()

    def _get_dataset_and_index(self, idx: int) -> tuple:
        """
        Convert global index to (dataset_idx, local_idx)

        Uses np.searchsorted for O(log n) lookup instead of O(n) linear search.

        Args:
            idx: Global frame index

        Returns:
            (dataset_idx, local_idx) tuple
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")

        # O(log n) binary search using searchsorted
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        return int(dataset_idx), int(local_idx)

    def __len__(self) -> int:
        """Return total number of frames across all datasets"""
        return self.total_length

    def __getitem__(self, idx: int) -> dict:
        """
        Get frame data by global index

        Args:
            idx: Global frame index

        Returns:
            Dictionary containing:
                - 'points': (N, 4) array [x, y, z, intensity]
                - 'pose': (4, 4) SE(3) transformation matrix
                - 'timestamp': float timestamp in seconds
                - 'idx': local frame index
                - 'global_idx': global frame index
                - 'dataset_idx': which dataset this came from
                - 'dataset_type': 'kitti' or 'nclt'
        """
        dataset_idx, local_idx = self._get_dataset_and_index(idx)

        # Get data from appropriate dataset
        data = self.datasets[dataset_idx][local_idx]

        # Add global information
        data['global_idx'] = idx
        data['dataset_idx'] = dataset_idx
        data['dataset_type'] = self.dataset_info[dataset_idx]['type']

        return data

    def get_dataset_info(self, dataset_idx: int) -> dict:
        """Get information about a specific dataset"""
        if dataset_idx < 0 or dataset_idx >= len(self.datasets):
            raise IndexError(f"Dataset index {dataset_idx} out of range")

        return self.dataset_info[dataset_idx]

    def get_all_info(self) -> List[dict]:
        """Get information about all loaded datasets"""
        return self.dataset_info

    def get_frames_by_dataset(self, dataset_type: str) -> List[int]:
        """
        Get list of global indices for frames from a specific dataset type

        Args:
            dataset_type: 'kitti' or 'nclt'

        Returns:
            List of global frame indices
        """
        indices = []
        for i, info in enumerate(self.dataset_info):
            if info['type'] == dataset_type:
                start_idx = self.cumulative_lengths[i]
                end_idx = self.cumulative_lengths[i + 1]
                indices.extend(range(start_idx, end_idx))

        return indices

    def split_by_dataset(self) -> Dict[str, List[int]]:
        """
        Split frames by dataset type

        Returns:
            Dictionary mapping dataset_type to list of global indices
        """
        splits = {'kitti': [], 'nclt': []}

        for i, info in enumerate(self.dataset_info):
            start_idx = self.cumulative_lengths[i]
            end_idx = self.cumulative_lengths[i + 1]
            indices = list(range(start_idx, end_idx))

            if info['type'] in splits:
                splits[info['type']].extend(indices)

        return splits


def create_multi_dataset_loader(config: dict, mode: str = 'train') -> Union[MultiDatasetLoader, KITTILoader, NCLTLoader]:
    """
    Factory function to create dataset loader from config

    Args:
        config: Configuration dict with 'data' section
        mode: 'train', 'val', or 'test'

    Returns:
        Dataset loader (MultiDatasetLoader or single dataset loader)
    """
    data_config = config['data']

    # Check if multi-dataset is configured
    if 'datasets' in data_config:
        # Multi-dataset mode
        datasets_config = data_config['datasets']

        if mode == 'train':
            datasets = datasets_config.get('train', [])
        elif mode == 'val':
            datasets = datasets_config.get('val', [])
        else:  # test
            datasets = datasets_config.get('test', [])

        if len(datasets) == 0:
            raise ValueError(f"No datasets configured for mode '{mode}'")

        return MultiDatasetLoader(datasets, lazy_load=data_config.get('lazy_load', True))

    else:
        # Single dataset mode (backward compatible with KITTI-only)
        root = data_config['kitti_root']

        if mode == 'train':
            sequences = [f"{i:02d}" for i in data_config['sequences_train']]
        elif mode == 'val':
            sequences = [f"{i:02d}" for i in data_config['sequences_val']]
        else:  # test
            sequences = [f"{i:02d}" for i in data_config['sequences_test']]

        if len(sequences) == 1:
            # Single sequence
            return KITTILoader(root, sequences[0], lazy_load=data_config.get('lazy_load', True))
        else:
            # Multiple KITTI sequences
            datasets = [{
                'type': 'kitti',
                'root': root,
                'sequences': sequences,
                'weight': 1.0
            }]
            return MultiDatasetLoader(datasets, lazy_load=data_config.get('lazy_load', True))
