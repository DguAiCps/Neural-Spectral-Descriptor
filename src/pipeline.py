"""
Main Pipeline - Algorithm 6

Full Neural Spectral Codec pipeline:

Offline (Training):
1. Load KITTI sequences [0-8]
2. Select keyframes (~1Hz rate)
3. Encode to spectral histograms
4. Build temporal graph
5. Train GNN with triplet loss
6. Validate on sequence [9]

Online (Inference):
1. Incremental keyframe selection
2. Spectral encoding
3. GNN embedding update
4. Two-stage loop closing (every 10 frames)
5. Export loop closures in g2o format

Target performance: 27ms latency @ 100K database
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import time

from data.kitti_loader import KITTILoader
from data.nclt_loader import NCLTLoader
from data.multi_dataset_loader import MultiDatasetLoader, create_multi_dataset_loader
from encoding.spectral_encoder import SpectralEncoder
from encoding.quantization import compress_descriptor, decompress_descriptor
from keyframe.selector import KeyframeSelector, Keyframe
from keyframe.graph_manager import TemporalGraphManager, build_graph_from_keyframes
from gnn.model import create_spectral_gnn
from gnn.trainer import GNNTrainer, create_trainer
from gnn.triplet_miner import create_triplet_miner
from retrieval.two_stage_retrieval import TwoStageRetrieval, create_two_stage_retrieval
from retrieval.geometric_verification import save_loop_closures_g2o


class NeuralSpectralCodecPipeline:
    """
    Complete Neural Spectral Codec pipeline

    Supports both offline training and online inference modes.
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline from config

        Args:
            config_path: Path to YAML configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = self.config['system']['device']

        # Initialize components
        bev_config = self.config['encoding'].get('bev', {})
        self.encoder = SpectralEncoder(
            n_elevation=self.config['encoding']['n_elevation'],
            n_azimuth=self.config['encoding']['n_azimuth'],
            n_bins=self.config['encoding']['n_bins'],
            alpha=self.config['encoding']['alpha'],
            learnable_alpha=self.config['encoding']['learnable_alpha'],
            target_elevation_bins=self.config['encoding']['target_elevation_bins'],
            bin_statistics=self.config['encoding'].get('bin_statistics', ['sum']),
            inter_bin_statistics=self.config['encoding'].get('inter_bin_statistics', []),
            projection_type=self.config['encoding'].get('projection_type', 'range_image'),
            max_range=self.config['encoding'].get('max_range', 80.0),
            min_range=self.config['encoding'].get('min_range', 1.0),
            z_min=bev_config.get('z_min', -3.0),
            zero_center=self.config['encoding'].get('zero_center', False),
            log_magnitude=self.config['encoding'].get('log_magnitude', False),
            binning_strategy=self.config['encoding'].get('binning_strategy', 'exponential'),
            normalize_channels=self.config['encoding'].get('normalize_channels', True),
        ).to(self.device)

        self.keyframe_selector = KeyframeSelector(
            distance_threshold=self.config['keyframe']['distance_threshold'],
            rotation_threshold=self.config['keyframe']['rotation_threshold'],
            overlap_threshold=self.config['keyframe']['overlap_threshold'],
            temporal_threshold=self.config['keyframe']['temporal_threshold']
        )

        self.graph_manager = TemporalGraphManager(
            temporal_neighbors=self.config['keyframe']['temporal_neighbors'],
            max_active_nodes=self.config['keyframe']['max_active_nodes'],
            feature_dim=self.encoder.output_dim,
            device=self.device
        )

        self.gnn = None  # Created during training or loaded from checkpoint

        self.retrieval_system = create_two_stage_retrieval(
            top_k=self.config['retrieval']['top_k'],
            spatial_filter_distance=self.config['retrieval']['spatial_filter_distance'],
            use_gpu=(self.device == 'cuda')
        )

    def train_offline(
        self,
        sequences_train: Optional[List[int]] = None,
        sequences_val: Optional[List[int]] = None,
        n_epochs: int = 50
    ):
        """
        Offline training on KITTI sequences

        Args:
            sequences_train: Training sequence IDs (default: [0-8])
            sequences_val: Validation sequence IDs (default: [9])
            n_epochs: Number of training epochs
        """
        if sequences_train is None:
            sequences_train = self.config['data']['sequences_train']
        if sequences_val is None:
            sequences_val = self.config['data']['sequences_val']

        print("=" * 80)
        print("OFFLINE TRAINING MODE")
        print("=" * 80)

        # Phase 1: Load and preprocess training data
        print("\n[Phase 1/4] Loading and preprocessing training data...")
        train_keyframes, train_poses, train_descriptors = self._load_and_process_sequences(
            sequences_train,
            mode='train'
        )

        print(f"Training keyframes: {len(train_keyframes)}")

        # Phase 2: Build temporal graph
        print("\n[Phase 2/4] Building temporal graph...")
        train_graph = build_graph_from_keyframes(
            train_keyframes,
            temporal_neighbors=self.config['keyframe']['temporal_neighbors'],
            device=self.device
        )

        print(f"Graph: {train_graph.num_nodes} nodes, {train_graph.edge_index.shape[1]} edges")

        # Phase 3: Load validation data
        print("\n[Phase 3/4] Loading validation data...")
        val_keyframes, val_poses, val_descriptors = self._load_and_process_sequences(
            sequences_val,
            mode='val'
        )

        val_graph = build_graph_from_keyframes(
            val_keyframes,
            temporal_neighbors=self.config['keyframe']['temporal_neighbors'],
            device=self.device
        )

        print(f"Validation keyframes: {len(val_keyframes)}")

        # Phase 4: Train GNN
        print("\n[Phase 4/4] Training GNN...")

        # Create GNN model
        self.gnn = create_spectral_gnn(
            input_dim=self.config['gnn']['input_dim'],
            hidden_dim=self.config['gnn']['hidden_dim'],
            context_dim=self.config['gnn']['context_dim'],
            n_layers=self.config['gnn']['n_layers'],
            n_heads=self.config['gnn'].get('n_heads', 4),
            dropout=self.config['gnn']['dropout'],
            use_local_updates=self.config['gnn']['use_local_updates'],
            local_update_hops=self.config['gnn']['local_update_hops'],
            sensor_gate_config=self.config['gnn'].get('sensor_gate'),
            diffattn_value_source=self.config['gnn'].get('diffattn_value_source', 'diff'),
        )

        # Create trainer
        trainer = create_trainer(
            model=self.gnn,
            device=self.device,
            learning_rate=self.config.get('training', {}).get('learning_rate', 5e-4),
            checkpoint_dir=self.config['system']['checkpoint_dir'],
            use_multi_gpu=self.config['gnn']['use_multi_gpu'],
            patience=self.config['gnn']['patience']
        )

        # Create triplet miner
        triplet_miner = create_triplet_miner(
            positive_distance_max=self.config.get('triplet', {}).get('positive_distance_max', 5.0),
            negative_distance_min=self.config.get('triplet', {}).get('negative_distance_min', 10.0)
        )

        # Train
        trainer.train(
            train_graph=train_graph,
            train_poses=train_poses,
            train_descriptors=train_descriptors,
            val_graph=val_graph,
            val_poses=val_poses,
            n_epochs=n_epochs,
            triplet_miner=triplet_miner
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

    def run_online(
        self,
        kitti_loader: KITTILoader,
        gnn_checkpoint_path: Optional[str] = None,
        loop_closing_interval: int = 10,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Online inference mode

        Args:
            kitti_loader: KITTI data loader
            gnn_checkpoint_path: Path to trained GNN checkpoint
            loop_closing_interval: Run loop closing every N frames
            output_path: Path to save loop closures (optional)

        Returns:
            List of loop closures
        """
        print("=" * 80)
        print("ONLINE INFERENCE MODE")
        print("=" * 80)

        # Load GNN if checkpoint provided
        if gnn_checkpoint_path is not None:
            self._load_gnn_checkpoint(gnn_checkpoint_path)

        loop_closures = []

        # Process scans incrementally
        for scan_id in range(len(kitti_loader)):
            data = kitti_loader[scan_id]

            # Keyframe selection
            selected, keyframe, _ = self.keyframe_selector.process_scan(
                scan_id=scan_id,
                points=data['points'],
                pose=data['pose'],
                timestamp=data['timestamp']
            )

            if not selected:
                continue

            # Encode to spectral histogram
            descriptor = self.encoder.encode_points(data['points']).detach().cpu().numpy()
            keyframe.descriptor = descriptor

            # Add to graph (optional: only if using GNN)
            if self.gnn is not None:
                self.graph_manager.add_keyframe(keyframe)

                # Update GNN embeddings
                graph = self.graph_manager.get_graph()
                with torch.no_grad():
                    embeddings = self.gnn(graph)
                    self.graph_manager.update_embeddings(embeddings)

            # Add to retrieval database
            self.retrieval_system.add_keyframe(keyframe)

            # Loop closing (every N frames)
            if scan_id % loop_closing_interval == 0 and scan_id > 0:
                start_time = time.time()

                lcs = self.retrieval_system.get_loop_closures(
                    query_keyframe=keyframe,
                    query_points=data['points']
                )

                query_time = (time.time() - start_time) * 1000  # ms

                if len(lcs) > 0:
                    print(f"Scan {scan_id}: Found {len(lcs)} loop closures (query time: {query_time:.1f}ms)")
                    loop_closures.extend(lcs)

        print(f"\nTotal loop closures found: {len(loop_closures)}")

        # Save to file
        if output_path is not None:
            save_loop_closures_g2o(loop_closures, output_path)

        return loop_closures

    def _load_and_process_sequences(
        self,
        sequence_ids: List[int],
        mode: str = 'train'
    ) -> tuple:
        """
        Load and process sequences from datasets

        Args:
            sequence_ids: List of sequence IDs to load (for backward compatibility)
            mode: 'train', 'val', or 'test'

        Returns:
            keyframes: List of keyframes
            poses: (n_keyframes, 4, 4) poses
            descriptors: (n_keyframes, n_bins) descriptors
        """
        all_keyframes = []

        # Load dataset (supports both single and multi-dataset)
        # For backward compatibility, check if sequence_ids is provided
        if 'datasets' in self.config['data']:
            # Multi-dataset mode
            loader = create_multi_dataset_loader(self.config, mode=mode)
            sequence_name = f"multi-dataset ({mode})"
        else:
            # Single dataset mode (backward compatibility)
            if len(sequence_ids) == 1:
                loader = KITTILoader(
                    data_root=self.config['data']['kitti_root'],
                    sequence=f"{sequence_ids[0]:02d}",
                    lazy_load=True
                )
                sequence_name = f"sequence {sequence_ids[0]:02d}"
            else:
                # Multiple KITTI sequences
                datasets_config = [{
                    'type': 'kitti',
                    'root': self.config['data']['kitti_root'],
                    'sequences': [f"{s:02d}" for s in sequence_ids],
                    'weight': 1.0
                }]
                loader = MultiDatasetLoader(datasets_config, lazy_load=True)
                sequence_name = "multi-sequence"

        print(f"Processing {sequence_name}...")

        # Reset selector
        self.keyframe_selector.reset()

        # Process all scans
        total_scans = len(loader)
        for scan_id in range(total_scans):
            if scan_id % 100 == 0:
                print(f"  Processing scan {scan_id}/{total_scans}...")

            data = loader[scan_id]

            selected, keyframe, _ = self.keyframe_selector.process_scan(
                scan_id=scan_id,
                points=data['points'],
                pose=data['pose'],
                timestamp=data['timestamp']
            )

            if selected:
                # Encode descriptor
                descriptor = self.encoder.encode_points(data['points']).detach().cpu().numpy()
                keyframe.descriptor = descriptor

                all_keyframes.append(keyframe)

        stats = self.keyframe_selector.get_statistics()
        print(f"  Selected {stats['num_keyframes']} keyframes from {stats['num_scans']} scans "
              f"({stats['compression_ratio']:.1f}x compression)")

        # Extract poses and descriptors
        poses = np.array([kf.pose for kf in all_keyframes])
        descriptors = np.array([kf.descriptor for kf in all_keyframes])

        return all_keyframes, poses, descriptors

    def _load_gnn_checkpoint(self, checkpoint_path: str):
        """Load GNN from checkpoint"""
        print(f"Loading GNN checkpoint: {checkpoint_path}")

        # Create GNN if not exists
        if self.gnn is None:
            self.gnn = create_spectral_gnn(
                input_dim=self.config['gnn']['input_dim'],
                hidden_dim=self.config['gnn']['hidden_dim'],
                context_dim=self.config['gnn']['context_dim'],
                n_layers=self.config['gnn']['n_layers'],
                n_heads=self.config['gnn'].get('n_heads', 4),
                sensor_gate_config=self.config['gnn'].get('sensor_gate'),
                diffattn_value_source=self.config['gnn'].get('diffattn_value_source', 'diff'),
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.gnn.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.gnn.load_state_dict(checkpoint)

        self.gnn.to(self.device)
        self.gnn.eval()

        print("GNN loaded successfully")


def run_pipeline(config_path: str, mode: str = 'train'):
    """
    Run pipeline from command line

    Args:
        config_path: Path to config file
        mode: 'train' or 'inference'
    """
    pipeline = NeuralSpectralCodecPipeline(config_path)

    if mode == 'train':
        pipeline.train_offline()
    elif mode == 'inference':
        # Load test sequence
        kitti_loader = KITTILoader(
            data_root=pipeline.config['data']['kitti_root'],
            sequence="00",
            lazy_load=True
        )

        loop_closures = pipeline.run_online(
            kitti_loader=kitti_loader,
            gnn_checkpoint_path='checkpoints/best_model.pth',
            output_path='outputs/loop_closures.g2o'
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Spectral Codec Pipeline")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'],
                        default='train', help='Pipeline mode')

    args = parser.parse_args()

    run_pipeline(args.config, args.mode)
