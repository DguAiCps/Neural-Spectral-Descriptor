#!/usr/bin/env python3
"""
Multi-Dataset Training Script for Neural Spectral Codec
Trains on KITTI + NCLT datasets with detailed profiling
"""

import sys
import os
import argparse
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import hashlib
import json
import numpy as np
import torch
import yaml
import time
import logging
import gc
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir='logs'):
    """Setup logging with timestamps"""
    os.makedirs(log_dir, exist_ok=True)

    # Create formatter with timestamp
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


# ============================================================================
# Profiling Utilities
# ============================================================================

class Profiler:
    """Performance profiler for tracking execution times"""

    def __init__(self):
        self.timings = {}
        self.counts = {}
        self.start_times = {}

    def start(self, name):
        """Start timing a section"""
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        """Stop timing and record"""
        if name not in self.start_times:
            return 0
        elapsed = time.perf_counter() - self.start_times[name]
        if name not in self.timings:
            self.timings[name] = 0
            self.counts[name] = 0
        self.timings[name] += elapsed
        self.counts[name] += 1
        return elapsed

    @contextmanager
    def profile(self, name):
        """Context manager for profiling"""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def get_stats(self, name):
        """Get statistics for a section"""
        if name not in self.timings:
            return None
        total = self.timings[name]
        count = self.counts[name]
        avg = total / count if count > 0 else 0
        return {'total': total, 'count': count, 'avg': avg}

    def summary(self):
        """Generate profiling summary"""
        lines = ["\n" + "=" * 80]
        lines.append("PROFILING SUMMARY")
        lines.append("=" * 80)

        # Sort by total time descending
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        total_time = sum(self.timings.values())

        lines.append(f"{'Section':<40} {'Total':>12} {'Count':>8} {'Avg':>12} {'%':>8}")
        lines.append("-" * 80)

        for name, total in sorted_items:
            count = self.counts[name]
            avg = total / count if count > 0 else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            lines.append(f"{name:<40} {total:>10.2f}s {count:>8} {avg:>10.4f}s {pct:>7.1f}%")

        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<40} {total_time:>10.2f}s")
        lines.append("=" * 80)

        return "\n".join(lines)


# Global profiler
profiler = Profiler()


# ============================================================================
# Data Processing
# ============================================================================

def load_config(config_path):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_encoder_preset(config, preset):
    """Apply compact closed-form encoder presets shared with fast KITTI eval."""
    enc = config['encoding']
    gnn = config['gnn']
    if preset == 'full':
        return config

    enc.setdefault('spectral_policy', {})['enabled'] = False
    enc['binning_strategy'] = 'octave'
    enc['zero_center'] = False
    enc['cross_spectrum'] = {'enabled': False, 'n_freqs': 0}

    if preset == 'no_interdiff':
        enc['target_elevation_bins'] = 16
        enc['bin_statistics'] = ['mean', 'std']
        enc['inter_bin_statistics'] = []
        gnn['input_dim'] = 16 * 9 * 2
    elif preset == 'cross4_no_interdiff':
        enc['target_elevation_bins'] = 16
        enc['bin_statistics'] = ['mean', 'std']
        enc['inter_bin_statistics'] = []
        enc['cross_spectrum'] = {'enabled': True, 'n_freqs': 4}
        gnn['input_dim'] = 16 * 9 * 2 + 15 * 4 * 2
    elif preset == 'cross8_no_interdiff':
        enc['target_elevation_bins'] = 16
        enc['bin_statistics'] = ['mean', 'std']
        enc['inter_bin_statistics'] = []
        enc['cross_spectrum'] = {'enabled': True, 'n_freqs': 8}
        gnn['input_dim'] = 16 * 9 * 2 + 15 * 8 * 2
    elif preset == 'mean_diff':
        enc['target_elevation_bins'] = 16
        enc['bin_statistics'] = ['mean']
        enc['inter_bin_statistics'] = ['diff']
        gnn['input_dim'] = 16 * (9 + 8)
    elif preset == 'rows12_full':
        enc['target_elevation_bins'] = 12
        enc['bin_statistics'] = ['mean', 'std']
        enc['inter_bin_statistics'] = ['diff']
        gnn['input_dim'] = 12 * (9 * 2 + 8 * 2)
    else:
        raise ValueError(f"Unknown encoder preset: {preset}")

    return config


def get_elevation_range(config, dataset_type):
    """Get sensor-specific elevation range, falling back to default"""
    sensor_ranges = config['encoding'].get('sensor_elevation_ranges', {})
    if dataset_type in sensor_ranges:
        return tuple(sensor_ranges[dataset_type])
    return tuple(config['encoding']['elevation_range'])


SENSOR_ID_BY_DATASET = {
    'kitti': 0,
    'nclt': 1,
    'helipr': 2,
    'mulran': 3,
}

BEAM_COUNT_BY_DATASET = {
    'kitti': 64.0,
    'mulran': 64.0,
    'nclt': 32.0,
    'helipr': 16.0,
}


def get_sensor_id(dataset_type):
    """Stable sensor id used by the sensor-aware GAT gate."""
    return SENSOR_ID_BY_DATASET.get(dataset_type, len(SENSOR_ID_BY_DATASET))


def get_beam_count(dataset_type):
    """Approximate vertical beam count used as continuous fallback metadata."""
    return BEAM_COUNT_BY_DATASET.get(dataset_type, 64.0)


def get_sensor_similarity_k(graph_config, dataset_type, default_max_k, default_min_k):
    """Return per-dataset similarity k values, falling back to scalar config.

    YAML format:
      keyframe:
        graph:
          sensor_similarity:
            enabled: true
            kitti:  {max_k: 10, min_k: 0}
            nclt:   {max_k: 16, min_k: 4}
            helipr: {max_k: 24, min_k: 6}
            mulran: {max_k: 12, min_k: 2}
    """
    cfg = graph_config.get('sensor_similarity', {})
    if not cfg or not cfg.get('enabled', False):
        return int(default_max_k), int(default_min_k)
    entry = cfg.get(dataset_type, {})
    return int(entry.get('max_k', default_max_k)), int(entry.get('min_k', default_min_k))


def attach_sensor_metadata(graph, sensor_ids, beam_counts, device):
    """Attach per-node sensor metadata consumed by sensor-aware GAT gate."""
    graph.sensor_id = torch.as_tensor(sensor_ids, dtype=torch.long, device=device).reshape(-1)
    graph.beam_count = torch.as_tensor(beam_counts, dtype=torch.float32, device=device).reshape(-1, 1)
    return graph


def compute_cache_key(config, dataset_type=None):
    """
    Compute SHA256 hash from encoding + keyframe config for cache invalidation.

    For range_image mode the key is per-dataset: it includes only the elevation
    range of the given `dataset_type` (not the whole sensor_elevation_ranges
    dict), so adding a new sensor does not invalidate unrelated caches.

    For bev mode the key is dataset-agnostic (sensor-independent projection).
    """
    enc = config['encoding']
    kf = config['keyframe']
    projection_type = enc.get('projection_type', 'range_image')
    key_params = {
        'projection_type': projection_type,
        'n_azimuth': enc['n_azimuth'],
        'n_bins': enc['n_bins'],
        'binning_strategy': enc.get('binning_strategy', 'exponential'),
        'bin_statistics': enc.get('bin_statistics', ['sum']),
        'inter_bin_statistics': enc.get('inter_bin_statistics', []),
        'max_range': enc.get('max_range', 80.0),
        'min_range': enc.get('min_range', 1.0),
        'zero_center': enc.get('zero_center', False),
        'log_magnitude': enc.get('log_magnitude', False),
        'normalize_channels': enc.get('normalize_channels', True),
        'spectral_policy_enabled': enc.get('spectral_policy', {}).get('enabled', False),
        'cross_spectrum': enc.get('cross_spectrum', {}),
        'phase_features': enc.get('phase_features', {}),
    }
    # Alpha affects bin edges for exponential strategy (octave ignores it)
    if key_params['binning_strategy'] != 'octave':
        key_params['alpha'] = enc.get('alpha', 2.0)
    key_params.update({
        'distance_threshold': kf['distance_threshold'],
        'rotation_threshold': kf['rotation_threshold'],
        'overlap_threshold': kf['overlap_threshold'],
        'temporal_threshold': kf['temporal_threshold'],
    })
    if projection_type == 'bev':
        key_params['bev'] = enc.get('bev', {})
    else:
        # Range image: include only the elevation range actually used for this
        # dataset. Fall back to the global default if no dataset_type given.
        key_params['n_elevation'] = enc['n_elevation']
        key_params['target_elevation_bins'] = enc['target_elevation_bins']
        if dataset_type is not None:
            key_params['dataset_type'] = dataset_type
            key_params['elevation_range'] = list(get_elevation_range(config, dataset_type))
        else:
            key_params['elevation_range'] = enc['elevation_range']
    return hashlib.sha256(json.dumps(key_params, sort_keys=True).encode()).hexdigest()[:8]


def get_cache_path(cache_dir, config, dataset_type, dataset_name):
    """Get cache file path for a dataset using a per-dataset cache key."""
    cache_key = compute_cache_key(config, dataset_type=dataset_type)
    return Path(cache_dir) / f"cache_{cache_key}_{dataset_name}.npz"


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # KB to MB
    except:
        return 0


def save_keyframes_cache(path, keyframes):
    """Save keyframes to disk as .npz"""
    descriptors = np.array([kf.descriptor for kf in keyframes])
    poses = np.array([kf.pose for kf in keyframes])
    timestamps = np.array([kf.timestamp for kf in keyframes])
    scan_ids = np.array([kf.scan_id for kf in keyframes])
    keyframe_ids = np.array([kf.keyframe_id for kf in keyframes])
    save_dict = dict(descriptors=descriptors, poses=poses, timestamps=timestamps,
                     scan_ids=scan_ids, keyframe_ids=keyframe_ids)
    # Save spectral entropy if available
    if keyframes[0].spectral_entropy is not None:
        entropies = np.array([kf.spectral_entropy for kf in keyframes])
        save_dict['spectral_entropies'] = entropies
    # Save FFT magnitudes for spectral policy (if computed)
    if keyframes[0].fft_magnitudes is not None:
        fft_mags = np.array([kf.fft_magnitudes for kf in keyframes])
        save_dict['fft_magnitudes'] = fft_mags
    if keyframes[0].phase_features is not None:
        phase_features = np.array([kf.phase_features for kf in keyframes])
        save_dict['phase_features'] = phase_features
    np.savez(path, **save_dict)
    logging.info(f"  Saved cache: {path} ({len(keyframes)} keyframes)")


def load_keyframes_cache(path):
    """Load keyframes from .npz cache"""
    from keyframe.selector import Keyframe
    with np.load(path) as data:
        # Load arrays once (not per-iteration)
        descriptors = data['descriptors']
        poses = data['poses']
        timestamps = data['timestamps']
        scan_ids = data['scan_ids']
        keyframe_ids = data['keyframe_ids']
        entropies = data['spectral_entropies'] if 'spectral_entropies' in data else None
        fft_mags = data['fft_magnitudes'] if 'fft_magnitudes' in data else None
        phase_features = data['phase_features'] if 'phase_features' in data else None

        keyframes = []
        for i in range(len(scan_ids)):
            kf = Keyframe(
                keyframe_id=int(keyframe_ids[i]),
                scan_id=int(scan_ids[i]),
                points=np.empty((0, 3)),
                pose=poses[i],
                timestamp=float(timestamps[i]),
                descriptor=descriptors[i],
                spectral_entropy=float(entropies[i]) if entropies is not None else None,
                fft_magnitudes=fft_mags[i] if fft_mags is not None else None,
                phase_features=phase_features[i] if phase_features is not None else None,
            )
            keyframes.append(kf)
        return keyframes


def process_dataset(loader, encoder, keyframe_selector, device, max_scans=None, dataset_name="",
                    compute_fft_magnitudes=False, compute_phase_features=False,
                    phase_feature_config=None):
    """Process dataset and extract keyframes with profiling"""
    keyframes = []
    num_scans = len(loader) if max_scans is None else min(len(loader), max_scans)

    logging.info(f"Processing {num_scans} scans from {dataset_name}...")

    load_times = []
    encode_times = []
    select_times = []

    for scan_id in range(num_scans):
        if scan_id % 500 == 0:
            avg_load = np.mean(load_times[-100:]) * 1000 if load_times else 0
            avg_encode = np.mean(encode_times[-100:]) * 1000 if encode_times else 0
            avg_select = np.mean(select_times[-100:]) * 1000 if select_times else 0
            logging.info(
                f"  Scan {scan_id}/{num_scans} | "
                f"Keyframes: {len(keyframes)} | "
                f"Avg: load={avg_load:.1f}ms, encode={avg_encode:.1f}ms, select={avg_select:.1f}ms"
            )

        try:
            # Load data
            t0 = time.perf_counter()
            data = loader[scan_id]
            load_times.append(time.perf_counter() - t0)

            # Keyframe selection
            t0 = time.perf_counter()
            selected, keyframe, _ = keyframe_selector.process_scan(
                scan_id=scan_id,
                points=data['points'],
                pose=data['pose'],
                timestamp=data['timestamp']
            )
            select_times.append(time.perf_counter() - t0)

            if selected:
                # Encode (with spectral entropy for adaptive prior)
                t0 = time.perf_counter()
                descriptor, entropy = encoder.encode_points(
                    data['points'], return_entropy=True
                )
                descriptor = descriptor.detach().cpu().numpy()
                encode_times.append(time.perf_counter() - t0)

                keyframe.descriptor = descriptor
                keyframe.spectral_entropy = entropy

                # Compute FFT magnitudes for spectral policy (before discarding points)
                if compute_fft_magnitudes:
                    keyframe.fft_magnitudes = encoder.compute_fft_magnitudes(data['points'])

                # Compute low-frequency phase inputs for the learned phase token.
                if compute_phase_features:
                    from encoding.phase_features import compute_phase_features_from_points
                    keyframe.phase_features = compute_phase_features_from_points(
                        data['points'],
                        encoder,
                        phase_feature_config or {},
                    )

                # Memory optimization: Discard point cloud after encoding
                # Points are only needed for encoding, not for GNN training
                keyframe.points = np.empty((0, 3), dtype=np.float32)

                keyframes.append(keyframe)

        except Exception as e:
            logging.warning(f"Failed to process scan {scan_id}: {e}")
            continue

    # Log statistics
    if load_times:
        profiler.timings[f'load_{dataset_name}'] = sum(load_times)
        profiler.counts[f'load_{dataset_name}'] = len(load_times)
    if encode_times:
        profiler.timings[f'encode_{dataset_name}'] = sum(encode_times)
        profiler.counts[f'encode_{dataset_name}'] = len(encode_times)
    if select_times:
        profiler.timings[f'select_{dataset_name}'] = sum(select_times)
        profiler.counts[f'select_{dataset_name}'] = len(select_times)

    logging.info(
        f"  Completed: {len(keyframes)}/{num_scans} keyframes "
        f"({len(keyframes)/num_scans*100:.1f}% selection rate)"
    )

    return keyframes


# ============================================================================
# Main Training
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-Dataset Training for Neural Spectral Codec')
    parser.add_argument('--config', type=str, default='configs/training_multi_dataset.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='src/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--validate-only', action='store_true',
                        help='Load best_model.pth and run validation only (no training)')
    parser.add_argument('--dump-per-query-dir', type=str, default=None,
                        help='Directory for per-query JSON dumps (validate-only mode)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for numpy/torch/cuda (default: nondeterministic)')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Optional model checkpoint to preload for fine-tuning')
    parser.add_argument(
        '--encoder-preset',
        type=str,
        default='full',
        choices=[
            'full',
            'no_interdiff',
            'cross4_no_interdiff',
            'cross8_no_interdiff',
            'mean_diff',
            'rows12_full',
        ],
        help='Closed-form encoder compression preset for ablation training',
    )
    parser.add_argument('--use-gated-context', action='store_true',
                        help='Enable learned alpha gate on GAT context output')
    parser.add_argument('--gate-initial-alpha', type=float, default=None,
                        help='Initial sigmoid alpha for learned context gate')
    args = parser.parse_args()

    # Setup
    start_time = time.perf_counter()
    logger, log_file = setup_logging('logs')

    # Reproducibility: seed all RNGs.
    if args.seed is not None:
        import random as _py_random
        _py_random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logging.info(f"Random seed set to {args.seed} (numpy/torch/cuda)")

    # Route uncaught exceptions to log file
    def _excepthook(exc_type, exc_value, exc_tb):
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    logging.info("=" * 80)
    logging.info("NEURAL SPECTRAL CODEC - MULTI-DATASET TRAINING")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")

    # Paths
    config_path = args.config
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load config
    logging.info(f"Loading configuration from: {config_path}")
    config = apply_encoder_preset(load_config(config_path), args.encoder_preset)
    if args.encoder_preset != 'full':
        logging.info(f"Applied encoder preset: {args.encoder_preset}")
        if args.resume_checkpoint is None:
            config.get('training', {})['resume_from_checkpoint'] = None
            logging.info("Disabled config resume checkpoint for non-full encoder preset")
    if args.use_gated_context:
        config['gnn']['use_residual_gate'] = True
        if args.gate_initial_alpha is not None:
            config['gnn']['gate_initial_alpha'] = args.gate_initial_alpha
        logging.info(
            "Enabled learned context gate "
            f"(initial_alpha={config['gnn'].get('gate_initial_alpha', 0.5)})"
        )

    # Log config summary
    logging.info(f"  n_elevation: {config['encoding']['n_elevation']}")
    logging.info(f"  n_azimuth: {config['encoding']['n_azimuth']}")
    logging.info(f"  n_bins: {config['encoding']['n_bins']}")
    logging.info(f"  distance_threshold: {config['keyframe']['distance_threshold']}m")
    logging.info(f"  n_epochs: {config['training']['n_epochs']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    if device == 'cuda':
        logging.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========================================================================
    # Stage 1: Create Encoder
    # ========================================================================
    logging.info("")
    logging.info("[1/6] Creating spectral encoder...")

    with profiler.profile('create_encoder'):
        from encoding.spectral_encoder import SpectralEncoder
        bin_statistics = config['encoding'].get('bin_statistics', ['sum'])
        inter_bin_statistics = config['encoding'].get('inter_bin_statistics', [])
        projection_type = config['encoding'].get('projection_type', 'range_image')
        bev_config = config['encoding'].get('bev', {})
        normalize_channels = config['encoding'].get('normalize_channels', True)
        cross_cfg = config['encoding'].get('cross_spectrum', {})
        encoder = SpectralEncoder(
            n_elevation=config['encoding']['n_elevation'],
            n_azimuth=config['encoding']['n_azimuth'],
            n_bins=config['encoding']['n_bins'],
            alpha=config['encoding']['alpha'],
            learnable_alpha=config['encoding']['learnable_alpha'],
            target_elevation_bins=config['encoding']['target_elevation_bins'],
            elevation_range=tuple(config['encoding']['elevation_range']),
            bin_statistics=bin_statistics,
            inter_bin_statistics=inter_bin_statistics,
            projection_type=projection_type,
            max_range=config['encoding'].get('max_range', 80.0),
            min_range=config['encoding'].get('min_range', 1.0),
            z_min=bev_config.get('z_min', -3.0),
            height_encoding=bev_config.get('height_encoding', 'max'),
            n_height_layers=bev_config.get('n_height_layers', 8),
            z_max=bev_config.get('z_max', 5.0),
            zero_center=config['encoding'].get('zero_center', False),
            log_magnitude=config['encoding'].get('log_magnitude', False),
            binning_strategy=config['encoding'].get('binning_strategy', 'exponential'),
            normalize_channels=normalize_channels,
            cross_spectrum_enabled=cross_cfg.get('enabled', False),
            cross_spectrum_n_freqs=cross_cfg.get('n_freqs', 0),
        ).to(device)

    logging.info(f"  Encoder created (projection={projection_type}, n_bins={config['encoding']['n_bins']}, "
                 f"bin_statistics={bin_statistics}, output_dim={encoder.output_dim})")

    # Check if spectral policy is enabled (need FFT magnitudes in cache)
    policy_config = config['encoding'].get('spectral_policy', {})
    need_fft_magnitudes = policy_config.get('enabled', False)
    phase_feature_config = config['encoding'].get('phase_features', {})
    phase_token_config = config['gnn'].get('phase_token', {})
    phase_edge_config = config['gnn'].get('phase_edge', {})
    phase_alignment_config = config['gnn'].get('phase_alignment_edge', {})
    phase_coherence_config = config['gnn'].get('phase_coherence', {})
    dual_stream_config = config['gnn'].get('dual_stream', {})
    need_phase_features = (
        phase_token_config.get('enabled', False)
        or phase_edge_config.get('enabled', False)
        or phase_alignment_config.get('enabled', False)
        or phase_coherence_config.get('enabled', False)
        or dual_stream_config.get('enabled', False)
    )
    if need_phase_features:
        from encoding.phase_features import (
            phase_feature_dim,
            prepare_raw_complex_phase_config,
        )
        raw_complex_consumers = {
            'phase_alignment_edge': phase_alignment_config,
            'phase_coherence': phase_coherence_config,
            'dual_stream': dual_stream_config,
        }
        if any(cfg.get('enabled', False) for cfg in raw_complex_consumers.values()):
            phase_feature_config = prepare_raw_complex_phase_config(
                phase_feature_config,
                raw_complex_consumers,
            )
            config['encoding']['phase_features'] = phase_feature_config
            logging.info(
                "  Raw complex phase features enabled: "
                f"source={phase_feature_config.get('source')}, "
                f"dim={phase_feature_dim(phase_feature_config)}"
            )
        computed_phase_dim = phase_feature_dim(phase_feature_config)
        if phase_token_config.get('enabled', False):
            phase_token_config['input_dim'] = phase_token_config.get(
                'input_dim',
                computed_phase_dim,
            )
            config['gnn']['phase_token'] = phase_token_config
            logging.info(
                "  Learned phase token enabled: "
                f"source={phase_feature_config.get('source', 'bev_complex')}, "
                f"input_dim={phase_token_config['input_dim']}, "
                f"token_dim={phase_token_config.get('token_dim', 64)}"
            )
        if phase_edge_config.get('enabled', False):
            phase_edge_config['input_dim'] = phase_edge_config.get(
                'input_dim',
                computed_phase_dim,
            )
            config['gnn']['phase_edge'] = phase_edge_config
            logging.info(
                "  Phase-aware GAT edge bias enabled: "
                f"source={phase_feature_config.get('source', 'bev_complex')}, "
                f"input_dim={phase_edge_config['input_dim']}, "
                f"key_dim={phase_edge_config.get('key_dim', 32)}, "
                f"max_logit={phase_edge_config.get('max_logit', 2.0)}"
            )
        if phase_alignment_config.get('enabled', False):
            phase_alignment_config['n_rows'] = int(
                phase_alignment_config.get(
                    'n_rows',
                    phase_feature_config.get('bev_rows', phase_feature_config.get('range_rows', 16)),
                )
            )
            phase_alignment_config['n_freqs'] = int(
                phase_alignment_config.get(
                    'n_freqs',
                    phase_feature_config.get('bev_freqs', phase_feature_config.get('range_freqs', 0)),
                )
            )
            phase_alignment_config['n_sectors'] = int(
                phase_alignment_config.get(
                    'n_sectors',
                    phase_feature_config.get('layout_sectors', config['encoding'].get('n_azimuth', 360)),
                )
            )
            config['gnn']['phase_alignment_edge'] = phase_alignment_config
            logging.info(
                "  Phase-alignment edge features enabled: "
                f"shape=({phase_alignment_config['n_rows']}, {phase_alignment_config['n_freqs']}), "
                f"n_sectors={phase_alignment_config['n_sectors']}, "
                f"include_score={phase_alignment_config.get('include_score', False)}, "
                f"value_scale={phase_alignment_config.get('value_scale', 0.0)}"
            )

    # Create keyframe selector
    from keyframe.selector import KeyframeSelector
    keyframe_selector = KeyframeSelector(
        distance_threshold=config['keyframe']['distance_threshold'],
        rotation_threshold=config['keyframe']['rotation_threshold'],
        overlap_threshold=config['keyframe']['overlap_threshold'],
        temporal_threshold=config['keyframe']['temporal_threshold']
    )

    # Descriptor cache setup — per-dataset keys are computed inside each loop
    cache_dir = config['data'].get('cache_dir', 'data/preprocessed')
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"  Cache dir: {cache_dir} (per-dataset cache keys)")

    # ========================================================================
    # Stage 2: Load Training Data
    # ========================================================================
    logging.info("")
    logging.info("[2/6] Loading training datasets...")

    train_datasets = config['data']['datasets']['train']
    all_train_keyframes = []
    train_sequence_ids = []
    train_sensor_ids = []
    train_beam_counts = []
    train_dataset_types = []
    current_seq_id = 0

    with profiler.profile('load_train_data'):
        for dataset_cfg in train_datasets:
            dataset_type = dataset_cfg['type']
            root = dataset_cfg['root']
            sequences = dataset_cfg['sequences']

            if projection_type != 'bev':
                elevation_range = get_elevation_range(config, dataset_type)
                encoder.set_elevation_range(elevation_range)
                logging.info(f"  Loading {dataset_type.upper()} from {root} (elevation: {elevation_range})")
            else:
                logging.info(f"  Loading {dataset_type.upper()} from {root} (BEV, sensor-agnostic)")

            if dataset_type == 'kitti':
                from data.kitti_loader import KITTILoader
                for seq in sequences:
                    ds_name = f"kitti_{seq}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        loader = KITTILoader(root, seq, lazy_load=True)
                        keyframes = process_dataset(
                            loader, encoder, keyframe_selector, device,
                            dataset_name=ds_name,
                            compute_fft_magnitudes=need_fft_magnitudes,
                            compute_phase_features=need_phase_features,
                            phase_feature_config=phase_feature_config,
                        )
                        save_keyframes_cache(cache_path, keyframes)
                        del loader  # Free loader memory
                    all_train_keyframes.extend(keyframes)
                    train_sequence_ids.extend([current_seq_id] * len(keyframes))
                    train_sensor_ids.extend([get_sensor_id(dataset_type)] * len(keyframes))
                    train_beam_counts.extend([get_beam_count(dataset_type)] * len(keyframes))
                    train_dataset_types.extend([dataset_type] * len(keyframes))
                    del keyframes  # Free temporary keyframes list
                    mem_mb = get_memory_usage_mb()
                    logging.info(f"    Sequence {seq}: seq_id={current_seq_id}, total_keyframes={len(all_train_keyframes)}, RAM={mem_mb:.0f}MB")
                    current_seq_id += 1

            elif dataset_type == 'nclt':
                from data.nclt_loader import NCLTLoader
                for date in sequences:
                    ds_name = f"nclt_{date}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        loader = NCLTLoader(root, date, lazy_load=True)
                        keyframes = process_dataset(
                            loader, encoder, keyframe_selector, device,
                            dataset_name=ds_name,
                            compute_fft_magnitudes=need_fft_magnitudes,
                            compute_phase_features=need_phase_features,
                            phase_feature_config=phase_feature_config,
                        )
                        save_keyframes_cache(cache_path, keyframes)
                        del loader  # Free loader memory
                    all_train_keyframes.extend(keyframes)
                    train_sequence_ids.extend([current_seq_id] * len(keyframes))
                    train_sensor_ids.extend([get_sensor_id(dataset_type)] * len(keyframes))
                    train_beam_counts.extend([get_beam_count(dataset_type)] * len(keyframes))
                    train_dataset_types.extend([dataset_type] * len(keyframes))
                    del keyframes  # Free temporary keyframes list
                    mem_mb = get_memory_usage_mb()
                    logging.info(f"    Date {date}: seq_id={current_seq_id}, total_keyframes={len(all_train_keyframes)}, RAM={mem_mb:.0f}MB")
                    current_seq_id += 1

            elif dataset_type == 'helipr':
                from data.helipr_loader import HeLiPRLoader
                for seq in sequences:
                    ds_name = f"helipr_{seq}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        seq_path = os.path.join(root, seq, seq)
                        try:
                            loader = HeLiPRLoader(seq_path, lazy_load=True)
                            keyframes = process_dataset(
                                loader, encoder, keyframe_selector, device,
                                dataset_name=ds_name,
                                compute_fft_magnitudes=need_fft_magnitudes,
                                compute_phase_features=need_phase_features,
                                phase_feature_config=phase_feature_config,
                            )
                            save_keyframes_cache(cache_path, keyframes)
                            del loader  # Free loader memory
                        except Exception as e:
                            logging.warning(f"    Failed to load HeLiPR {seq}: {e}")
                            current_seq_id += 1
                            continue
                    all_train_keyframes.extend(keyframes)
                    train_sequence_ids.extend([current_seq_id] * len(keyframes))
                    train_sensor_ids.extend([get_sensor_id(dataset_type)] * len(keyframes))
                    train_beam_counts.extend([get_beam_count(dataset_type)] * len(keyframes))
                    train_dataset_types.extend([dataset_type] * len(keyframes))
                    del keyframes  # Free temporary keyframes list
                    mem_mb = get_memory_usage_mb()
                    logging.info(f"    Sequence {seq}: seq_id={current_seq_id}, total_keyframes={len(all_train_keyframes)}, RAM={mem_mb:.0f}MB")
                    current_seq_id += 1

            elif dataset_type == 'mulran':
                from data.mulran_loader import MulRanLoader
                for seq in sequences:
                    ds_name = f"mulran_{seq}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        try:
                            loader = MulRanLoader(root, seq, lazy_load=True)
                            keyframes = process_dataset(
                                loader, encoder, keyframe_selector, device,
                                dataset_name=ds_name,
                                compute_fft_magnitudes=need_fft_magnitudes,
                                compute_phase_features=need_phase_features,
                                phase_feature_config=phase_feature_config,
                            )
                            save_keyframes_cache(cache_path, keyframes)
                            del loader  # Free loader memory
                        except Exception as e:
                            logging.warning(f"    Failed to load MulRan {seq}: {e}")
                            current_seq_id += 1
                            continue
                    all_train_keyframes.extend(keyframes)
                    train_sequence_ids.extend([current_seq_id] * len(keyframes))
                    train_sensor_ids.extend([get_sensor_id(dataset_type)] * len(keyframes))
                    train_beam_counts.extend([get_beam_count(dataset_type)] * len(keyframes))
                    train_dataset_types.extend([dataset_type] * len(keyframes))
                    del keyframes  # Free temporary keyframes list
                    mem_mb = get_memory_usage_mb()
                    logging.info(f"    Sequence {seq}: seq_id={current_seq_id}, total_keyframes={len(all_train_keyframes)}, RAM={mem_mb:.0f}MB")
                    current_seq_id += 1

    # Opt 8: Single gc.collect() after all training data loaded (not per-sequence)
    gc.collect()

    train_sequence_ids = np.array(train_sequence_ids)
    train_sensor_ids = np.array(train_sensor_ids, dtype=np.int64)
    train_beam_counts = np.array(train_beam_counts, dtype=np.float32)
    if not (
        len(all_train_keyframes)
        == len(train_sequence_ids)
        == len(train_sensor_ids)
        == len(train_beam_counts)
        == len(train_dataset_types)
    ):
        raise RuntimeError(
            "Training metadata length mismatch: "
            f"keyframes={len(all_train_keyframes)}, seq_ids={len(train_sequence_ids)}, "
            f"sensor_ids={len(train_sensor_ids)}, beam_counts={len(train_beam_counts)}, "
            f"dataset_types={len(train_dataset_types)}"
        )
    logging.info(f"Total training keyframes: {len(all_train_keyframes)} across {current_seq_id} sequences")

    # ========================================================================
    # Stage 3: Load Validation Data (per-dataset)
    # ========================================================================
    logging.info("")
    logging.info("[3/6] Loading validation datasets...")

    val_datasets_config = config['data']['datasets']['val']
    val_datasets_info = {}  # {dataset_name: {'keyframes': [...], 'poses': np.array}}

    with profiler.profile('load_val_data'):
        for dataset_cfg in val_datasets_config:
            dataset_type = dataset_cfg['type']
            root = dataset_cfg['root']
            sequences = dataset_cfg['sequences']

            if projection_type != 'bev':
                elevation_range = get_elevation_range(config, dataset_type)
                encoder.set_elevation_range(elevation_range)
                logging.info(f"  Loading {dataset_type.upper()} validation (elevation: {elevation_range})")
            else:
                logging.info(f"  Loading {dataset_type.upper()} validation (BEV, sensor-agnostic)")

            if dataset_type == 'kitti':
                from data.kitti_loader import KITTILoader
                for seq in sequences:
                    ds_name = f"kitti_val_{seq}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        loader = KITTILoader(root, seq, lazy_load=True)
                        keyframes = process_dataset(
                            loader, encoder, keyframe_selector, device,
                            dataset_name=ds_name,
                            compute_fft_magnitudes=need_fft_magnitudes,
                            compute_phase_features=need_phase_features,
                            phase_feature_config=phase_feature_config,
                        )
                        save_keyframes_cache(cache_path, keyframes)
                    dataset_name = f"KITTI_{seq}"
                    val_datasets_info[dataset_name] = {
                        'keyframes': keyframes,
                        'dataset_type': dataset_type,
                    }
                    logging.info(f"  Validation {dataset_name}: {len(keyframes)} keyframes")

            elif dataset_type == 'nclt':
                from data.nclt_loader import NCLTLoader
                for date in sequences:
                    ds_name = f"nclt_val_{date}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        loader = NCLTLoader(root, date, lazy_load=True)
                        keyframes = process_dataset(
                            loader, encoder, keyframe_selector, device,
                            dataset_name=ds_name,
                            compute_fft_magnitudes=need_fft_magnitudes,
                            compute_phase_features=need_phase_features,
                            phase_feature_config=phase_feature_config,
                        )
                        save_keyframes_cache(cache_path, keyframes)
                    dataset_name = f"NCLT_{date}"
                    val_datasets_info[dataset_name] = {
                        'keyframes': keyframes,
                        'dataset_type': dataset_type,
                    }
                    logging.info(f"  Validation {dataset_name}: {len(keyframes)} keyframes")

            elif dataset_type == 'helipr':
                from data.helipr_loader import HeLiPRLoader
                for seq in sequences:
                    ds_name = f"helipr_val_{seq}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        seq_path = os.path.join(root, seq, seq)
                        try:
                            loader = HeLiPRLoader(seq_path, lazy_load=True)
                            keyframes = process_dataset(
                                loader, encoder, keyframe_selector, device,
                                dataset_name=ds_name,
                                compute_fft_magnitudes=need_fft_magnitudes,
                                compute_phase_features=need_phase_features,
                                phase_feature_config=phase_feature_config,
                            )
                            save_keyframes_cache(cache_path, keyframes)
                        except Exception as e:
                            logging.warning(f"  Failed to load HeLiPR {seq}: {e}")
                            continue
                    dataset_name = f"HeLiPR_{seq}"
                    val_datasets_info[dataset_name] = {
                        'keyframes': keyframes,
                        'dataset_type': dataset_type,
                    }
                    logging.info(f"  Validation {dataset_name}: {len(keyframes)} keyframes")

            elif dataset_type == 'mulran':
                from data.mulran_loader import MulRanLoader
                for seq in sequences:
                    ds_name = f"mulran_val_{seq}"
                    cache_path = get_cache_path(cache_dir, config, dataset_type, ds_name)
                    if cache_path.exists():
                        keyframes = load_keyframes_cache(cache_path)
                        logging.info(f"    Loaded from cache: {len(keyframes)} keyframes ({ds_name})")
                    else:
                        keyframe_selector.reset()
                        try:
                            loader = MulRanLoader(root, seq, lazy_load=True)
                            keyframes = process_dataset(
                                loader, encoder, keyframe_selector, device,
                                dataset_name=ds_name,
                                compute_fft_magnitudes=need_fft_magnitudes,
                                compute_phase_features=need_phase_features,
                                phase_feature_config=phase_feature_config,
                            )
                            save_keyframes_cache(cache_path, keyframes)
                        except Exception as e:
                            logging.warning(f"  Failed to load MulRan {seq}: {e}")
                            continue
                    dataset_name = f"MulRan_{seq}"
                    val_datasets_info[dataset_name] = {
                        'keyframes': keyframes,
                        'dataset_type': dataset_type,
                    }
                    logging.info(f"  Validation {dataset_name}: {len(keyframes)} keyframes")

    total_val_keyframes = sum(len(v['keyframes']) for v in val_datasets_info.values())
    logging.info(f"Total validation keyframes: {total_val_keyframes} ({len(val_datasets_info)} datasets)")

    # ========================================================================
    # Stage 4: Build Graphs (with edge distances)
    # ========================================================================
    logging.info("")
    logging.info("[4/6] Building temporal graphs with edge distances...")

    # Extract poses first (needed for edge distance computation)
    train_poses = np.array([kf.pose for kf in all_train_keyframes])
    train_descriptors = np.array([kf.descriptor for kf in all_train_keyframes])

    from keyframe.graph_manager import build_graph_from_keyframes_batch

    graph_config = config['keyframe'].get('graph', {})
    similarity_threshold = graph_config.get('similarity_threshold', 0.993)
    similarity_max_k = graph_config.get('similarity_max_k', 10)
    similarity_min_k = graph_config.get('similarity_min_k', 0)
    if graph_config.get('sensor_similarity', {}).get('enabled', False):
        train_similarity_max_k = np.array([
            get_sensor_similarity_k(graph_config, dt, similarity_max_k, similarity_min_k)[0]
            for dt in train_dataset_types
        ], dtype=np.int64)
        train_similarity_min_k = np.array([
            get_sensor_similarity_k(graph_config, dt, similarity_max_k, similarity_min_k)[1]
            for dt in train_dataset_types
        ], dtype=np.int64)
        logging.info(
            "  Sensor-aware similarity k enabled: "
            f"max_k range=[{train_similarity_max_k.min()}, {train_similarity_max_k.max()}], "
            f"min_k range=[{train_similarity_min_k.min()}, {train_similarity_min_k.max()}]"
        )
    else:
        train_similarity_max_k = similarity_max_k
        train_similarity_min_k = similarity_min_k
    similarity_exclude_temporal = graph_config.get('similarity_exclude_temporal', True)
    temporal_edge_mode = graph_config.get('temporal_edge_mode', 'bidirectional')
    temporal_direction_mode = graph_config.get('temporal_direction_mode', 'none')

    # Adaptive prior signal and multi-scale consistency config
    prior_signal = graph_config.get('prior_signal', 'density')
    multiscale_min_consistency = graph_config.get('multiscale_min_consistency', 0.0)

    # Compute channel splits for multi-scale consistency from encoding config
    channel_splits = None
    if multiscale_min_consistency > 0:
        enc_cfg = config.get('encoding', {})
        bin_stats = enc_cfg.get('bin_statistics', ['sum'])
        inter_stats = enc_cfg.get('inter_bin_statistics', [])
        proj_type = enc_cfg.get('projection_type', 'range_image')
        n_bins_cfg = enc_cfg.get('n_bins', 16)
        if proj_type == 'bev':
            n_rows = int(enc_cfg.get('max_range', 80.0) - enc_cfg.get('min_range', 1.0))
        else:
            n_rows = enc_cfg.get('target_elevation_bins', 16)
        base_dim = n_rows * n_bins_cfg
        inter_base_dim = n_rows * (n_bins_cfg - 1)
        channel_splits = []
        offset = 0
        for _ in bin_stats:
            channel_splits.append((offset, offset + base_dim))
            offset += base_dim
        for _ in inter_stats:
            for _ in bin_stats:
                channel_splits.append((offset, offset + inter_base_dim))
                offset += inter_base_dim
        logging.info(f"  Multi-scale consistency: {len(channel_splits)} channels, "
                     f"min_consistency={multiscale_min_consistency}")

    # Similarity metric: 'cosine' (original) or 'l2' (standardized Euclidean)
    similarity_metric = graph_config.get('similarity_metric', 'cosine')
    standardization_stats = None

    if similarity_metric == 'l2':
        from utils.standardization_stats import StandardizationStats

        std_cache_path = os.path.join(checkpoint_dir, 'standardization_stats.npz')
        if os.path.exists(std_cache_path):
            logging.info(f"  Loading cached standardization stats from {std_cache_path}")
            standardization_stats = StandardizationStats().load(std_cache_path)
        else:
            logging.info("  Fitting standardization stats from training descriptors...")
            standardization_stats = StandardizationStats().fit(train_descriptors)
            standardization_stats.save(std_cache_path)

    # Ground-truth pose-based similarity edges (train only). When enabled,
    # select same-place pairs via pose distance during initial graph build —
    # perfectly clean supervision edges for the GNN. Val/test graphs stay
    # temporal-only (realistic inference). Mutually exclusive with two_pass.
    use_pose_gt_edges = bool(graph_config.get('use_pose_gt_edges', False))
    pose_gt_edge_max_k = int(graph_config.get('pose_gt_edge_max_k', 10))
    if use_pose_gt_edges:
        logging.info(
            f"  Pose-GT similarity edges enabled (train only): "
            f"pos_dist={config.get('triplet', {}).get('positive_distance_max', 5.0)}m, "
            f"min_temporal_gap={config.get('triplet', {}).get('min_temporal_distance', 30)}, "
            f"max_k={pose_gt_edge_max_k}"
        )

    # Two-pass similarity edge refinement: when enabled, skip initial Bayesian
    # fitting on raw descriptors (structurally impossible, see plan) and build
    # the initial graph with temporal edges only. The GNN will periodically
    # refit on its own ctx embeddings during training.
    two_pass_cfg = graph_config.get('two_pass', {})
    two_pass_enabled = bool(two_pass_cfg.get('enabled', False)) and not use_pose_gt_edges
    if use_pose_gt_edges and bool(two_pass_cfg.get('enabled', False)):
        logging.info("  (two_pass disabled: superseded by use_pose_gt_edges)")
    if two_pass_enabled:
        logging.info(
            f"  Two-pass refinement enabled: warmup={two_pass_cfg.get('warmup_epochs', 10)} ep, "
            f"refine_every={two_pass_cfg.get('refine_every', 5)} ep, "
            f"refine_space='{two_pass_cfg.get('refine_space', 'ctx')}' "
            f"-> skipping initial similarity_dist fit"
        )

    # Bayesian edge selection: fit similarity distribution from training data
    edge_method = graph_config.get('edge_method', 'threshold')
    similarity_dist = None
    # Bayesian config is always constructed (also used by two-pass refinement)
    bayesian_config = {
        'confidence_level': graph_config.get('confidence_level', 0.95),
        'base_prior': graph_config.get('base_prior', 0.01),
        'density_k': graph_config.get('density_k', 50),
        'density_beta': graph_config.get('density_beta', 10.0),
    } if edge_method == 'bayesian' else {}

    if edge_method == 'bayesian' and not two_pass_enabled and not use_pose_gt_edges:
        from utils.similarity_stats import SimilarityDistribution

        dist_cache_path = os.path.join(checkpoint_dir, 'similarity_dist.npz')
        if os.path.exists(dist_cache_path):
            logging.info(f"  Loading cached similarity distribution from {dist_cache_path}")
            similarity_dist = SimilarityDistribution(metric=similarity_metric).load(dist_cache_path)
        else:
            logging.info(f"  Fitting Bayesian similarity distribution (metric={similarity_metric})...")
            triplet_cfg = config.get('triplet', {})
            # For L2 metric, fit on z-scored descriptors
            if similarity_metric == 'l2':
                fit_descriptors = standardization_stats.transform(train_descriptors)
            else:
                fit_descriptors = train_descriptors
            similarity_dist = SimilarityDistribution(metric=similarity_metric).fit(
                fit_descriptors, train_poses,
                sequence_ids=train_sequence_ids,
                pos_dist=triplet_cfg.get('positive_distance_max', 5.0),
                neg_dist=triplet_cfg.get('negative_distance_min', 10.0),
                min_temporal_gap=triplet_cfg.get('min_temporal_distance', 30),
                n_samples=graph_config.get('n_samples', 100000),
            )
            if similarity_dist.fitted:
                similarity_dist.save(dist_cache_path)

        if not similarity_dist.fitted:
            logging.warning("  Bayesian distribution fitting failed. Falling back to threshold mode.")
            similarity_dist = None

        logging.info(f"  Edge method: {edge_method}, metric: {similarity_metric}, "
                     f"prior_signal: {prior_signal}, "
                     f"confidence: {bayesian_config['confidence_level']}")
    elif edge_method == 'bayesian' and two_pass_enabled:
        logging.info(f"  Edge method: {edge_method} (two-pass), metric: {similarity_metric}, "
                     f"prior_signal: {prior_signal}, "
                     f"confidence: {bayesian_config['confidence_level']} (applied during training)")
    elif use_pose_gt_edges:
        logging.info(f"  Edge method: pose-GT (train only), metric: {similarity_metric}")
    else:
        logging.info(f"  Edge method: threshold ({similarity_threshold}), metric: {similarity_metric}")

    # Extract spectral entropies if using entropy-based prior
    def _extract_entropies(kfs):
        """Extract spectral entropies from keyframes, or None if unavailable."""
        if prior_signal != 'entropy':
            return None
        if kfs[0].spectral_entropy is None:
            logging.warning("  Spectral entropy not available in keyframes. "
                            "Falling back to density prior.")
            return None
        return np.array([kf.spectral_entropy for kf in kfs])

    train_entropies = _extract_entropies(all_train_keyframes)

    # Under two-pass mode OR pose-GT edges mode, the initial graph holds only
    # temporal edges. Similarity edges are attached afterward (pose-GT) or
    # during training via the refinement hook (two-pass).
    initial_descriptors = None if (two_pass_enabled or use_pose_gt_edges) else train_descriptors
    with profiler.profile('build_train_graph'):
        train_graph = build_graph_from_keyframes_batch(
            all_train_keyframes,
            temporal_neighbors=config['keyframe']['temporal_neighbors'],
            device=device,
            poses=train_poses,
            descriptors=initial_descriptors,
            similarity_threshold=similarity_threshold,
            similarity_max_k=train_similarity_max_k,
            similarity_min_k=train_similarity_min_k,
            similarity_exclude_temporal=similarity_exclude_temporal,
            similarity_dist=similarity_dist,
            spectral_entropies=train_entropies,
            prior_signal=prior_signal,
            channel_splits=channel_splits,
            multiscale_min_consistency=multiscale_min_consistency,
            similarity_metric=similarity_metric,
            standardization_stats=standardization_stats,
            sequence_ids=train_sequence_ids,
            temporal_edge_mode=temporal_edge_mode,
            temporal_direction_mode=temporal_direction_mode,
            **bayesian_config,
        )
    train_graph = attach_sensor_metadata(train_graph, train_sensor_ids, train_beam_counts, device)
    # Attach ground-truth pose-based similarity edges (train only).
    if use_pose_gt_edges:
        from keyframe.graph_manager import attach_pose_gt_similarity_edges
        triplet_cfg = config.get('triplet', {})
        with profiler.profile('attach_pose_gt_edges'):
            train_graph, n_pose_gt = attach_pose_gt_similarity_edges(
                train_graph,
                train_poses,
                train_descriptors,
                sequence_ids=train_sequence_ids,
                pos_dist=triplet_cfg.get('positive_distance_max', 5.0),
                min_temporal_gap=triplet_cfg.get('min_temporal_distance', 30),
                similarity_max_k=pose_gt_edge_max_k,
            )
        logging.info(f"  Pose-GT similarity edges attached: {n_pose_gt:,}")

    # Attach FFT magnitudes to graph for spectral policy
    if need_fft_magnitudes:
        fft_mags = np.array([kf.fft_magnitudes for kf in all_train_keyframes])
        train_graph.x_fft = torch.from_numpy(fft_mags.reshape(len(fft_mags), -1)).float().to(device)
        logging.info(f"  Attached x_fft: {train_graph.x_fft.shape} ({train_graph.x_fft.nbytes / 1e6:.1f} MB)")
    if need_phase_features:
        phase_feats = np.array([kf.phase_features for kf in all_train_keyframes])
        train_graph.x_phase = torch.from_numpy(phase_feats).float().to(device)
        logging.info(
            f"  Attached x_phase: {train_graph.x_phase.shape} "
            f"({train_graph.x_phase.element_size() * train_graph.x_phase.nelement() / 1e6:.1f} MB)"
        )
        phase_edges_cfg = graph_config.get('phase_edges', {})
        if phase_edges_cfg.get('enabled', False):
            from keyframe.graph_manager import attach_phase_similarity_edges
            with profiler.profile('attach_phase_edges_train'):
                train_graph, n_phase_edges = attach_phase_similarity_edges(
                    train_graph,
                    phase_feats,
                    descriptors=train_descriptors,
                    poses=train_poses,
                    sequence_ids=train_sequence_ids,
                    max_k=phase_edges_cfg.get('max_k', 5),
                    min_similarity=phase_edges_cfg.get('min_similarity', 0.0),
                    temporal_exclude=phase_edges_cfg.get('temporal_exclude', 30),
                )
            logging.info(f"  Phase-neighbor similarity edges attached: {n_phase_edges:,}")

    train_graph_time = profiler.get_stats('build_train_graph')['total']
    has_edge_attr = train_graph.edge_attr is not None
    n_temporal = int((train_graph.edge_type == 0).sum()) if hasattr(train_graph, 'edge_type') else 0
    n_similarity = int((train_graph.edge_type == 1).sum()) if hasattr(train_graph, 'edge_type') else 0
    logging.info(
        f"  Training graph: {train_graph.num_nodes:,} nodes, "
        f"{train_graph.edge_index.shape[1]:,} edges "
        f"(temporal={n_temporal:,}, similarity={n_similarity:,}), "
        f"edge_attr={'yes' if has_edge_attr else 'no'} (built in {train_graph_time:.2f}s)"
    )

    # Build validation graphs per dataset
    with profiler.profile('build_val_graphs'):
        for dataset_name, info in val_datasets_info.items():
            keyframes = info['keyframes']
            dataset_type = info.get('dataset_type', 'kitti')
            poses = np.array([kf.pose for kf in keyframes])
            val_descs = np.array([kf.descriptor for kf in keyframes])
            val_entropies = _extract_entropies(keyframes)
            val_similarity_max_k, val_similarity_min_k = get_sensor_similarity_k(
                graph_config, dataset_type, similarity_max_k, similarity_min_k
            )
            # Two-pass mode OR pose-GT mode: build graph with temporal edges only
            # initially. Pose-GT edges attached afterward below; two-pass refines
            # during training via validate() hook.
            val_initial_descriptors = None if (two_pass_enabled or use_pose_gt_edges) else val_descs
            graph = build_graph_from_keyframes_batch(
                keyframes,
                temporal_neighbors=config['keyframe']['temporal_neighbors'],
                device=device,
                poses=poses,
                descriptors=val_initial_descriptors,
                similarity_threshold=similarity_threshold,
                similarity_max_k=val_similarity_max_k,
                similarity_min_k=val_similarity_min_k,
                similarity_exclude_temporal=similarity_exclude_temporal,
                similarity_dist=similarity_dist,
                spectral_entropies=val_entropies,
                prior_signal=prior_signal,
                channel_splits=channel_splits,
                multiscale_min_consistency=multiscale_min_consistency,
                similarity_metric=similarity_metric,
                standardization_stats=standardization_stats,
                temporal_edge_mode=temporal_edge_mode,
                temporal_direction_mode=temporal_direction_mode,
                **bayesian_config,
            )
            graph = attach_sensor_metadata(
                graph,
                [get_sensor_id(dataset_type)] * len(keyframes),
                [get_beam_count(dataset_type)] * len(keyframes),
                device,
            )
            # Attach pose-GT similarity edges to val graph for upper-bound
            # experiment (train/val structural symmetry). Each val graph holds
            # a single sequence, so no sequence_ids needed.
            if use_pose_gt_edges:
                from keyframe.graph_manager import attach_pose_gt_similarity_edges
                triplet_cfg = config.get('triplet', {})
                graph, n_val_pose_gt = attach_pose_gt_similarity_edges(
                    graph, poses, val_descs,
                    sequence_ids=None,  # single sequence per val graph
                    pos_dist=triplet_cfg.get('positive_distance_max', 5.0),
                    min_temporal_gap=triplet_cfg.get('min_temporal_distance', 30),
                    similarity_max_k=pose_gt_edge_max_k,
                )
            info['poses'] = poses
            info['graph'] = graph
            # Attach FFT magnitudes for spectral policy
            if need_fft_magnitudes:
                val_fft = np.array([kf.fft_magnitudes for kf in keyframes])
                graph.x_fft = torch.from_numpy(val_fft.reshape(len(val_fft), -1)).float().to(device)
            if need_phase_features:
                val_phase = np.array([kf.phase_features for kf in keyframes])
                graph.x_phase = torch.from_numpy(val_phase).float().to(device)
                phase_edges_cfg = graph_config.get('phase_edges', {})
                if phase_edges_cfg.get('enabled', False):
                    from keyframe.graph_manager import attach_phase_similarity_edges
                    graph, n_val_phase = attach_phase_similarity_edges(
                        graph,
                        val_phase,
                        descriptors=val_descs,
                        poses=poses,
                        sequence_ids=None,
                        max_k=phase_edges_cfg.get('max_k', 5),
                        min_similarity=phase_edges_cfg.get('min_similarity', 0.0),
                        temporal_exclude=phase_edges_cfg.get('temporal_exclude', 30),
                    )
                    logging.info(f"  {dataset_name} phase-neighbor edges attached: {n_val_phase:,}")
            vt = int((graph.edge_type == 0).sum()) if hasattr(graph, 'edge_type') else 0
            vs = int((graph.edge_type == 1).sum()) if hasattr(graph, 'edge_type') else 0
            logging.info(
                f"  {dataset_name} graph: {graph.num_nodes:,} nodes, "
                f"{graph.edge_index.shape[1]:,} edges (temporal={vt:,}, similarity={vs:,})"
            )

    # ========================================================================
    # Stage 5: Create GNN Model
    # ========================================================================
    logging.info("")
    logging.info("[5/6] Creating GNN model...")

    from gnn.model import create_spectral_gnn

    edge_enc_config = config['gnn'].get('edge_encoding', None)

    # Create spectral policy if enabled
    spectral_policy = None
    if need_fft_magnitudes:
        from encoding.spectral_policy import create_spectral_policy
        # Determine n_rings and n_freqs from the cached FFT magnitudes
        sample_fft = all_train_keyframes[0].fft_magnitudes
        n_rings_fft, n_freqs_fft = sample_fft.shape
        spectral_policy = create_spectral_policy(
            policy_config, n_rings=n_rings_fft, n_freqs=n_freqs_fft
        )
        logging.info(f"  Spectral policy: {policy_config.get('type', 'soft_binning')} "
                     f"(n_rings={n_rings_fft}, n_freqs={n_freqs_fft}, "
                     f"output_dim={spectral_policy.output_dim}, "
                     f"params={sum(p.numel() for p in spectral_policy.parameters()):,})")

    with profiler.profile('create_gnn'):
        gnn = create_spectral_gnn(
            input_dim=config['gnn']['input_dim'],
            hidden_dim=config['gnn']['hidden_dim'],
            context_dim=config['gnn']['context_dim'],
            n_layers=config['gnn']['n_layers'],
            n_heads=config['gnn'].get('n_heads', 4),
            dropout=config['gnn']['dropout'],
            edge_encoder_config=edge_enc_config,
            spectral_policy=spectral_policy,
            norm_type=config['gnn'].get('norm_type', 'batch_norm'),
            use_residual_gate=config['gnn'].get('use_residual_gate', False),
            gate_hidden_dim=config['gnn'].get('gate_hidden_dim', 64),
            gate_initial_alpha=config['gnn'].get('gate_initial_alpha', 0.5),
            use_edge_confidence_gate=config['gnn'].get('use_edge_confidence_gate', False),
            edge_gate_hidden_dim=config['gnn'].get('edge_gate_hidden_dim', 16),
            phase_token_config=config['gnn'].get('phase_token'),
            phase_edge_config=config['gnn'].get('phase_edge'),
            phase_alignment_config=config['gnn'].get('phase_alignment_edge'),
            phase_coherence_config=config['gnn'].get('phase_coherence'),
            dual_stream_config=config['gnn'].get('dual_stream'),
            sensor_gate_config=config['gnn'].get('sensor_gate'),
            diffattn_value_source=config['gnn'].get('diffattn_value_source', 'diff'),
        ).to(device)

    # Get effective input_dim (may differ from config if policy overrides it)
    base_gnn = gnn.gnn if hasattr(gnn, 'gnn') else gnn
    effective_input_dim = base_gnn.input_dim

    n_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    logging.info(f"  GNN parameters: {n_params:,}")
    d_edge = edge_enc_config['d_edge'] if edge_enc_config else None
    logging.info(f"  Architecture: DiffAttnConv + EdgeEncoder(d_edge={d_edge}), "
                 f"{config['gnn']['n_layers']} layers, "
                 f"{config['gnn']['hidden_dim']} hidden, {config['gnn']['context_dim']} context, "
                 f"{config['gnn'].get('n_heads', 4)} heads")
    phase_dim = config['gnn'].get('phase_token', {}).get('token_dim', 0) if need_phase_features else 0
    logging.info(f"  Output: cat(raw_{effective_input_dim}, context_{config['gnn']['context_dim']}"
                 f"{', phase_' + str(phase_dim) if phase_dim else ''}) = "
                 f"{effective_input_dim + config['gnn']['context_dim'] + phase_dim}D")

    # Optional fine-tune initialization. This intentionally loads model weights
    # only, not optimizer/epoch state, so new graph policies can start from the
    # paper checkpoint without inheriting stale optimizer moments.
    resume_checkpoint = args.resume_checkpoint or config.get('training', {}).get('resume_from_checkpoint')
    if resume_checkpoint and not args.validate_only:
        ckpt_path = Path(resume_checkpoint)
        if not ckpt_path.exists():
            logging.warning(f"  Resume checkpoint not found, training from scratch: {ckpt_path}")
        else:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            source_state = ckpt.get('model_state_dict', ckpt)
            target_state = gnn.state_dict()
            compatible = {}
            skipped = []
            for key, value in source_state.items():
                clean_key = key[len('module.'):] if key.startswith('module.') else key
                if clean_key in target_state and target_state[clean_key].shape == value.shape:
                    compatible[clean_key] = value
                else:
                    skipped.append(clean_key)
            missing, unexpected = gnn.load_state_dict(compatible, strict=False)
            logging.info(
                f"  Fine-tune init: loaded {len(compatible)}/{len(target_state)} tensors "
                f"from {ckpt_path} (epoch={ckpt.get('epoch', '?')}, "
                f"best_val_metric={ckpt.get('best_val_metric', '?')})"
            )
            if skipped:
                logging.warning(f"  Skipped incompatible checkpoint tensors: {skipped[:20]}")
            if missing:
                logging.warning(f"  Missing after fine-tune init: {missing[:20]}")
            if unexpected:
                logging.warning(f"  Unexpected after fine-tune init: {unexpected[:20]}")

    # Create trainer
    from gnn.trainer import GNNTrainer

    smoothap_cfg = config['training'].get('smoothap', {})
    trainer = GNNTrainer(
        model=gnn,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        temperature=config['training']['temperature'],
        checkpoint_dir=checkpoint_dir,
        patience=config['gnn']['patience'],
        use_multi_gpu=False,
        use_amp=config['training'].get('use_amp', True),
        policy_lr_scale=policy_config.get('lr_scale', 1.0),
        policy_warmup_epochs=policy_config.get('warmup_epochs', 0),
        loss_type=config['training'].get('loss_type', 'infonce'),
        smoothap_tau=smoothap_cfg.get('tau', 0.01),
        smoothap_n_pos=smoothap_cfg.get('n_pos', 8),
        smoothap_n_neg=smoothap_cfg.get('n_neg', 32),
        smoothap_batch_anchors=smoothap_cfg.get('batch_anchors', 64),
        edge_aux_lambda=config['training'].get('edge_aux_lambda', 0.0),
        phase_edge_aux_lambda=config['training'].get('phase_edge_aux_lambda', 0.0),
        phase_edge_aux_balance=config['training'].get('phase_edge_aux_balance', False),
        phase_edge_aux_focal_gamma=config['training'].get('phase_edge_aux_focal_gamma', 0.0),
        phase_alignment_aux_lambda=config['training'].get('phase_alignment_aux_lambda', 0.0),
        phase_alignment_aux_balance=config['training'].get('phase_alignment_aux_balance', False),
        phase_alignment_aux_focal_gamma=config['training'].get('phase_alignment_aux_focal_gamma', 0.0),
        context_aux_lambda=config['training'].get('context_aux_lambda', 0.0),
        phase_token_aux_lambda=config['training'].get('phase_token_aux_lambda', 0.0),
        checkpoint_metric=config['training'].get('checkpoint_metric', 'average_recall@1'),
        recall_k_values=config['training'].get('recall_k_values', [1, 5, 10]),
    )

    # ========================================================================
    # Stage 6: Training (or validate-only)
    # ========================================================================

    if args.validate_only:
        # ── Validate-only mode ─────────────────────────────────────────────
        logging.info("")
        logging.info("[6/6] VALIDATE-ONLY MODE")
        logging.info("=" * 80)

        # Load model weights only (skip optimizer — not needed for eval)
        ckpt_path = os.path.join(checkpoint_dir, 'best_model.pth')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        missing, unexpected = gnn.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            logging.warning(f"Missing keys (new params, using defaults): {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys (removed): {unexpected}")
        logging.info(f"  Loaded model from {ckpt_path} (epoch {ckpt.get('epoch', '?')}, "
                     f"best R@1={ckpt.get('best_val_metric', '?')})")
        logging.info(f"  Graph params: confidence_level={bayesian_config.get('confidence_level', 'N/A')}, "
                     f"similarity_max_k={similarity_max_k}, "
                     f"similarity_min_k={similarity_min_k}, "
                     f"temporal_edge_mode={temporal_edge_mode}, "
                     f"temporal_direction_mode={temporal_direction_mode}, "
                     f"multiscale_min_consistency={multiscale_min_consistency}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = trainer.validate_all(
            val_datasets_info,
            per_query_dump_dir=args.dump_per_query_dir,
        )
        logging.info("")
        logging.info("VALIDATE-ONLY COMPLETE")
        logging.info(f"  Avg R@1: {metrics['_average']['recall@1']:.4f} "
                     f"(raw: {metrics['_average']['raw_recall@1']:.4f}, "
                     f"ctx: {metrics['_average']['ctx_recall@1']:.4f})")

    else:
        # ── Normal training mode ───────────────────────────────────────────
        logging.info("")
        logging.info("[6/6] Starting GNN training...")
        logging.info("=" * 80)

        # poses and descriptors already extracted in Stage 4
        logging.info(f"Training configuration:")
        logging.info(f"  Epochs: {config['training']['n_epochs']}")
        logging.info(f"  Learning rate: {config['training']['learning_rate']}")
        logging.info(f"  Temperature: {config['training']['temperature']}")
        logging.info(f"  Sequences: {current_seq_id} (per-sequence mining enabled)")
        logging.info("")

        # Create triplet miner from config (with GPU acceleration)
        from gnn.triplet_miner import create_triplet_miner
        triplet_config = config.get('triplet', {})
        triplet_miner = create_triplet_miner(
            positive_distance_max=triplet_config.get('positive_distance_max', 5.0),
            negative_distance_min=triplet_config.get('negative_distance_min', 10.0),
            positive_temporal_min=triplet_config.get('min_temporal_distance', 30),
            negative_temporal_min=triplet_config.get('min_temporal_distance', 30),
            mining_strategy=triplet_config.get('mining_strategy', 'hard'),
        )
        logging.info(f"  Triplet mining device: {triplet_miner.device}")

        # Free VRAM used during graph building / data loading before training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Two-pass refinement kwargs: params forwarded to trainer._refine_similarity_edges
        refine_fit_kwargs = {
            'pos_dist': triplet_config.get('positive_distance_max', 5.0),
            'neg_dist': triplet_config.get('negative_distance_min', 10.0),
            'min_temporal_gap': triplet_config.get('min_temporal_distance', 30),
            'n_samples': graph_config.get('n_samples', 1_000_000),
        }
        refine_edge_kwargs = {
            'similarity_threshold': similarity_threshold,
            'similarity_max_k': similarity_max_k,
            'similarity_min_k': similarity_min_k,
            'similarity_exclude_temporal': similarity_exclude_temporal,
            'similarity_metric': similarity_metric,
            'prior_signal': prior_signal,
            'channel_splits': channel_splits,
            'multiscale_min_consistency': multiscale_min_consistency,
            **bayesian_config,  # confidence_level, base_prior, density_k, density_beta
        }

        try:
            training_config = config.get('training', {})
            with profiler.profile('training'):
                trainer.train(
                    train_graph=train_graph,
                    train_poses=train_poses,
                    train_descriptors=train_descriptors,
                    train_sequence_ids=train_sequence_ids,
                    val_datasets=val_datasets_info,
                    n_epochs=training_config.get('n_epochs', 100),
                    triplet_miner=triplet_miner,
                    mine_every_n_epochs=training_config.get('mine_every_n_epochs', 1),
                    validate_every_n_epochs=training_config.get('validate_every_n_epochs', 1),
                    n_triplets_per_anchor=triplet_config.get('n_triplets_per_anchor', 1),
                    two_pass_cfg=two_pass_cfg if two_pass_enabled else None,
                    refine_fit_kwargs=refine_fit_kwargs,
                    refine_edge_kwargs=refine_edge_kwargs,
                    temperature_schedule=training_config.get('temperature_schedule'),
                    triplet_sampling_config=training_config.get('triplet_sampling'),
                )
        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
            raise

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.perf_counter() - start_time

    logging.info("")
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total runtime: {total_time/3600:.2f} hours ({total_time:.0f} seconds)")
    logging.info(f"Log file: {log_file}")

    # Print profiling summary
    logging.info(profiler.summary())


if __name__ == '__main__':
    main()
