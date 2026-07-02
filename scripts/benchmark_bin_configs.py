#!/usr/bin/env python3
"""
Bin Configuration Benchmark — binning strategy, n_bins, alpha 조합 실험.

Phase 1 (Screening, 20 epochs):
  bin_baseline_octave:  octave 9 bins (현재 baseline)
  bin_exp_a2_b9:        exponential alpha=2.0, 9 bins (dead dim 제거)
  bin_exp_a3_b9:        exponential alpha=3.0, 9 bins (강한 저주파 집중)
  bin_exp_a1_b9:        exponential alpha=1.0, 9 bins (거의 linear)
  bin_exp_a2_b6:        exponential alpha=2.0, 6 bins (compact)
  bin_exp_a2_b4:        exponential alpha=2.0, 4 bins (최소)
  bin_exp_a2_b12:       exponential alpha=2.0, 12 bins (세밀)

Usage:
  python scripts/benchmark_bin_configs.py --suite screening --epochs 20
  python scripts/benchmark_bin_configs.py --suite quick --epochs 5
  python scripts/benchmark_bin_configs.py --experiments bin_exp_a2_b9 bin_exp_a3_b9 --epochs 100
  python scripts/benchmark_bin_configs.py --list
  python scripts/benchmark_bin_configs.py --analyze-only --results-dir results/bin_benchmark_XXXX
"""

import sys
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import copy
import json
import time
import logging
import yaml
import torch
import gc
from pathlib import Path
from datetime import datetime


# ============================================================================
# Config Helpers
# ============================================================================

def compute_input_dim(n_rows, n_bins, n_stats, n_inter_stats):
    """Compute descriptor dimension from binning params."""
    base_dim = n_rows * n_bins * n_stats
    inter_dim = n_rows * (n_bins - 1) * n_inter_stats * n_stats
    return base_dim + inter_dim


def effective_n_bins(binning_strategy, n_bins, n_azimuth=360):
    """Get actual number of bins (octave auto-computes)."""
    if binning_strategy == 'octave':
        n_freqs = n_azimuth // 2 + 1
        power = 0
        while 2 ** power < n_freqs:
            power += 1
        return power + 1
    return n_bins


def make_bin_config(base: dict, binning_strategy: str, n_bins: int,
                    alpha: float = 2.0) -> dict:
    """Create config with specific bin configuration."""
    cfg = copy.deepcopy(base)
    enc = cfg['encoding']

    enc['binning_strategy'] = binning_strategy
    enc['n_bins'] = n_bins
    enc['alpha'] = alpha

    # Disable spectral policy (pure binning comparison)
    enc['spectral_policy']['enabled'] = False

    # Compute effective bins and input_dim
    n_azimuth = enc.get('n_azimuth', 360)
    eff_bins = effective_n_bins(binning_strategy, n_bins, n_azimuth)

    projection_type = enc.get('projection_type', 'range_image')
    if projection_type == 'bev':
        n_rows = int(enc.get('max_range', 80.0) - enc.get('min_range', 1.0))
    else:
        n_rows = enc.get('target_elevation_bins', 16)

    n_stats = len(enc.get('bin_statistics', ['sum']))
    n_inter = len(enc.get('inter_bin_statistics', []))

    input_dim = compute_input_dim(n_rows, eff_bins, n_stats, n_inter)
    cfg['gnn']['input_dim'] = input_dim
    cfg['encoding']['descriptor_size'] = input_dim

    return cfg


# ============================================================================
# Experiment Definitions
# ============================================================================

BIN_EXPERIMENTS = {
    'bin_baseline_octave': {
        'description': 'Octave binning (9 auto-bins, current baseline)',
        'config_fn': lambda base: make_bin_config(base, 'octave', 9),
    },
    'bin_exp_a2_b9': {
        'description': 'Exponential alpha=2.0, 9 bins (same dim, no dead dims)',
        'config_fn': lambda base: make_bin_config(base, 'exponential', 9, alpha=2.0),
    },
    'bin_exp_a3_b9': {
        'description': 'Exponential alpha=3.0, 9 bins (stronger low-freq emphasis)',
        'config_fn': lambda base: make_bin_config(base, 'exponential', 9, alpha=3.0),
    },
    'bin_exp_a1_b9': {
        'description': 'Exponential alpha=1.0, 9 bins (near-linear)',
        'config_fn': lambda base: make_bin_config(base, 'exponential', 9, alpha=1.0),
    },
    'bin_exp_a2_b6': {
        'description': 'Exponential alpha=2.0, 6 bins (compact, 352D)',
        'config_fn': lambda base: make_bin_config(base, 'exponential', 6, alpha=2.0),
    },
    'bin_exp_a2_b4': {
        'description': 'Exponential alpha=2.0, 4 bins (minimal, 224D)',
        'config_fn': lambda base: make_bin_config(base, 'exponential', 4, alpha=2.0),
    },
    'bin_exp_a2_b12': {
        'description': 'Exponential alpha=2.0, 12 bins (fine-grained, 736D)',
        'config_fn': lambda base: make_bin_config(base, 'exponential', 12, alpha=2.0),
    },
}

# ============================================================================
# Learnable Bin Experiments — SoftBinning with auto-computed output_dim
# ============================================================================

def make_soft_config(base: dict, n_soft_bins: int, init_mode: str = 'octave',
                     shared: bool = True, stats=None, inter_stats=None,
                     inter_ring=None) -> dict:
    """Create config with learnable soft binning.

    output_dim and GNN input_dim are auto-computed from n_soft_bins to ensure
    no projection layer is needed (previous SoftBinning failure root cause).

    Args:
        inter_ring: dict with keys 'enabled', 'channels', 'kernel_size' for
                    inter-ring 1D conv. None or {'enabled': False} to disable.
    """
    stats = stats if stats is not None else ['mean', 'std']
    inter_stats = inter_stats if inter_stats is not None else ['diff']
    cfg = copy.deepcopy(base)
    enc = cfg['encoding']

    # Compute auto output_dim
    projection_type = enc.get('projection_type', 'range_image')
    if projection_type == 'bev':
        n_rings = int(enc.get('max_range', 80.0) - enc.get('min_range', 1.0))
    else:
        n_rings = enc.get('target_elevation_bins', 16)

    n_stats = len(stats)
    n_inter = len(inter_stats)
    per_ring = n_soft_bins * n_stats + (n_soft_bins - 1) * n_inter * n_stats

    # Inter-ring conv adds channels per ring
    ir_total = 0
    if inter_ring and inter_ring.get('enabled', False):
        ir_total = n_rings * inter_ring.get('channels', 8)

    output_dim = n_rings * per_ring + ir_total

    # Enable spectral policy with soft_binning
    sb_config = {
        'n_soft_bins': n_soft_bins,
        'init_mode': init_mode,
        'stats': stats,
        'inter_stats': inter_stats,
        'alpha': enc.get('alpha', 2.0),
    }
    if inter_ring:
        sb_config['inter_ring'] = inter_ring

    enc['spectral_policy'] = {
        'enabled': True,
        'type': 'soft_binning',
        'output_dim': output_dim,
        'shared_across_rings': shared,
        'lr_scale': 0.1,
        'warmup_epochs': 3,
        'soft_binning': sb_config,
    }

    cfg['gnn']['input_dim'] = output_dim
    cfg['encoding']['descriptor_size'] = output_dim
    return cfg


LEARNABLE_EXPERIMENTS = {
    'learn_baseline_octave': {
        'description': 'Fixed octave binning baseline (no learnable policy)',
        'config_fn': lambda base: make_bin_config(base, 'octave', 9),
    },
    'learn_soft9_octave_shared': {
        'description': 'Soft 9 bins, octave init, shared (18 params)',
        'config_fn': lambda base: make_soft_config(base, 9, 'octave', shared=True),
    },
    'learn_soft9_octave_perring': {
        'description': 'Soft 9 bins, octave init, per-ring (288 params)',
        'config_fn': lambda base: make_soft_config(base, 9, 'octave', shared=False),
    },
    'learn_soft9_exp_shared': {
        'description': 'Soft 9 bins, exp init, shared (18 params)',
        'config_fn': lambda base: make_soft_config(base, 9, 'exponential', shared=True),
    },
    'learn_soft16_octave_shared': {
        'description': 'Soft 16 bins, octave init, shared (32 params)',
        'config_fn': lambda base: make_soft_config(base, 16, 'octave', shared=True),
    },
    'learn_soft16_octave_perring': {
        'description': 'Soft 16 bins, octave init, per-ring (512 params)',
        'config_fn': lambda base: make_soft_config(base, 16, 'octave', shared=False),
    },
}

# ============================================================================
# Inter-Ring Experiments — 1D Conv along ring axis (vertical structure)
# All use: soft9, octave init, per-ring (winner from learnable benchmark)
# ============================================================================

INTER_RING_EXPERIMENTS = {
    'ir_baseline': {
        'description': 'Current best: soft9 perring + inter-bin diff (544D)',
        'config_fn': lambda base: make_soft_config(
            base, 9, 'octave', shared=False,
            inter_stats=['diff'], inter_ring=None),
    },
    'ir_nointer': {
        'description': 'Base only: no inter-bin, no inter-ring (288D)',
        'config_fn': lambda base: make_soft_config(
            base, 9, 'octave', shared=False,
            inter_stats=[], inter_ring=None),
    },
    'ir_conv_k8': {
        'description': 'Inter-ring conv k=8 ks=3, no inter-bin (416D)',
        'config_fn': lambda base: make_soft_config(
            base, 9, 'octave', shared=False,
            inter_stats=[],
            inter_ring={'enabled': True, 'channels': 8, 'kernel_size': 3}),
    },
    'ir_conv_k16': {
        'description': 'Inter-ring conv k=16 ks=3, no inter-bin (544D)',
        'config_fn': lambda base: make_soft_config(
            base, 9, 'octave', shared=False,
            inter_stats=[],
            inter_ring={'enabled': True, 'channels': 16, 'kernel_size': 3}),
    },
    'ir_conv_k8_ks5': {
        'description': 'Inter-ring conv k=8 ks=5, wider receptive field (416D)',
        'config_fn': lambda base: make_soft_config(
            base, 9, 'octave', shared=False,
            inter_stats=[],
            inter_ring={'enabled': True, 'channels': 8, 'kernel_size': 5}),
    },
}

# ============================================================================
# Regularization Experiments — BN vs LN, Dropout sweep
# Base: ir_conv_k8 (soft9, octave, per-ring, inter_ring k=8 ks=3, 416D)
# ============================================================================

def make_reg_config(base: dict, dropout: float, norm_type: str = 'batch_norm') -> dict:
    """ir_conv_k8 base with GNN regularization variant."""
    cfg = make_soft_config(
        base, n_soft_bins=9, init_mode='octave', shared=False,
        inter_stats=[],
        inter_ring={'enabled': True, 'channels': 8, 'kernel_size': 3},
    )
    cfg['gnn']['dropout'] = dropout
    cfg['gnn']['norm_type'] = norm_type
    cfg['gnn']['edge_encoding']['dropout'] = dropout
    return cfg


REGULARIZATION_EXPERIMENTS = {
    'reg_baseline': {
        'description': 'ir_conv_k8 + BN + drop=0.1 (control)',
        'config_fn': lambda base: make_reg_config(base, 0.1, 'batch_norm'),
    },
    'reg_bn_drop0': {
        'description': 'BN + dropout=0.0',
        'config_fn': lambda base: make_reg_config(base, 0.0, 'batch_norm'),
    },
    'reg_bn_drop02': {
        'description': 'BN + dropout=0.2',
        'config_fn': lambda base: make_reg_config(base, 0.2, 'batch_norm'),
    },
    'reg_ln_drop01': {
        'description': 'LN + dropout=0.1',
        'config_fn': lambda base: make_reg_config(base, 0.1, 'layer_norm'),
    },
    'reg_ln_drop0': {
        'description': 'LN + dropout=0.0',
        'config_fn': lambda base: make_reg_config(base, 0.0, 'layer_norm'),
    },
}

# Merge all experiments into one registry
ALL_EXPERIMENTS = {
    **BIN_EXPERIMENTS, **LEARNABLE_EXPERIMENTS,
    **INTER_RING_EXPERIMENTS, **REGULARIZATION_EXPERIMENTS,
}

BIN_SUITES = {
    'screening': list(BIN_EXPERIMENTS.keys()),
    'quick': ['bin_baseline_octave', 'bin_exp_a2_b9', 'bin_exp_a3_b9'],
    'alpha_sweep': ['bin_exp_a1_b9', 'bin_exp_a2_b9', 'bin_exp_a3_b9'],
    'bins_sweep': ['bin_exp_a2_b4', 'bin_exp_a2_b6', 'bin_exp_a2_b9', 'bin_exp_a2_b12'],
    'phase2': ['bin_baseline_octave', 'bin_exp_a3_b9', 'bin_exp_a2_b12'],
    'learnable': list(LEARNABLE_EXPERIMENTS.keys()),
    'learnable_quick': ['learn_baseline_octave', 'learn_soft9_octave_shared', 'learn_soft9_octave_perring'],
    'inter_ring': list(INTER_RING_EXPERIMENTS.keys()),
    'inter_ring_quick': ['ir_baseline', 'ir_nointer', 'ir_conv_k8'],
    'regularization': list(REGULARIZATION_EXPERIMENTS.keys()),
    'reg_quick': ['reg_baseline', 'reg_bn_drop0', 'reg_ln_drop01'],
}


# ============================================================================
# Benchmark Runner (adapted from benchmark_policies.py)
# ============================================================================

class BinBenchmarkRunner:
    """Runs bin configuration experiments and collects results."""

    def __init__(self, base_config_path, results_dir, n_epochs=None,
                 device='cuda', seed=42):
        with open(base_config_path) as f:
            self.base_config = yaml.safe_load(f)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_epochs = n_epochs
        self.device = device
        self.seed = seed
        self.results = {}

    def run_experiment(self, exp_name, exp_def):
        """Run a single experiment."""
        logging.info("=" * 80)
        logging.info(f"EXPERIMENT: {exp_name}")
        logging.info(f"  {exp_def['description']}")
        logging.info("=" * 80)

        config = exp_def['config_fn'](self.base_config)
        if self.n_epochs is not None:
            config['training']['n_epochs'] = self.n_epochs
        config['system']['seed'] = self.seed
        config['system']['device'] = self.device
        if self.device == 'cpu':
            config['training']['use_amp'] = False

        exp_dir = self.results_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        config['system']['checkpoint_dir'] = str(exp_dir / 'checkpoints')
        config['system']['log_dir'] = str(exp_dir / 'logs')
        os.makedirs(config['system']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['system']['log_dir'], exist_ok=True)

        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Log config details
        enc = config['encoding']
        input_dim = config['gnn']['input_dim']
        sp = enc.get('spectral_policy', {})
        if sp.get('enabled') and sp.get('type') == 'soft_binning':
            sb = sp.get('soft_binning', {})
            ir = sb.get('inter_ring', {})
            ir_info = (f", inter_ring=k{ir.get('channels')}ks{ir.get('kernel_size')}"
                       if ir.get('enabled') else "")
            logging.info(f"  soft_binning: n_soft_bins={sb.get('n_soft_bins')}, "
                         f"init_mode={sb.get('init_mode', 'exp')}, "
                         f"shared={sp.get('shared_across_rings', True)}, "
                         f"inter_stats={sb.get('inter_stats', [])}{ir_info}, "
                         f"input_dim={input_dim}")
        else:
            logging.info(f"  strategy={enc['binning_strategy']}, n_bins={enc['n_bins']}, "
                         f"alpha={enc.get('alpha', 'n/a')}, input_dim={input_dim}")

        gnn_cfg = config['gnn']
        norm_type = gnn_cfg.get('norm_type', 'batch_norm')
        dropout = gnn_cfg.get('dropout', 0.1)
        edge_drop = gnn_cfg.get('edge_encoding', {}).get('dropout', 0.1)
        logging.info(f"  GNN: norm={norm_type}, dropout={dropout}, edge_dropout={edge_drop}")

        start_time = time.perf_counter()
        try:
            metrics = self._run_training(config, exp_name, exp_dir)
            elapsed = time.perf_counter() - start_time
            metrics['elapsed_seconds'] = elapsed
            metrics['status'] = 'success'
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logging.error(f"Experiment {exp_name} failed: {e}", exc_info=True)
            metrics = {
                'status': 'failed',
                'error': str(e),
                'elapsed_seconds': elapsed,
            }

        # Add config metadata
        metrics['input_dim'] = input_dim
        if sp.get('enabled') and sp.get('type') == 'soft_binning':
            sb = sp.get('soft_binning', {})
            metrics['binning_strategy'] = 'soft_binning'
            metrics['n_bins'] = sb.get('n_soft_bins', '?')
            metrics['alpha'] = sb.get('alpha', None)
            metrics['init_mode'] = sb.get('init_mode', 'exponential')
            metrics['shared'] = sp.get('shared_across_rings', True)
        else:
            metrics['binning_strategy'] = enc['binning_strategy']
            metrics['n_bins'] = enc['n_bins']
            metrics['alpha'] = enc.get('alpha', None)

        metrics_path = exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.results[exp_name] = metrics
        self._save_summary()
        return metrics

    def _run_training(self, config, exp_name, exp_dir):
        """Run training as subprocess."""
        import subprocess

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        train_script = os.path.join(project_root, 'train_multi_dataset.py')
        cmd = [
            sys.executable, train_script,
            '--config', str(config_path),
            '--checkpoint-dir', config['system']['checkpoint_dir'],
        ]

        logging.info(f"  Running: {' '.join(cmd)}")

        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=project_root,
            text=True,
            bufsize=1,
        )

        log_lines = []
        log_path = exp_dir / 'training.log'
        with open(log_path, 'w') as log_f:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                log_lines.append(line.rstrip())
                if any(kw in line for kw in ['Epoch ', 'AVERAGE', 'best model',
                                              'Early stopping', 'TRAINING COMPLETE',
                                              'Encoder created', 'Cache']):
                    logging.info(f"  [{exp_name}] {line.rstrip()}")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Training subprocess exited with code {proc.returncode}. "
                f"Check {log_path}"
            )

        metrics = self._parse_log(log_lines, config)
        metrics['log_path'] = str(log_path)
        return metrics

    def _parse_log(self, log_lines, config):
        """Parse training log for metrics."""
        import re
        metrics = {
            'n_epochs_config': config['training']['n_epochs'],
        }

        epoch_metrics = []
        best_r1 = 0.0
        best_epoch = -1
        final_loss = None

        for line in log_lines:
            # Epoch X/Y | Loss: 0.XXXX | Avg R@1: 0.XXXX
            if 'Epoch ' in line and '| Loss:' in line:
                try:
                    parts = line.split('|')
                    epoch_part = [p for p in parts if 'Epoch' in p][0]
                    loss_part = [p for p in parts if 'Loss:' in p][0]
                    epoch_num = int(epoch_part.strip().split()[-1].split('/')[0])
                    loss_val = float(loss_part.strip().split()[-1])

                    entry = {'epoch': epoch_num, 'loss': loss_val}
                    r1_parts = [p for p in parts if 'R@1:' in p]
                    if r1_parts:
                        r1_val = float(r1_parts[0].strip().split()[-1])
                        entry['avg_recall@1'] = r1_val
                        if r1_val > best_r1:
                            best_r1 = r1_val
                            best_epoch = epoch_num

                    final_loss = loss_val
                    epoch_metrics.append(entry)
                except (ValueError, IndexError):
                    continue

            # Per-dataset R@1
            if '| R@1:' in line and 'AVERAGE' not in line and 'Epoch' not in line:
                try:
                    parts = line.split('|')
                    ds_name = parts[0].strip().split()[-1]
                    r1_match = re.search(r'R@1:\s*([\d.]+)', parts[1])
                    raw_match = re.search(r'raw:\s*([\d.]+)', parts[1])
                    if r1_match:
                        metrics[f'last_val_{ds_name}_recall@1'] = float(r1_match.group(1))
                    if raw_match:
                        metrics[f'last_val_{ds_name}_raw_recall@1'] = float(raw_match.group(1))
                except (ValueError, IndexError):
                    continue

            # AVERAGE
            if 'AVERAGE' in line and 'R@1:' in line:
                try:
                    r1_match = re.search(r'R@1:\s*([\d.]+)', line)
                    raw_match = re.search(r'raw:\s*([\d.]+)', line)
                    if r1_match:
                        metrics['last_avg_recall@1'] = float(r1_match.group(1))
                    if raw_match:
                        metrics['last_avg_raw_recall@1'] = float(raw_match.group(1))
                except (ValueError, IndexError):
                    continue

            # Best model
            if 'New best model' in line:
                try:
                    m = re.search(r'R@1:\s*([\d.]+)', line)
                    if m:
                        best_r1 = float(m.group(1))
                except (ValueError, IndexError):
                    continue

            # GNN params
            if 'GNN parameters:' in line:
                try:
                    m = re.search(r'([\d,]+)', line.split('GNN parameters:')[1])
                    if m:
                        metrics['n_params'] = int(m.group(1).replace(',', ''))
                except (ValueError, IndexError):
                    continue

            # Encoder output dim (from log)
            if 'output_dim=' in line and 'Encoder' in line:
                try:
                    m = re.search(r'output_dim=(\d+)', line)
                    if m:
                        metrics['encoder_output_dim'] = int(m.group(1))
                except (ValueError, IndexError):
                    continue

        metrics['best_recall@1'] = best_r1
        metrics['best_epoch'] = best_epoch
        metrics['final_loss'] = final_loss
        metrics['n_epochs_actual'] = epoch_metrics[-1]['epoch'] if epoch_metrics else 0
        metrics['epoch_history'] = epoch_metrics
        return metrics

    def _save_summary(self):
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def run_suite(self, experiment_names):
        """Run experiments sequentially."""
        logging.info(f"Running {len(experiment_names)} bin config experiments")
        logging.info(f"Results: {self.results_dir}")

        for i, name in enumerate(experiment_names):
            if name not in ALL_EXPERIMENTS:
                logging.warning(f"Unknown experiment: {name}, skipping")
                continue

            logging.info(f"\n[{i+1}/{len(experiment_names)}] {name}")

            # Skip completed
            metrics_path = self.results_dir / name / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path) as f:
                    existing = json.load(f)
                if existing.get('status') == 'success':
                    logging.info(f"  Skipping (completed, best R@1={existing.get('best_recall@1', '?')})")
                    self.results[name] = existing
                    continue

            self.run_experiment(name, ALL_EXPERIMENTS[name])

            torch.cuda.empty_cache()
            gc.collect()

        self._save_summary()
        self._print_table()

    def _print_table(self):
        """Print comparison table."""
        logging.info("\n" + "=" * 100)
        logging.info("BIN CONFIGURATION BENCHMARK RESULTS")
        logging.info("=" * 100)

        header = (
            f"{'Experiment':<25} {'Strategy':<12} {'Bins':>5} {'Alpha':>6} "
            f"{'Dim':>5} {'Best R@1':>10} {'Raw R@1':>10} {'Loss':>10} "
            f"{'Epoch':>6} {'Time':>8} {'Status':<8}"
        )
        logging.info(header)
        logging.info("-" * 100)

        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('best_recall@1', 0),
            reverse=True
        )

        for name, m in sorted_results:
            strategy = m.get('binning_strategy', '?')
            n_bins = m.get('n_bins', '?')
            alpha = m.get('alpha')
            alpha_str = f"{alpha:.1f}" if alpha is not None else "n/a"
            dim = m.get('input_dim', '?')
            best_r1 = m.get('best_recall@1', 0)
            raw_r1 = m.get('last_avg_raw_recall@1', 0)
            loss = m.get('final_loss', 0)
            epoch = m.get('best_epoch', -1)
            elapsed = m.get('elapsed_seconds', 0)
            status = m.get('status', '?')
            time_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"

            logging.info(
                f"{name:<25} {strategy:<12} {n_bins:>5} {alpha_str:>6} "
                f"{dim:>5} {best_r1:>10.4f} {raw_r1:>10.4f} "
                f"{loss if loss else 0:>10.4f} {epoch:>6} {time_str:>8} {status:<8}"
            )

        logging.info("=" * 100)

        # Per-dataset breakdown
        logging.info("\nPER-DATASET R@1:")
        datasets = set()
        for m in self.results.values():
            for k in m:
                if k.startswith('last_val_') and k.endswith('_recall@1') and 'raw' not in k:
                    datasets.add(k.replace('last_val_', '').replace('_recall@1', ''))
        datasets = sorted(datasets)

        if datasets:
            header = f"{'Experiment':<25}" + "".join(f" {ds:>12}" for ds in datasets)
            logging.info(header)
            logging.info("-" * (25 + 13 * len(datasets)))
            for name, m in sorted_results:
                if m.get('status') != 'success':
                    continue
                vals = []
                for ds in datasets:
                    v = m.get(f'last_val_{ds}_recall@1')
                    vals.append(f"{v:.4f}" if v is not None else "n/a")
                logging.info(f"{name:<25}" + "".join(f" {v:>12}" for v in vals))

        # Save CSV
        csv_path = self.results_dir / 'comparison.csv'
        with open(csv_path, 'w') as f:
            f.write('experiment,strategy,n_bins,alpha,input_dim,best_recall@1,'
                    'last_avg_raw_recall@1,final_loss,best_epoch,n_params,'
                    'elapsed_seconds,status\n')
            for name, m in sorted_results:
                f.write(
                    f"{name},{m.get('binning_strategy','?')},"
                    f"{m.get('n_bins','?')},{m.get('alpha','')},"
                    f"{m.get('input_dim','?')},{m.get('best_recall@1',0):.6f},"
                    f"{m.get('last_avg_raw_recall@1',0):.6f},"
                    f"{m.get('final_loss',0) or 0:.6f},{m.get('best_epoch',-1)},"
                    f"{m.get('n_params',0)},{m.get('elapsed_seconds',0):.1f},"
                    f"{m.get('status','?')}\n"
                )
        logging.info(f"\nCSV: {csv_path}")


# ============================================================================
# Main
# ============================================================================

def list_experiments():
    """Show all experiments with computed dimensions."""
    dummy_enc = {
        'n_azimuth': 360,
        'projection_type': 'range_image',
        'target_elevation_bins': 16,
        'bin_statistics': ['mean', 'std'],
        'inter_bin_statistics': ['diff'],
    }
    dummy_base = {
        'encoding': {
            **dummy_enc,
            'binning_strategy': 'octave',
            'n_bins': 16,
            'alpha': 2.0,
            'spectral_policy': {'enabled': False},
            'descriptor_size': 544,
        },
        'gnn': {'input_dim': 544, 'edge_encoding': {'dropout': 0.1}},
    }

    # Fixed binning experiments
    print("\nFixed Binning Experiments:")
    print("=" * 100)
    print(f"  {'Name':<30} {'Strategy':<12} {'Bins':>5} {'Alpha':>6} {'Dim':>6} Description")
    print("-" * 100)
    for name, exp in BIN_EXPERIMENTS.items():
        cfg = exp['config_fn'](copy.deepcopy(dummy_base))
        strategy = cfg['encoding']['binning_strategy']
        n_bins = cfg['encoding']['n_bins']
        alpha = cfg['encoding'].get('alpha')
        eff_bins = effective_n_bins(strategy, n_bins)
        input_dim = cfg['gnn']['input_dim']
        alpha_str = f"{alpha:.1f}" if (alpha is not None and strategy != 'octave') else "n/a"
        print(f"  {name:<30} {strategy:<12} {eff_bins:>5} {alpha_str:>6} {input_dim:>6} {exp['description']}")

    # Learnable binning experiments
    print(f"\nLearnable Bin Experiments:")
    print("=" * 100)
    print(f"  {'Name':<30} {'Init':<12} {'Bins':>5} {'Shared':>7} {'Dim':>6} Description")
    print("-" * 100)
    for name, exp in LEARNABLE_EXPERIMENTS.items():
        cfg = exp['config_fn'](copy.deepcopy(dummy_base))
        sp = cfg['encoding'].get('spectral_policy', {})
        if sp.get('enabled'):
            sb = sp.get('soft_binning', {})
            init_mode = sb.get('init_mode', 'exp')
            n_bins = sb.get('n_soft_bins', '?')
            shared = 'yes' if sp.get('shared_across_rings', True) else 'no'
        else:
            init_mode = 'fixed'
            n_bins = effective_n_bins(cfg['encoding']['binning_strategy'],
                                     cfg['encoding']['n_bins'])
            shared = '-'
        input_dim = cfg['gnn']['input_dim']
        print(f"  {name:<30} {init_mode:<12} {n_bins:>5} {shared:>7} {input_dim:>6} {exp['description']}")

    # Inter-ring experiments
    print(f"\nInter-Ring Experiments:")
    print("=" * 100)
    print(f"  {'Name':<25} {'InterBin':<10} {'InterRing':<15} {'Dim':>6} Description")
    print("-" * 100)
    for name, exp in INTER_RING_EXPERIMENTS.items():
        cfg = exp['config_fn'](copy.deepcopy(dummy_base))
        sp = cfg['encoding'].get('spectral_policy', {})
        sb = sp.get('soft_binning', {})
        inter_stats = sb.get('inter_stats', [])
        ir = sb.get('inter_ring', {})
        ib_str = ','.join(inter_stats) if inter_stats else 'none'
        if ir.get('enabled'):
            ir_str = f"k={ir['channels']} ks={ir['kernel_size']}"
        else:
            ir_str = 'none'
        input_dim = cfg['gnn']['input_dim']
        print(f"  {name:<25} {ib_str:<10} {ir_str:<15} {input_dim:>6} {exp['description']}")

    # Regularization experiments
    print(f"\nRegularization Experiments:")
    print("=" * 100)
    print(f"  {'Name':<25} {'Norm':<12} {'Dropout':>8} {'Dim':>6} Description")
    print("-" * 100)
    for name, exp in REGULARIZATION_EXPERIMENTS.items():
        cfg = exp['config_fn'](copy.deepcopy(dummy_base))
        norm_type = cfg['gnn'].get('norm_type', 'batch_norm')
        dropout = cfg['gnn'].get('dropout', 0.1)
        input_dim = cfg['gnn']['input_dim']
        print(f"  {name:<25} {norm_type:<12} {dropout:>8.2f} {input_dim:>6} {exp['description']}")

    print(f"\nSuites:")
    for suite, exps in BIN_SUITES.items():
        print(f"  {suite:<15} ({len(exps)}): {', '.join(exps)}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Bin Configuration Benchmark')
    parser.add_argument('--config', default='configs/training_multi_dataset.yaml')
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--suite', choices=list(BIN_SUITES.keys()))
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--list', action='store_true')

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    results_dir = args.results_dir or f"results/bin_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.analyze_only:
        # Reuse analysis from benchmark_policies
        runner = BinBenchmarkRunner(args.config, results_dir)
        summary_path = Path(results_dir) / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                runner.results = json.load(f)
            runner._print_table()
        else:
            logging.error(f"No summary.json in {results_dir}")
        return

    if args.experiments:
        experiment_names = args.experiments
    elif args.suite:
        experiment_names = BIN_SUITES[args.suite]
    else:
        parser.error("Specify --suite or --experiments (or --list)")

    # Setup logging
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    logging.info(f"Bin Config Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Results: {results_dir}")
    logging.info(f"Experiments: {experiment_names}")
    if args.epochs:
        logging.info(f"Epochs: {args.epochs}")

    runner = BinBenchmarkRunner(
        base_config_path=args.config,
        results_dir=results_dir,
        n_epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )

    total_start = time.perf_counter()
    runner.run_suite(experiment_names)
    total_elapsed = time.perf_counter() - total_start

    logging.info(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    logging.info(f"Results: {results_dir}")


if __name__ == '__main__':
    main()
