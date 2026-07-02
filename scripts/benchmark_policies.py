#!/usr/bin/env python3
"""
Spectral Policy Benchmark Runner

실험 계획 (Plan의 Exp 0~5 + Ablations):
  Exp 0: Baseline (fixed binning, no policy)
  Exp 1: SoftBinning (init_from_fixed=true)
  Exp 2: LearnedFilterbank (linear projection)
  Exp 3: ConvSpectralPool (1D conv)
  Exp 4: GatedFrequencySelection
  Exp 5: CrossAttentionPool

  Ablation A: Policy-only (GNN frozen)
  Ablation B: GNN-only (policy frozen, warmup=999)
  Ablation C: Per-ring vs shared (SoftBinning)
  Ablation D: Output dim sweep (512, 1106, 2048)
  Ablation E: lr_scale sweep (0.01, 0.1, 0.5, 1.0)

Usage:
  # 전체 벤치마크 (Exp 0~5)
  python scripts/benchmark_policies.py --suite main

  # Ablation만
  python scripts/benchmark_policies.py --suite ablation

  # 특정 실험만
  python scripts/benchmark_policies.py --experiments exp0 exp1_soft_binning

  # 짧은 테스트 (5 epochs)
  python scripts/benchmark_policies.py --suite main --epochs 5

  # 결과 분석만 (이미 실행된 결과)
  python scripts/benchmark_policies.py --analyze-only --results-dir results/benchmark
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
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# ============================================================================
# Experiment Definitions
# ============================================================================

def get_base_config(config_path: str) -> dict:
    """Load and return base config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_baseline_config(base: dict) -> dict:
    """Exp 0: Fixed binning baseline (no spectral policy)."""
    cfg = copy.deepcopy(base)
    cfg['encoding']['spectral_policy']['enabled'] = False
    return cfg


def make_policy_config(base: dict, policy_type: str, **overrides) -> dict:
    """Create a config with spectral policy enabled."""
    cfg = copy.deepcopy(base)
    sp = cfg['encoding']['spectral_policy']
    sp['enabled'] = True
    sp['type'] = policy_type

    # Apply overrides (e.g., lr_scale, warmup_epochs, output_dim, sub-config)
    for k, v in overrides.items():
        if k in sp:
            sp[k] = v
        elif '.' in k:
            # Nested: e.g., 'soft_binning.n_soft_bins'
            parts = k.split('.', 1)
            if parts[0] not in sp:
                sp[parts[0]] = {}
            sp[parts[0]][parts[1]] = v
        else:
            sp[k] = v

    return cfg


# ---- Main experiments ----

MAIN_EXPERIMENTS = {
    'exp0_baseline': {
        'description': 'Fixed binning baseline (no policy)',
        'config_fn': lambda base: make_baseline_config(base),
    },
    'exp1_soft_binning': {
        'description': 'SoftBinning (init_from_fixed=true, default params)',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning',
            lr_scale=0.1, warmup_epochs=3,
            **{'soft_binning.init_from_fixed': True, 'soft_binning.n_soft_bins': 4}
        ),
    },
    'exp2_linear': {
        'description': 'LearnedFilterbank (shared, d_per_ring=14)',
        'config_fn': lambda base: make_policy_config(
            base, 'linear',
            lr_scale=0.1, warmup_epochs=3,
            shared_across_rings=True,
            **{'linear.d_per_ring': 14}
        ),
    },
    'exp3_conv1d': {
        'description': 'ConvSpectralPool (kernel=7, channels=2)',
        'config_fn': lambda base: make_policy_config(
            base, 'conv1d',
            lr_scale=0.1, warmup_epochs=3,
            **{'conv1d.kernel_size': 7, 'conv1d.channels_per_group': 2}
        ),
    },
    'exp4_gated': {
        'description': 'GatedFrequencySelection (gate_hidden=64)',
        'config_fn': lambda base: make_policy_config(
            base, 'gated',
            lr_scale=0.1, warmup_epochs=3,
            **{'gated.gate_hidden': 64}
        ),
    },
    'exp5_attention': {
        'description': 'CrossAttentionPool (n_queries=7, n_heads=2)',
        'config_fn': lambda base: make_policy_config(
            base, 'attention',
            lr_scale=0.1, warmup_epochs=3,
            **{'attention.n_queries': 7, 'attention.n_heads': 2, 'attention.head_dim': 32}
        ),
    },
}

# ---- Ablation experiments ----

ABLATION_EXPERIMENTS = {
    'abl_a_policy_only': {
        'description': 'SoftBinning: policy-only (GNN lr=0)',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning',
            lr_scale=1.0, warmup_epochs=0,
        ),
        'post_config_fn': lambda cfg: _set_gnn_lr_zero(cfg),
    },
    'abl_b_gnn_only': {
        'description': 'SoftBinning: GNN-only (policy frozen)',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning',
            lr_scale=0.0, warmup_epochs=999,
        ),
    },
    'abl_c_per_ring': {
        'description': 'SoftBinning: per-ring (not shared)',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning',
            lr_scale=0.1, warmup_epochs=3,
            shared_across_rings=False,
        ),
    },
    'abl_d_dim512': {
        'description': 'SoftBinning: output_dim=512',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning',
            lr_scale=0.1, warmup_epochs=3, output_dim=512,
        ),
    },
    'abl_d_dim2048': {
        'description': 'SoftBinning: output_dim=2048',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning',
            lr_scale=0.1, warmup_epochs=3, output_dim=2048,
        ),
    },
    'abl_e_lr001': {
        'description': 'SoftBinning: lr_scale=0.01',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning', lr_scale=0.01, warmup_epochs=3,
        ),
    },
    'abl_e_lr05': {
        'description': 'SoftBinning: lr_scale=0.5',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning', lr_scale=0.5, warmup_epochs=3,
        ),
    },
    'abl_e_lr10': {
        'description': 'SoftBinning: lr_scale=1.0',
        'config_fn': lambda base: make_policy_config(
            base, 'soft_binning', lr_scale=1.0, warmup_epochs=3,
        ),
    },
    'abl_f_conv_k3': {
        'description': 'Conv1D: kernel_size=3 (local)',
        'config_fn': lambda base: make_policy_config(
            base, 'conv1d', lr_scale=0.1, warmup_epochs=3,
            **{'conv1d.kernel_size': 3}
        ),
    },
    'abl_f_conv_k15': {
        'description': 'Conv1D: kernel_size=15 (global)',
        'config_fn': lambda base: make_policy_config(
            base, 'conv1d', lr_scale=0.1, warmup_epochs=3,
            **{'conv1d.kernel_size': 15}
        ),
    },
}


def _set_gnn_lr_zero(cfg: dict) -> dict:
    """Post-config: mark GNN base lr as ~zero for policy-only ablation."""
    cfg['training']['learning_rate'] = 1e-10
    cfg['encoding']['spectral_policy']['lr_scale'] = 5e+6  # policy gets real LR
    return cfg


ALL_EXPERIMENTS = {**MAIN_EXPERIMENTS, **ABLATION_EXPERIMENTS}

SUITES = {
    'main': list(MAIN_EXPERIMENTS.keys()),
    'ablation': list(ABLATION_EXPERIMENTS.keys()),
    'all': list(ALL_EXPERIMENTS.keys()),
    'quick': ['exp0_baseline', 'exp1_soft_binning'],  # 빠른 비교
}


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Runs multiple training experiments and collects results."""

    def __init__(
        self,
        base_config_path: str,
        results_dir: str,
        n_epochs: int = None,
        device: str = 'cuda',
        seed: int = 42,
    ):
        self.base_config = get_base_config(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_epochs = n_epochs
        self.device = device
        self.seed = seed
        self.results = {}

    def run_experiment(self, exp_name: str, exp_def: dict) -> dict:
        """Run a single experiment and return metrics."""
        logging.info("=" * 80)
        logging.info(f"EXPERIMENT: {exp_name}")
        logging.info(f"  {exp_def['description']}")
        logging.info("=" * 80)

        # Generate config
        config = exp_def['config_fn'](self.base_config)
        if 'post_config_fn' in exp_def:
            config = exp_def['post_config_fn'](config)

        # Override epochs if specified
        if self.n_epochs is not None:
            config['training']['n_epochs'] = self.n_epochs

        # Set seed and device
        config['system']['seed'] = self.seed
        config['system']['device'] = self.device

        # Disable AMP on CPU (not supported)
        if self.device == 'cpu':
            config['training']['use_amp'] = False

        # Create experiment directory
        exp_dir = self.results_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Set checkpoint dir per experiment
        config['system']['checkpoint_dir'] = str(exp_dir / 'checkpoints')
        config['system']['log_dir'] = str(exp_dir / 'logs')
        os.makedirs(config['system']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['system']['log_dir'], exist_ok=True)

        # Run training
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

        # Save metrics
        metrics_path = exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.results[exp_name] = metrics
        self._save_summary()

        return metrics

    def _run_training(self, config: dict, exp_name: str, exp_dir: Path) -> dict:
        """Run the actual training pipeline and extract metrics."""
        import subprocess

        # Determine project root (parent of scripts/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        checkpoint_dir = config['system']['checkpoint_dir']

        # Use absolute path for train_multi_dataset.py
        train_script = os.path.join(project_root, 'train_multi_dataset.py')

        cmd = [
            sys.executable, train_script,
            '--config', str(config_path),
            '--checkpoint-dir', checkpoint_dir,
        ]

        logging.info(f"  Running: {' '.join(cmd)}")

        # Run in subprocess for GPU memory isolation
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

        # Stream output and capture log
        log_lines = []
        log_path = exp_dir / 'training.log'
        with open(log_path, 'w') as log_f:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                log_lines.append(line.rstrip())
                # Print key lines to main console
                if any(kw in line for kw in ['Epoch ', 'AVERAGE', 'best model', 'Early stopping', 'TRAINING COMPLETE']):
                    logging.info(f"  [{exp_name}] {line.rstrip()}")

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Training subprocess exited with code {proc.returncode}. "
                f"Check {log_path} for details."
            )

        # Parse metrics from log
        metrics = self._parse_training_log(log_lines, config)
        metrics['log_path'] = str(log_path)
        metrics['config_path'] = str(config_path)

        return metrics

    def _parse_training_log(self, log_lines: list, config: dict) -> dict:
        """Parse training log to extract key metrics."""
        metrics = {
            'policy_type': config['encoding']['spectral_policy'].get('type', 'none'),
            'policy_enabled': config['encoding']['spectral_policy'].get('enabled', False),
            'output_dim': config['encoding']['spectral_policy'].get('output_dim', config['gnn']['input_dim']),
            'lr_scale': config['encoding']['spectral_policy'].get('lr_scale', 1.0),
            'warmup_epochs': config['encoding']['spectral_policy'].get('warmup_epochs', 0),
            'n_epochs_config': config['training']['n_epochs'],
        }

        # Parse epoch-by-epoch metrics
        epoch_metrics = []
        best_recall_at_1 = 0.0
        best_epoch = -1
        final_loss = None
        n_epochs_actual = 0

        for line in log_lines:
            # Parse: Epoch X/Y | Loss: 0.XXXX | Avg R@1: 0.XXXX
            if 'Epoch ' in line and '| Loss:' in line:
                try:
                    parts = line.split('|')
                    epoch_part = [p for p in parts if 'Epoch' in p][0]
                    loss_part = [p for p in parts if 'Loss:' in p][0]
                    epoch_str = epoch_part.strip().split()[-1].split('/')[0]
                    epoch_num = int(epoch_str)
                    loss_val = float(loss_part.strip().split()[-1])
                    n_epochs_actual = epoch_num

                    entry = {'epoch': epoch_num, 'loss': loss_val}

                    r1_parts = [p for p in parts if 'R@1:' in p]
                    if r1_parts:
                        r1_val = float(r1_parts[0].strip().split()[-1])
                        entry['avg_recall@1'] = r1_val
                        if r1_val > best_recall_at_1:
                            best_recall_at_1 = r1_val
                            best_epoch = epoch_num

                    final_loss = loss_val
                    epoch_metrics.append(entry)
                except (ValueError, IndexError):
                    continue

            # Parse per-dataset validation: KITTI_00 | R@1: 0.XXXX (raw: 0.XXXX, ctx: 0.XXXX) | R@5: 0.XXXX
            if '| R@1:' in line and 'AVERAGE' not in line and 'Epoch' not in line:
                try:
                    parts = line.split('|')
                    dataset_name = parts[0].strip().split()[-1]
                    r1_str = parts[1].strip()

                    # Extract R@1 components
                    import re
                    r1_match = re.search(r'R@1:\s*([\d.]+)', r1_str)
                    raw_match = re.search(r'raw:\s*([\d.]+)', r1_str)
                    ctx_match = re.search(r'ctx:\s*([\d.]+)', r1_str)
                    policy_match = re.search(r'policy:\s*([\d.]+)', r1_str)
                    r5_match = re.search(r'R@5:\s*([\d.]+)', parts[2].strip() if len(parts) > 2 else '')

                    key_prefix = f'last_val_{dataset_name}'
                    if r1_match:
                        metrics[f'{key_prefix}_recall@1'] = float(r1_match.group(1))
                    if raw_match:
                        metrics[f'{key_prefix}_raw_recall@1'] = float(raw_match.group(1))
                    if ctx_match:
                        metrics[f'{key_prefix}_ctx_recall@1'] = float(ctx_match.group(1))
                    if policy_match:
                        metrics[f'{key_prefix}_policy_recall@1'] = float(policy_match.group(1))
                    if r5_match:
                        metrics[f'{key_prefix}_recall@5'] = float(r5_match.group(1))
                except (ValueError, IndexError):
                    continue

            # Parse AVERAGE line
            if 'AVERAGE' in line and 'R@1:' in line:
                try:
                    import re
                    r1_match = re.search(r'R@1:\s*([\d.]+)', line)
                    raw_match = re.search(r'raw:\s*([\d.]+)', line)
                    if r1_match:
                        metrics['last_avg_recall@1'] = float(r1_match.group(1))
                    if raw_match:
                        metrics['last_avg_raw_recall@1'] = float(raw_match.group(1))
                except (ValueError, IndexError):
                    continue

            # Parse: New best model! Avg R@1: X.XXXX
            if 'New best model' in line:
                try:
                    import re
                    m = re.search(r'R@1:\s*([\d.]+)', line)
                    if m:
                        best_recall_at_1 = float(m.group(1))
                except (ValueError, IndexError):
                    continue

            # Parse GNN parameters count
            if 'GNN parameters:' in line:
                try:
                    import re
                    m = re.search(r'([\d,]+)', line.split('GNN parameters:')[1])
                    if m:
                        metrics['n_params'] = int(m.group(1).replace(',', ''))
                except (ValueError, IndexError):
                    continue

            # Parse policy params
            if 'Spectral policy:' in line and 'params=' in line:
                try:
                    import re
                    m = re.search(r'params=([\d,]+)', line)
                    if m:
                        metrics['policy_params'] = int(m.group(1).replace(',', ''))
                except (ValueError, IndexError):
                    continue

        metrics['best_recall@1'] = best_recall_at_1
        metrics['best_epoch'] = best_epoch
        metrics['final_loss'] = final_loss
        metrics['n_epochs_actual'] = n_epochs_actual
        metrics['epoch_history'] = epoch_metrics

        return metrics

    def _save_summary(self):
        """Save running summary of all experiments."""
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def run_suite(self, experiment_names: list):
        """Run a list of experiments sequentially."""
        logging.info(f"Running {len(experiment_names)} experiments")
        logging.info(f"Results directory: {self.results_dir}")

        for i, name in enumerate(experiment_names):
            if name not in ALL_EXPERIMENTS:
                logging.warning(f"Unknown experiment: {name}, skipping")
                continue

            logging.info(f"\n{'='*80}")
            logging.info(f"[{i+1}/{len(experiment_names)}] Starting: {name}")
            logging.info(f"{'='*80}\n")

            # Skip if already completed
            metrics_path = self.results_dir / name / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path) as f:
                    existing = json.load(f)
                if existing.get('status') == 'success':
                    logging.info(f"  Skipping {name} (already completed, "
                                 f"best R@1={existing.get('best_recall@1', '?')})")
                    self.results[name] = existing
                    continue

            self.run_experiment(name, ALL_EXPERIMENTS[name])

            # Force GPU memory cleanup between experiments
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        self._save_summary()
        self._print_comparison_table()

    def _print_comparison_table(self):
        """Print comparison table of all results."""
        logging.info("\n" + "=" * 100)
        logging.info("BENCHMARK RESULTS COMPARISON")
        logging.info("=" * 100)

        header = (
            f"{'Experiment':<25} {'Policy':<15} {'Best R@1':>10} {'Raw R@1':>10} "
            f"{'Final Loss':>12} {'Epoch':>6} {'Params':>10} {'Time':>10} {'Status':<8}"
        )
        logging.info(header)
        logging.info("-" * 100)

        # Sort by best_recall@1 descending
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('best_recall@1', 0),
            reverse=True
        )

        for name, m in sorted_results:
            policy = m.get('policy_type', 'none') if m.get('policy_enabled') else 'fixed'
            best_r1 = m.get('best_recall@1', 0)
            raw_r1 = m.get('last_avg_raw_recall@1', 0)
            final_loss = m.get('final_loss', 0)
            best_epoch = m.get('best_epoch', -1)
            n_params = m.get('n_params', 0)
            elapsed = m.get('elapsed_seconds', 0)
            status = m.get('status', 'unknown')

            time_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
            params_str = f"{n_params:,}" if n_params else "?"

            logging.info(
                f"{name:<25} {policy:<15} {best_r1:>10.4f} {raw_r1:>10.4f} "
                f"{final_loss:>12.4f} {best_epoch:>6} {params_str:>10} {time_str:>10} {status:<8}"
            )

        logging.info("=" * 100)

        # Save table as CSV
        csv_path = self.results_dir / 'comparison.csv'
        with open(csv_path, 'w') as f:
            f.write('experiment,policy_type,policy_enabled,best_recall@1,last_avg_raw_recall@1,'
                    'final_loss,best_epoch,n_params,policy_params,elapsed_seconds,status\n')
            for name, m in sorted_results:
                f.write(
                    f"{name},{m.get('policy_type','none')},{m.get('policy_enabled',False)},"
                    f"{m.get('best_recall@1',0):.6f},{m.get('last_avg_raw_recall@1',0):.6f},"
                    f"{m.get('final_loss',0):.6f},{m.get('best_epoch',-1)},"
                    f"{m.get('n_params',0)},{m.get('policy_params',0)},"
                    f"{m.get('elapsed_seconds',0):.1f},{m.get('status','unknown')}\n"
                )
        logging.info(f"\nCSV saved to: {csv_path}")


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(results_dir: str):
    """Analyze and compare results from completed benchmark runs."""
    results_dir = Path(results_dir)
    summary_path = results_dir / 'summary.json'

    if not summary_path.exists():
        # Rebuild from individual metrics
        results = {}
        for exp_dir in sorted(results_dir.iterdir()):
            metrics_path = exp_dir / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path) as f:
                    results[exp_dir.name] = json.load(f)
    else:
        with open(summary_path) as f:
            results = json.load(f)

    if not results:
        print("No results found.")
        return

    # Print main comparison table
    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 110)

    header = (
        f"{'Experiment':<25} {'Policy':<15} {'Best R@1':>10} {'Raw R@1':>10} "
        f"{'Policy R@1':>12} {'Loss':>10} {'Epoch':>6} {'Params':>10} {'Time':>10}"
    )
    print(header)
    print("-" * 110)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    baseline_r1 = None
    for name, m in sorted_results:
        if m.get('status') != 'success':
            continue
        if not m.get('policy_enabled', False):
            baseline_r1 = m.get('best_recall@1', 0)
            break

    for name, m in sorted_results:
        if m.get('status') != 'success':
            status_str = f"  [FAILED: {m.get('error', '?')[:40]}]"
            print(f"{name:<25} {status_str}")
            continue

        policy = m.get('policy_type', 'none') if m.get('policy_enabled') else 'fixed'
        best_r1 = m.get('best_recall@1', 0)
        raw_r1 = m.get('last_avg_raw_recall@1', 0)
        final_loss = m.get('final_loss', 0)
        best_epoch = m.get('best_epoch', -1)
        n_params = m.get('n_params', 0)
        elapsed = m.get('elapsed_seconds', 0)

        # Policy R@1 (last logged)
        policy_r1_keys = [k for k in m.keys() if 'policy_recall@1' in k and 'last_val' in k]
        policy_r1 = np.mean([m[k] for k in policy_r1_keys]) if policy_r1_keys else None

        time_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
        params_str = f"{n_params:,}" if n_params else "?"
        policy_r1_str = f"{policy_r1:.4f}" if policy_r1 is not None else "n/a"

        # Delta vs baseline
        delta = ""
        if baseline_r1 is not None and m.get('policy_enabled', False):
            d = best_r1 - baseline_r1
            delta = f" ({'+' if d >= 0 else ''}{d:.4f})"

        print(
            f"{name:<25} {policy:<15} {best_r1:>10.4f}{delta:>0} {raw_r1:>10.4f} "
            f"{policy_r1_str:>12} {final_loss:>10.4f} {best_epoch:>6} {params_str:>10} {time_str:>10}"
        )

    print("=" * 110)

    # Per-dataset breakdown for top 3
    print("\n" + "-" * 80)
    print("PER-DATASET BREAKDOWN (Top 3 experiments)")
    print("-" * 80)

    top3 = [name for name, m in sorted_results if m.get('status') == 'success'][:3]
    datasets = set()
    for name in top3:
        m = results[name]
        for k in m.keys():
            if k.startswith('last_val_') and k.endswith('_recall@1') and 'raw' not in k and 'ctx' not in k and 'policy' not in k:
                ds = k.replace('last_val_', '').replace('_recall@1', '')
                datasets.add(ds)

    datasets = sorted(datasets)
    if datasets:
        header = f"{'Experiment':<25}" + "".join(f" {ds:>15}" for ds in datasets)
        print(header)
        print("-" * (25 + 16 * len(datasets)))

        for name in top3:
            m = results[name]
            vals = []
            for ds in datasets:
                v = m.get(f'last_val_{ds}_recall@1', None)
                vals.append(f"{v:.4f}" if v is not None else "n/a")
            print(f"{name:<25}" + "".join(f" {v:>15}" for v in vals))

    # Convergence comparison
    print("\n" + "-" * 80)
    print("CONVERGENCE (loss at epoch 1, 5, 10, 20, best)")
    print("-" * 80)

    header = f"{'Experiment':<25} {'E1':>8} {'E5':>8} {'E10':>8} {'E20':>8} {'Best':>8}"
    print(header)

    for name, m in sorted_results[:6]:
        if m.get('status') != 'success':
            continue
        history = m.get('epoch_history', [])
        if not history:
            continue
        losses = {e['epoch']: e['loss'] for e in history}
        vals = []
        for ep in [1, 5, 10, 20]:
            v = losses.get(ep)
            vals.append(f"{v:.4f}" if v is not None else "n/a")
        best_loss = min(e['loss'] for e in history) if history else None
        vals.append(f"{best_loss:.4f}" if best_loss is not None else "n/a")
        print(f"{name:<25}" + "".join(f" {v:>8}" for v in vals))

    print()


# ============================================================================
# Main
# ============================================================================

def setup_logging(log_dir: str):
    """Setup logging."""
    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    log_file = os.path.join(log_dir, f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description='Spectral Policy Benchmark Runner')
    parser.add_argument('--config', type=str,
                        default='configs/training_multi_dataset.yaml',
                        help='Base config file')
    parser.add_argument('--results-dir', type=str,
                        default=None,
                        help='Results directory (default: results/benchmark_YYYYMMDD_HHMMSS)')
    parser.add_argument('--suite', type=str, choices=list(SUITES.keys()),
                        default=None, help='Experiment suite to run')
    parser.add_argument('--experiments', nargs='+', type=str, default=None,
                        help='Specific experiments to run')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        print("-" * 70)
        for name, exp in ALL_EXPERIMENTS.items():
            print(f"  {name:<30} {exp['description']}")
        print(f"\nSuites:")
        for suite, exps in SUITES.items():
            print(f"  {suite:<15} ({len(exps)} experiments): {', '.join(exps[:3])}...")
        return

    # Results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.analyze_only:
        analyze_results(results_dir)
        return

    # Determine experiments to run
    if args.experiments:
        experiment_names = args.experiments
    elif args.suite:
        experiment_names = SUITES[args.suite]
    else:
        parser.error("Specify --suite or --experiments (or --list to see options)")

    setup_logging(os.path.join(results_dir, 'logs'))

    logging.info(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Base config: {args.config}")
    logging.info(f"Results dir: {results_dir}")
    logging.info(f"Experiments: {experiment_names}")
    if args.epochs:
        logging.info(f"Epoch override: {args.epochs}")

    runner = BenchmarkRunner(
        base_config_path=args.config,
        results_dir=results_dir,
        n_epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )

    total_start = time.perf_counter()
    runner.run_suite(experiment_names)
    total_elapsed = time.perf_counter() - total_start

    logging.info(f"\nTotal benchmark time: {total_elapsed/3600:.2f} hours")
    logging.info(f"Results saved to: {results_dir}")

    # Final analysis
    analyze_results(results_dir)


if __name__ == '__main__':
    main()
