#!/usr/bin/env python3
"""
Benchmark Results Analyzer & Visualizer

벤치마크 결과를 분석하고 비교 차트를 생성합니다.

Usage:
  # 결과 분석 + 차트 생성
  python scripts/analyze_benchmark.py results/benchmark_XXXXXXXX

  # 특정 결과 디렉토리 비교
  python scripts/analyze_benchmark.py results/benchmark_A results/benchmark_B

  # CSV만 출력
  python scripts/analyze_benchmark.py results/benchmark_XXXXXXXX --csv

  # LaTeX 테이블 생성
  python scripts/analyze_benchmark.py results/benchmark_XXXXXXXX --latex
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# Data Loading
# ============================================================================

def load_results(results_dir: str) -> dict:
    """Load all experiment results from a benchmark directory."""
    results_dir = Path(results_dir)
    results = {}

    # Try summary.json first
    summary_path = results_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    # Fall back to individual metrics.json files
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                results[exp_dir.name] = json.load(f)

    return results


def filter_successful(results: dict) -> dict:
    """Filter to successful experiments only."""
    return {k: v for k, v in results.items() if v.get('status') == 'success'}


# ============================================================================
# Tables
# ============================================================================

def print_main_table(results: dict):
    """Print main comparison table to stdout."""
    results = filter_successful(results)
    if not results:
        print("No successful experiments found.")
        return

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    # Find baseline for delta
    baseline_r1 = None
    for name, m in sorted_results:
        if not m.get('policy_enabled', False):
            baseline_r1 = m.get('best_recall@1', 0)
            break

    print("\n" + "=" * 120)
    print("                        SPECTRAL POLICY BENCHMARK COMPARISON")
    print("=" * 120)

    header = (
        f"{'#':>2} {'Experiment':<26} {'Policy':<14} "
        f"{'Best R@1':>9} {'Delta':>8} {'Raw R@1':>9} {'Pol R@1':>9} "
        f"{'Loss':>8} {'Ep':>4} {'Params':>9} {'Time':>8}"
    )
    print(header)
    print("-" * 120)

    for rank, (name, m) in enumerate(sorted_results, 1):
        policy = m.get('policy_type', 'none') if m.get('policy_enabled') else 'fixed'
        best_r1 = m.get('best_recall@1', 0)
        raw_r1 = m.get('last_avg_raw_recall@1', 0)
        final_loss = m.get('final_loss', 0) or 0
        best_epoch = m.get('best_epoch', -1)
        n_params = m.get('n_params', 0)
        elapsed = m.get('elapsed_seconds', 0) or 0

        # Policy R@1
        policy_r1_keys = [k for k in m.keys() if 'policy_recall@1' in k and 'last_val' in k]
        policy_r1 = np.mean([m[k] for k in policy_r1_keys]) if policy_r1_keys else None

        # Delta
        delta_str = ""
        if baseline_r1 is not None:
            d = best_r1 - baseline_r1
            if m.get('policy_enabled', False):
                delta_str = f"{'+'if d>=0 else ''}{d:.4f}"
            else:
                delta_str = "(base)"

        time_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.0f}m"
        params_str = f"{n_params:,}" if n_params else "?"
        pol_str = f"{policy_r1:.4f}" if policy_r1 is not None else "n/a"

        print(
            f"{rank:>2} {name:<26} {policy:<14} "
            f"{best_r1:>9.4f} {delta_str:>8} {raw_r1:>9.4f} {pol_str:>9} "
            f"{final_loss:>8.4f} {best_epoch:>4} {params_str:>9} {time_str:>8}"
        )

    print("=" * 120)


def print_per_dataset_table(results: dict):
    """Print per-dataset R@1 breakdown."""
    results = filter_successful(results)
    if not results:
        return

    # Collect all dataset names
    datasets = set()
    for m in results.values():
        for k in m.keys():
            if k.startswith('last_val_') and k.endswith('_recall@1') and \
               'raw' not in k and 'ctx' not in k and 'policy' not in k:
                ds = k.replace('last_val_', '').replace('_recall@1', '')
                datasets.add(ds)
    datasets = sorted(datasets)

    if not datasets:
        return

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    print("\n" + "=" * (30 + 13 * len(datasets)))
    print("PER-DATASET R@1 (GNN-enhanced)")
    print("-" * (30 + 13 * len(datasets)))

    header = f"{'Experiment':<30}" + "".join(f" {ds:>12}" for ds in datasets)
    print(header)
    print("-" * (30 + 13 * len(datasets)))

    for name, m in sorted_results:
        vals = []
        for ds in datasets:
            v = m.get(f'last_val_{ds}_recall@1')
            vals.append(f"{v:.4f}" if v is not None else "n/a")
        print(f"{name:<30}" + "".join(f" {v:>12}" for v in vals))

    # Also print raw baseline
    print()
    print("PER-DATASET R@1 (Raw baseline)")
    print("-" * (30 + 13 * len(datasets)))
    print(header)
    print("-" * (30 + 13 * len(datasets)))

    for name, m in sorted_results:
        vals = []
        for ds in datasets:
            v = m.get(f'last_val_{ds}_raw_recall@1')
            vals.append(f"{v:.4f}" if v is not None else "n/a")
        print(f"{name:<30}" + "".join(f" {v:>12}" for v in vals))

    print("=" * (30 + 13 * len(datasets)))


def print_convergence_table(results: dict):
    """Print convergence comparison (loss and R@1 at key epochs)."""
    results = filter_successful(results)
    if not results:
        return

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    print("\n" + "=" * 90)
    print("CONVERGENCE: R@1 at Epoch 1, 5, 10, 20, Final")
    print("-" * 90)
    print(f"{'Experiment':<26} {'E1':>10} {'E5':>10} {'E10':>10} {'E20':>10} {'Best':>10}")
    print("-" * 90)

    for name, m in sorted_results:
        history = m.get('epoch_history', [])
        if not history:
            continue
        r1_map = {e['epoch']: e.get('avg_recall@1') for e in history}
        vals = []
        for ep in [1, 5, 10, 20]:
            v = r1_map.get(ep)
            vals.append(f"{v:.4f}" if v is not None else "-")
        best = m.get('best_recall@1', 0)
        vals.append(f"{best:.4f}")
        print(f"{name:<26}" + "".join(f" {v:>10}" for v in vals))

    print("\nCONVERGENCE: Loss at Epoch 1, 5, 10, 20, Final")
    print("-" * 90)
    print(f"{'Experiment':<26} {'E1':>10} {'E5':>10} {'E10':>10} {'E20':>10} {'Final':>10}")
    print("-" * 90)

    for name, m in sorted_results:
        history = m.get('epoch_history', [])
        if not history:
            continue
        loss_map = {e['epoch']: e.get('loss') for e in history}
        vals = []
        for ep in [1, 5, 10, 20]:
            v = loss_map.get(ep)
            vals.append(f"{v:.4f}" if v is not None else "-")
        final_loss = m.get('final_loss', 0)
        vals.append(f"{final_loss:.4f}" if final_loss else "-")
        print(f"{name:<26}" + "".join(f" {v:>10}" for v in vals))

    print("=" * 90)


# ============================================================================
# CSV / LaTeX Export
# ============================================================================

def export_csv(results: dict, output_path: str):
    """Export results to CSV."""
    results = filter_successful(results)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    with open(output_path, 'w') as f:
        f.write(
            'rank,experiment,policy_type,policy_enabled,output_dim,'
            'best_recall@1,last_avg_raw_recall@1,final_loss,'
            'best_epoch,n_epochs_actual,n_params,policy_params,'
            'lr_scale,warmup_epochs,elapsed_seconds\n'
        )
        for rank, (name, m) in enumerate(sorted_results, 1):
            f.write(
                f"{rank},{name},{m.get('policy_type','none')},"
                f"{m.get('policy_enabled',False)},{m.get('output_dim',0)},"
                f"{m.get('best_recall@1',0):.6f},{m.get('last_avg_raw_recall@1',0):.6f},"
                f"{m.get('final_loss',0):.6f},"
                f"{m.get('best_epoch',-1)},{m.get('n_epochs_actual',0)},"
                f"{m.get('n_params',0)},{m.get('policy_params',0)},"
                f"{m.get('lr_scale',1.0)},{m.get('warmup_epochs',0)},"
                f"{m.get('elapsed_seconds',0):.1f}\n"
            )
    print(f"CSV saved to: {output_path}")


def export_latex(results: dict, output_path: str):
    """Export results as LaTeX table."""
    results = filter_successful(results)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    # Find best R@1 for bold highlighting
    best_r1 = max(m.get('best_recall@1', 0) for m in results.values()) if results else 0

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Spectral Policy Benchmark Results}',
        r'\label{tab:benchmark}',
        r'\begin{tabular}{l l r r r r r}',
        r'\toprule',
        r'Experiment & Policy & Best R@1 & Raw R@1 & Loss & Epoch & Params \\',
        r'\midrule',
    ]

    for name, m in sorted_results:
        policy = m.get('policy_type', 'none') if m.get('policy_enabled') else 'fixed'
        r1 = m.get('best_recall@1', 0)
        raw_r1 = m.get('last_avg_raw_recall@1', 0)
        loss = m.get('final_loss', 0) or 0
        epoch = m.get('best_epoch', -1)
        params = m.get('n_params', 0)

        # Escape underscores for LaTeX
        name_tex = name.replace('_', r'\_')
        policy_tex = policy.replace('_', r'\_')

        r1_str = f"\\textbf{{{r1:.4f}}}" if abs(r1 - best_r1) < 1e-6 else f"{r1:.4f}"
        params_str = f"{params:,}" if params else "?"

        lines.append(
            f"{name_tex} & {policy_tex} & {r1_str} & {raw_r1:.4f} & "
            f"{loss:.4f} & {epoch} & {params_str} \\\\"
        )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table saved to: {output_path}")


# ============================================================================
# Plots
# ============================================================================

def plot_recall_comparison(results: dict, output_dir: str):
    """Bar chart comparing Best R@1 across experiments."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots.")
        return

    results = filter_successful(results)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    names = [n for n, _ in sorted_results]
    best_r1 = [m.get('best_recall@1', 0) for _, m in sorted_results]
    raw_r1 = [m.get('last_avg_raw_recall@1', 0) for _, m in sorted_results]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_r1, width, label='GNN R@1', color='#2196F3')
    bars2 = ax.bar(x + width/2, raw_r1, width, label='Raw R@1', color='#FF9800', alpha=0.7)

    ax.set_ylabel('Recall@1')
    ax.set_title('Spectral Policy Benchmark: R@1 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, max(max(best_r1), max(raw_r1)) * 1.15)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'recall_comparison.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {out_path}")


def plot_convergence(results: dict, output_dir: str):
    """Training curves: loss and R@1 over epochs."""
    if not HAS_MPL:
        return

    results = filter_successful(results)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_recall@1', 0),
        reverse=True
    )

    # Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(sorted_results), 10)))

    for idx, (name, m) in enumerate(sorted_results[:10]):
        history = m.get('epoch_history', [])
        if not history:
            continue
        epochs = [e['epoch'] for e in history]
        losses = [e['loss'] for e in history]
        r1_vals = [e.get('avg_recall@1', 0) for e in history]

        ax1.plot(epochs, losses, label=name, color=colors[idx], linewidth=1.5)
        if any(v > 0 for v in r1_vals):
            ax2.plot(epochs, r1_vals, label=name, color=colors[idx], linewidth=1.5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('InfoNCE Loss')
    ax1.set_title('Training Loss Convergence')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg R@1')
    ax2.set_title('Validation R@1 Convergence')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'convergence.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {out_path}")


def plot_param_efficiency(results: dict, output_dir: str):
    """Scatter: R@1 vs parameter count (efficiency frontier)."""
    if not HAS_MPL:
        return

    results = filter_successful(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, m in results.items():
        n_params = m.get('n_params', 0)
        r1 = m.get('best_recall@1', 0)
        policy = m.get('policy_type', 'none') if m.get('policy_enabled') else 'fixed'

        if n_params == 0:
            continue

        marker = 'o' if m.get('policy_enabled') else 's'
        ax.scatter(n_params, r1, s=80, marker=marker, zorder=5)
        ax.annotate(name, (n_params, r1), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel('Total Parameters')
    ax.set_ylabel('Best R@1')
    ax.set_title('Parameter Efficiency: R@1 vs Model Size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'param_efficiency.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {out_path}")


def plot_ablation_grid(results: dict, output_dir: str):
    """Ablation grid: lr_scale and output_dim effects."""
    if not HAS_MPL:
        return

    results = filter_successful(results)

    # lr_scale ablation
    lr_exps = {k: v for k, v in results.items() if k.startswith('abl_e_') or k == 'exp1_soft_binning'}
    if len(lr_exps) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        lr_scales = []
        r1_vals = []
        for name, m in sorted(lr_exps.items()):
            lr = m.get('lr_scale', 0.1)
            r1 = m.get('best_recall@1', 0)
            lr_scales.append(lr)
            r1_vals.append(r1)
            ax.annotate(name, (lr, r1), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))

        ax.plot(lr_scales, r1_vals, 'o-', markersize=8)
        ax.set_xlabel('Policy LR Scale')
        ax.set_ylabel('Best R@1')
        ax.set_title('Ablation: Policy Learning Rate')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(output_dir, 'ablation_lr_scale.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {out_path}")

    # output_dim ablation
    dim_exps = {k: v for k, v in results.items() if k.startswith('abl_d_') or k == 'exp1_soft_binning'}
    if len(dim_exps) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        dims = []
        r1_vals = []
        for name, m in sorted(dim_exps.items()):
            dim = m.get('output_dim', 1106)
            r1 = m.get('best_recall@1', 0)
            dims.append(dim)
            r1_vals.append(r1)
            ax.annotate(name, (dim, r1), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))

        ax.plot(dims, r1_vals, 's-', markersize=8, color='#E91E63')
        ax.set_xlabel('Output Dimension')
        ax.set_ylabel('Best R@1')
        ax.set_title('Ablation: Descriptor Dimension')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(output_dir, 'ablation_output_dim.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('results_dirs', nargs='+', type=str,
                        help='Results directory/directories to analyze')
    parser.add_argument('--csv', type=str, default=None,
                        help='Export to CSV (path)')
    parser.add_argument('--latex', type=str, default=None,
                        help='Export to LaTeX (path)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')

    args = parser.parse_args()

    # Load and merge results from all directories
    all_results = {}
    for rd in args.results_dirs:
        results = load_results(rd)
        all_results.update(results)

    if not all_results:
        print("No results found.")
        return

    successful = filter_successful(all_results)
    print(f"\nLoaded {len(all_results)} experiments ({len(successful)} successful)")

    # Print tables
    print_main_table(all_results)
    print_per_dataset_table(all_results)
    print_convergence_table(all_results)

    # Exports
    output_dir = args.results_dirs[0]

    if args.csv:
        export_csv(all_results, args.csv)
    else:
        export_csv(all_results, os.path.join(output_dir, 'results.csv'))

    if args.latex:
        export_latex(all_results, args.latex)

    # Plots
    if not args.no_plots and HAS_MPL:
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_recall_comparison(all_results, plots_dir)
        plot_convergence(all_results, plots_dir)
        plot_param_efficiency(all_results, plots_dir)
        plot_ablation_grid(all_results, plots_dir)
        print(f"\nPlots saved to: {plots_dir}/")


if __name__ == '__main__':
    main()
