"""KITTI 08 failure decomposition for paper Appendix K.

Consumes per-query dumps produced by `train_multi_dataset.py --validate-only
--dump-per-query-dir <dir>` and the existing `results/sim_edge_impact/summary.json`.

Outputs (under --output-dir):
  - summary.json                : aggregated statistics
  - table_failure_breakdown.tex : LaTeX table for Appendix K
  - fig_yaw_dist.pdf            : |Δyaw| histogram on KITTI 08
  - fig_descriptor_dist.pdf     : top-1 cosine sim, success vs failure split

Run example:
  python scripts/analyze_kitti08_failure.py \
    --per-query-dir results/per_query_v06 \
    --sim-edge-summary results/sim_edge_impact/summary.json \
    --output-dir results/kitti08_failure
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COMPARISON_SEQUENCES = ('KITTI_00', 'KITTI_05', 'KITTI_08')
TARGET = 'KITTI_08'


def load_per_query(per_query_dir: Path, dataset: str) -> dict:
    path = per_query_dir / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-query dump: {path}")
    with open(path) as f:
        return json.load(f)


def failure_breakdown(records: list) -> dict:
    """Decompose where the true match landed in the retrieval ranking."""
    n = len(records)
    if n == 0:
        return {}
    ranks = np.array([r['true_match_rank'] for r in records])
    geo = np.array([r['top1_geo_dist_m'] for r in records])
    cos = np.array([r['top1_cosine_sim'] for r in records])
    success_k1 = np.array([r['success_at_k1'] for r in records])

    # Note: success_at_k1 is geo-based (top-1 within 5m), not rank-based.
    # rank==1 means "the SPECIFIC pose used as gt-anchor was top-1"; we may
    # have other valid revisits within threshold. Both views are useful.
    return {
        'n_queries': int(n),
        'recall_at_1_geo': float(success_k1.mean()),
        'true_match_top1_rate': float((ranks == 1).mean()),
        'true_match_top5_rate': float(((ranks >= 1) & (ranks <= 5)).mean()),
        'true_match_top10_rate': float(((ranks >= 1) & (ranks <= 10)).mean()),
        'true_match_never_in_topK_rate': float((ranks == -1).mean()),
        'top1_cosine_mean': float(cos.mean()),
        'top1_cosine_std': float(cos.std()),
        'top1_geo_p50_m': float(np.median(geo)),
        'top1_geo_p90_m': float(np.percentile(geo, 90)),
    }


def yaw_breakdown(records: list) -> dict:
    if not records:
        return {}
    dyaw = np.abs(np.array([r['delta_yaw_deg'] for r in records]))
    return {
        'abs_dyaw_mean': float(dyaw.mean()),
        'abs_dyaw_p50': float(np.median(dyaw)),
        'abs_dyaw_p90': float(np.percentile(dyaw, 90)),
        'reverse_loop_rate': float((dyaw > 90.0).mean()),
        'forward_loop_rate': float((dyaw <= 30.0).mean()),
    }


def plot_yaw_histogram(records: list, out_path: Path) -> None:
    dyaw = np.abs(np.array([r['delta_yaw_deg'] for r in records]))
    success = np.array([r['success_at_k1'] for r in records])

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bins = np.arange(0, 181, 10)
    ax.hist(dyaw[success], bins=bins, color='#2c7fb8', alpha=0.65,
            label=f'success (n={success.sum()})')
    ax.hist(dyaw[~success], bins=bins, color='#d7301f', alpha=0.65,
            label=f'failure (n={(~success).sum()})')
    ax.axvline(90.0, color='black', linestyle='--', linewidth=0.8,
               label='reverse threshold')
    ax.set_xlabel(r'$|\Delta\mathrm{yaw}|$ (deg)')
    ax.set_ylabel('# queries')
    ax.set_title('KITTI 08: yaw difference at revisit')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_descriptor_distribution(records_by_seq: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    colors = {'KITTI_00': '#1b9e77', 'KITTI_05': '#7570b3', 'KITTI_08': '#d95f02'}
    for seq, records in records_by_seq.items():
        cos = np.array([r['top1_cosine_sim'] for r in records])
        ax.hist(cos, bins=30, alpha=0.55, color=colors.get(seq, 'gray'),
                label=f'{seq} (μ={cos.mean():.3f})', density=True)
    ax.set_xlabel('top-1 cosine similarity')
    ax.set_ylabel('density')
    ax.set_title('NSD+GNN top-1 cosine: 08 vs 00/05')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def emit_latex_table(per_seq: dict, sim_stats: dict, out_path: Path) -> None:
    rows = []
    for seq in COMPARISON_SEQUENCES:
        s = per_seq.get(seq, {})
        e = sim_stats.get(seq, {})
        rows.append(
            f"{seq.replace('_', ' ')} & {s.get('n_queries', 0)} & "
            f"{s.get('recall_at_1_geo', 0.0):.3f} & "
            f"{s.get('true_match_top1_rate', 0.0):.3f} & "
            f"{s.get('true_match_top10_rate', 0.0):.3f} & "
            f"{s.get('true_match_never_in_topK_rate', 0.0):.3f} & "
            f"{e.get('sim_per_node', 0.0):.2f} & "
            f"{e.get('raw_r1', 0.0):.3f} \\\\")

    body = '\n'.join(rows)
    table = (
        "\\begin{table}[h]\n"
        "\\caption{KITTI 08 failure decomposition (Appendix K). "
        "Columns: queries, R@1 (geo), top-1/top-10/never-in-top-K rate of the anchor true match, "
        "similarity edges per node, raw-encoder R@1.}\n"
        "\\label{tab:kitti08_failure}\n"
        "\\centering\\small\n"
        "\\begin{tabular}{lccccccc}\n"
        "\\toprule\n"
        "Seq & $|Q|$ & R@1 & top-1 & top-10 & never & sim/node & raw R@1 \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    out_path.write_text(table)


def load_sim_edge_stats(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {row['name']: row for row in data['datasets']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-query-dir', type=Path, required=True,
                        help='Directory of per-query JSON dumps (one per dataset).')
    parser.add_argument('--sim-edge-summary', type=Path,
                        default=Path('results/sim_edge_impact/summary.json'),
                        help='Existing sim-edge impact summary JSON.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Where to write summary.json + plots + LaTeX table.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim_stats = load_sim_edge_stats(args.sim_edge_summary)

    per_seq_summary = {}
    records_by_seq = {}
    for seq in COMPARISON_SEQUENCES:
        try:
            data = load_per_query(args.per_query_dir, seq)
        except FileNotFoundError as e:
            print(f"[warn] {e}; skipping {seq}")
            continue
        records = data['records']
        records_by_seq[seq] = records
        per_seq_summary[seq] = failure_breakdown(records)
        if seq == TARGET:
            per_seq_summary[seq]['yaw'] = yaw_breakdown(records)

    summary = {
        'per_sequence': per_seq_summary,
        'sim_edge_stats': {s: sim_stats.get(s) for s in COMPARISON_SEQUENCES},
        'target_sequence': TARGET,
    }
    summary_path = args.output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] {summary_path}")

    if TARGET in records_by_seq:
        plot_yaw_histogram(records_by_seq[TARGET],
                           args.output_dir / 'fig_yaw_dist.pdf')
        print(f"[ok] {args.output_dir / 'fig_yaw_dist.pdf'}")

    if records_by_seq:
        plot_descriptor_distribution(records_by_seq,
                                     args.output_dir / 'fig_descriptor_dist.pdf')
        print(f"[ok] {args.output_dir / 'fig_descriptor_dist.pdf'}")

    emit_latex_table(per_seq_summary, sim_stats,
                     args.output_dir / 'table_failure_breakdown.tex')
    print(f"[ok] {args.output_dir / 'table_failure_breakdown.tex'}")


if __name__ == '__main__':
    main()
