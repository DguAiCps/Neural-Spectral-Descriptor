"""Octave binning panel — analogous to fig2's Gaussian binning visualization.

Hard octave binning used in `no_interdiff` ablation preset.
Each frequency f is assigned to exactly one bin (no overlap).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def octave_edges(n_freqs=181):
    edges = [0]
    p = 0
    while 2 ** p < n_freqs:
        edges.append(2 ** p)
        p += 1
    edges.append(n_freqs)
    return edges  # [0, 1, 2, 4, 8, 16, 32, 64, 128, 181]


def gaussian_panel(ax, n_freqs=181, n_bins=9):
    """Reference: soft Gaussian binning (paper canonical)."""
    centers = np.array([0, 1, 2.5, 6, 12, 24, 48, 96, 154])
    widths  = np.array([0.5, 0.5, 1, 2, 4, 8, 16, 32, 27])
    f = np.linspace(0, n_freqs - 1, 1000)

    cmap = plt.cm.viridis
    for i, (c, w) in enumerate(zip(centers, widths)):
        weight = np.exp(-0.5 * ((f - c) / w) ** 2)
        weight /= weight.max()
        ax.plot(f, weight, color=cmap(i / (n_bins - 1)), lw=1.6)
        ax.fill_between(f, 0, weight, color=cmap(i / (n_bins - 1)), alpha=0.15)
    ax.set_xlim(0, n_freqs)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('frequency index $f$', fontsize=11)
    ax.set_ylabel('bin weight $W_b(f)$', fontsize=11)
    ax.set_title('(a) Soft Gaussian binning  —  paper canonical (learnable)',
                 fontsize=11)
    ax.grid(alpha=0.3)


def octave_panel(ax, n_freqs=181):
    """Hard octave binning (current ablation: no_interdiff preset)."""
    edges = octave_edges(n_freqs)
    n_bins = len(edges) - 1

    cmap = plt.cm.viridis
    # Step functions: each bin = rectangle of weight=1 in its frequency range
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Filled rectangle
        ax.add_patch(Rectangle(
            (lo, 0), hi - lo, 1.0,
            facecolor=cmap(i / (n_bins - 1)), alpha=0.45,
            edgecolor=cmap(i / (n_bins - 1)), lw=1.6,
        ))
        # Bin label
        x_label = (lo + hi) / 2
        ax.text(max(x_label, lo + 0.3), 1.05, f'bin {i}',
                ha='center', va='bottom', fontsize=8, color='black')

    # Vertical lines at edges
    for e in edges:
        ax.axvline(e, color='gray', lw=0.5, alpha=0.4)

    # Annotations (boundaries)
    edge_labels = [str(e) for e in edges]
    for i, e in enumerate(edges):
        ax.text(e, -0.10, edge_labels[i], ha='center', va='top',
                fontsize=8, color='gray')

    ax.set_xlim(0, n_freqs)
    ax.set_ylim(0, 1.30)
    ax.set_xlabel('frequency index $f$', fontsize=11)
    ax.set_ylabel('bin weight $W_b(f)$', fontsize=11)
    ax.set_title('(b) Hard octave binning  —  ablation (closed-form, fixed)',
                 fontsize=11)
    ax.grid(alpha=0.3, axis='y')

    # Add a small text box showing bin ranges
    text = (
        'bins (frequency ranges):\n'
        '0: {0}    1: {1}    2: {2}–{3}\n'
        '3: {4}–{5}    4: {6}–{7}    5: {8}–{9}\n'
        '6: {10}–{11}    7: {12}–{13}    8: {14}–{15}'
    ).format(0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 180)
    ax.text(0.97, 0.55, text, transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='gray', alpha=0.9))


fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
gaussian_panel(axes[0])
octave_panel(axes[1])
plt.tight_layout()

out = 'figs/octave_vs_gaussian_binning.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved {out} and pdf version')
