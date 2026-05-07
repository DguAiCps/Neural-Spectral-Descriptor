"""Compact octave binning inset for fig2 panel (analogous to existing Gaussian inset)."""
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
    return edges


def render(filename, transparent=True):
    fig, ax = plt.subplots(figsize=(2.4, 1.0))   # small, fits inside panel
    n_freqs = 181
    edges = octave_edges(n_freqs)
    n_bins = len(edges) - 1
    cmap = plt.cm.viridis

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        color = cmap(i / max(n_bins - 1, 1))
        ax.add_patch(Rectangle(
            (lo, 0), hi - lo, 1.0,
            facecolor=color, alpha=0.55,
            edgecolor=color, lw=1.2,
        ))

    # Vertical edges (very thin)
    for e in edges:
        ax.axvline(e, color='gray', lw=0.4, alpha=0.5)

    ax.set_xlim(0, n_freqs)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout(pad=0.05)
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=transparent)
    print(f'Saved {filename}')


render('figs/octave_inset.png', transparent=True)
render('figs/octave_inset.pdf', transparent=True)
render('figs/octave_inset_white.png', transparent=False)
