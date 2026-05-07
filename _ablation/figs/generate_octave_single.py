"""Single-panel octave binning visualization for fig2 alternative."""
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


fig, ax = plt.subplots(figsize=(7.5, 3.0))

n_freqs = 181
edges = octave_edges(n_freqs)
n_bins = len(edges) - 1
cmap = plt.cm.viridis

# Draw rectangles for each bin
for i in range(n_bins):
    lo, hi = edges[i], edges[i + 1]
    color = cmap(i / max(n_bins - 1, 1))
    ax.add_patch(Rectangle(
        (lo, 0), hi - lo, 1.0,
        facecolor=color, alpha=0.55,
        edgecolor=color, lw=1.5,
    ))
    # Bin index label inside the rectangle
    cx = (lo + hi) / 2
    if hi - lo >= 4:
        ax.text(cx, 0.5, f'bin {i}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
    else:
        # narrow bins -> label above
        ax.text(cx, 1.08, f'{i}', ha='center', va='bottom',
                fontsize=8, color='black')

# Vertical lines at edges
for e in edges:
    ax.axvline(e, color='gray', lw=0.6, alpha=0.5)

# Edge labels
for e in edges:
    ax.text(e, -0.13, str(e), ha='center', va='top',
            fontsize=8, color='gray')

ax.set_xlim(-2, n_freqs + 2)
ax.set_ylim(0, 1.25)
ax.set_xlabel('frequency index $f$  (rfft of 360-azimuth column)', fontsize=10)
ax.set_ylabel('bin weight $W_b(f)$', fontsize=10)
ax.set_title(
    r'Hard octave binning: 9 non-overlapping bins, '
    r'edges $\{0, 1, 2, 4, 8, 16, 32, 64, 128, 181\}$',
    fontsize=10
)
ax.grid(alpha=0.25, axis='y')
ax.set_yticks([0, 0.5, 1.0])

# Legend / explanation box
text = (
    'each frequency $f$ assigned to exactly one bin\n'
    r'$W_b(f) = 1$ if $f \in [\text{edge}_b, \text{edge}_{b+1})$, else $0$'
)
ax.text(0.98, 0.97, text, transform=ax.transAxes,
        ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.35',
                  facecolor='white', edgecolor='gray', alpha=0.92))

plt.tight_layout()
out = 'figs/octave_binning_panel.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved {out} and pdf version')
