"""Render real-data panels for Figure 2 (Stage 1 of NSD).

Produces five PNG panels from a single 64-ring KITTI scan:
  panel1_pointcloud.png  - top-down point cloud
  panel2_range_64.png    - 64x360 range image (same height as panel3)
  panel3_range_16.png    - 16x360 range image (after elevation pooling)
  panel4_fft.png         - 1D FFT magnitude line plot (single elevation)
  panel5_bins.png        - 9-bin Gaussian-binning bar chart (single elevation)

All panels: white background, no text/axes/colorbars, 300 DPI.

Usage (inside container):
    python scripts/render_stage1_panels.py --frame 1400
    python scripts/render_stage1_panels.py --auto-pick
    python scripts/render_stage1_panels.py --list-candidates  # render thumbnails
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.kitti_loader import KITTILoader
from encoding.range_image import RangeImageProjector
from encoding.spectral_policy import SoftBinning

KITTI_ROOT = os.path.join(os.environ.get("NSD_DATA_ROOT", str(REPO_ROOT / "data")), "kitti", "dataset")
OUT_DIR = REPO_ROOT / "docs/paper/body_v1.1/figs/stage1_real"

# KITTI HDL-64E elevation range
ELEV_RANGE = (-24.8, 2.0)
N_ELEV_NATIVE = 64
N_ELEV_TARGET = 16
A = 360  # azimuth bins

# Candidate intersection frames in KITTI 00 (urban Karlsruhe).
INTERSECTION_CANDIDATES = [
    50, 100, 150, 200, 250, 300, 400, 500, 800, 1100, 1300, 1400, 1500,
    1700, 1900, 2050, 2200, 2400, 2600, 2700, 2900, 3100, 3300, 3450,
    3600, 3700, 3900, 4000, 4200,
]

ACCENT_BLUE = "#4A6FA5"   # matches fig1.png modern minimal style
DARK_GRAY = "#444444"


def load_scan(loader: KITTILoader, idx: int) -> np.ndarray:
    return loader[idx]["points"]


def project_range_64(points: np.ndarray) -> np.ndarray:
    proj = RangeImageProjector(
        n_elevation=N_ELEV_NATIVE, n_azimuth=A,
        elevation_range=ELEV_RANGE, max_range=80.0,
    )
    out = proj.project(points.astype(np.float32), keep_intensity=False)
    img = out[0] if isinstance(out, tuple) else out
    return np.asarray(img)


def pool_to_16(image_64: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(image_64).float()[None, None]
    t16 = F.adaptive_avg_pool2d(t, (N_ELEV_TARGET, A))[0, 0]
    return t16.numpy()


def per_row_fft_magnitude(image_16: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(image_16).float()
    spec = torch.fft.rfft(t, dim=-1, norm="ortho") * np.sqrt(A)
    return spec.abs().numpy()


def gaussian_binning(magnitudes: np.ndarray):
    soft = SoftBinning(
        n_rings=N_ELEV_TARGET, n_freqs=181,
        output_dim=0, n_soft_bins=9,
        init_mode="octave", shared_across_rings=True,
    )
    soft.eval()
    with torch.no_grad():
        weights = soft._compute_weights(0)
        fft_t = torch.from_numpy(magnitudes).float()
        binned = fft_t @ weights.t()
    return binned.cpu().numpy(), weights.cpu().numpy()


def four_way_score(loader: KITTILoader, idx: int) -> float:
    pts = load_scan(loader, idx)
    xy = pts[:, :2]
    m = (np.abs(xy) <= 40.0).all(axis=1)
    xy = xy[m]
    if len(xy) < 100:
        return 0.0
    corr_w = 3.0
    n_e = ((xy[:, 1] > -corr_w) & (xy[:, 1] < corr_w) & (xy[:, 0] > 5)).sum()
    n_w = ((xy[:, 1] > -corr_w) & (xy[:, 1] < corr_w) & (xy[:, 0] < -5)).sum()
    n_n = ((xy[:, 0] > -corr_w) & (xy[:, 0] < corr_w) & (xy[:, 1] > 5)).sum()
    n_s = ((xy[:, 0] > -corr_w) & (xy[:, 0] < corr_w) & (xy[:, 1] < -5)).sum()
    return float(((n_e + 1) * (n_w + 1) * (n_n + 1) * (n_s + 1)) ** 0.25)


def auto_pick_intersection(loader: KITTILoader) -> int:
    scores = []
    for idx in INTERSECTION_CANDIDATES:
        if idx >= len(loader):
            continue
        scores.append((idx, four_way_score(loader, idx)))
    scores.sort(key=lambda s: s[1], reverse=True)
    print("Top 10 intersection scores:")
    for i, s in scores[:10]:
        print(f"  frame {i:5d}: score={s:8.1f}")
    return scores[0][0]


def list_candidates(loader: KITTILoader, out_dir: Path):
    """Render small top-down thumbnails for all candidates so the user can pick."""
    thumb_dir = out_dir / "candidate_thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for idx in INTERSECTION_CANDIDATES:
        if idx >= len(loader):
            continue
        s = four_way_score(loader, idx)
        scores.append((idx, s))
        pts = load_scan(loader, idx)
        save_pointcloud_topdown(pts, thumb_dir / f"frame_{idx:05d}_score_{s:.0f}.png",
                                fig_w=1.4, fig_h=1.4)
    scores.sort(key=lambda x: x[1], reverse=True)
    print("\nCandidates ranked by 4-way score:")
    for i, s in scores:
        print(f"  frame {i:5d}: score={s:8.1f}  -> {thumb_dir.name}/frame_{i:05d}_score_{s:.0f}.png")


def _clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_image(arr: np.ndarray, path: Path, *, cmap: str, fig_w: float, fig_h: float,
               vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest")
    _clean_axes(ax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path.name}  ({arr.shape}, {fig_w:.2f}x{fig_h:.2f} in)")


def save_cylindrical_projection(points: np.ndarray, path: Path, *,
                                fig_w: float, fig_h: float,
                                lim: float = 30.0,
                                cyl_radius: float = 18.0,
                                cyl_z_range: tuple = (-2.5, 4.0),
                                n_vertical: int = 24,
                                n_horizontal: int = 8):
    """3D perspective: point cloud (range-shaded) + translucent cylinder grid.

    Cylinder is shown with a dense (theta, z) grid so the (azimuth x elevation)
    discretization that produces the range image is visually explicit.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = points
    xy = pts[:, :2]; z = pts[:, 2]
    m = (np.abs(xy) <= lim).all(axis=1) & (z > cyl_z_range[0]) & (z < cyl_z_range[1])
    pts = pts[m]
    if len(pts) > 6000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), size=6000, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Camera-perspective occlusion:
    #   axis xy = (cos azim, sin azim) points from origin toward the camera.
    #   - s     = projection along that direction (positive = nearer to camera)
    #   - perp  = lateral distance from the camera-axis line through origin
    #   A point outside the cylinder is occluded by it when s < 0 AND its
    #   lateral offset is within the cylinder silhouette AND its z falls
    #   within the cylinder height.
    view_azim_deg = -58
    azim_rad = np.radians(view_azim_deg)
    cam_x = np.cos(azim_rad)
    cam_y = np.sin(azim_rad)
    s_proj = pts[:, 0] * cam_x + pts[:, 1] * cam_y
    perp_xy = np.abs(pts[:, 0] * (-cam_y) + pts[:, 1] * cam_x)

    r_h = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    inside_cyl = r_h < cyl_radius
    z_overlap = (pts[:, 2] > cyl_z_range[0] - 1.5) & (pts[:, 2] < cyl_z_range[1] + 1.5)
    behind_cyl = (~inside_cyl) & (s_proj < 0) & (perp_xy < cyl_radius) & z_overlap

    faded = inside_cyl | behind_cyl   # rendered behind cylinder
    vivid = ~faded                     # rendered in front
    r_full = np.linalg.norm(pts[:, :2], axis=1)

    # 1) Faded points (inside cylinder OR view-occluded by cylinder): muted but visible
    ax.scatter(pts[faded, 0], pts[faded, 1], pts[faded, 2],
               color="#9b9b9b", s=0.45, alpha=0.40,
               linewidths=0, depthshade=False)

    # 2) Cylinder surface (slightly more opaque so the occlusion reads)
    theta = np.linspace(0, 2 * np.pi, 120)
    z_cyl_dense = np.linspace(cyl_z_range[0], cyl_z_range[1], 40)
    Theta, Z = np.meshgrid(theta, z_cyl_dense)
    Xc = cyl_radius * np.cos(Theta)
    Yc = cyl_radius * np.sin(Theta)
    ax.plot_surface(Xc, Yc, Z,
                    color=ACCENT_BLUE, alpha=0.22,
                    linewidth=0, antialiased=True, shade=False)

    # Dense horizontal rings (n_horizontal levels along z)
    h_levels = np.linspace(cyl_z_range[0], cyl_z_range[1], n_horizontal)
    rim_x = cyl_radius * np.cos(theta)
    rim_y = cyl_radius * np.sin(theta)
    for zi, zv in enumerate(h_levels):
        is_edge = zi == 0 or zi == len(h_levels) - 1
        lw = 0.7 if is_edge else 0.45
        alpha = 0.55 if is_edge else 0.35
        ax.plot(rim_x, rim_y, np.full_like(rim_x, zv),
                color=ACCENT_BLUE, linewidth=lw, alpha=alpha)

    # Dense vertical guide lines (n_vertical evenly spaced angles)
    for ang in np.linspace(0, 2 * np.pi, n_vertical, endpoint=False):
        ax.plot([cyl_radius * np.cos(ang)] * 2,
                [cyl_radius * np.sin(ang)] * 2,
                cyl_z_range,
                color=ACCENT_BLUE, linewidth=0.4, alpha=0.4)

    # 3) Vivid points: drawn LAST -> in front / not occluded by the cylinder
    ax.scatter(pts[vivid, 0], pts[vivid, 1], pts[vivid, 2],
               c=r_full[vivid], cmap="Greys", s=0.65, alpha=0.9,
               linewidths=0, vmin=0, vmax=lim, depthshade=False)

    ax.set_axis_off()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(cyl_z_range[0] - 1.0, cyl_z_range[1] + 1.0)
    ax.view_init(elev=18, azim=-58)
    ax.set_box_aspect((1, 1, 0.35))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path.name}  (cylindrical 3D, grid {n_vertical}v x {n_horizontal}h, {fig_w:.2f}x{fig_h:.2f} in)")


def save_pointcloud_topdown(points: np.ndarray, path: Path, *,
                            fig_w: float, fig_h: float, lim: float = 40.0):
    xy = points[:, :2]; z = points[:, 2]
    m = (np.abs(xy) <= lim).all(axis=1) & (z > -3.0) & (z < 3.0)
    xy = xy[m]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    r = np.linalg.norm(xy, axis=1)
    ax.scatter(xy[:, 0], xy[:, 1], c=r, cmap="Greys", s=0.4,
               linewidths=0, alpha=0.85, vmin=0, vmax=lim)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    _clean_axes(ax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path.name}  (point cloud, {fig_w:.2f}x{fig_h:.2f} in)")


def save_fft_lineplot(spectrum_1d: np.ndarray, path: Path, *,
                      fig_w: float, fig_h: float):
    """Single-elevation FFT magnitude as a log-scale line plot with soft fill below."""
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    x = np.arange(len(spectrum_1d))
    y = np.log1p(spectrum_1d)  # log(1 + |F|) preserves monotonicity, handles zero
    ax.fill_between(x, 0, y, color=ACCENT_BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, y, color=DARK_GRAY, linewidth=1.4)
    ax.set_xlim(0, len(spectrum_1d) - 1)
    ax.set_ylim(0, y.max() * 1.05)
    _clean_axes(ax)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {path.name}  (FFT log-scale, n={len(spectrum_1d)}, {fig_w:.2f}x{fig_h:.2f} in)")


def save_bins_gaussians(weights: np.ndarray, path: Path, *,
                        fig_w: float, fig_h: float, clip_rank: int = 3):
    """9 Gaussian assignment curves over the frequency axis.

    Plots the actual sum=1 normalized weights (no per-curve renormalization).
    Tall narrow curves are clipped at the (clip_rank+1)-th largest peak, so the
    shorter wide curves remain visible while the narrow ones simply truncate
    at the top edge.
    """
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    n_bins, n_freqs = weights.shape
    x = np.arange(n_freqs)
    peaks = weights.max(axis=1)
    sorted_peaks = np.sort(peaks)[::-1]   # descending
    clip_idx = min(clip_rank, n_bins - 1)
    y_max = sorted_peaks[clip_idx] * 1.05
    for b in range(n_bins):
        y = np.clip(weights[b], 0, y_max)
        ax.fill_between(x, 0, y, color=ACCENT_BLUE, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=ACCENT_BLUE, linewidth=1.2, alpha=0.85)
    ax.set_xlim(0, n_freqs - 1)
    ax.set_ylim(0, y_max)
    _clean_axes(ax)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {path.name}  (Gaussian curves, {n_bins} bins x {n_freqs} freq, "
          f"clipped at rank-{clip_rank+1} peak={y_max:.4f}, {fig_w:.2f}x{fig_h:.2f} in)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--auto-pick", action="store_true")
    ap.add_argument("--list-candidates", action="store_true",
                    help="render thumbnails of all candidates and exit")
    ap.add_argument("--sequence", default="00")
    ap.add_argument("--elev-row", type=int, default=8,
                    help="elevation row [0,16) for FFT/bins single-row plots")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = KITTILoader(KITTI_ROOT, args.sequence, lazy_load=True)

    if args.list_candidates:
        list_candidates(loader, OUT_DIR)
        return

    if args.auto_pick or args.frame is None:
        frame = auto_pick_intersection(loader)
        print(f"-> picked frame {frame}")
    else:
        frame = args.frame

    pts = load_scan(loader, frame)
    print(f"Loaded frame {frame}: {len(pts)} points")

    img_64 = project_range_64(pts)
    img_16 = pool_to_16(img_64)
    fft_mag = per_row_fft_magnitude(img_16)            # (16, 181)
    bins_16x9, gauss_weights = gaussian_binning(fft_mag)   # (16, 9), (9, 181)

    # Single elevation row for FFT line plot
    row = max(0, min(N_ELEV_TARGET - 1, args.elev_row))
    fft_row = fft_mag[row]                              # (181,)
    print(f"FFT line plot uses elevation row {row}")

    # ---- panel sizes ----
    # NeurIPS textwidth is 5.5 in; five panels with gaps -> ~1.0-1.4 in each.
    # All 'wide' panels (range64, range16, fft) share height = 1.6 in.
    H = 1.6
    save_pointcloud_topdown(pts, OUT_DIR / "panel1_pointcloud.png", fig_w=H, fig_h=H)
    save_cylindrical_projection(pts, OUT_DIR / "panel1b_cylindrical.png",
                                fig_w=2.0, fig_h=H)
    save_image(img_64, OUT_DIR / "panel2_range_64.png",
               cmap="viridis", fig_w=2.4, fig_h=H, vmin=0, vmax=60)
    save_image(img_16, OUT_DIR / "panel3_range_16.png",
               cmap="viridis", fig_w=2.4, fig_h=H, vmin=0, vmax=60)
    save_fft_lineplot(fft_row, OUT_DIR / "panel4_fft.png",
                      fig_w=2.4, fig_h=H)
    save_bins_gaussians(gauss_weights, OUT_DIR / "panel5_bins.png",
                        fig_w=2.4, fig_h=H, clip_rank=6)

    print(f"\nAll panels written to {OUT_DIR}")
    print(f"Frame: {frame} (KITTI {args.sequence}), elev-row: {row}")


if __name__ == "__main__":
    main()
