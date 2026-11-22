"""Spectral encoding (method section "Spectral encoding and place indexing").

For one keyframe point cloud this computes, with no learned parameters,

  d  float32 [288]       invariant spectral key: per-row octave-band mean and standard deviation
                         of the azimuthal DFT magnitudes of a 16 x 360 range image;
  z  complex64 [16, 8]   complex DFT features: azimuthal DFT coefficients k = 1..8 of a max-height
                         polar BEV image clipped at zero, averaged from 79 rings to 16 range bins.

Steps (one function each):
  range_image -> fill_empty_cells -> row_dft_magnitudes -> octave_band_statistics  => d
  max_height_bev -> average_rings -> low_frequency_coefficients                   => z
"""
from __future__ import annotations

import numpy as np

# Range image (paper: "Range projection and invariant spectral key").
N_ROWS = 16
N_AZIMUTH = 360
MIN_RANGE_M = 1.0
MAX_RANGE_M = 80.0
OCTAVE_EDGES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 181)  # 9 bands over the 181 rfft frequencies
STD_EPS = 1e-8

# Vertical field of view [deg] used to place the 16 rows, per sensor. NCLT's HDL-32E is mounted
# upside down; after the reader flips z its window is mirrored to [-10.67, 30.67].
ELEVATION_RANGE_DEG = {
    "kitti": (-24.8, 2.0),       # Velodyne HDL-64E
    "kitti360": (-24.8, 2.0),    # Velodyne HDL-64E
    "nclt": (-10.67, 30.67),     # Velodyne HDL-32E, inverted mount
    "helipr": (-15.0, 15.0),     # Velodyne VLP-16
    "mulran": (-16.6, 16.6),     # Ouster OS1-64 (gen 1)
    "mcd": (-22.5, 22.5),        # Ouster OS1-64 (gen 2)
}

# Polar BEV (paper: "Complex DFT features").
BEV_SECTORS = 60          # 6 degree sectors
BEV_MIN_RANGE_M = 1.0     # 79 rings of 1 m between 1 m and 80 m
BEV_MAX_RANGE_M = 80.0
BEV_MIN_HEIGHT_M = -3.0   # points at or below -3 m are ignored
BEV_RANGE_BINS = 16
N_COMPLEX_FREQS = 8       # frequencies k = 1..8


def _azimuth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Azimuth in [0, 2 pi), measured from the -x axis."""
    return (np.arctan2(y, x) + np.pi) % (2 * np.pi)


def _finite_xyz(points: np.ndarray):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[keep], y[keep], z[keep]


# ----------------------------------------------------------------------------- invariant key d
def range_image(points: np.ndarray, elevation_range_deg) -> np.ndarray:
    """16 x 360 elevation-azimuth range image; each cell holds the range of its nearest point.

    Points with 3D range in [1, 80] m are kept. Rows split the sensor's vertical field of view
    evenly (points outside it go to the top or bottom row); columns are 1 degree wide. Empty
    cells are 0.
    """
    x, y, z = _finite_xyz(points)
    horizontal_sq = x ** 2 + y ** 2
    rng = np.sqrt(horizontal_sq + z ** 2)
    azimuth = _azimuth(x, y)
    elevation = np.arctan2(z, np.sqrt(horizontal_sq))
    keep = (rng >= MIN_RANGE_M) & (rng <= MAX_RANGE_M) & np.isfinite(rng)
    rng, azimuth, elevation = rng[keep], azimuth[keep], elevation[keep]

    low, high = np.deg2rad(elevation_range_deg[0]), np.deg2rad(elevation_range_deg[1])
    row = np.clip(np.floor((elevation - low) / (high - low) * N_ROWS).astype(int), 0, N_ROWS - 1)
    col = np.clip(np.floor(azimuth / (2 * np.pi) * N_AZIMUTH).astype(int), 0, N_AZIMUTH - 1)

    image = np.full(N_ROWS * N_AZIMUTH, np.inf, dtype=np.float32)
    np.minimum.at(image, row * N_AZIMUTH + col, rng)
    image[image == np.inf] = 0.0
    return image.reshape(N_ROWS, N_AZIMUTH)


def fill_empty_cells(image: np.ndarray) -> np.ndarray:
    """Fill empty (zero) cells: circular linear interpolation along each row; a row with no
    return copies the nearest non-empty row (searching one row down, then up, then two, ...)."""
    out = image.copy()
    n_rows, n_cols = out.shape
    cols = np.arange(n_cols)
    for row in out:
        valid = row > 0
        if valid.any() and not valid.all():
            known = np.flatnonzero(valid)
            wrapped = np.concatenate([known - n_cols, known, known + n_cols])
            row[~valid] = np.interp(cols[~valid], wrapped, np.tile(row[known], 3))
    for r in range(n_rows):
        if (out[r] > 0).any():
            continue
        for offset in range(1, n_rows):
            if r - offset >= 0 and (out[r - offset] > 0).any():
                out[r] = out[r - offset]
                break
            if r + offset < n_rows and (out[r + offset] > 0).any():
                out[r] = out[r + offset]
                break
    return out


def row_dft_magnitudes(image: np.ndarray) -> np.ndarray:
    """|DFT| of each row along azimuth for frequencies 0..180: [16, 360] -> [16, 181].
    A cyclic shift of the columns (a yaw rotation by whole degrees) leaves these unchanged."""
    return np.abs(np.fft.rfft(image.astype(np.float32), axis=1))


def octave_band_statistics(magnitudes: np.ndarray) -> np.ndarray:
    """Invariant spectral key d [288]: mean (first 144) and standard deviation (last 144) of the
    magnitudes in each of the 9 octave bands, row-major over (row, band).
    The standard deviation is sqrt(variance + 1e-8)."""
    means, stds = [], []
    for lo, hi in zip(OCTAVE_EDGES[:-1], OCTAVE_EDGES[1:]):
        band = magnitudes[:, lo:hi]
        mean = band.mean(axis=1)
        var = ((band - mean[:, None]) ** 2).mean(axis=1)
        means.append(mean)
        stds.append(np.sqrt(var + np.float32(STD_EPS)))
    return np.concatenate([np.stack(means, 1).reshape(-1),
                           np.stack(stds, 1).reshape(-1)]).astype(np.float32)


def spectral_key(points: np.ndarray, dataset: str) -> np.ndarray:
    """d_i in R^288 for one point cloud of `dataset`."""
    image = fill_empty_cells(range_image(points, ELEVATION_RANGE_DEG[dataset]))
    return octave_band_statistics(row_dft_magnitudes(image))


# ----------------------------------------------------------------------------- complex features z
def max_height_bev(points: np.ndarray) -> np.ndarray:
    """79 x 60 polar BEV image of the maximum point height above the sensor, clipped at zero.

    Rings are 1 m wide over horizontal range [1, 80) m, sectors are 6 degrees; points at or
    below -3 m are ignored. Empty cells and cells with only returns below the sensor are 0.
    """
    x, y, z = _finite_xyz(points)
    radius = np.sqrt(x ** 2 + y ** 2)
    keep = (radius >= BEV_MIN_RANGE_M) & (radius < BEV_MAX_RANGE_M) & (z > BEV_MIN_HEIGHT_M)
    x, y, z, radius = x[keep], y[keep], z[keep], radius[keep]
    n_rings = int(BEV_MAX_RANGE_M - BEV_MIN_RANGE_M)
    ring = np.clip(np.floor(radius - BEV_MIN_RANGE_M).astype(int), 0, n_rings - 1)
    sector = np.clip(np.floor(_azimuth(x, y) / (2 * np.pi) * BEV_SECTORS).astype(int),
                     0, BEV_SECTORS - 1)
    image = np.zeros(n_rings * BEV_SECTORS, dtype=np.float32)   # zero floor = clip at zero
    np.maximum.at(image, ring * BEV_SECTORS + sector, z.astype(np.float32))
    return image.reshape(n_rings, BEV_SECTORS)


def average_rings(image: np.ndarray, n_bins: int = BEV_RANGE_BINS) -> np.ndarray:
    """Average the 79 rings into 16 range bins (adaptive average pooling: bin i covers rings
    floor(79 i / 16) .. ceil(79 (i + 1) / 16) - 1)."""
    n = image.shape[0]
    return np.stack([image[(i * n) // n_bins: -((-(i + 1) * n) // n_bins)].mean(axis=0)
                     for i in range(n_bins)]).astype(np.float32)


def low_frequency_coefficients(bins: np.ndarray) -> np.ndarray:
    """Orthonormal azimuthal DFT of each range bin, coefficients k = 1..8: [16, 60] -> [16, 8]."""
    coeffs = np.fft.rfft(bins.astype(np.float32), axis=1, norm="ortho")
    return coeffs[:, 1:N_COMPLEX_FREQS + 1].astype(np.complex64)


def complex_features(points: np.ndarray) -> np.ndarray:
    """z_i in C^{16 x 8} for one point cloud (sensor independent)."""
    return low_frequency_coefficients(average_rings(max_height_bev(points)))


def encode_scan(points: np.ndarray, dataset: str):
    """(d, z) of one keyframe point cloud."""
    return spectral_key(points, dataset), complex_features(points)
