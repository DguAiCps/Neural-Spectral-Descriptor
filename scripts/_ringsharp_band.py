#!/usr/bin/env python3
"""RING#-L connector: per-dataset height band and the paper angle grid.

(1) height band, decision 2026-10-11:

NSCBandTrainingDataset is our NSC connector class
(_external/RINGSharp/glnet/datasets/nsc/nsc_dataset.py, NSCTrainingDataset)
with one change: the BEV bounds are set per sample from the sample's sensor
(scripts/_ring_official_lib.ringsharp_bounds = DATASET_TABLE band, the same
as RING / RING++) instead of the fixed NSC_BOUNDS (z [1, 20] m for every
sensor).  Loader, augmentation (official RandomTransformation), official
generate_bev and the returned tuple are the parent class unchanged.

(2) paper angle grid (project rule 2026-10-11, paper value wins):
compute_spec_paper replaces RINGSharpL.compute_spec (see its docstring).
install_model() is used by training, validation and test alike.

install_dataset() points the 'nsc' branch of glnet/datasets/dataset_utils.py at
the band class: that branch imports NSCTrainingDataset from the nsc module when
make_datasets() runs, so swapping the module attribute before the trainer
builds its datasets is enough.  No RINGSharp file is edited, and the
existing connector (used by the earlier runs) keeps its behaviour.

Import inside the RINGSharp environment (glnet on sys.path).
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_radon import ParallelBeam  # noqa: E402

import _ring_official_lib as rol  # noqa: E402  (torch-free)
from glnet.datasets.nsc import nsc_dataset as nd  # noqa: E402
from glnet.models.localizer.ring_sharp_l import RINGSharpL  # noqa: E402


class NSCBandTrainingDataset(nd.NSCTrainingDataset):
    """NSCTrainingDataset with per-sensor BEV bounds."""

    def __getitem__(self, ndx):
        dtype, _, _ = nd.parse_virtual_path(self.queries[ndx].rel_scan_filepath)
        self.bounds = rol.ringsharp_bounds(dtype)
        return super().__getitem__(ndx)


def compute_spec_paper(self, x, rot_ang=2 * np.pi):
    """RINGSharpL.compute_spec with the paper angle grid (only change).

    RING# paper (arXiv 2409.00206v2, Sec. IV-C Eq. 11, and the sinogram
    S in R^{X x Y x C}): theta in [0, 2pi) sampled by the X = 160 rows.
    Overrides glnet/models/localizer/ring_sharp_l.py:114
    np.linspace(0, rot_ang, n_angles) (0 and 2pi both included, 2pi/159);
    the repo's own YawLoss (glnet/models/loss.py:181) and evaluator
    (tools/evaluate_ours_gl.py:199) already convert shifts with 2pi/160."""
    n_angles = x.shape[-2]
    image_size = x.shape[-1]
    angles = torch.FloatTensor(np.linspace(0, rot_ang, n_angles, endpoint=False).astype(np.float32))
    radon = ParallelBeam(image_size, angles)
    sinogram = radon.forward(x)
    spec, fft_result = self.forward_row_fft(sinogram)
    return spec, sinogram


def install_dataset():
    nd.NSCTrainingDataset = NSCBandTrainingDataset
    return NSCBandTrainingDataset


def install_model():
    RINGSharpL.compute_spec = compute_spec_paper
    return RINGSharpL


def install():
    """Both swaps (training): per-sensor band and paper angle grid."""
    install_model()
    return install_dataset()
