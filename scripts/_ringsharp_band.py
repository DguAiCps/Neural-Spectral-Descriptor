#!/usr/bin/env python3
"""RING#-L training data with the per-dataset height band (decision 2026-10-11).

NSCBandTrainingDataset is our NSC connector class
(_external/RINGSharp/glnet/datasets/nsc/nsc_dataset.py, NSCTrainingDataset)
with one change: the BEV bounds are set per sample from the sample's sensor
(scripts/_ring_official_lib.ringsharp_bounds = DATASET_TABLE band, the same
as RING / RING++) instead of the fixed NSC_BOUNDS (z [1, 20] m for every
sensor).  Loader, augmentation (official RandomTransformation), official
generate_bev and the returned tuple are the parent class unchanged.

install() points the 'nsc' branch of glnet/datasets/dataset_utils.py at
this class: that branch imports NSCTrainingDataset from the nsc module when
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

import _ring_official_lib as rol  # noqa: E402  (torch-free)
from glnet.datasets.nsc import nsc_dataset as nd  # noqa: E402


class NSCBandTrainingDataset(nd.NSCTrainingDataset):
    """NSCTrainingDataset with per-sensor BEV bounds."""

    def __getitem__(self, ndx):
        dtype, _, _ = nd.parse_virtual_path(self.queries[ndx].rel_scan_filepath)
        self.bounds = rol.ringsharp_bounds(dtype)
        return super().__getitem__(ndx)


def install():
    nd.NSCTrainingDataset = NSCBandTrainingDataset
    return NSCBandTrainingDataset
