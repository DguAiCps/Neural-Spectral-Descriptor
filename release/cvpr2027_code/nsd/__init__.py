"""NSD: reference implementation of the Neural Spectral Descriptor for LiDAR place recognition.

Modules follow the order of the method section:

    data, odometry, keyframes   input scans, LiDAR odometry and keyframe selection
    encoding                    spectral encoding: invariant spectral key d, complex DFT features z
    model, train                DiffAttnConv and temporal spectrum refining
    index, matching, metrics    place indexing, matching, evaluation

Per-sequence interface (keyframes only, in acquisition order):

    scan_ids    int64      [N]        index of the keyframe scan in the sequence reader
    timestamps  float64    [N]        scan time in seconds
    poses       float64    [N, 4, 4]  LiDAR odometry poses (the only poses the method sees)
    poses_gt    float64    [N, 4, 4]  dataset reference poses (evaluation only)
    d           float32    [N, 288]   invariant spectral key
    z           complex64  [N, 16, 8] complex DFT features
"""


def deterministic(num_threads: int = 1) -> None:
    """Torch settings of all reported runs: deterministic kernels, no TF32, fixed CPU thread count.

    Without them the GPU sums over graph neighbours in a non-deterministic order and the refined spectra
    differ in the last bits from run to run. Call before any CUDA work.
    """
    import os

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(num_threads)
