# Reproducibility Environment

The paper results were produced in NVIDIA NGC PyG container `26.01` on Ubuntu
24.04 with Python 3.12.3, PyTorch 2.10, CUDA 13.1, cuDNN 9.17, PyTorch
Geometric 2.8.0, FAISS 1.14.3, and NumPy 1.26.4. Hardware was an RTX 5080
(16 GB), AMD Ryzen 7 9800X3D, and 32 GB RAM.

`requirements.txt` and `setup.py` intentionally retain the older Python
3.8-compatible development environment (PyTorch 2.1 / PyG 2.4). It is useful
for unit tests but is not a byte-for-byte reproduction of the paper container.
For paper reruns, start the NGC container and install any project-only missing
dependencies from `requirements.txt` without downgrading the container's CUDA,
PyTorch, PyG, FAISS, or NumPy packages.

Set the dataset root explicitly before data-dependent commands:

```bash
export NSD_DATA_ROOT=/absolute/path/to/NSD_datasets
```

Absent that variable, paper scripts resolve data under `./data`. This repository
does not distribute the datasets, external baseline checkpoints, or cached
paper results; see `DATA.md`, `artifacts/MANIFEST.md`, and
`docs/reproducibility_manifest.yaml` for the required inputs.
