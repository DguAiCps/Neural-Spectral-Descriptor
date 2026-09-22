#!/usr/bin/env python3
"""Launch the OFFICIAL ImLPR trainer on the pd2026 clean split.

Thin wrapper around _external/ImLPR/training/train.py (which is NOT modified).
It exists for one reason: the official release is broken at its own
mid-training eval — training/train.py calls
    evaluate(model, device, params, intra_flg=...)
every 5 epochs, but eval/evaluate.py:evaluate() has no `intra_flg` parameter
(TypeError at epoch 5), and it would anyway demand hard-coded
db_helipr_town/query_helipr_town eval pickles we do not have. This launcher
stubs `evaluate` IN THE TRAINER'S NAMESPACE with a no-op returning
ave_recall=0.0: the `_best.pth` branch never fires and no eval pickles are
needed. Checkpoint selection is OURS, post-hoc and pre-declared
(scripts/_imlpr_pd_select.py) over the save_freq=10 snapshots
_external/ImLPR/weights/ImLPR_pd2026_{10..100}.pth (+ _final.pth, a
duplicate of _100).

It also pins CWD to _external/ImLPR so the relative dataset_folder
(../imlpr_data_pd2026/) in config_pd.txt resolves identically on host and in
docker, and validates config_pd.txt against the canonical text in
scripts/_imlpr_pd_prepare.py before starting.

Everything after the script name is passed through to the official trainer
CLI (e.g. --seed 0 --deterministic --debug).

Usage (inside docker ring_baseline, GPU):
    python3 scripts/_imlpr_pd_train.py --seed 0
"""
from __future__ import annotations

import os
import sys

# NumPy 2.x compat shim for the OFFICIAL ImLPR code (upstream unmodified):
# their loss uses np.NINF, removed in NumPy 2.0. Restoring the aliases is
# semantically identical (np.NINF was literally -inf).
import numpy as _np
for _name, _val in (("NINF", -_np.inf), ("PINF", _np.inf), ("Inf", _np.inf),
                    ("infty", _np.inf), ("float_", _np.float64)):
    if not hasattr(_np, _name):
        setattr(_np, _name, _val)
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMLPR = REPO / "_external" / "ImLPR"
CONFIG = REPO / "_external" / "imlpr_data_pd2026" / "config_pd.txt"
MODEL_CONFIG = IMLPR / "config" / "config_model_imlpr.txt"


def main() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from _imlpr_pd_prepare import PICKLE_DEFAULT, ensure_config

    if not ensure_config(write=False):
        raise SystemExit("config_pd.txt missing or drifted from the canonical "
                         "CONFIG_TEXT — reconcile via scripts/"
                         "_imlpr_pd_prepare.py --stage config first")
    train_pickle = CONFIG.parent / PICKLE_DEFAULT
    if not train_pickle.is_file():
        raise SystemExit(f"{train_pickle} missing — run scripts/"
                         f"_imlpr_pd_prepare.py --stage riv then --stage "
                         f"pickle first")

    os.chdir(IMLPR)  # dataset_folder in config_pd.txt is relative to here
    sys.path.insert(0, str(IMLPR))

    # ImLPR's top-level packages (misc/models/datasets/eval/training) ship
    # without __init__.py, i.e. as namespace packages — and a namespace
    # portion LOSES to any installed regular package of the same name
    # (docker ring_baseline has HuggingFace `datasets`, which shadows
    # ImLPR's datasets/ and breaks `from datasets.dataset_utils import ...`).
    # Pre-register them in sys.modules with explicit __path__ so the official
    # imports resolve into _external/ImLPR deterministically.
    import importlib.machinery
    import types
    for pkg in ("misc", "models", "datasets", "eval", "training"):
        spec = importlib.machinery.ModuleSpec(pkg, None, is_package=True)
        spec.submodule_search_locations = [str(IMLPR / pkg)]
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(IMLPR / pkg)]
        mod.__spec__ = spec
        sys.modules[pkg] = mod

    import training.train as trainer

    def _eval_disabled(model, device, params, **kwargs):
        print("[pd2026] official mid-training eval disabled (broken "
              "intra_flg signature + hard-coded helipr_town pickles; "
              "selection = scripts/_imlpr_pd_select.py)", flush=True)
        return {"ave_recall": 0.0, "ave_one_percent_recall": 0.0}

    trainer.evaluate = _eval_disabled

    sys.argv = ["train.py",
                "--config", str(CONFIG),
                "--model_config", str(MODEL_CONFIG)] + sys.argv[1:]
    trainer._main()


if __name__ == "__main__":
    main()
