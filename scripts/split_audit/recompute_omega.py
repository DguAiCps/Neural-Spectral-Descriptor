# Read-only recomputation of pd2026 key re-metrization omega (mirrors scripts/_pd_fit_key_remetrize.py)
import glob, io, sys, hashlib
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data/preprocessed_pd2026"
DTH, SKIP = 5.0, 30

def harvest(files):
    deltas = []
    for f in files:
        blob = np.load(f)
        d = blob["descriptors"].astype(np.float64)
        pos = blob["poses"][:, :3, 3].astype(np.float64)
        n = len(d)
        for q in range(SKIP, n):
            prior = np.arange(0, q - SKIP)
            if prior.size == 0: continue
            geo = np.linalg.norm(pos[prior] - pos[q], axis=1)
            positives = prior[geo < DTH]
            if positives.size == 0: continue
            m = positives[np.argmin(np.linalg.norm(pos[positives] - pos[q], axis=1))]
            deltas.append(d[q] - d[m])
    return np.array(deltas)

def fit(deltas):
    W = 0.5 * np.mean(deltas ** 2, axis=0)
    eps = 1e-6 * W.mean()
    r = 1.0 / np.sqrt(W + eps)
    r /= r.mean()
    return r

def npy_bytes(r):
    b = io.BytesIO(); np.save(b, r); return b.getvalue()

allf = sorted(glob.glob(str(CACHE / "cache_*.npz")))
train = [f for f in allf if "_val_" not in Path(f).name and "_test_" not in Path(f).name]
print("train caches:", [Path(f).name for f in train])
variants = {
  "pd_trainonly": train,
  "pdnn_trainonly": [f for f in train if "_nclt_" not in Path(f).name],
  "pd_bareglob_now(train+val+test)": allf,
  "pd_train+val": [f for f in allf if "_test_" not in Path(f).name],
  "pd_train+test": [f for f in allf if "_val_" not in Path(f).name],
}
cache = {}
for f in allf:
    pass
ref = {"pd": np.load(REPO/"artifacts/key_remetrize_r_pd2026.npy"),
       "pdnn": np.load(REPO/"artifacts/key_remetrize_r_pdnn2026.npy")}
refbytes = {k: (REPO/f"artifacts/key_remetrize_r_{k}2026.npy").read_bytes() for k in ref}
# cache per-file deltas for speed
per = {}
for f in allf:
    per[f] = harvest([f])
    print(f"  {Path(f).name}: {len(per[f])} pairs", flush=True)
for name, fl in variants.items():
    ds = [per[f] for f in fl if len(per[f])]
    r = fit(np.concatenate(ds, axis=0))
    b = npy_bytes(r)
    for k in ref:
        print(f"{name:35s} vs artifact {k:5s}: bytes_equal={b==refbytes[k]} max|diff|={np.max(np.abs(r-ref[k])):.3e} sha={hashlib.sha256(b).hexdigest()[:16]}")
