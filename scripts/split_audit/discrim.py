import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
REPO = Path(__file__).resolve().parents[2]; PD = REPO/"data/preprocessed_pd2026"
exec(open(Path(__file__).resolve().parent / "recompute_omega.py").read().split("allf = sorted")[0])
allf = sorted(glob.glob(str(PD / "cache_*.npz")))
per = {f: harvest([f]) for f in allf}
def mk(fl): return fit(np.concatenate([per[f] for f in fl if len(per[f])], axis=0))
alt = {"train+val": mk([f for f in allf if "_test_" not in Path(f).name]),
       "train+test": mk([f for f in allf if "_val_" not in Path(f).name]),
       "all": mk(allf)}
def cosrows(a,b): return (a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)
for kf, cpat in [("keys_pd_s1_kitti_09.npz","cache_*_kitti_val_09.npz"),("keys_pd_s1_mulran_DCC01.npz","cache_*_mulran_test_DCC01.npz")]:
    k = np.load(REPO/"results/pd2026"/kf); c = np.load(glob.glob(str(PD/cpat))[0])
    idx = {v:i for i,v in enumerate(c["keyframe_ids"])}; sel = np.array([idx[v] for v in k["keyframe_ids"]])
    d = c["descriptors"][sel].astype(np.float64); e = k["emb416"][:, :288].astype(np.float64)
    print(kf, " ; ".join(f"{n}: min={cosrows(e,d*r).min():.7f}" for n,r in alt.items()))
