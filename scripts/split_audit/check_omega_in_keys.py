# Check which omega was applied when 08-04 key dumps were produced: emb416[:, :288] should be collinear with d288*omega
import glob
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; PD = REPO/"data/preprocessed_pd2026"
r_cur = np.load(REPO/"artifacts/key_remetrize_r_pd2026.npy")
r_pdnn = np.load(REPO/"artifacts/key_remetrize_r_pdnn2026.npy")
r_frozen = np.load(REPO/"artifacts/key_remetrize_r.npy")
def cosrows(a,b): return (a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)
for kf, cpat in [("keys_pd_s1_kitti_09.npz","cache_*_kitti_val_09.npz"),
                 ("keys_pd_s1_mulran_DCC01.npz","cache_*_mulran_test_DCC01.npz"),
                 ("keys_pd_s3_helipr_Town02.npz","cache_*_helipr_test_Town02.npz"),
                 ("keys_pdnn_s1_mulran_DCC01.npz","cache_*_mulran_test_DCC01.npz")]:
    k = np.load(REPO/"results/pd2026"/kf); c = np.load(glob.glob(str(PD/cpat))[0])
    idx = {v:i for i,v in enumerate(c["keyframe_ids"])}
    sel = np.array([idx[v] for v in k["keyframe_ids"]])
    d = c["descriptors"][sel].astype(np.float64); e = k["emb416"][:, :288].astype(np.float64)
    out = []
    for nm, r in [("pd_cur", r_cur), ("pdnn", r_pdnn), ("paper_frozen", r_frozen), ("none", np.ones(288))]:
        cs = cosrows(e, d*r); out.append(f"{nm}: min={cs.min():.7f} mean={cs.mean():.7f}")
    print(kf, "|", " ; ".join(out))
    print("   meta:", str(k["meta"])[:300])
