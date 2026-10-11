import glob
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
fd = REPO/"results/_edge_cls_pd2026"
HOLD = {"kitti_07","nclt_2013-02-23","helipr_Bridge01","mulran_Sejong02"}
tr=ho=trp=hop=0
for f in sorted(fd.glob("data_*.npz")):
    k = f.stem[5:]
    d = np.load(f); n=len(d["labels"]); p=int(d["labels"].sum())
    if k in HOLD: ho+=n; hop+=p
    else: tr+=n; trp+=p
    print(k, n, p)
print(f"fit {tr:,} (pos {trp:,})  holdout {ho:,} (pos {hop:,})")
# operating-cache identity with pd2026 train caches
PD = REPO/"data/preprocessed_pd2026"; OP = REPO/"data/preprocessed_cross_sensor_operating"
for sensor, seqs in {"kitti":["01","02","06","07"],"helipr":["Bridge01","Roundabout01"],"mulran":["Sejong01","Sejong02"]}.items():
    for s in seqs:
        pdf = glob.glob(str(PD/f"cache_*_{sensor}_{s}.npz")); assert len(pdf)==1
        a = np.load(pdf[0]); b = np.load(OP/f"{sensor}_operating_{s}_layout60_stride1.npz")
        ida, idb = a["keyframe_ids"], b["keyframe_ids"]
        same = ida.shape==idb.shape and bool((ida==idb).all())
        da, db = a["descriptors"].astype(np.float32), b["descriptors"].astype(np.float32)
        cos = (da*db).sum(1)/(np.linalg.norm(da,axis=1)*np.linalg.norm(db,axis=1)+1e-12) if same else np.array([np.nan])
        pa = a["poses"][:,:3,3]; pb = b["poses"][:,:3,3] if "poses" in b.files else None
        pdiff = float(np.abs(pa-pb).max()) if pb is not None and pb.shape==pa.shape else None
        print(f"{sensor}/{s}: pd={Path(pdf[0]).name} n={len(ida)} op_n={len(idb)} ids_equal={same} cos_min={float(cos.min()):.6f} pose_maxdiff={pdiff} op_keys={b.files[:8]}")
