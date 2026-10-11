# Verify KITTI odometry 09 (raw 2011_09_30_drive_0033) vs KITTI-360 test drives route overlap via GPS
import glob
from pathlib import Path
import numpy as np
S = Path("/mnt/d/NSD_datasets/kitti_raw_oxts")   # fetched by fetch_kitti_oxts.py
def ll(dirp, sub=1):
    fs = sorted(glob.glob(str(Path(dirp)/"*.txt")))[::sub]
    return np.array([[float(x) for x in open(f).read().split()[:2]] for f in fs])
def hav(a, b):
    la1, lo1 = np.radians(a[:,0])[:,None], np.radians(a[:,1])[:,None]
    la2, lo2 = np.radians(b[:,0])[None,:], np.radians(b[:,1])[None,:]
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*6371000*np.arcsin(np.sqrt(h))
k09 = ll(S/"2011_09_30_drive_0033_sync/oxts/data", 5)
print("k09 pts", len(k09))
for d in ["0000","0004","0006"]:
    k = ll(f"/mnt/d/NSD_datasets/kitti360/data_poses/2013_05_28_drive_{d}_sync/oxts/data", 10)
    dm = hav(k09, k).min(1)
    print(f"KITTI 09 vs KITTI-360 {d}: min {dm.min():.1f} m, frac(09 pts)<50m {(dm<50).mean():.3f}, <10m {(dm<10).mean():.3f}")
