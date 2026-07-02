#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""END-TO-END bispectrum reranking on the ACTUAL full-pipeline context-refined key f.
For KITTI 00/05/08: load the full-pipeline dump (refined_key f, poses) + the matching cache
(scan_ids, same order) -> raw scans -> bispec (aligned). Compare R@1 of:
  - raw d (recomputed), f (dumped context-refined key), f+bispec rerank, d+bispec rerank.
This measures the bispectrum incremental ON the deployed retrieval key, not a proxy."""
import sys
sys.path.insert(0, "src")
import numpy as np, torch, yaml
from data.kitti_loader import KITTILoader
from encoding.spectral_encoder import SpectralEncoder

POS_THR, SKIP, K, W = 5.0, 30, 5, 0.5
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def make_encoder(elev, dev):
    cfg=yaml.safe_load(open("configs/training_kitti_phase_alignment_gat_fast.yaml")); e=cfg["encoding"]; b=e.get("bev",{})
    return SpectralEncoder(n_elevation=e.get("n_elevation",16),n_azimuth=e.get("n_azimuth",360),n_bins=e.get("n_bins",16),
        alpha=e.get("alpha",2.0),learnable_alpha=False,epsilon=e.get("epsilon",1e-8),
        target_elevation_bins=e.get("target_elevation_bins",16),elevation_range=tuple(elev),
        bin_statistics=e.get("bin_statistics",["mean","std"]),inter_bin_statistics=[],device=dev,
        projection_type=e.get("projection_type","range_image"),max_range=e.get("max_range",80.0),min_range=e.get("min_range",1.0),
        z_min=b.get("z_min",-3.0),height_encoding=b.get("height_encoding","iris"),n_height_layers=b.get("n_height_layers",8),
        z_max=b.get("z_max",5.0),zero_center=e.get("zero_center",False),log_magnitude=e.get("log_magnitude",False),
        binning_strategy=e.get("binning_strategy","octave"),normalize_channels=e.get("normalize_channels",False),
        cross_spectrum_enabled=False,cross_spectrum_n_freqs=0).to(dev)

def r1_rerank(base_key, rer_add, pos, K, W):
    """R@1 using base_key for top-K, reordered by base_dist + W*rer_dist. If W=0 -> plain base."""
    M=len(pos); Q=[i for i in range(M) if (i-SKIP>0 and np.linalg.norm(pos[:i-SKIP]-pos[i],axis=1).min()<=POS_THR)]
    ok=[]
    for i in Q:
        dd=np.linalg.norm(base_key-base_key[i],axis=1); a,b=max(0,i-SKIP),min(M,i+SKIP+1); dd[a:b]=np.inf
        order=np.argsort(dd); cand=order[:K]
        if W>0:
            sc=np.linalg.norm(base_key[cand]-base_key[i],axis=1)+W*np.linalg.norm(rer_add[cand]-rer_add[i],axis=1)
            j=int(cand[int(np.argmin(sc))])
        else:
            j=int(cand[0])
        ok.append(np.linalg.norm(pos[j]-pos[i])<=POS_THR)
    return np.mean(ok), len(Q)

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    enc=make_encoder([-24.8,2.0],dev)
    for seq in ["00","05","08"]:
        dump=np.load(f"step0_out/dumps_full/kitti_operating_{seq}_layout60_dump.npz")
        f=l2(dump["refined_key"].astype(np.float64)); pos=dump["poses"][:, :3, 3].astype(np.float64) if dump["poses"].ndim==3 else dump["poses"].astype(np.float64)
        cache=np.load(f"data/preprocessed_kitti_full_pipe/kitti_operating_{seq}_layout60.npz")
        sids=cache["scan_ids"]; d=l2(cache["descriptors"].astype(np.float64)[:, :288]) if cache["descriptors"].shape[1]>=288 else l2(cache["descriptors"].astype(np.float64))
        assert len(sids)==len(f)==len(pos), f"align {len(sids)},{len(f)},{len(pos)}"
        loader=KITTILoader("/rise/RISE1/workspace/data/kitti/dataset", seq, lazy_load=True)
        BS=np.array([enc.compute_bispectrum(loader[int(s)]["points"]).astype(np.float64) for s in sids])
        bn=l2(BS)
        r1_f,_=r1_rerank(f, None, pos, 1, 0.0)
        r1_fb,q=r1_rerank(f, bn, pos, K, W)
        r1_d,_=r1_rerank(d, None, pos, 1, 0.0)
        r1_db,_=r1_rerank(d, bn, pos, K, W)
        print(f"KITTI{seq} q={q}: f(ctx)={r1_f:.4f} -> f+bispec={r1_fb:.4f} (delta {r1_fb-r1_f:+.4f}) | d={r1_d:.4f} -> d+bispec={r1_db:.4f} (delta {r1_db-r1_d:+.4f})")

if __name__=="__main__":
    main()
