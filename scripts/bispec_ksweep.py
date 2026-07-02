#!/usr/bin/env python3
"""K/w sweep for bispectrum reranking ceiling (self-contained, direct imports). KITTI+MulRan.
Reports per-seq R@1 for K in {5,10,25,50}, w in {0.5,1.0}; then the best SINGLE fixed (K,w)
averaged across all sequences (honest, non-cherry-picked)."""
import sys
sys.path.insert(0, "src")
import numpy as np, torch, yaml
from data.kitti_loader import KITTILoader
from data.mulran_loader import MulRanLoader
from encoding.spectral_encoder import SpectralEncoder

POS_THR, SKIP = 5.0, 30
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def make_encoder(elev, device):
    cfg=yaml.safe_load(open("configs/training_kitti_phase_alignment_gat_fast.yaml")); enc=cfg["encoding"]; bev=enc.get("bev",{})
    return SpectralEncoder(n_elevation=enc.get("n_elevation",16),n_azimuth=enc.get("n_azimuth",360),
        n_bins=enc.get("n_bins",16),alpha=enc.get("alpha",2.0),learnable_alpha=False,epsilon=enc.get("epsilon",1e-8),
        target_elevation_bins=enc.get("target_elevation_bins",16),elevation_range=tuple(elev),
        bin_statistics=enc.get("bin_statistics",["mean","std"]),inter_bin_statistics=[],device=device,
        projection_type=enc.get("projection_type","range_image"),max_range=enc.get("max_range",80.0),
        min_range=enc.get("min_range",1.0),z_min=bev.get("z_min",-3.0),height_encoding=bev.get("height_encoding","iris"),
        n_height_layers=bev.get("n_height_layers",8),z_max=bev.get("z_max",5.0),zero_center=enc.get("zero_center",False),
        log_magnitude=enc.get("log_magnitude",False),binning_strategy=enc.get("binning_strategy","octave"),
        normalize_channels=enc.get("normalize_channels",False),cross_spectrum_enabled=False,cross_spectrum_n_freqs=0).to(device)

def feats(enc, loader):
    N=len(loader); STRIDE=max(1,N//1000); idxs=list(range(0,N,STRIDE))
    D,BS,POS=[],[],[]
    for i in idxs:
        it=loader[i]
        D.append(enc.encode_points(it["points"]).detach().cpu().numpy().astype(np.float64))
        BS.append(enc.compute_bispectrum(it["points"]).astype(np.float64))
        POS.append(np.asarray(it["pose"])[:3,3])
    return l2(np.array(D)), l2(np.array(BS)), np.array(POS)

def rr(dn,bn,POS,K,w):
    M=len(dn); Q=[i for i in range(M) if (i-SKIP>0 and np.linalg.norm(POS[:i-SKIP]-POS[i],axis=1).min()<=POS_THR)]
    base=0; rer=0
    for i in Q:
        dd=np.linalg.norm(dn-dn[i],axis=1); a,b=max(0,i-SKIP),min(M,i+SKIP+1); dd[a:b]=np.inf
        order=np.argsort(dd)
        base+= np.linalg.norm(POS[int(order[0])]-POS[i])<=POS_THR
        cand=order[:K]; sc=np.linalg.norm(dn[cand]-dn[i],axis=1)+w*np.linalg.norm(bn[cand]-bn[i],axis=1)
        rer+= np.linalg.norm(POS[int(cand[int(np.argmin(sc))])]-POS[i])<=POS_THR
    n=len(Q); return base/n, rer/n, n

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    ek=make_encoder([-24.8,2.0],dev); em=make_encoder([-16.6,16.6],dev)
    jobs=[("KITTI00",ek,KITTILoader("/rise/RISE1/workspace/data/kitti/dataset","00")),
          ("KITTI05",ek,KITTILoader("/rise/RISE1/workspace/data/kitti/dataset","05")),
          ("KITTI08",ek,KITTILoader("/rise/RISE1/workspace/data/kitti/dataset","08")),
          ("MulDCC03",em,MulRanLoader("/rise/RISE1/workspace/data/mulran","DCC03")),
          ("MulKAIST03",em,MulRanLoader("/rise/RISE1/workspace/data/mulran","KAIST03")),
          ("MulRiv03",em,MulRanLoader("/rise/RISE1/workspace/data/mulran","Riverside03"))]
    Ks=[5,10,25,50]; ws=[0.5,1.0]
    cache={}
    for nm,enc,ld in jobs:
        dn,bn,POS=feats(enc,ld); cache[nm]=(dn,bn,POS)
        b,_,n=rr(dn,bn,POS,5,0.5); print(f"{nm}: d-only={b:.4f} (q={n})")
    print("\n=== fixed (K,w) mean delta across all 6 seqs ===")
    best=None
    for K in Ks:
        for w in ws:
            ds=[]
            for nm in cache:
                dn,bn,POS=cache[nm]; b,r,_=rr(dn,bn,POS,K,w); ds.append(r-b)
            md=np.mean(ds)
            print(f"  K={K:2d} w={w:g}: mean delta={md:+.4f}  per-seq={[round(x,3) for x in ds]}")
            if best is None or md>best[0]: best=(md,K,w)
    print(f"\nBEST fixed: K={best[1]} w={best[2]} mean delta={best[0]:+.4f}")

if __name__=="__main__":
    main()
