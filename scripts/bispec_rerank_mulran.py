#!/usr/bin/env python3
"""Self-contained MulRan bispectrum-reranking check (direct imports only, avoids the
run_kitti_operating_point/evaluate_dump import path that shadows the loader). FIXED K=5,w=0.5,
paired-bootstrap 95% CI on the R@1 delta."""
import sys
sys.path.insert(0, "src")
import numpy as np, torch, yaml
from data.mulran_loader import MulRanLoader
from encoding.spectral_encoder import SpectralEncoder

POS_THR, SKIP = 5.0, 30
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def make_encoder(elev, device):
    cfg = yaml.safe_load(open("configs/training_kitti_phase_alignment_gat_fast.yaml"))
    enc = cfg["encoding"]
    bev = enc.get("bev", {})
    return SpectralEncoder(
        n_elevation=enc.get("n_elevation",16), n_azimuth=enc.get("n_azimuth",360),
        n_bins=enc.get("n_bins",16), alpha=enc.get("alpha",2.0),
        learnable_alpha=False, epsilon=enc.get("epsilon",1e-8),
        target_elevation_bins=enc.get("target_elevation_bins",16),
        elevation_range=tuple(elev),
        bin_statistics=enc.get("bin_statistics",["mean","std"]),
        inter_bin_statistics=[],  # no_interdiff -> 288D
        device=device, projection_type=enc.get("projection_type","range_image"),
        max_range=enc.get("max_range",80.0), min_range=enc.get("min_range",1.0),
        z_min=bev.get("z_min",-3.0), height_encoding=bev.get("height_encoding","iris"),
        n_height_layers=bev.get("n_height_layers",8), z_max=bev.get("z_max",5.0),
        zero_center=enc.get("zero_center",False), log_magnitude=enc.get("log_magnitude",False),
        binning_strategy=enc.get("binning_strategy","octave"),
        normalize_channels=enc.get("normalize_channels",False),
        cross_spectrum_enabled=False, cross_spectrum_n_freqs=0,
    ).to(device)

def eval_seq(enc, root, seq, K=5, w=0.5):
    loader=MulRanLoader(root, seq, lazy_load=True); N=len(loader)
    STRIDE=max(1, N//1000); idxs=list(range(0,N,STRIDE))
    D,BS,POS=[],[],[]
    for i in idxs:
        it=loader[i]
        D.append(enc.encode_points(it["points"]).detach().cpu().numpy().astype(np.float64))
        BS.append(enc.compute_bispectrum(it["points"]).astype(np.float64))
        POS.append(np.asarray(it["pose"])[:3,3])
    D=np.array(D); BS=np.array(BS); POS=np.array(POS)
    dn=l2(D); bn=l2(BS); M=len(dn)
    Q=[i for i in range(M) if (i-SKIP>0 and np.linalg.norm(POS[:i-SKIP]-POS[i],axis=1).min()<=POS_THR)]
    Q=np.array(Q)
    base_ok=[]; rr_ok=[]
    for i in Q:
        dd=np.linalg.norm(dn-dn[i],axis=1); a,b=max(0,i-SKIP),min(M,i+SKIP+1); dd[a:b]=np.inf
        order=np.argsort(dd)
        base_ok.append(np.linalg.norm(POS[int(order[0])]-POS[i])<=POS_THR)
        cand=order[:K]
        score=np.linalg.norm(dn[cand]-dn[i],axis=1)+w*np.linalg.norm(bn[cand]-bn[i],axis=1)
        rr_ok.append(np.linalg.norm(POS[int(cand[int(np.argmin(score))])]-POS[i])<=POS_THR)
    base_ok=np.array(base_ok,float); rr_ok=np.array(rr_ok,float)
    rng=np.random.RandomState(0); n=len(Q); deltas=[]
    for _ in range(2000):
        idx=rng.randint(0,n,n); deltas.append(rr_ok[idx].mean()-base_ok[idx].mean())
    lo,hi=np.percentile(deltas,[2.5,97.5]); d=rr_ok.mean()-base_ok.mean()
    sig="SIG" if (lo>0 or hi<0) else "ns"
    print(f"  mulran/{seq:12s} q={n:4d} d-only={base_ok.mean():.4f} rerank={rr_ok.mean():.4f} delta={d:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] {sig}")

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    enc=make_encoder([-16.6,16.6], dev)  # Ouster OS1-64
    print("=== MulRan bispectrum reranking (K=5,w=0.5) ===")
    for seq in ["DCC03","KAIST03","Riverside03"]:
        try: eval_seq(enc,"/rise/RISE1/workspace/data/mulran",seq)
        except Exception as e:
            import traceback; print(f"  mulran/{seq} ERR {repr(e)[:120]}")

if __name__=="__main__":
    main()
