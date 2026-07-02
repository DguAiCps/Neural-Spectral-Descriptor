#!/usr/bin/env python3
"""Verify bispectrum top-K reranking on NCLT + MulRan (cross-sensor), FIXED (K=5, w=0.5)
plus a couple settings, with a paired-bootstrap 95% CI on the R@1 delta for significance."""
import sys
sys.path.insert(0, "src")
import numpy as np, torch
from run_kitti_operating_point import _make_encoder
from evaluate_dump import _apply_encoder_preset, _load_config

POS_THR, SKIP = 5.0, 30
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def get_loader(sensor, root, seq):
    if sensor=="nclt":
        from data.nclt_loader import NCLTLoader; return NCLTLoader(root, seq, lazy_load=True)
    if sensor=="mulran":
        from data.mulran_loader import MulRanLoader; return MulRanLoader(root, seq, lazy_load=True)

def eval_seq(enc, sensor, root, seq, K=5, w=0.5):
    loader=get_loader(sensor, root, seq); N=len(loader)
    STRIDE=max(1, N//1000); idxs=list(range(0,N,STRIDE))
    D,BS,POS=[],[],[]
    for i in idxs:
        it=loader[i]
        D.append(enc.encode_points(it["points"]).detach().cpu().numpy().astype(np.float64))
        BS.append(enc.compute_bispectrum(it["points"]).astype(np.float64))
        POS.append(np.asarray(it["pose"])[:3,3])
    D=np.array(D); BS=np.array(BS); POS=np.array(POS)
    dn=l2(D); bn=l2(BS); M=len(dn)
    Q=[]
    for i in range(M):
        lo=i-SKIP
        if lo<=0: continue
        if np.linalg.norm(POS[:lo]-POS[i],axis=1).min()<=POS_THR: Q.append(i)
    Q=np.array(Q)
    base_ok=[]; rr_ok=[]
    for i in Q:
        dd=np.linalg.norm(dn-dn[i],axis=1); a,b=max(0,i-SKIP),min(M,i+SKIP+1); dd[a:b]=np.inf
        order=np.argsort(dd)
        base_ok.append(np.linalg.norm(POS[int(order[0])]-POS[i])<=POS_THR)
        cand=order[:K]
        score=np.linalg.norm(dn[cand]-dn[i],axis=1)+w*np.linalg.norm(bn[cand]-bn[i],axis=1)
        j=int(cand[int(np.argmin(score))])
        rr_ok.append(np.linalg.norm(POS[j]-POS[i])<=POS_THR)
    base_ok=np.array(base_ok,float); rr_ok=np.array(rr_ok,float)
    # paired bootstrap CI on delta
    rng=np.random.RandomState(0); n=len(Q); deltas=[]
    for _ in range(2000):
        idx=rng.randint(0,n,n); deltas.append(rr_ok[idx].mean()-base_ok[idx].mean())
    lo,hi=np.percentile(deltas,[2.5,97.5])
    d=rr_ok.mean()-base_ok.mean()
    sig="SIG" if (lo>0 or hi<0) else "ns"
    print(f"  {sensor}/{seq:14s} q={n:4d}  d-only={base_ok.mean():.4f}  rerank(K{K},w{w:g})={rr_ok.mean():.4f}  delta={d:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}] {sig}")
    return base_ok, rr_ok

def make_enc(dev, elev):
    import copy
    cfg=_apply_encoder_preset(_load_config("configs/training_kitti_phase_alignment_gat_fast.yaml"),"no_interdiff")
    cfg=copy.deepcopy(cfg); cfg["encoding"]["elevation_range"]=elev
    return _make_encoder(cfg,dev)

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print("=== bispectrum reranking, FIXED K=5 w=0.5, cross-sensor (paired-bootstrap CI) ===")
    enc_n=make_enc(dev,[-30.67,10.67])
    for seq in ["2012-01-08","2013-01-10"]:
        try: eval_seq(enc_n,"nclt","/rise/RISE1/workspace/data/nclt",seq)
        except Exception as e: print(f"  nclt/{seq} ERR {e}")
    enc_m=make_enc(dev,[-16.6,16.6])
    for seq in ["DCC03","KAIST03","Riverside03"]:
        try: eval_seq(enc_m,"mulran","/rise/RISE1/workspace/data/mulran",seq)
        except Exception as e: print(f"  mulran/{seq} ERR {e}")

if __name__=="__main__":
    main()
