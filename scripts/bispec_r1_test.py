#!/usr/bin/env python3
"""DECISIVE bispectrum test: does [d;bispec] PRESERVE true positives (same-place revisits) and
improve/hold Recall@1, or does the bispectrum separate EVERYTHING (including positives) -> R@1 drop?
Runs on KITTI 00/05/08. Reports R@1 for d alone vs [d; w*bispec] over several weights w, plus
same-place vs distant bispec-distance separability (a good feature: same-place SMALL, distant LARGE)."""
import sys, os
sys.path.insert(0, "src")
import numpy as np, torch
from data.kitti_loader import KITTILoader
from run_kitti_operating_point import _make_encoder
from evaluate_dump import _apply_encoder_preset, _load_config

POS_THR, SKIP = 5.0, 30
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def r1(z, pos):
    N=len(pos); hit=0; q=0
    for i in range(N):
        lo=i-SKIP
        if lo<=0: continue
        if np.linalg.norm(pos[:lo]-pos[i],axis=1).min()>POS_THR: continue
        q+=1
        dd=np.linalg.norm(z-z[i],axis=1); a,b=max(0,i-SKIP),min(N,i+SKIP+1); dd[a:b]=np.inf
        j=int(np.argmin(dd))
        if np.linalg.norm(pos[j]-pos[i])<=POS_THR: hit+=1
    return hit/max(1,q), q

def run(enc, root, seq):
    loader=KITTILoader(root, seq, lazy_load=True); N=len(loader)
    STRIDE=max(1, N//1200); idxs=list(range(0,N,STRIDE))
    D,BS,POS=[],[],[]
    for i in idxs:
        it=loader[i]
        D.append(enc.encode_points(it["points"]).detach().cpu().numpy().astype(np.float64))
        BS.append(enc.compute_bispectrum(it["points"]).astype(np.float64))
        POS.append(np.asarray(it["pose"])[:3,3])
    D=np.array(D); BS=np.array(BS); POS=np.array(POS)
    dn=l2(D); bn=l2(BS)
    # same-place vs distant bispec distances
    rng=np.random.RandomState(0); M=len(dn)
    a=rng.randint(0,M,150000); b=rng.randint(0,M,150000); m=(np.abs(a-b)>SKIP); a,b=a[m],b[m]
    dpos=np.linalg.norm(POS[a]-POS[b],axis=1)
    same=dpos<=POS_THR; dist=dpos>=25.0
    bsame=np.linalg.norm(bn[a[same]]-bn[b[same]],axis=1) if same.sum() else np.array([np.nan])
    bdist=np.linalg.norm(bn[a[dist]]-bn[b[dist]],axis=1) if dist.sum() else np.array([np.nan])
    r1d,q=r1(dn,POS)
    print(f"### KITTI{seq}: N={M} q={q}")
    print(f"  bispec dist: SAME-place mean={np.nanmean(bsame):.4f} (n={same.sum()})  DISTANT mean={np.nanmean(bdist):.4f} (n={dist.sum()})")
    print(f"  (good feature => SAME << DISTANT; if SAME ~ DISTANT, bispec separates positives too -> harmful)")
    print(f"  R@1  d-only = {r1d:.4f}")
    for w in [0.25,0.5,1.0]:
        joint=l2(np.concatenate([dn, w*bn],axis=1))
        r1j,_=r1(joint,POS)
        print(f"  R@1  [d; {w:g}*bispec] = {r1j:.4f}   delta={r1j-r1d:+.4f}")

def main():
    cfg=_apply_encoder_preset(_load_config("configs/training_kitti_phase_alignment_gat_fast.yaml"),"no_interdiff")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    enc=_make_encoder(cfg,dev)
    root="/rise/RISE1/workspace/data/kitti/dataset"
    for seq in ["00","05","08"]:
        run(enc, root, seq)

if __name__=="__main__":
    main()
