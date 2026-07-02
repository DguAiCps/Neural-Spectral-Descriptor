#!/usr/bin/env python3
"""GAMBLE: use the bispectrum as a top-K RERANKER (not a retrieval key).
Rationale from the earlier test: raw-concat fails because bispec separates ALL pairs, but WITHIN
d's top-K (already d-similar) the TRUE revisit has smaller bispec distance (1.09) than false
d-collisions (1.34). So reranking top-K by (d + w*bispec) may raise R@1 without the global
positive-destruction. Cheap: reuses encoder, no retrain. KITTI 00/05/08."""
import sys
sys.path.insert(0, "src")
import numpy as np, torch
from data.kitti_loader import KITTILoader
from run_kitti_operating_point import _make_encoder
from evaluate_dump import _apply_encoder_preset, _load_config

POS_THR, SKIP = 5.0, 30
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def eval_seq(enc, root, seq, Ks=(5,10,25), ws=(0.5,1.0,2.0)):
    loader=KITTILoader(root, seq, lazy_load=True); N=len(loader)
    STRIDE=max(1, N//1200); idxs=list(range(0,N,STRIDE))
    D,BS,POS=[],[],[]
    for i in idxs:
        it=loader[i]
        D.append(enc.encode_points(it["points"]).detach().cpu().numpy().astype(np.float64))
        BS.append(enc.compute_bispectrum(it["points"]).astype(np.float64))
        POS.append(np.asarray(it["pose"])[:3,3])
    D=np.array(D); BS=np.array(BS); POS=np.array(POS)
    dn=l2(D); bn=l2(BS); M=len(dn)
    # queries = frames with a past revisit
    Q=[]
    for i in range(M):
        lo=i-SKIP
        if lo<=0: continue
        if np.linalg.norm(POS[:lo]-POS[i],axis=1).min()<=POS_THR: Q.append(i)
    Q=np.array(Q)
    def hit(idx_list_top1):
        return np.mean([np.linalg.norm(POS[j]-POS[i])<=POS_THR for i,j in idx_list_top1])
    # baseline d-only top1
    base=[]
    dtopk={K:[] for K in Ks}
    for i in Q:
        dd=np.linalg.norm(dn-dn[i],axis=1); a,b=max(0,i-SKIP),min(M,i+SKIP+1); dd[a:b]=np.inf
        order=np.argsort(dd)
        base.append((i,int(order[0])))
        for K in Ks: dtopk[K].append((i,order[:K]))
    r1_base=hit(base)
    print(f"### KITTI{seq}: q={len(Q)}  R@1 d-only = {r1_base:.4f}")
    # rerank top-K by d + w*bispec
    best=(r1_base,"d-only",0,0)
    for K in Ks:
        for w in ws:
            top1=[]
            for (i,cand) in dtopk[K]:
                score=np.linalg.norm(dn[cand]-dn[i],axis=1) + w*np.linalg.norm(bn[cand]-bn[i],axis=1)
                top1.append((i,int(cand[int(np.argmin(score))])))
            r=hit(top1); tag=f"rerank K={K} w={w:g}"
            flag=" *" if r>r1_base else ""
            print(f"    {tag:18s} R@1={r:.4f}  delta={r-r1_base:+.4f}{flag}")
            if r>best[0]: best=(r,tag,K,w)
    # also: rerank by bispec ALONE within top-K
    for K in Ks:
        top1=[]
        for (i,cand) in dtopk[K]:
            score=np.linalg.norm(bn[cand]-bn[i],axis=1)
            top1.append((i,int(cand[int(np.argmin(score))])))
        r=hit(top1); print(f"    bispec-only K={K:2d}      R@1={r:.4f}  delta={r-r1_base:+.4f}")
    print(f"  BEST: {best[1]} -> {best[0]:.4f} (delta {best[0]-r1_base:+.4f})")

def main():
    cfg=_apply_encoder_preset(_load_config("configs/training_kitti_phase_alignment_gat_fast.yaml"),"no_interdiff")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    enc=_make_encoder(cfg,dev)
    root="/rise/RISE1/workspace/data/kitti/dataset"
    for seq in ["00","05","08"]:
        eval_seq(enc, root, seq)

if __name__=="__main__":
    main()
