#!/usr/bin/env python3
"""RING# 원래 정규화·FFT의 지도 쪽 반복 계산만 캐시한다."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]


class CachedRerank:
    """공식 batch128·정규화·FFT·점수·정렬 순서를 그대로 둔다."""

    def __init__(self, base, trans, specs, bts):
        import torch
        import torch.nn as nn
        self.base, self.trans = base, trans
        self.h, self.w = specs.shape[-2:]
        self.norm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
        self.rotation = torch.empty_like(specs, dtype=torch.complex64)
        self.translation = torch.empty_like(bts, dtype=torch.complex64)
        with torch.no_grad():
            for start in range(0, len(specs), 128):
                self.rotation[start:start+128] = torch.fft.fft2(
                    self.norm(specs[start:start+128]), dim=-2, norm="ortho")
                self.translation[start:start+128] = torch.fft.fft2(
                    self.norm(bts[start:start+128]), dim=(-2,-1), norm="ortho")

    def rotation_branch(self, query: int, cand: np.ndarray) -> tuple:
        """정규화한 FFT 피연산자를 재사용해 원래 회전 점수·각도를 만든다."""
        import torch
        dists, angles = [], []
        with torch.no_grad():
            for start in range(0, len(cand), 128):
                others = self.rotation[cand[start:start+128]]
                corr = torch.fft.ifft2(self.rotation[query:query+1] * others.conj(), dim=-2, norm="ortho")
                corr = torch.sqrt(corr.real**2 + corr.imag**2 + 1e-15)
                corr = torch.sum(corr, dim=-3)
                corr = torch.sum(corr, dim=-1)
                score = torch.max(corr, dim=-1)[0] / (0.15*self.h*self.w)
                corr = torch.fft.fftshift(corr, dim=-1)
                shift = corr.shape[-1]//2 - torch.argmax(corr, dim=-1)
                dists.append(1-score.cpu().numpy().reshape(-1))
                angles.append((shift*2*np.pi/self.h).cpu().numpy().reshape(-1))
        return np.concatenate(dists), np.concatenate(angles)

    def translation_branch(self, qfeat, cand: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """원래 고유 각도 추론과 두 방향 평행이동 오차의 최솟값을 계산한다."""
        import torch
        errors = []
        with torch.no_grad():
            a1 = torch.tensor(angles, dtype=torch.float)
            a2 = torch.tensor(angles-np.pi, dtype=torch.float)
            unique, inverse = torch.unique(torch.cat([a1,a2]), return_inverse=True)
            qmaps = torch.cat([self.trans(self.base.rotate_bev_batch(qfeat,unique[k:k+128]))
                               for k in range(0,len(unique),128)])
            qfft = torch.cat([torch.fft.fft2(self.norm(qmaps[k:k+128]),dim=(-2,-1),norm="ortho")
                              for k in range(0,len(qmaps),128)])
            inv1, inv2 = inverse[:len(a1)], inverse[len(a1):]
            for start in range(0,len(cand),128):
                cfft = self.translation[cand[start:start+128]]
                pair = []
                for inv in (inv1,inv2):
                    query = qfft[inv[start:start+128].to(qfft.device)]
                    corr = torch.fft.ifft2(query*cfft.conj(),dim=(-2,-1),norm="ortho")
                    corr = torch.sqrt(corr.real**2+corr.imag**2+1e-15)
                    corr = torch.sum(corr,dim=-3)
                    corr = torch.fft.fftshift(corr,dim=(-2,-1))
                    pair.append((1-corr.amax(dim=(-2,-1))/(0.15*self.h*self.w)).float().cpu().numpy().reshape(-1))
                errors.append(np.minimum(*pair))
        return np.concatenate(errors)


def verify() -> None:
    """저장된 실제 큰 시퀀스와 동률 피연산자의 점수·순서를 원경로와 대조한다."""
    import torch
    import _ringsharp_official as original
    from _review_selection_bev import read, write, sha
    from _review_selection_full_inputs import OUT, INPUT

    torch.set_num_threads(4)
    home = OUT/"ring_cache_audit"
    home.mkdir(parents=True,exist_ok=True)
    candidate = read(OUT/"retained_protocol.json")["methods"]["ringsharp"]["1"]["candidates"][0]
    assert sha(ROOT/candidate["path"]) == candidate["sha256"]
    features = OUT/f"retained/val/ringsharp/s1/{candidate['id']}/features_mulran_Sejong01.pt"
    saved = torch.load(features,map_location="cuda",weights_only=False)
    specs,bts = saved["specs"],saved["bts"]
    base = original._base()
    model,trans = base.load_model(str(ROOT/candidate["path"]),"cuda")
    packed = np.load(INPUT/"ringsharp/mulran_Sejong01/packed.npy",mmap_mode="r")
    tick=time.time(); cached=CachedRerank(base,trans,specs,bts); torch.cuda.synchronize()
    build_s=time.time()-tick
    checks=[]

    def check(q, candidates, label, cache=cached, bank=specs, translations=bts):
        tick=time.time(); d1,a1=original.rotation_branch(base,bank,bank[q:q+1],candidates)
        rotation_original_s=time.time()-tick
        tick=time.time(); d2,a2=cache.rotation_branch(q,candidates)
        rotation_cached_s=time.time()-tick
        order1,order2=np.argsort(d1),np.argsort(d2)
        top=order1[:original._depth(len(candidates))]
        qfeat=original.query_features(base,model,packed[q:q+1],"cuda")
        tick=time.time(); e1=original.translation_branch(base,trans,translations,qfeat,candidates[top],a1[top],False)
        translation_original_s=time.time()-tick
        tick=time.time(); e2=cache.translation_branch(qfeat,candidates[top],a1[top])
        translation_cached_s=time.time()-tick
        equal={"rotation_score":bool(np.array_equal(d1,d2)),"angle":bool(np.array_equal(a1,a2)),
               "rotation_order":bool(np.array_equal(order1,order2)),"translation_score":bool(np.array_equal(e1,e2)),
               "winner":bool(np.argmin(e1)==np.argmin(e2))}
        record=dict(label=label,query=q,n_candidates=len(candidates),equal=equal,
            max_rotation_diff=float(np.max(np.abs(d1-d2))),max_translation_diff=float(np.max(np.abs(e1-e2))),
            rotation_original_s=rotation_original_s,rotation_cached_s=rotation_cached_s,
            translation_original_s=translation_original_s,translation_cached_s=translation_cached_s)
        checks.append(record);write(home/"progress.json",checks)
        print("RING_CACHE_CHECK",record,flush=True)
        assert all(equal.values()),record

    for q in [30,31,128,1029,4096,8192,12030,16030,len(specs)-16]:
        check(q,np.arange(q-29),f"actual_{q}")
    # 같은 지도 특징으로 생기는 동률 및 한 ULP 차이도 후보 순서까지 대조한다.
    for label in ("duplicate","near_tie"):
        ss=specs[:700].clone();bb=bts[:700].clone()
        ss[:512]=ss[0];bb[:512]=bb[0]
        if label=="near_tie":
            ss[1:512:2]=torch.nextafter(ss[1:512:2],torch.full_like(ss[1:512:2],float("inf")))
            bb[1:512:2]=torch.nextafter(bb[1:512:2],torch.full_like(bb[1:512:2],float("inf")))
        synthetic=CachedRerank(base,trans,ss,bb)
        check(650,np.arange(512),label,synthetic,ss,bb)
        del synthetic,ss,bb
    write(home/"validated.json",dict(completed_unix=time.time(),all_exact=True,checks=checks,build_s=build_s,
        implementation_sha256=sha(Path(__file__)),original_sha256=sha(ROOT/"scripts/_ringsharp_official.py"),
        cuda_peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
        rule="지도별 InstanceNorm+FFT 캐시만; batch128·공식1000재정렬·FP32점수·원래동률 유지"))


if __name__=="__main__":
    os.environ["NVIDIA_TF32_OVERRIDE"]="0"
    verify()
