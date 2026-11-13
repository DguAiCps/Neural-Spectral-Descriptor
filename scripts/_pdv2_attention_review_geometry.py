#!/usr/bin/env python3
"""거리·회전 입력이 간선 부호기와 층별 편향에서 어느 정도 남는지 관측한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(REPO/'scripts')]
import _pdv2_attention_review as run
from _pdv2_input_alignment_lib import temporal_edges,pose_attributes
from _pdv2_no_context_ablation import sha256,write_json


def centered_rms(x:torch.Tensor)->float:
    """같은 중심의 이웃 평균을 뺀 성분의 제곱평균제곱근이다."""
    return float((x-x.mean(1,keepdim=True)).square().mean().sqrt())


def main()->None:
    """동결 모델 다섯 개의 같은 검증 간선 표본을 비교한다."""
    spec=run.protocol(True);old=run.source.previous.previous
    raw,sources={}, {str(Path(__file__).relative_to(REPO)):sha256(Path(__file__))}
    for seq in spec['val']:
        _,_,paths,_=run.base.exp.old.base.inputs('val',seq)
        cache=run.base.exp.arrays(paths['cache']);n=len(cache['poses'])
        edges=temporal_edges(np.zeros(n,np.int64));degree=np.bincount(edges[1],minlength=n)
        centers=np.flatnonzero(degree==10);centers=centers[np.linspace(0,len(centers)-1,512).astype(int)]
        idx=np.flatnonzero(np.isin(edges[1],centers));idx=idx[np.argsort(edges[1,idx]*n+edges[0,idx])]
        attr=pose_attributes(cache['poses'],edges[:,idx])
        raw[seq]=torch.from_numpy(attr).cuda()
        sources[str(Path(paths['cache']).relative_to(REPO))]=sha256(Path(paths['cache']))
    records={}
    with torch.no_grad():
        for arm in run.CFG['diagnostic_models']:
            parent,name=(old,'diff') if arm=='original_diff' else (run.source,arm)
            selected=run.read(parent.OUT/'summary.json')['per_seed']['1'][name]['selected']
            checkpoint=parent.OUT/'s1'/name/'checkpoints'/f"checkpoint_epoch_{selected['epoch']}.pth"
            sources[str(checkpoint.relative_to(REPO))]=sha256(checkpoint)
            net=parent.model(name).cuda().eval()
            net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'])
            records[arm]={'epoch':selected['epoch'],'sequences':{},'bias_last_weight_norms':[
                float(c.bias_mlp[-1].weight.norm()) for c in net.gnn.convs]}
            for seq,attr in raw.items():
                encoded=net.gnn.edge_encoder(attr,torch.zeros(len(attr),dtype=torch.long,device='cuda'))
                values={'raw_distance_centered_rms':centered_rms(attr[:,0].reshape(512,10,1)),
                    'raw_rotation_centered_rms':centered_rms(attr[:,1].reshape(512,10,1)),
                    'encoded_centered_rms':centered_rms(encoded.reshape(512,10,32)),'layers':{}}
                for i,c in enumerate(net.gnn.convs):
                    hidden=c.bias_mlp[0](encoded).relu()
                    bias=c.bias_mlp[-1](hidden)
                    values['layers'][str(i)]={'relu_active_fraction':float((hidden>0).float().mean()),
                        'hidden_centered_rms':centered_rms(hidden.reshape(512,10,32)),
                        'bias_centered_rms':centered_rms(bias.reshape(512,10,4))}
                records[arm]['sequences'][seq]=values
            print('GEOMETRY_OBSERVED',arm,flush=True)
    for p,h in sources.items():assert sha256(REPO/p)==h
    write_json(run.OUT/'geometry_diagnosis.json',{'models':records,'sources':sources,'test_used':False,
        'training_performed':False,'scope':'같은 검증 간선의 순전파 관측이다. 학습 기울기나 원인 확증 결과가 아니다.'})

if __name__=='__main__':
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    main()
