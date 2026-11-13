#!/usr/bin/env python3
"""표2의 공식 RING#-L 경로를 실제 도착 단위로 계측한다."""
import argparse
import gc
import resource
import time
from pathlib import Path
import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _ringsharp_official as official


def settings() -> dict:
    """기존 선택 가중치와 현재 공식 경로만 사용한다."""
    selection=bench.read(official.SEL)
    relative=str(selection['selected_weight']).split('Neural-Spectral-Codec/')[-1]
    path=bench.REPO/relative
    assert bench.sha(path)==selection['weight_sha256']
    return {'weight':str(path),'weight_sha256':selection['weight_sha256'],
            'source_sha256':bench.sha(Path(__file__)),
            'selection_sha256':bench.sha(official.SEL),
            'parent_protocol_sha256':bench.sha(bench.OUT/'protocol.json'),
            'retrieval':'official exhaustive rotation; top1000 translation; rotate raw128channel query then trans_cnn; exact vectorized translation errors',
            'source_files':{str(p.relative_to(bench.REPO)):bench.sha(p) for p in (Path(official.__file__),bench.REPO/'scripts/_ringsharp_eval.py')}}


def load_models(config:dict):
    """공식 factory의 가중치를 읽는다."""
    base=official._base()
    model,trans=base.load_model(config['weight'],'cuda')
    model.eval();trans.eval()
    return base,model,trans


def encode(base,model,trans,points):
    """공식 BEV 생성과 신경망·map translation 특징을 만든다."""
    pts=np.ascontiguousarray(points[:,:3],dtype=np.float64)
    bev=base.generate_bev(pts,Z=base.NSC_Z,Y=base.NSC_Y,X=base.NSC_X,
                          bounds=official.rol.ringsharp_bounds('mulran'))
    packed=np.packbits(bev.numpy().reshape(-1)>0)
    with torch.no_grad():
        out=model({'pc':bev[None].cuda(),'img':None})
        spec=out['spec'][0].float()
        bt=trans(out['bev'])[0].float()
    return spec,bt,packed


def cache(n_map:int)->None:
    """원시 스캔을 공식 batch1 경로로 초기지도 준비한다."""
    from _pd_build_test_caches import _make_loader
    config=settings();target=bench.OUT/f'ringsharp_cache_n{n_map}'
    meta=target.with_suffix('.json')
    if meta.exists():
        old=bench.read(meta);assert old['config']==config
        return
    work=bench.Workload(bench.read(bench.OUT/'protocol.json'),n_map)
    loader=_make_loader('mulran','Sejong01')
    base,model,trans=load_models(config)
    first=encode(base,model,trans,loader[int(work.ids[0])]['points'])
    files={}
    arrays=[]
    for name,value in zip(('spec','bt','packed'),first):
        shape=value.shape;dtype=np.uint8 if name=='packed' else np.float32
        path=target.with_name(target.name+'_'+name+'.npy')
        arrays.append(np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=(work.end,*shape)))
        files[name]=str(path.relative_to(bench.REPO))
    started=time.time()
    for i,sid in enumerate(work.ids):
        result=first if i==0 else encode(base,model,trans,loader[int(sid)]['points'])
        for dst,value in zip(arrays,result):
            dst[i]=value.cpu().numpy() if isinstance(value,torch.Tensor) else value
        if i%100==0:
            print('RINGSHARP_CACHE',i,work.end,round(time.time()-started,1),flush=True)
            bench.write(bench.OUT/'status.json',{'status':'ringsharp_batch1_initial_cache','done':i,'total':work.end})
    for a in arrays:a.flush()
    bench.write(meta,{'config':config,'files':files,'sha256':{k:bench.sha(bench.REPO/v) for k,v in files.items()},
                     'frames':work.end,'seconds':time.time()-started})


class Runtime:
    """공식 검색을 위한 GPU스펙트럼·map특징과 점유비트 상태다."""
    def __init__(self,work):
        self.work=work;self.config=settings()
        meta=bench.read(bench.OUT/f'ringsharp_cache_n{work.n_map}.json')
        assert meta['config']==self.config
        self.sources={k:np.load(bench.REPO/v,mmap_mode='r') for k,v in meta['files'].items()}
        for k,v in meta['files'].items():assert bench.sha(bench.REPO/v)==meta['sha256'][k]
        self.base,self.model,self.trans=load_models(self.config)
        self.spec=torch.empty(self.sources['spec'].shape,device='cuda',dtype=torch.float32)
        self.bt=torch.empty(self.sources['bt'].shape,device='cuda',dtype=torch.float32)
        self.packed=np.empty_like(self.sources['packed'])
        self.reset()

    def reset(self):
        self.n=self.work.initial
        self.spec[:self.n]=torch.from_numpy(np.asarray(self.sources['spec'][:self.n]).copy()).cuda()
        self.bt[:self.n]=torch.from_numpy(np.asarray(self.sources['bt'][:self.n]).copy()).cuda()
        self.packed[:self.n]=self.sources['packed'][:self.n]

    def encode(self,points):
        return encode(self.base,self.model,self.trans,points)

    def append(self,values):
        spec,bt,packed=values
        self.spec[self.n]=spec;self.bt[self.n]=bt;self.packed[self.n]=packed;self.n+=1

    def query(self,q,reference=False):
        candidates=np.arange(q-29)
        dist,angle=official.rotation_branch(self.base,self.spec,self.spec[q:q+1],candidates)
        order=np.argsort(dist);top=order[:official._depth(len(candidates))]
        query=official.query_features(self.base,self.model,self.packed[q:q+1],'cuda')
        error=official.translation_branch(self.base,self.trans,self.bt,query,candidates[top],angle[top],reference,dedup=not reference)
        best=int(np.argmin(error))
        return int(top[best]),-float(error[best]),len(top)

    def memory(self):
        gpu=torch.cuda.memory_allocated()
        model=bench.tensor_bytes(self.model);trans=bench.tensor_bytes(self.trans)
        # trans_cnn is also a submodule of model in the official loader: its storage
        # is already contained in gpu allocated and is never added a second time.
        cpu=self.packed.nbytes+model['cpu']
        return {'cpu':cpu,'gpu':gpu,'total':cpu+gpu,
                'breakdown':{'packed_occupancy_capacity':self.packed.nbytes,'model_cpu':model['cpu']},
                'gpu_spec_storage':self.spec.untyped_storage().nbytes(),'gpu_translation_storage':self.bt.untyped_storage().nbytes(),
                'model_gpu':model['gpu'],'trans_model_gpu_diagnostic':trans['gpu'],
                'logical_feature_payload_bytes_per_frame':int(self.sources['spec'][0].nbytes+self.sources['bt'][0].nbytes),
                'timepoint':'final arrival; full preallocated map+45support capacity; GPU allocated includes model and persistent workspace'}


def measure(n_map:int)->None:
    config=settings();work=bench.Workload(bench.read(bench.OUT/'protocol.json'),n_map)
    runtime=Runtime(work);checks=[]
    for i in range(bench.WARMUP):
        t=work.initial+i;values=runtime.encode(work.points[i])
        for key,value in zip(('spec','bt','packed'),values):
            actual=value.cpu().numpy() if isinstance(value,torch.Tensor) else value
            np.testing.assert_allclose(actual,runtime.sources[key][t],atol=1e-6,rtol=1e-6)
        runtime.append(values);answer=runtime.query(t-15)
        if i in (0,bench.WARMUP-1):
            ref=runtime.query(t-15,True)
            assert answer[0]==ref[0] and abs(answer[1]-ref[1])<1e-5
            checks.append({'query':t-15,'winner':answer[0],'reference_score_error':abs(answer[1]-ref[1])})
    verification={'status':'passed','raw_batch1_feature_checks':bench.WARMUP,'official_loop_checks':checks,
                  'checkpoint_sha256':config['weight_sha256']}
    bench.write(bench.OUT/'verification'/f'ringsharp_n{n_map}.json',verification)
    rows=[];memories=[]
    bench.write(bench.OUT/'status.json',{'status':'measuring','method':'ringsharp','n_map':n_map})
    for repeat in range(bench.REPEATS):
        runtime.reset();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();before=bench.rss()
        for i,points in enumerate(work.points):
            t=work.initial+i
            encoded,enc=bench.timed(lambda:runtime.encode(points))
            _,state=bench.timed(lambda:runtime.append(encoded))
            answer,query=bench.timed(lambda:runtime.query(t-15))
            rows.append({'repeat':repeat,'arrival':t,'query':t-15,'n_candidates':t-44,'encode':enc,'state_update':state,
                         'update':enc+state,'query_ms':query,'processing':enc+state+query,'winner':answer[0],'confidence':answer[1],
                         'rerank_candidates':answer[2]})
            if i%24==0:print('RINGSHARP_MEASURE',repeat,i,round(query,2),flush=True)
        mem=runtime.memory();mem.update({'process_rss_before':before,'process_rss_after':bench.rss(),
            'process_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cuda_allocated':torch.cuda.memory_allocated(),
            'cuda_reserved':torch.cuda.memory_reserved(),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved':torch.cuda.max_memory_reserved()});memories.append(mem)
    for i in range(bench.SAMPLES):
        matched=[rows[k*bench.SAMPLES+i] for k in range(bench.REPEATS)]
        assert len({r['winner'] for r in matched})==1
        assert np.ptp([r['confidence'] for r in matched])<1e-6
    result={'schema':'pdv2-table3-measurement-v1','method':'ringsharp','n_map':n_map,'status':'passed','publish_ready':True,
            'protocol_sha256':bench.sha(bench.OUT/'protocol.json'),'source_sha256':bench.sha(Path(__file__)),
            'method_config':config,'sequence':bench.SEQUENCE,'seed':1,'candidate_range':[n_map,n_map+bench.SAMPLES-1],
            'mean_candidates':n_map+(bench.SAMPLES-1)/2,'samples':bench.SAMPLES,'repeats':bench.REPEATS,'warmup':bench.WARMUP,
            'timings_ms':{k:bench.stats([r[f] for r in rows]) for k,f in (('encode','encode'),('state_update','state_update'),
                ('update','update'),('query','query_ms'),('processing','processing'))},
            'resident_bytes':memories[-1],'memory_repeats':memories,'verification':verification,'raw_samples':rows,
            'raw_input_fixture_bytes':work.input_bytes,'raw_input_sha256':work.input_hash,'completed_unix':time.time(),
            'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'threads':1,'tf32':False},
            'scope':'official batch1 160x160x20 BEV, selected checkpoint; full prefix rotation and1000 translation; no labels or tuning'}
    bench.write(bench.OUT/'methods'/f'ringsharp_n{n_map}.json',result);bench.summarize()
    print('COMPLETE_RINGS HARP',n_map,result['timings_ms'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('cache','measure'))
    parser.add_argument('--n-map',type=int,default=1000);args=parser.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    globals()[args.action](args.n_map)
