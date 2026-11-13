#!/usr/bin/env python3
"""문맥 인코더의 장소 제외와 학습량 대조를 고정 예산으로 수행한다."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src'), str(REPO/'scripts')]
import _pdv2_temporal_focus as focus
import _pdv2_train_failure as previous
import _pdv2_context_accept as accept
from _pdv2_context_place_lib import PLACES
from _pdv2_encoder_place_lib import restricted_triplets, temporal_subgraph
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from _xs_eval_lib import normalize

OUT = REPO/'results/pdv2_context_encoder_place_20261019'
SAMPLES = focus.old.SAMPLES
SEED, EPOCHS = 1, 20
REFRESH = (1, 6, 11, 16)


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def folder(arm: str) -> Path:
    """모든 새 산출물을 독립 실험 폴더에 둔다."""
    return OUT/arm


def model():
    """현재 288차원·문맥 게이트·표준 attention 모델을 새로 만든다."""
    from gnn.model import create_spectral_gnn
    cfg = yaml.safe_load((focus.OUT/'standard.yaml').read_text())
    return create_spectral_gnn(input_dim=288, hidden_dim=128, context_dim=128,
        n_layers=2, n_heads=4, dropout=.1, norm_type='batch_norm',
        use_residual_gate=True, gate_hidden_dim=64, gate_initial_alpha=.0625,
        diffattn_value_source='standard', edge_encoder_config=cfg['gnn']['edge_encoding'])


def protocol() -> dict:
    """학습 전에 고정한 코드·규칙이 변하지 않았는지 확인한다."""
    spec = read(OUT/'protocol.json')
    for p, h in spec['runtime'].items():
        assert sha256(REPO/p) == h, p
    return spec


def prepare() -> None:
    """장소 제외·학습량 대조·검증 적용 조건을 학습 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        protocol()
        return
    state = torch.load(previous.PREPARED, map_location='cpu', weights_only=False)
    graph = torch.load(previous.GRAPH, map_location='cpu', weights_only=False)
    seq = state['sequences']
    roster = [a+'/'+b for a,b in previous.TRAIN]
    np.testing.assert_array_equal(graph.x.numpy(), state['descriptors'])
    original, inputs = {}, {previous.PREPARED, previous.GRAPH, SAMPLES/'audit.json',
                          focus.OUT/'standard.yaml', focus.base.RVEC,
                          REPO/'results/pdv2_context_place_20261018/place_audit.json',
                          REPO/'results/pdv2_context_accept_20261018/protocol.json'}
    audit = read(SAMPLES/'audit.json')
    for ep in REFRESH:
        p = SAMPLES/f'original_epoch{ep}.npz'
        assert sha256(p) == audit['sample_sha256'][p.name]
        original[ep] = np.load(p)['triplets']
        a,b,c = original[ep].T
        assert np.all(seq[a] == seq[b]) and np.all(seq[a] == seq[c])
        inputs.add(p)
    plans = {'full': {'heldout': [], 'budget_place': None}}
    pairs = {}
    for place, held in PLACES.items():
        leave = 'leave_'+place
        ids = np.flatnonzero(~np.isin(seq, [roster.index(s) for s in held]))
        n = len(restricted_triplets(original[1], ids, len(seq)))
        matched = 'full' if n == len(original[1]) else 'matched_'+place
        plans[leave] = {'heldout': list(held), 'budget_place': place}
        if matched != 'full':
            plans[matched] = {'heldout': [], 'budget_place': place}
        pairs[place] = {'leave': leave, 'matched': matched, 'n_triplets': n}
    for arm, plan in plans.items():
        root = folder(arm)
        root.mkdir(parents=True, exist_ok=True)
        ids = np.flatnonzero(~np.isin(seq, [roster.index(s) for s in plan['heldout']]))
        np.save(root/'training_node_ids.npy', ids)
        part = temporal_subgraph(graph, ids)
        assert np.all(seq[ids[part.edge_index[0]]] == seq[ids[part.edge_index[1]]])
        counts = []
        for ep in REFRESH:
            trips = restricted_triplets(original[ep], ids, len(seq))
            if arm.startswith('matched_'):
                count = pairs[plan['budget_place']]['n_triplets']
                rows = np.random.default_rng(71000+ep).choice(len(trips), count, replace=False)
                trips = trips[rows]
            if plan['budget_place']:
                assert len(trips) == pairs[plan['budget_place']]['n_triplets']
            np.savez_compressed(root/f'triplets_epoch{ep}.npz', triplets=trips)
            counts.append(len(trips))
        plan.update(nodes=len(ids), triplets_per_epoch=counts[0],
                    updates_per_epoch=int(np.ceil(counts[0]/4096)),
                    graph_hashes={k:array_hash(getattr(part,k).numpy())
                                  for k in ('x','edge_index','edge_attr','edge_type')})
        assert len(set(counts)) == 1
        inputs.update(root.glob('*.npz'))
        inputs.add(root/'training_node_ids.npy')
    # 원 삼중쌍의 정답·오답은 시퀀스 내부에서만 뽑으므로 전체 시퀀스 제외와 독립이다.
    inputs.update(REPO/p for p in read(REPO/'results/pdv2_context_accept_20261018/protocol.json')['sources']
                  if p.startswith('data/') or p.endswith('freeze.json'))
    runtime = {Path(__file__), REPO/'scripts/_pdv2_encoder_place_lib.py',
               REPO/'scripts/_pdv2_encoder_place_run.py',
               REPO/'tests/gnn/test_encoder_place.py', REPO/'src/gnn/model.py',
               REPO/'src/gnn/trainer.py', REPO/'src/keyframe/graph_manager.py',
               REPO/'scripts/_pdv2_context_accept_lib.py', REPO/'scripts/_pdv2_context_place_lib.py',
               REPO/'src/retrieval/sequence_evidence.py', REPO/'scripts/_pdv2_train_failure.py',
               REPO/'scripts/_pdv2_no_context_ablation.py', REPO/'scripts/_pr_eval_lib.py'}
    spec = {'created_unix':time.time(), 'seed':SEED, 'epochs':EPOCHS, 'plans':plans,
        'pairs':pairs, 'train':roster, 'places':PLACES, 'test_used':False,
        'training':'모든 조건 처음부터, 같은 초기 상태·InfoNCE·20에포크 LR. 장소 노드·간선·삼중쌍을 제거. 남은 표본 반복 추가 없음. matched는 전체 원 표본의 행을 비복원 추출해 각 제외 조건과 같은 배치·업데이트 수. 시드1 예비.',
        'checkpoint':'결과 미관측으로 20에포크 고정. 10에포크는 저장만 하고 선택/평가하지 않음. 검증/제외 장소로 조기종료 안함.',
        'encoder':'no_interdiff 288차원 캐시, 표준 attention2층, 시간±5, gate초기.0625. 직접 GNNTrainer 사용, spectral policy/재계량 학습 없음. TF32 비활성.',
        'evaluation':'GT학습8에서 동일 고정 승자 문맥 판별 대조와 각 모델의 실제100+100검색/최종판정 둘 다. 재계량·융합·판정은 기존 시드1 상수 고정. q+15,j<=q-30. 인코더 제외이며 전체 파이프라인 설정의 미관측장소 확증 아님.',
        'calibration':'full모델의 학습 최종 원장8 대 제외모델 OOF 최종 원장8로 각각 baseline+context_arc 양수 BCE/L2.001 적합. 학습 장소에서만 적합, 계열→시퀀스→질의 균등. 추가 계수100% 사전 고정, 가중치 격자 탐색 없음.',
        'validation':'두 적합을 동결 후 새 full모델의 KISS검증9 최종 원장에 적용. 같은 full모델 baseline/in_sample/crossfit 세조건. 현재 선택 모델의 시드1 수치는 별도 참고. 주조건crossfit.',
        'promotion':'시드1 crossfit가 같은full baseline과in_sample 각각 AP+.001이상,R100손실<=.001,계열AP손실<=.0025일 때만 3시드 확증 후보. 이번 실험에서 시험 평가/원고 교체 안함.',
        'limits':'장소 제외와 자료·센서 분포 이동이 함께 달라짐. 학습량 대조도 학습 구성·그래프 크기까지 같지는 않음. OOF는 서로 다른 모델의 출력. Bridge 제외는KITTI만이며40업데이트. 고정 재계량/융합/기존승자는 기존 전체학습·검증 선택에 노출.',
        'runtime':{str(p.relative_to(REPO)):sha256(p) for p in sorted(runtime)},
        'inputs':{str(p.relative_to(REPO)):sha256(p) for p in sorted(inputs)}}
    write_json(OUT/'protocol.json', spec)
    print('PREPARED', {a:(p['nodes'],p['triplets_per_epoch'],p['updates_per_epoch']*20) for a,p in plans.items()}, flush=True)


def train(arm: str) -> None:
    """제외한 노드가 없는 그래프에서 고정된 마지막 에포크까지 학습한다."""
    from gnn.trainer import GNNTrainer
    spec = protocol()
    plan, root = spec['plans'][arm], folder(arm)
    if (root/'training_complete.json').exists():
        done = read(root/'training_complete.json')
        assert sha256(root/'checkpoints/final_model.pth') == done['checkpoint_sha256']
        return
    assert not (root/'checkpoints').exists(), '미완료 실행을 덮어쓰지 않는다'
    state = torch.load(previous.PREPARED, map_location='cpu', weights_only=False)
    original = torch.load(previous.GRAPH, map_location='cpu', weights_only=False)
    ids = np.load(root/'training_node_ids.npy')
    graph = temporal_subgraph(original, ids)
    for key, h in plan['graph_hashes'].items():
        assert array_hash(getattr(graph,key).numpy()) == h
    del original
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    net = model()
    write_json(root/'initial_model.json', {k:array_hash(v.numpy()) for k,v in net.state_dict().items()})
    trainer = GNNTrainer(net, learning_rate=5e-4, weight_decay=1e-5, temperature=.1,
        checkpoint_dir=str(root/'checkpoints'), use_multi_gpu=False, patience=100,
        use_amp=True, loss_type='infonce')
    trainer.scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=1000000)
    updates, records = [], []
    trainer.optimizer.register_step_post_hook(lambda *args: updates.append(1))

    class FixedMiner:
        """학습 장소 내부의 사전 고정 삼중쌍만 돌려준다."""
        def mine_triplets(self, **kwargs):
            np.testing.assert_array_equal(kwargs['descriptors'], state['descriptors'][ids])
            ep = trainer.epoch+1
            assert ep in REFRESH
            p = root/f'triplets_epoch{ep}.npz'
            assert sha256(p) == spec['inputs'][str(p.relative_to(REPO))]
            return np.load(p)['triplets'].tolist()

    original_epoch = trainer.train_epoch

    def audited_epoch(*args, **kwargs):
        epoch = trainer.epoch+1
        seed = SEED*100000+epoch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        value = original_epoch(*args, **kwargs)
        assert np.isfinite(value) and len(updates) == epoch*plan['updates_per_epoch']
        assert all(torch.isfinite(v).all() for v in net.state_dict().values())
        records.append({'epoch':epoch,'loss':float(value),'updates':len(updates)})
        write_json(root/'training_progress.json', {'records':records})
        print('EPOCH',arm,epoch,float(value),len(updates),flush=True)
        return value

    trainer.train_epoch = audited_epoch
    tick = time.time()
    trainer.train(graph, state['poses'][ids], state['descriptors'][ids], state['sequences'][ids],
                  val_datasets=None, n_epochs=EPOCHS, triplet_miner=FixedMiner(),
                  mine_every_n_epochs=5, n_triplets_per_anchor=3)
    write_json(root/'training_complete.json', {'runtime_s':time.time()-tick,'epochs':EPOCHS,
        'updates':len(updates),'checkpoint_sha256':sha256(root/'checkpoints/final_model.pth'),
        'initial_sha256':sha256(root/'initial_model.json'),'test_used':False})


def training_keys(arm: str) -> None:
    """학습 종료 뒤 전체 학습 장소를 추론하고 접두 그래프로 관측 범위를 검사한다."""
    spec = protocol()
    root = folder(arm)
    done = root/'keys_complete.json'
    if done.exists():
        assert sha256(root/'retrieval_keys.npy') == read(done)['sha256']
        return
    checkpoint = root/'checkpoints/final_model.pth'
    assert sha256(checkpoint) == read(root/'training_complete.json')['checkpoint_sha256']
    net = model().cuda().eval()
    net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    state = torch.load(previous.PREPARED,map_location='cpu',weights_only=False)
    source = torch.load(previous.GRAPH,map_location='cpu',weights_only=False)
    graph = temporal_subgraph(source,np.arange(source.num_nodes)).cuda()
    with torch.no_grad():
        raw = net(graph).float().cpu().numpy()
    assert raw.shape == (72306,416) and np.isfinite(raw).all()
    checks=[]
    for ordinal,sequence in enumerate(spec['train']):
        ids=np.flatnonzero(state['sequences']==ordinal)
        q=int(ids[len(ids)//2]); end=q+16
        mask=(graph.edge_index < end).all(dim=0)
        from torch_geometric.data import Data
        small=Data(x=graph.x[:end],edge_index=graph.edge_index[:,mask],
                   edge_attr=graph.edge_attr[mask],edge_type=graph.edge_type[mask])
        with torch.no_grad():one=net(small)[q].float().cpu().numpy()
        np.testing.assert_allclose(one,raw[q],atol=3e-6,rtol=3e-5)
        checks.append({'sequence':sequence,'query':q-int(ids[0]),'max_error':float(np.abs(one-raw[q]).max())})
    raw[:,:288]=normalize(state['descriptors']*np.load(focus.base.RVEC)[None])
    key=normalize(raw)
    np.save(root/'retrieval_keys.npy',key)
    np.savez_compressed(root/'training_metadata.npz',positions=state['poses'][:,:3,3],
                        sequences=state['sequences'],coordinate_groups=state['frames'])
    write_json(done,{'sha256':sha256(root/'retrieval_keys.npy'),'prefix':checks,
                     'checkpoint_sha256':sha256(checkpoint)})
    print('KEYS',arm,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('prepare','train','keys'))
    p.add_argument('--arm')
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    logging.basicConfig(level=logging.WARNING)
    if a.action=='prepare':prepare()
    elif a.action=='train':train(a.arm)
    else:training_keys(a.arm)
