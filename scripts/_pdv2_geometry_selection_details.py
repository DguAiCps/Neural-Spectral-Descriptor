#!/usr/bin/env python3
"""KITTI 정답 손실의 고정 후보 쌍에서 원 특징·문맥 키 점수를 분해한다."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """저장된 키만 사용하며 최종 판정이나 후보를 바꾸지 않는다."""
    root = REPO / 'results/pdv2_geometry_interaction_20261023'
    usage = REPO / 'results/pdv2_geometry_interaction_usage_20261023'
    source = root / 'mechanism_diagnosis.json'
    diagnosis = json.loads(source.read_text())
    sources = {str(source.relative_to(REPO)): sha256(source),
               str(Path(__file__).relative_to(REPO)): sha256(Path(__file__))}
    records = []
    for seq, result in diagnosis['final']['comparisons']['interaction']['sequences'].items():
        cases = result.get('lost_correct_queries', [])
        if not cases:
            continue
        key = {}
        for arm in ('additive', 'interaction'):
            p = root / 's1' / arm / 'epoch20' / ('keys_' + seq.replace('/', '_') + '.npz')
            sources[str(p.relative_to(REPO))] = sha256(p)
            with np.load(p) as z:
                key[arm] = z['embedding'].copy()
        probes = {}
        for arm in ('interaction_off', 'geometry_off'):
            p = usage / 's1' / arm / 'epoch20' / (seq.replace('/', '_') + '.npz')
            sources[str(p.relative_to(REPO))] = sha256(p)
            with np.load(p) as z:
                probes[arm] = {int(q): bool(d <= 5) for q, d in zip(z['query_idx'], z['top1_geo_dist'])}
        for case in cases:
            q, true, false = case['query'], case['old_winner'], case['new_winner']
            row = {'sequence': seq, **case, 'fixed_pair_true': true, 'fixed_pair_false': false,
                   'probe_recovers': {a: p[q] for a, p in probes.items()}, 'margins': {}}
            for arm, emb in key.items():
                x, positive, negative = [emb[i].astype(np.float64) for i in (q, true, false)]
                raw = x[:288] @ (positive[:288] - negative[:288])
                context = x[288:] @ (positive[288:] - negative[288:])
                total = x @ (positive - negative)
                assert abs(raw + context - total) < 1e-12
                normalized_raw = (x[:288] / np.linalg.norm(x[:288])) @ (
                    positive[:288] / np.linalg.norm(positive[:288]) -
                    negative[:288] / np.linalg.norm(negative[:288]))
                row['margins'][arm] = {'raw_block': float(raw), 'context_block': float(context),
                    'total': float(total), 'raw_cosine_only': float(normalized_raw)}
            np.testing.assert_allclose(row['margins']['additive']['raw_cosine_only'],
                                       row['margins']['interaction']['raw_cosine_only'], atol=1e-6, rtol=0)
            row['context_margin_change'] = row['margins']['interaction']['context_block'] - row['margins']['additive']['context_block']
            row['key_margin_change'] = row['margins']['interaction']['total'] - row['margins']['additive']['total']
            records.append(row)
    summary = {'lost_queries': len(records),
        'farther_than_25m': sum(r['new_winner_distance_m'] > 25 for r in records),
        'raw_cosine_prefers_true': sum(r['margins']['interaction']['raw_cosine_only'] > 0 for r in records),
        'new_total_key_prefers_true_in_this_pair': sum(r['margins']['interaction']['total'] > 0 for r in records),
        'context_margin_deteriorates': sum(r['context_margin_change'] < 0 for r in records),
        'total_margin_deteriorates': sum(r['key_margin_change'] < 0 for r in records),
        'recovered_by_probe': {a: sum(r['probe_recovers'][a] for r in records) for a in ('interaction_off', 'geometry_off')}}
    for p, digest in sources.items():
        assert sha256(REPO / p) == digest
    write_json(root / 'selection_details.json', {'summary': summary, 'cases': records, 'sources': sources,
        'test_used': False, 'passed': True,
        'scope': '사후선정한KITTI정답손실8개. 기준최종정답과새최종오답의고정두후보. 키원특징288/문맥128내적의합항등식이며블록제거재학습기여가아니다. 전체키정규화·문맥게이트가블록크기에영향을준다. 원특징코사인은별도정규화한다.'})
    print('SELECTION_DETAILS', summary, flush=True)


if __name__ == '__main__':
    main()
