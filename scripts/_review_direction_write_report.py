#!/usr/bin/env python3
"""완료된 방향·회전·원인 분리 결과로 별도 실험 일지를 작성한다."""
from pathlib import Path
import _review_direction_metrics as base

OUT = base.OUT / 'expanded46'
TARGET = base.ROOT / 'docs/RAW/실험일지/2026-11-05_46시퀀스-방향학습과-yaw-원인분리.md'
LABEL = dict(forward_only='순방향 전용', reverse_only='역방향 전용', mixed='혼합', other_positive='나머지 양성')


def interval(value: dict) -> str:
    """%p 점추정과 구간을 같은 정밀도로 표현한다."""
    low, high = value['ci95_pp']
    return f"{value['delta_pp']:+.3f} [{low:+.3f}, {high:+.3f}]"


def main() -> None:
    """검산이 끝난 JSON만 사용해 보고서를 만든다."""
    direction = base.read(base.OUT / 'direction_summary.json')
    sequence = base.read(base.OUT / 'direction_learning/sequence_summary.json')
    block = base.read(base.OUT / 'direction_learning/summary.json')
    six = base.read(base.OUT / 'direction_learning/fixed_city_block_summary.json')
    yaw = base.read(OUT / 'yaw_summary.json')
    causes = base.read(OUT / 'causes/summary.json')
    grid = base.read(OUT / 'grid_probe/summary.json')
    proposal = base.read(OUT / 'phase_covariance_proposal.json')
    roll = base.read(OUT / 'roll_check/summary.json')
    spec = base.read(OUT / 'yaw_protocol.json')
    for record in (sequence, block, six, yaw, causes, grid, proposal, roll):
        assert record['status'] == 'passed'
    lines = [
        '# 46시퀀스의 방향별 학습 기여와 원시 yaw 회전 원인 분리', '',
        '- 날짜: 2026-11-05',
        '- 기준: `635594f` 이후 작업 트리. 기존 원고·그림·현재 모델·동결 실행기는 수정하지 않는다.',
        '- 결과 루트: `results/pdv2_review_direction_yaw_20261105/`', '',
        '## 왜 (진행 이유)', '',
        '리뷰가 요구한 실제 순·역방향 재방문 성능과 회전 강건성을 현재 `diff/matching_only` 모델에서 확인한다. 먼저 수행한 8시퀀스·512질의 진단을 포함해 시험 46시퀀스로 원시 회전 평가를 확장하고, 정수 칸 회전에서도 바뀌는 승자와 점수의 원인을 내부 대조로 분리한다. 실제 방향 평가는 전체 195,726질의이며 회전 평가는 사전 고정 2,944질의다.', '',
        '## 설정 (파라미터)', '',
        '### 실제 재방문 방향과 분모', '',
        '정답은 j≤q−30이며 GT 위치가 질의의 5m 이내인 모든 후보다. 과거 5키프레임 GT 이동이 1m 이상일 때 진행 방향을 정의한다. 모든 정답의 방향이 유효하고 질의와 60° 이하이면 순방향 전용, 모두 120° 이상이면 역방향 전용이다. 60° 이하와 120° 이상 정답이 각각 하나 이상 있으면 혼합이며 중간각·유효하지 않은 정답이 함께 있을 수 있다. 나머지는 별도 양성군이다.', '',
        '선택한 방향의 양성과 공통 음성(dmin>25m)만 PR 계산에 유지한다. 다른 방향 양성과 5m<dmin≤25m 질의는 제외한다. 반환 승자는 원래처럼 GT 거리≤5m이면 정답이며 승자의 방향으로 정답을 바꾸지 않는다. 방향 양성이 없는 시퀀스를 제외하고 시퀀스→도시→3시작점으로 평균한다. 혼합은 스톡홀름에 양성이 없어 5도시, 순방향·역방향은 6도시다.', '',
        '양성 61,682개는 순방향 33,031·역방향 23,422·혼합 2,997·나머지 2,232개다. 공통 음성은 101,270개, 모호 구간은 32,774개다. 원래 총계와 기존 방향 라벨을 재현했으며 현재 NSD 결과를 이전 v2.2 결과와 섞지 않았다.', '',
        '### 회전 표본과 물리적 의미', '',
        '46개 시퀀스 각각의 유효 질의 배열에서 `np.linspace(0, len(q)-1, 64).astype(int)`로 64개 순서 위치를 고른다. 초 단위 시간이나 이동거리의 등간격 표본은 아니다. 기존 8시퀀스·512질의를 정확히 포함하고 GT·성능으로 고르지 않는다. 각도는 0°, 0.5°, 1°, 6°, 7.3°, 90°, 180°이고 현재 최종 체크포인트 3개를 고정했다. 거리 영상의 각도 칸은 1°, BEV는 6°이므로 1°는 거리 영상에만 정수 칸이며 6°·90°·180°는 둘 다 정수 칸이다.', '',
        '질의마다 q±15 원시 점군을 회전해 다시 투영·인코딩하고 q±5의 수정 계수를 최종 시간 판정에 사용한다. 열벡터 기준 p′=Rz(θ)p, 자세 T′=T·diag(Rz(−θ),1)이므로 T′p′=Tp이며 병진과 주행 거리는 그대로다. 점 좌표는 원래 float32 저장 정밀도를 유지한다. 캐시를 단순 roll한 값을 원시 평가로 쓰지 않는다. KISS-ICP를 재실행하거나 실제 반대 방향에서 새 점군을 취득한 실험은 아니다.', '',
        'gallery의 원키·원보조·전체 수정 계수·후보측 시간 문맥은 원본으로 고정하고 query 표현만 독립적으로 회전한다. 후보 중심 j≤q−30이라도 후보 시간 대응과 두 층 문맥의 보수적 원시 상한 j+20이 query 시작 q−15와 겹칠 수 있다. 이 경계에서도 원본 gallery를 유지한다. 따라서 전체 연속 궤적 하나를 모두 회전한 실험으로 해석하지 않는다.', '',
        '원인 대조는 각 시퀀스의 64개 중 매 네 번째인 16개, 총 736질의에서 3시작점·6°/90°/180°·6조건으로 고정했다. 격자 검사는 각 시퀀스의 0·16·32·48번째 중심 184스캔이다. 대조는 문제 원인을 확인하는 내부 대체이며 모델 변경이나 추가 성능 채택이 아니다.', '',
        '센서·데이터셋별 표본 범위는 아래와 같고, 기존 시험의 6도시를 모두 포함한다.', '',
        '| 데이터셋 | 시퀀스 | 질의 | 고유 원시 지원 프레임 |', '|---|---:|---:|---:|',
    ]
    for dataset in sorted({s.split('/')[0] for s in spec['queries']}):
        sequences = [s for s in spec['queries'] if s.startswith(dataset+'/')]
        support = sum(spec['support_frames'][s] for s in sequences)
        lines.append(f'| {dataset} | {len(sequences)} | {64*len(sequences)} | {support:,} |')
    lines += [
        '', '## 결과', '',
        '### 전체 방향별 최종 성능', '',
        '| 방향 | 방법 | AP (%) | 최대 F1 (%) | R@100P (%) | R@1 (%) |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for group in LABEL:
        for method in ('identity', 'fixed', 'convs_only', 'abs_kv', 'diff', 'bev_native'):
            values = direction['methods'][method]['mean_std'][group]
            lines.append('| ' + ' | '.join([LABEL[group], method, *[f"{100*values[k]['mean']:.3f}" for k in base.KEYS]]) + ' |')
    lines += ['', '출처: `direction_summary.json`, `direction_metrics.csv`, `direction_city_dataset.csv`. 세 시작점·도시·데이터셋별 수치와 모든 추가 대조는 원천 JSON/CSV에 보존한다. 역방향에서 현재 NSD는 BEVPlace++보다 AP·F1·R@100P가 높지만 R@1은 낮다. 순방향은 네 지표 모두 BEVPlace++가 높다.', '',
        '### 학습 기여의 짝지은 조건부 구간', '',
        '아래는 기존 통합 보고서와 같은 도시 내 시퀀스 복원추출 10,000회의 95% percentile 구간이다. 해당 방향 양성이 있는 시퀀스만 도시 내 추출 모집단으로 삼고 그 수를 유지한다. 세 시작점은 평균하고 다시 추출하지 않으며 같은 방향의 비교 간 같은 표본을 사용한다. 전체 방향을 합친 기존 구간도 재현했다.', '',
        '| 방향 | 비교 | AP 차이와 95% 구간 (%p) | F1 | R@100P | R@1 |', '|---|---|---:|---:|---:|---:|']
    for group in ('all', 'forward_only', 'reverse_only'):
        for pair in ('convs_only-fixed', 'diff-fixed', 'diff-identity'):
            row = sequence['comparisons'][pair][group]['overall']
            lines.append('| ' + ' | '.join([LABEL.get(group, '전체'), pair, *[interval(row[k]) for k in base.KEYS]]) + ' |')
    lines += ['', '추가 민감도 분석은 도시 내 시퀀스와 연속 100질의 원형 블록을 2,000회 함께 재표집한다. 같은 replicate에서 모든 방법·시작점·방향에 같은 질의를 적용한다. 아래는 `convs_only−fixed`이며 유효 도시만 평균한 구간과 6도시가 모두 남은 replicate만 사용한 구간을 함께 보존한다.', '',
        '| 방향 | 구간 방식 | 유효 replicate | AP (%p) | F1 | R@100P | R@1 |', '|---|---|---:|---:|---:|---:|---:|']
    for group in ('all', 'forward_only', 'reverse_only'):
        for name, source, total in [('가용 도시', block, 2000), ('6도시 유지', six, six['comparisons']['convs_only-fixed'][group]['complete_city_replicates'])]:
            row = source['comparisons']['convs_only-fixed'][group]['overall']
            lines.append('| ' + ' | '.join([LABEL.get(group, '전체'), name, str(total), *[interval(row[k]) for k in base.KEYS]]) + ' |')
    lines += ['', '블록 분석은 도시의 전체 시퀀스를 추출한 뒤 해당 방향 양성이 남은 항목만 평균한다. 따라서 시퀀스 전용 분석과는 질의 블록뿐 아니라 시퀀스 추출 모집단과 분모 처리도 다르다. 두 구간의 차이를 시간 상관의 영향 하나로 단정하지 않는다.', '',
        '전체 및 역방향 AP의 블록 구간은 0을 포함한다. 해당 F1·R@100P의 양의 구간과 AP의 불확실성을 함께 보고하며 유리한 재표집 방식만 선택하지 않는다. 전체 F1 블록 구간의 하단 +0.002%p는 0에 가깝다. 시퀀스 전용 구간에서 확인한 AP 개선은 그 재표집 규약에 조건부인 결과다. 원래 블록 분석의 도시별 누락 횟수와 전체·방향·도시별 세 비교는 `direction_learning/{summary,sequence_summary,fixed_city_block_summary}.json` 및 `learning_intervals.csv`에 있다. 이 구간은 고정 데이터·체크포인트·세 시작점에 조건부인 사후 분석이며 독립 확증, 새 학습 변동, 개발 선택, 다중 비교 보정이나 동등성 검정이 아니다.', '',
        '| 도시 | 순방향 convs_only−fixed AP (%p) | 역방향 AP (%p) | 역방향 R@100P (%p) |', '|---|---:|---:|---:|']
    for city, values in sequence['comparisons']['convs_only-fixed']['reverse_only']['cities'].items():
        fwd = sequence['comparisons']['convs_only-fixed']['forward_only']['cities'][city]
        lines.append(f"| {city} | {fwd['AP']['delta_pp']:+.3f} | {values['AP']['delta_pp']:+.3f} | {values['R@100P']['delta_pp']:+.3f} |")
    lines += ['', '전체 모델과 identity의 차이에는 고정 감산 자체의 기여도 포함된다. DiffAttnConv 학습만의 기여는 같은 초기 출력의 `convs_only−fixed`로 판정한다.', '', '### 46시퀀스 원시 회전의 최종 인식 성능', '']
    c = yaw['counts']
    lines += [f"표본은 {c['queries']:,}질의, 양성 {c['positive']:,}·음성 {c['negative']:,}·모호 {c['ambiguous']:,}개다. 전체 195,726질의를 회전한 결과는 아니다. 표본 양성이 없는 시퀀스는 {', '.join(yaw['zero_positive_sequences'])}이며 AP/재현율 평균에서 제외하고 음성·승자 기록은 보존한다.", '',
        '| 각도 | AP (%) | 최대 F1 (%) | R@100P (%) | R@1 (%) | 0° 대비 AP (%p) | 승자 일치 (%) |', '|---:|---:|---:|---:|---:|---:|---:|']
    for angle, row in yaw['final_recognition'].items():
        p = row['pooled_query_seed_pairs']
        lines.append('| ' + ' | '.join([f'{float(angle):g}°', *[f"{100*row['mean_std'][k]['mean']:.3f}" for k in base.KEYS], f"{row['delta_vs_zero_pp']['AP']:+.3f}", f"{100*p['winner_agreement']/p['queries']:.3f}"]) + ' |')
    lines += ['', '출처: `expanded46/{yaw_summary.json,yaw_metrics.csv,yaw_city_dataset.csv}`. 모든 각도·시작점·도시·데이터셋·시퀀스 수치, 정답 복구/실패 변화, 승자 일치율, 점수 차이, 원시 지원 경계 기록을 보존한다. 도시·시퀀스 균등 평균이므로 질의를 합산한 복구·실패 건수와 지표 변화의 부호는 다를 수 있다. 정수 칸 회전에서도 최종 출력이 엄밀히 같지는 않다.', '', '### 정수 칸 회전의 원인 분리', '',
        '원인 대조의 actual은 별도 회전 원장과 승자 및 최종 점수를 다시 맞춘다. ideal_grid는 원래 원키와 BEV 계수의 해석적 위상 회전으로 원시 재투영·인코딩 잔차를 없앤다. 주된 격자 배정 효과와 함께 인코더의 부동소수점 잔차도 제거되므로 아래 별도 원형 이동 검산으로 그 크기를 확인한다. ideal_transport는 실제 원시 회전 특징을 유지하며 전달 위상만 yaw0−θ로 고정한다. ideal_both는 둘을 함께 적용한다. exact_geometry는 추가로 원래 기하 간선 속성을 쓰며 direct_output_phase는 원래 GNN 출력을 직접 위상 회전해 이후 검색·정렬·판정만 검사한다. 효과의 가산성을 가정하지 않는다.', '',
        '| 각도 | 조건 | 승자 변화 / 2,208 | 원후보 고정 승자 변화 | 원승자 최종점수 변화>8e−6 | 원승자 점수 최대 차이 | 출력 위상 L2 최대 |', '|---:|---|---:|---:|---:|---:|---:|']
    for angle, row in causes['results'].items():
        for arm, value in row['arms'].items():
            m = value['measurements']
            lines.append(f"| {float(angle):g}° | {arm} | {value['winner_changes']} | {value['fixed_pool_winner_changes']} | {value['fixed_winner_confidence_changes_gt_8e6']} | {m['fixed_winner_confidence_change']['max']:.6g} | {m['output_phase_l2_max']['max']:.6g} |")
    lines += ['', 'actual의 검색·정렬 단계별 최대 차이는 다음과 같다. 후보 변화는 query-시작점 2,208건 중 합집합이 달라진 건수이며, 네 채널 오차는 원래 후보 집합을 고정해서 계산한다. 최대치는 서로 다른 질의에서 나올 수 있다.', '',
        '| 각도 | 후보 집합 변화 | 원키 코사인 | BEV 크기 코사인 | 스펙트럼 정렬 | 열 정렬 | minmax 후 4채널 최대 | 융합 점수 최대 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for angle, row in causes['results'].items():
        actual = row['arms']['actual']
        values = [actual['raw_channels'][str(k)]['max'] for k in range(4)]
        norm = max(v['max'] for v in actual['minmax_channels'].values())
        fused = actual['measurements']['fixed_pool_fusion_max_error']['max']
        lines.append('| ' + ' | '.join([f'{float(angle):g}°', str(actual['candidate_set_changes']), *[f'{v:.6g}' for v in values], f'{norm:.6g}', f'{fused:.6g}']) + ' |')
    joint_changes = sum(v['arms']['ideal_both']['winner_changes'] for v in causes['results'].values())
    joint_scores = sum(v['arms']['ideal_both']['fixed_winner_confidence_changes_gt_8e6'] for v in causes['results'].values())
    direct_changes = sum(v['arms']['direct_output_phase']['winner_changes'] for v in causes['results'].values())
    direct_scores = sum(v['arms']['direct_output_phase']['fixed_winner_confidence_changes_gt_8e6'] for v in causes['results'].values())
    lines += ['', f'세 각도를 합친 6,624건에서 ideal_both의 승자 변화는 {joint_changes}건, 원승자 점수 오차 8e−6 초과는 {joint_scores}건이다. direct_output_phase는 각각 {direct_changes}건과 {direct_scores}건이다. 이 결과는 사전 고정한 원인 대조 표본에 대한 검산이며 전체 입력 공간의 정리는 아니다.']
    geometry_error = max(v['arms']['actual']['measurements']['geometry_attribute_error']['max'] for v in causes['results'].values())
    lines += ['', f'자세 행렬을 함께 변환한 뒤 시간 간선 기하 속성의 최대 절대 오차는 {geometry_error:.6g}다. 이 반올림 오차까지 없앤 ideal_both_exact_geometry를 함께 비교하므로 물리 좌표의 일치, 간선 기하의 수치 차이, yaw 위상의 차이를 분리한다.']
    lines += ['', '1. **원시 격자 배정**: 원시 점을 float32로 회전한 뒤 `atan2`와 `floor`로 칸을 다시 고르면 경계점이 인접 칸으로 이동할 수 있다. 6°에서는 거리 범위 판정과 BEV 반경 행 경계도 일부 바뀐다. 칸의 min 거리/max 높이 집계 및 빈 칸 보간이 영향을 확대한다. 아래에서 각도 배정을 고정하고, 이어 행·유효 범위도 고정한 내부 대조로 각 단계를 분리했다. float64도 정확한 경계에서는 무조건 일치를 보장하지 않는다.', '',
        '| 각도 | 투영 | 각도 칸 불일치 점 | float64 불일치 | 행 변화 | 유효 범위 변화 | 실제 특징 최대 차이 | 각도 고정 후 | 모든 칸·범위 고정 후 |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for angle, row in grid['results'].items():
        for name, values in row.items():
            c = values['counts']; e = values['feature_max_errors']
            lines.append(f"| {float(angle):g}° | {name} | {c['angular_assignment_mismatch']} | {c['angular_assignment_mismatch_float64']} | {c['row_assignment_mismatch']} | {c['range_mask_mismatch']} | {e['actual']['max']:.6g} | {e['angle_locked']['max']:.6g} | {e['all_bins_locked']['max']:.6g} |")
    edge_sentence = '같은 736질의의 q±15 구간에서 ±5 시간 간선의 최대 상대 잔차는 ' + ', '.join(
        f"{float(a):g}° 회전에서 {v['max']:.6g}°" for a, v in causes['transport_edge_residual_deg_max_per_query'].items()) + '다.'
    lines += ['', '위 특징은 거리 영상의 비정규화 288D 값 또는 BEV 복소 계수의 절대 오차다. 서로 다른 척도이므로 두 열의 크기를 성능 영향처럼 비교하지 않는다. 모든 칸·범위를 고정한 BEV는 동일하며 거리 영상의 6° 잔차는 거리값·FFT 연산의 부동소수점 정밀도 수준이다. 격자 검사는 184스캔의 원리 검사이고 최종 성능 표본과 구분한다.', '',
        '같은 184스캔의 정확한 원형 이동을 CPU에서 별도로 검산했다. 원형 빈칸 보간은 모든 552건에서 이동과 정확히 교환된다. 아래 단위벡터 오차는 이론적인 크기 불변 또는 복소 위상 등변 결과와의 차이다. 원시 회전 평가를 이 내부 roll로 대체하지 않았다 (`expanded46/roll_check/summary.json`).', '',
        '| 각도 | 거리 키 상대 L2 최대 | 거리 키 단위벡터 L2 최대 | BEV 위상보정 단위벡터 L2 최대 |', '|---:|---:|---:|---:|']
    for angle, values in roll['results'].items():
        lines.append(f"| {float(angle):g}° | {values['range_descriptor_relative_l2']['max']:.6g} | {values['range_descriptor_unit_l2']['max']:.6g} | {values['bev_phase_unit_l2']['max']:.6g} |")
    lines += ['',
        '2. **yaw 전용 전달의 비공변성**: 현재 각도는 상대 자세 R에서 `atan2(R10,R00)`다. roll/pitch가 있는 R에서 센서 좌표 회전은 R′=R·Rz(−θ)이며 일반적으로 yaw(R′)≠yaw(R)−θ다. 잔차 δ_i가 모든 시간 이웃에서 같은 상수라면 공통 위상은 상쇄된다. 복소 전달에 영향을 주는 것은 스캔마다 달라지는 상대 잔차 δ_i−δ_j다. 물리 좌표 T′p′=Tp를 유지해도 이 상대 위상 오차가 생길 수 있다. 180°는 평면 투영 벡터의 부호 반전으로 관계가 성립하지만 일반 각도는 그렇지 않다. 두 원인을 제거한 조건과 실제 조건의 비교로 이를 원시 격자 효과에서 분리했다.', '',
        edge_sentence, '',
        '3. **검색·정렬 채널과 최종 판정**: actual의 원키/보조 후보 목록 변경을 기록하고, 원래 후보 집합에 고정한 4채널 원점수·minmax 점수·융합 점수도 계산했다. 원래 후보에서 승자가 바뀐 경우 원래 1·2위 점수 차이가 융합 최대 오차의 두 배 이하인지 검산했다. 새 승자 점수와 원래 승자를 고정한 최종 시간 점수를 모두 비교하여 승자 교체와 판정 자체의 변화를 구분한다. ideal_both와 direct_output_phase는 원인 제거 후 남는 정렬·수치 오차를 보여준다. 전체 39,744행과 각 단계 측정은 `expanded46/causes/`에 있다.', '',
        '### 학술적 주장 범위와 필요한 수정 제안', '',
        'DFT 크기의 불변성은 입력 격자가 정확히 원형 이동할 때 성립한다. 원시 점의 재투영이 그 정확한 이동을 만들지 않으면 이 정리의 전제가 깨진다. 복소 문맥 전달의 회전 등변성도 입력 위상뿐 아니라 전달에 사용하는 자세 위상이 같은 센서 좌표 회전에 공변이어야 한다. 현재 Euler yaw는 평면 자세에서는 이를 만족하지만 일반 roll/pitch 자세에서는 만족하지 않는다. 따라서 원시 입력부터 최종 출력까지의 무조건적·엄밀한 yaw 불변성을 주장할 수 없다. 보고한 것은 고정된 실제 모델의 표본 회전 강건성과 원인 대조다.', '',
        '수정 후보로 `φ(R)=atan2(R10−R01,R00+R11)`를 검산했다. 복소수 `(R00+R11)+i(R10−R01)`는 우측 Rz(−θ)에 대해 exp(−iθ)를 곱한 값이 되어 φ(RRz(−θ))=φ(R)−θ가 성립한다. proper SO(3)에서 분모 벡터의 크기는 1+R22이므로 기울기 180°에서 특이하다. 이 식은 기존 Euler yaw와 0°에서도 달라질 수 있어 채택하려면 학습/검증 단계에서 재학습·선택·성능 검증이 필요하다. 현재 시험 성능을 보고 이 수정이나 더 유리한 모델을 선택하지 않았다.', '',
        f"수학적 후보는 저장 자세 {sum(v['poses'] for v in proposal['sequences'].values()):,}개×7각도와 합성 50,000건에서 공변성을 확인했다. 합성 최대 오차는 {proposal['synthetic_max_error_rad']:.3g}rad이다. 이는 개선 모델의 최종 인식 성능을 입증한 결과가 아니다 (`expanded46/phase_covariance_proposal.json`).", '',
        '### 검산과 출처', '',
        f"0°의 2,944질의×3시작점=8,832판정은 원래 원장의 승자와 모두 같고 최대 점수 오차는 {yaw['baseline_validation']['max_score_error']:.6g}이다. 첫 512질의의 최대 오차는 1.43051e−6이었다. 사전 고정 허용치 8e−6은 전체 0° 재현 오차의 약 4.2배이며, 원시 재인코딩·부분 그래프와 전체 그래프·float32 연산에 따른 재현 오차를 기준으로 대조의 수치 복원 여부를 판정한다. 모든 입력에 대한 이론적 오차 상한은 아니다. 원시 지원 프레임은 77,595개, 각도별 재인코딩은 총 543,165회, 저장 인코딩 묶음 322개, 최종 원장 966개·61,824행이다. 기존 8시퀀스의 99,918회 인코딩과 168원장은 해시 확인 후 재사용했다.", '',
        f"원인 actual 재현 {causes['actual_ledger_checks']:,}건의 최대 점수 오차는 {causes['actual_max_score_error']:.6g}다. 세계 좌표 검사는 시퀀스별 첫 지원 스캔의 모든 점×7각도=322스캔-각도이며 최대 점 좌표 오차는 {max(r['world_coordinate_first_scan_max_error_m'] for a in yaw['coordinate_audit'].values() for r in a.values()):.6g}m다. 모든 원시 프레임의 모든 점을 수치 검산한 것은 아니다. 각도의 원시 점 해시가 같고 인코딩·원장 해시와 동결 코드 해시를 확인했다. 최종 검사 목록은 `expanded46/completion_validation.json`에 둔다.", '',
        '## 결론 / 다음', '',
        '현재 모델의 전체 방향별 성능, 방향별 모듈 학습 기여, 46시퀀스 원시 query 회전, 정수 칸 회전의 단계별 원인을 확인했다. 실제 반대 방향 재방문과 합성 좌표 회전은 서로 다른 평가다. 원시 회전 표본은 2,944질의로 제한되며 모든 각도·모든 가림·새 센서·새 도시의 보장을 제공하지 않는다. 원고에는 검증된 범위의 회전 강건성을 쓰고 엄밀한 불변성의 전제를 명시해야 한다. 자세 위상 수정은 제안으로만 남기고 이 작업에서 GPU 실험을 추가하지 않는다. 공통 위키 반영과 원고 편집은 부모 작업에서 담당한다.', '',
    ]
    TARGET.write_text('\n'.join(lines), encoding='utf-8')
    print(TARGET)


if __name__ == '__main__':
    main()
