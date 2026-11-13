#!/usr/bin/env python3
"""검증된 표 3 측정으로 효율 위키와 실험 일지를 갱신한다."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/pdv2_table3_20261029'
NAMES = {'nsd': 'NSD', 'bevplace': 'BEVPlace++', 'seqot': 'SeqOT',
         'ringsharp': 'RING#-L', 'scpp': 'SC++', 'ring': 'RING',
         'ringpp': 'RING++', 'disco': 'DiSCO', 'deepring': 'DeepRING',
         'helios': 'HeLiOS', 'imlpr': 'ImLPR'}


def main() -> None:
    """완료한 수치와 조건만 문서에 기록한다."""
    data = json.loads((OUT / 'summary.json').read_text())
    rows = [r for r in data['measurements']
            if r['status'] == 'passed' and r.get('publish_ready', True)]
    expected = {(method, size) for method in NAMES for size in (1000, 5000, 10000)}
    completed = {(row['method'], row['n_map']) for row in rows}
    complete_all = completed == expected
    settings = '''- 지도 크기 N은 첫 질의에서 검색 가능한 후보 수다. 96번 도착하면서 N부터 N+95까지 늘어난다. 최근 45개 관측은 후보 시차와 질의 지연을 지원한다. 이 지원 상태도 메모리에 포함한다.
- 최초 N은 1,000·5,000·10,000이다. 처음 12번 도착으로 검증과 워밍업을 한 뒤 같은 96번 도착을 세 번 반복한다. 반복마다 같은 초기 상태로 돌아간다.
- RTX 5080, `ring_baseline`, CPU 수치 연산 스레드 1개, 배치 1이다. GPU 구간 앞뒤를 동기화하고 TF32를 끈다. SC++만 같은 PC의 호스트 MATLAB 공식 구현을 단일 스레드로 사용한다.
- 원시 점군은 메모리에 미리 올린다. 디스크 I/O와 공통 KISS-ICP 주행 추정은 모든 방법에서 제외한다. 관측 15키프레임 지연은 계산 시간과 별개다.
- 업데이트에는 원시 인코딩, 실제 상태 추가와 색인 갱신을 포함한다. 초기 지도 구축 시간/N을 업데이트 비용으로 쓰지 않는다. 초기 구축은 따로 기록한다.
- N마다 도착하는 96개 스캔이 다르다. 인코딩 시간 변화에는 점군 크기 차이도 섞이므로 이를 지도 크기만의 효과로 해석하지 않는다.
- SeqOT은 같은 키프레임 도착 사이의 연속 원시 스캔도 쓴다. 각 스캔은 한 번씩 투영·인코딩하고 최근 3개 거리 영상과 20개 부분 특징을 재사용한다. centered 문맥은 필요한 원시 스캔이 실제 도착했을 때만 완성한다. N=1,000 구간에는 96키프레임 사이에 원시 스캔 220개가 들어온다.'''
    table = ['| 방법 | 최초 N | 상주 상태 MB | 업데이트 ms | 질의 ms | 전체 처리 ms |',
             '|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda r: (r['n_map'], r['method'])):
        cells = ['%.3f (%.3f)' % (row['timings_ms'][k]['mean'],
                                 row['timings_ms'][k]['p95'])
                 for k in ('update', 'query', 'processing')]
        memory = row['resident_bytes'].get('total')
        memory_text = '미확정' if memory is None or not row.get('memory_publish_ready', True) else '%.3f' % (memory/1e6)
        table.append('| %s | %d | %s | %s |' % (
            NAMES[row['method']], row['n_map'],
            memory_text, ' | '.join(cells)))
    table_text = '\n'.join(table)
    body = '''# 효율 측정

> [!NOTE]
> 기준: 2026-10-30 작업 트리(바로 앞 커밋 `635594f`). 현재 시간 전용·25% 상관 보정 주모델을 실제 증분 실행으로 측정한다.

같은 MulRan Sejong01 검증 경로와 KISS-ICP 키프레임 도착에서 저장량과 계산 시간을 비교한다. 원천은 `results/pdv2_table3_20261029/{protocol,summary}.json`과 `methods/*.json`이다. 완료한 방법과 크기만 표시하며 시험 성능이나 선택 설정은 바꾸지 않는다.

## 측정 조건

```mermaid
flowchart LR
  P["원시 도착 스캔<br/>디스크 읽기 제외"] --> E["스캔 인코딩"]
  E --> U["실제 상태·색인 추가<br/>NSD는 영향받는 11개 키 갱신"]
  U --> Q["질의 q = 도착 t − 15<br/>후보 j ≤ q − 30"]
  Q --> F["검색·재정렬·최종 판정"]
  E --> T["구간별 평균·p95"]
  U --> T
  F --> T
  U --> M["CPU 고유 배열 + GPU 현재 할당<br/>RSS·최대값 별도"]
```

''' + settings + '''

## 완료 수치

시간은 평균(p95) 밀리초, 메모리는 십진 MB다. 상주 상태는 마지막 도착 후 실제 버퍼 용량과 GPU 현재 할당량이다. 본문 표 3은 최초 N=1,000을 표시하고 부록은 완료한 모든 크기를 표시한다. 미완료 방법은 대시로 둔다.

''' + table_text + '''

## 저장량과 메모리 정의

NSD의 활성 특징은 416실수 키와 16×8복소 BEV 계수, 총 2,688B다. 이는 논리 특징 크기이며 전체 메모리는 아니다. 실제 실행은 원본·보정 정렬 뱅크, 색인용 키, 원계수, 자세와 누적 거리, 최근 31개 원특징, 32,768B 보정 행렬을 보관한다. 새 경로는 쓰지 않는 원통 계수를 만들지 않는다.

상주 상태는 CPU의 고유 숫자 배열 저장소와 GPU의 현재 할당량을 더한다. 모델의 등록 버퍼 밖에 있는 텐서·NumPy 배열도 객체와 저장 주소를 재귀 검사해 중복 없이 포함한다. 독립 감사로 BEVPlace++에 32B, RING#-L에 4,541B를 추가했고 다른 기존 모델은 누락이 없었다. 원래 계측과 타이밍은 보존했다. GPU 모델 저장소는 별도로 기록하지만 전체 GPU 할당에 다시 더하지 않는다. 지속 작업 공간과 기타 할당을 포함하며 이를 특정 라이브러리의 메모리라고 단정하지 않는다. CUDA 예약량과 최대 할당량은 별도로 기록한다.

지도 버퍼는 마지막 도착에 필요한 용량을 미리 확보하고 그 용량 전부를 센다. FAISS는 마지막 후보 수까지 할당한 뒤 비우고 증분 추가하며, 저장 주소가 변하지 않는지 검사한다. Python 객체, 라이브러리, 공통 입력과 감사 자료를 포함한 프로세스 RSS는 상주 상태와 구별한다. 외부 라이브러리의 지속 할당이 미확정인 방법은 상주 메모리를 대시로 두고 검증된 시간만 표시한다. GPU 최대값은 PyTorch 할당자 기준이며 외부 CUDA 라이브러리의 일시 할당까지 포괄하는 총 장치 최대값은 아니다.

## 정확성 확인

NSD 원시 특징은 기존 캐시와 대조한다. 새 도착의 증분 키는 전체 접두 그래프 추론과 최대 허용오차 3×10⁻⁶에서 일치하고 승자는 정확히 같다. 각 크기의 96개 승자와 25% 최종 점수는 고정 검증 원장과 비트 단위로 일치해야 완료 처리한다. 세 시드의 서로 다른 최종 계수 채널도 별도로 검증했다. 비용 대표는 시드 1이며 표 2의 세 시드 시험 평균과 구별한다.

BEVPlace++는 공식 PNG 전처리와 선택 가중치로 배치 1 특징을 새로 생성했다. 이전 배치 32 캐시와 섞지 않고 새 지도에서 공식 FAISS L2 검색의 승자와 점수를 검사한다. SeqOT은 공식 centered 창의 배치 1 참조와 투영·부분 특징·최종 특징을 대조하고, 필요한 미래 원시 스캔이 관측되기 전에 특징을 완성하지 않는지 검사한다.

ImLPR·HeLiOS의 시간 측정에는 표 2와 같은 정확한 접두 코사인 검색을 사용한다. DiSCO는 float64 접두 유클리드 검색과 추린 후보의 정확한 재계산을 유지한다. 이 세 방법의 공식 KDTree 1위는 시간 측정 밖의 정확성 참조이며 실제 측정 검색 구현과 구분한다. 소스와 선택 규약은 `scripts/_pdv2_table3_global_v2.py` 및 각 방법의 `*_protocol_v2.json`에 고정했다.

FAISS 1.11.0의 기존 GENERIC 빌드는 SIMD 최적화가 없었다. 같은 버전의 AVX512 CPU 빌드로 공식 IndexFlatL2 알고리즘을 유지하고 NSD와 BEVPlace++를 모두 다시 쟀다. 주표는 이 교정 측정만 쓴다. 이전 GENERIC 결과는 `revision1/`, GPU 지속 할당을 누락한 초기 결과는 `revision0/`, 게시 빌드와 겹친 측정은 `concurrent_build_diagnostic/`에 보존한다.

RING++ 공식 CUDA 특징 생성에는 원자적 연산이 아닌 최대값 갱신이 있다. 같은 입력을 반복 인코딩하면 특징과 승자가 달라질 수 있어, 서로 다른 실행의 점수 일치와 같은 실행의 검색 정확성을 구별한다. RINGPP_REPEAT_NOTE 실제 각 실행을 공식 전수 상관과 비교한 288질의는 모두 같은 승자를 냈다. 원본 코드는 바꾸지 않았으며 이 진단은 표 2의 시험 성능 재평가가 아니다.

RING++ native 특징 함수의 미해제 CUDA 요청은 인코딩당 360B다. 추적기로 확인한 이 저장량을 PyTorch 밖의 지속 버퍼로 더한다. 초기 지도를 캐시에서 읽은 새 프로세스의 워밍업 12회와 측정 288회, 총 300회에서 108,000B이며 캐시 생성 프로세스의 누수는 이어받지 않는다. 장기 실행하면 누적량이 더 커진다. native 요청 용량은 드라이버의 실제 할당 단위나 최대 장치 메모리가 아니다. Cython manager 객체의 포인터·스칼라 메타데이터는 수치 배열 범위 밖이며 RSS에 포함된다. 상세 근거와 실행 소스는 `native_trace/`, `ringpp_v2_audit/`, `ringpp_v3_audit/`에 보존한다.

RING++의 초기 검색 청크 1,024개는 후보 수가 변할 때 큰 임시 배열을 GPU 할당자 캐시에 계속 남겼다. 예약량이 19.705GB로 커지고 드라이버의 여유 메모리가 0이 되면서 검색이 약 512ms로 느려졌다. 같은 후보 수를 고정하면 약 15ms였고, 청크를 128개로 제한한 재생도 약 15ms·예약량 1.244GB로 안정적이었다. 공식 점수와 승자는 그대로였고, 모델·특징·검색 수식을 바꾸지 않고 작업 메모리만 제한한 v3의 실제 계측을 검증했다. 이전 실제 측정은 `before_ringpp_workspace_audit/`, 원인 분리 결과는 `ring_query_workspace_diagnostic/`에 보존한다.

RINGSHARP_WORKSPACE_NOTE

DeepRING은 표 2와 같은 논문 기반 재구현이며 공식 공개 구현으로 표시하지 않는다.

HeLiOS는 동일한 GPU 입력에서도 작은 FP32 특징 변동이 생긴다. 초기 후보 1천 개의 원인 진단에서 입력·양자화 좌표·모델 상태는 같았고, 최종 점수 허용오차 10⁻⁷을 넘은 사례는 최대 1.1921×10⁻⁷이었다. 기존 실패와 소스를 보존하고, 실제 실행의 검색 정확성과 독립 추론의 수치 차이를 별도로 확인했다. 완료한 각 크기에서 실제 실행의 FP32·FP64 전수 검색·KDTree·최종 내적과 고정 캐시의 승자는 288회 모두 같았으며, 특징 오차로 계산한 점수 변화 상한도 1·2위 간격보다 작았다. 모델·전처리·정밀도·검색 계산은 바꾸지 않았다. 원천은 `helios_diagnostic_n1000/`, `helios_protocol_v3.json` 및 `verification/helios_v3_n*_repeat*.json`이다.

## 자동 반영

공통 계측기와 방법별 어댑터가 검증 결과와 JSON을 쓴다. `scripts/_paper_efficiency.py`는 passed·publish_ready 상태와 원천을 검사해 표를 만든다. `scripts/_paper_figure4.py`는 같은 요약으로 편집 가능한 PPTX를 만든다. `scripts/_paper_publish_v22.py`가 PDF 빌드, Dropbox 복사 해시와 Cloud Files 동기화를 확인한다. 기존 그림 1~3은 유지한다.

## 관련 문서

- [논문 작업](../논문%20작업.md)
- [비교 방법](비교%20방법.md)
- [평가 개요](평가%20개요.md)
- [실행 환경](../실행%20환경.md)
- [할 일](../할%20일.md)
'''
    current_ringpp = next((r for r in rows if r['method'] == 'ringpp' and r['n_map'] == 1000), None)
    if current_ringpp:
        variation = current_ringpp['verification']['official_native_repeat_variation']
        note = '최초 N=1,000의 96질의에서 세 반복 중 %d질의의 승자가 바뀌었고 최종 점수 범위는 최대 %.6f였다.' % (
            variation['winner_changed_queries'], variation['max_confidence_spread'])
    else:
        note = '보존한 v2 진단의 96질의 중 8질의에서 승자가 바뀌었고 최종 점수 범위는 최대 0.001392였다.'
    body = body.replace('RINGPP_REPEAT_NOTE', note)
    current_ringsharp = next((r for r in rows if r['method'] == 'ringsharp' and r['n_map'] == 1000), None)
    note = 'RING#-L의 N=5,000 측정에서 회전한 질의 특징의 CNN 배치가 GPU 임시 저장량을 과도하게 늘리는 문제가 확인되어 게시를 보류했다.'
    if current_ringsharp and current_ringsharp.get('method_config', {}).get('angle_cnn_batch') == 16:
        note += ' 각도 CNN 배치를 128개에서 16개로 제한한 v2를 모든 지도 크기에 동일하게 적용했다. 전체 회전 검색·번역 후보 1,000개·후보 상관 청크 128개는 유지한다. 실제 288질의의 모든 후보 번역 점수와 최종 승자를 원본 구현과 대조했다. 기존 측정은 `before_ringsharp_workspace_audit/`, 원인 분리 결과는 `ringsharp_workspace_diagnostic/`에 보존한다. 고유 질의 회전 수는 N=1,000에서 평균 155.10개, N=5,000에서 22.57개였다. 구간별 장면에 따라 회전 CNN 작업량도 달라지므로 큰 지도에서 검색 시간이 반드시 늘어나는 것은 아니다.'
    body = body.replace('RINGSHARP_WORKSPACE_NOTE', note)
    provenance = OUT/'cache_provenance_manifest.json'
    if provenance.exists() and any(r['mode'] == 'verified_prefix_reuse' for r in json.loads(provenance.read_text()).get('records', [])):
        body = body.replace('## 자동 반영', '## 초기 지도 준비 재사용\n\nImLPR·HeLiOS의 큰 지도 준비에서는 5,000개 조건의 앞 5,140행을 그대로 복사하고 나머지만 같은 인코더로 계산한다. 재사용 전후 원시 파일·키프레임·자세·가중치·소스 해시를 확인하며 12개 접두 특징도 재생 검사한다. 준비 시간과 재사용 근거는 별도 provenance에 기록한다. 실제 측정의 96개 새 도착과 검색은 생략하지 않는다. 원천은 `cache_provenance_manifest.json`과 각 `*_preparation/`이다.\n\n## 자동 반영')
    if complete_all:
        body = body.replace('완료한 방법과 크기만 표시하며 시험 성능이나 선택 설정은 바꾸지 않는다.',
                            '11종×세 지도 크기의 33개 조건을 모두 측정·검증했다. 시험 성능이나 선택 설정은 바꾸지 않는다.')
    (REPO / 'docs/평가/효율 측정.md').write_text(body)
    journal = '''# 표 3 실제 증분 효율 측정

## 왜

현재 주모델과 표 2에 보고한 비교 방법의 최종 처리·전체 상태 비용을 같은 조건에서 측정하고 원고에 반영한다.

## 설정

''' + settings + '''

## 결과

시간은 평균(p95) ms, 상주 상태는 마지막 도착 후 십진 MB다.

''' + table_text + '''

원천: `results/pdv2_table3_20261029/summary.json`. 측정 표본, 해시, 검증과 방법별 소스는 원천 JSON에 보존한다. 컨테이너의 오래된 드라이버 마운트 문제는 다른 GPU 작업이 없음을 확인하고 재시작해 복구했다.

## 결론·다음

완료 행만 표·그림·PDF·Dropbox에 자동 반영한다. NSD와 BEVPlace++의 세 크기 측정 후 SeqOT·RING#-L·SC++와 나머지 방법을 순차 진행한다. 초기 누락, SIMD 없는 FAISS, 빌드와 겹친 측정은 별도 개정에 보존하고 주표에서 제외했다. 교정된 주측정의 조건과 한계는 [효율 측정](../../평가/효율%20측정.md)에 정리한다.
'''
    if complete_all:
        journal = journal.replace('NSD와 BEVPlace++의 세 크기 측정 후 SeqOT·RING#-L·SC++와 나머지 방법을 순차 진행한다.',
                                  'NSD·BEVPlace++를 포함한 11종×세 지도 크기의 33개 조건을 모두 검증했다. 본문 표 3은 초기 후보 1천 개, 부록과 그림 4는 세 크기를 모두 담는다.')
    (REPO / 'docs/RAW/실험일지/2026-10-29_table3-실제-증분-효율.md').write_text(journal)


if __name__ == '__main__':
    main()
