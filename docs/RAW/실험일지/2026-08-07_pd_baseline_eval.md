# Place-disjoint 42시퀀스 베이스라인 전체 평가 (6종)

- 날짜: 2026-08-05 ~ 08-07
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

동결 test에서 NSD 수치는 확보했으나 비교 대상이 없으면 표가 성립하지 않음.
6개 베이스라인을 NSD와 동일한 42시퀀스·동일 프로토콜(5 m / skip 30,
동일 pd2026 캐시 로스터)로 평가.

## 설정 (파라미터)

- SC++ / RING / RING++: training-free. RING·RING++는 BEV 캐시 공유
  (`_pdb_ring_eval.py`), 공식 TIRING 경로.
- BEVPlace++ / SeqOT: **pd train으로 재학습** (paper-era 체크포인트는 test
  장소 10개를 학습에 노출 → 비교 무효). BEVPlace++ 5ep fine-tune(공식 KITTI
  ckpt 초기화, 레시피 계승), SeqOT 2-stage 15+10ep.
- RING#: 공식 재현 가중치 zero-shot (자체 pretrained 미배포). paper-era 학습에
  노출된 test 10개 시퀀스는 `train_overlap` 플래그로 매크로에서 제외.
- 검증: 전 평가기가 paper-era 커밋 수치 재현 통과 (SC++/BEVPlace++/RING#/SeqOT
  델타 0.0000~bit-일치, RING++ Town01 델타 0.005 게이트 통과).
- 장애 2건: OOM(→시퀀스당 프로세스 격리), WSL2 GPU 드롭(→컨테이너 재시작 후
  재개; 체인이 시퀀스 단위 재개 구조라 손실 없음). 총 소요 ~2.5일.

## 결과 (시퀀스-매크로 R@1, `results/pd2026_baselines/summary_final.json`)

| 그룹 | SC++ | BEVP++ | SeqOT | RING# | RING | RING++ | NSD | NSD-H |
|---|---|---|---|---|---|---|---|---|
| KITTI 00/05/08 | .915 | .960 | .710 | .690 | .922 | .919 | .915 | **.975** |
| Town+B4 | .622 | .803 | .509 | .590 | **.873** | .766 | .820 | .834 |
| HeLiPR 대전 | .732 | .847 | .625 | .638 | .869 | .835 | **.915** | .903 |
| MulRan 대전 | **.863** | .835 | .797 | .733 | .823 | .821 | .852 | .861 |
| NCLT 페어 | .929 | .930 | .736 | .522 | .686 | .757 | **.975** | .970 |
| KITTI-360 | .851 | .950 | .506 | .545 | .826 | .761 | .881 | **.966** |
| MCD | .819 | .861 | .468 | .385 | .809 | .799 | **.940** | .934 |
| **전체** | .819 | .884 | .622 | .586 | .830 | .808 | .900 | **.920** |

## 정정 (2026-08-07, 표 발표 후 검증에서 발견)

42시퀀스 중 3개는 엄밀한 place-disjoint가 아님이 확인됨 (GPS 교차검증):
- **kitti/00**: 학습 seq 07과 경로 13.6% 공유 (min 0.04 m) — GPS 감사로 신규 발견.
  기존 `prepare_place_disjoint_nsd.py`의 가정도 동일 오류였음.
- **helipr/Bridge04**: 학습 Bridge01–03과 같은 다리 (의도된 역방향 하드 케이스).
- **nclt 페어**: 같은 캠퍼스 (pd 구성 기준 세션-disjoint일 뿐; pdnn만 완전 unseen).
- (kitti/05↔train 06은 최소 49.9 m, 접점 0.2% — 사실상 분리로 판정)

추가 발견 (같은 날, val 전수 감사): **val kitti 09(2011_09_30_drive_0033)가
KITTI-360 test 드라이브 0004와 동일 경로** (min 0.0 m, 100% <50 m) — tuple·
체크포인트 선택이 0004의 장소를 본 것. 0004도 strict에서 제외+공시.
val의 다른 시퀀스는 전 test 대비 청정 확인 (09↔나머지 ≥1.2 km,
Roundabout03↔Town 31 km GPS, Sejong↔대전 도시 분리).

4종(kitti/00, kitti360/0004, Bridge04, NCLT 페어) 제외한 **strict-v2 매크로
(37시퀀스): NSD-H 0.915 > NSD 0.886 > BEVPlace++ 0.876 > RING 0.867 >
RING++ 0.825 > SC++ 0.799** — 순위 결론 불변
(`results/pd2026_baselines/summary_strict.json`). 본문 표는 strict 그룹으로,
제외 4종은 공시된 별도 행(각주)으로 제시할 것.

**재학습 필요성 검토 결론: 0건.** train 구성은 재분류와 무관; ω/classifier는
train 전용; val 오염(09↔0004)은 해당 시퀀스 제외로 해결 (tuple 재선택은
체크포인트 선택 오염이 잔존해 불충분, 완전 재학습은 3–4일 비용에 시퀀스
2개 복구뿐이라 기각).

## 결론 / 다음

1. **NSD-H가 7그룹 중 5승 1무 1패, 전체 매크로 1위** (0.920 vs 차점
   BEVPlace++ 0.884). 무: MulRan 대전 (SC++ 0.863 vs 0.861). 패: Town에서
   RING(0.873 vs 0.834) — RING은 대신 NCLT 0.686으로 32빔에서 크게 무너져
   일관성 부족 (NSD-H 최저 그룹 0.83 vs RING 최저 0.69).
2. 학습형 베이스라인의 zero-shot 취약성 대비 NSD의 강건함: MCD에서 SeqOT
   0.468 / RING# 0.385 vs NSD 0.940.
3. 스토리지·검색 비용 우위 유지: RING 계열은 120×120 dense 상관(쿼리
   65–900 ms), NSD는 800/1,024 float + 이중 인덱스.
4. 다음: 본문 표 갱신 + Town 열세 분석(좁은 FOV VLP-16 사이트, RING 상관의
   강점 구간) 각주, cross-session fusion 확장은 여전히 open.
