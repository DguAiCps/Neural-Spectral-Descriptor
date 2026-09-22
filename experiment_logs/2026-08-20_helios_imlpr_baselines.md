# HeLiOS/ImLPR 베이스라인 추가 — 공식 코드 zero-shot, 전 프로토콜(Table 1·2·13)

- 날짜: 2026-08-19 ~ 08-20
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

리뷰어가 요구할 최신(2025) 베이스라인 2종 추가. HeLiOS(ICRA25)·ImLPR(CoRL25)
모두 HeLiPR 저자 그룹 방법으로, 공식 git 코드+가중치를 기존 3개 평가
인프라(Strict-v2 intra / xs / causal PR)에 그대로 편입.

## 설정 (파라미터·통합 결정)

- **오염 공시**: 두 방법 모두 저자 학습 데이터 = HeLiPR DCC/KAIST/Riverside
  04-06 = 우리 helipr_daejeon test 그룹 9시퀀스 → 전 산출물 train_overlap
  플래그 + Table 1은 †셀 표기 + clean 5그룹 매크로 병기.
- **HeLiOS 포팅** (공식 스택 torch1.13/CUDA11.6은 sm_120 실행 불가):
  MinkowskiEngine 0.5.4를 CUDA-13/CCCL-3.0 패치 5건으로 소스 빌드 +
  sptr/torch_scatter/torch_cluster sm_120 빌드 (레시피
  `scripts/_helios_port_setup.sh`; upstream 코드 0줄 수정). 입력 = 센서
  프레임 XYZ(intensity 미사용), crop 100 m, ×0.01, ME quantize 0.01.
  eval 피클이 비변환 `/LiDAR/` 참조임을 코드로 확정(LiDAR_transformed는
  학습 overlap 전용) — GT 누출 없음. 단일 키프레임 스캔(서브맵 아님) 공시.
- **ImLPR**: 순수 파이썬 스택, 가중치 GDrive(README 링크 404 → 데이터
  폴더 내 발견). sm_120 로드 missing/unexpected 0. 멀티센서 RIV
  (`_imlpr_riv_lib.py`): 공식 함수 재사용 + 공식 'mulran' zero-shot 경로
  (선형 0..255) 전 센서 적용, mulran은 공식 ring%64 그대로, 나머지 angle
  모드(kitti 64 2/24.8, nclt 32 10.67/30.67, helipr VLP-16 15/15, mcd 64
  22.5/22.5). **normal 채널의 공식 구현(포인트별 파이썬 SVD)이 프레임당
  분 단위 → 수학 동치 벡터화(공유 kNN + 배치 eigvalsh)로 교체, 공식
  구현과 실프레임 2센서 수치 동등성 검증(오차 1.5e-5 ≪ 허용 1e-3,
  10/10 tests)**. 파이프라인 pad28+resize126+/255 공식 그대로.
- 서술자 인터페이스 통일(fp16 8448D, scan_ids=캐시)로 3드라이버를
  `--method {imlpr,helios}` 일반화. workflow adversarial 검증 PASS
  (kitti/05에서 bevplace와 n_queries 377 정확 일치 = 프로토콜 미러 증명).
- 장애: WSL2 GPU 드롭 2회(스모크 중 1, HeLiOS 추출 31/42 지점 1) —
  컨테이너 재시작으로 복구, 전 체인 재개형이라 손실 0.

## 결과

**Table 1 (Strict-v2 rev4, 37seq 6그룹 macro; †=오염 셀, 괄호=clean 5그룹):**
NSD-H 0.915 > NSD 0.886 > BEVP++ 0.876 > RING 0.867 > RING++ 0.825 >
SC++ 0.799 > **HeLiOS 0.722†(0.711)** > **ImLPR 0.677†(0.701)** >
SeqOT 0.588 > RING# 0.546.
- ImLPR: helipr_town(VLP-16) **0.280 붕괴** — 16빔 RIV vs 128빔 학습 도메인
  격차; 64빔 계열은 견실(mcd 0.873, mulran 0.852, kitti360 0.834).
- HeLiOS: 그룹 간 균등(0.66-0.81) — heterogeneous 설계 반영; 오염 셀 0.781.

**Table 2 (cross-session, @5m macro):** HeLiOS 0.352, ImLPR 0.337 — 둘 다
NSD/NSD-H 0.383 아래.

**Table 13 (causal PR, AP/R@100P):** **ImLPR 0.560/0.328** ≈ NSD causal
0.561/0.222 (R@100P는 ImLPR 우위 — DINOv2 cosine의 cross-query
캘리브레이션 강점), HeLiOS 0.554/0.201. 상위권(BEVP++ 0.769, RING 0.747,
SC++ R@100P 0.417)은 불변.

## 결론 / 다음

1. 최신 2방법 추가 후에도 **주 표 결론 불변** (NSD-H 1위, 멀티센서
   place-disjoint 로스터에서 HeLiPR-특화 방법들은 중하위권).
2. ImLPR의 helipr_town 붕괴는 "빔 수 도메인 격차" 사례로 서사 가치
   (NSD의 캘리브레이티드 풀링 대비점); zero-shot 공시 하에 정직 보고.
3. 필요시 후속: ImLPR pd 재학습(+2-3일)으로 clean 비교 — 현 zero-shot
   결과 기준 우선순위는 사용자 판단 사항.
4. 논문 반영 완료: Table 1(†+이중 매크로+각주), Table 2, Table 13에
   두 행씩 추가, 빌드 클린(28p, undefined 0).
