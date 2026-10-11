# MCD (Multi-Campus Dataset) 도입: rosbag → MulRan 스타일 추출

- 날짜: 2026-08-03 ~ 08-04
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

"완전히 새로운 도시" zero-shot 테스트 확보 (스톡홀름 KTH + 함부르크 TUHH,
학습 센서와 같은 계열 OS1-64). 기존 4개 데이터셋 어디에도 없는 미학습
대륙/도시 + 주·야간 세션 → 장소 일반화 주장의 핵심 근거.

## 설정 (파라미터)

- 다운로드: Ouster os1_64 bag 12개 + GT(pose_inW.csv, spline.csv) 12개 +
  핸드헬드 캘리브레이션 (Google Drive; gdown 6.1은 **파일 ID만** 전달해야 함
  — URL 형태는 HTML 페이지가 저장됨. 쿼터 걸린 3개는 브라우저 수동)
- 추출: `scripts/extract_mcd_scans.py` — 토픽 `/os_cloud_node/points`
  (64×1024, 48-byte point), MulRan 규약 65536×4 float32 bin +
  `scan_poses.csv` (GT는 epoch-seconds, bag 스탬프와 동일 시간축, 선형 보간)
- 출력: `/mnt/d/NSD_datasets/mcd_extracted/<seq>/`

## 결과

- 12/12 시퀀스, 총 75,013 스캔; 드롭 시퀀스당 14–20개 (GT 경계)
- 도보 속도 1.35–1.67 m/s, 경로 289 m–1.4 km — 전 시퀀스 물리적으로 타당
- **센서 발견: 고도각 p1/p99 ≈ ±21.1° (12개 전부) → OS1-64 2세대 ±22.5° FOV.**
  MulRan의 OS1-64(±16.6°, 1세대)와 다름 → 신규 센서 항목 `mcd: [-22.5, 22.5]`
  필요 (같은 "OS1-64" 이름이라도 unseen-FOV 테스트가 됨)
- z-up 확인 (NCLT식 z-flip 불필요); body→sensor 오프셋 ≤6 cm 무시

## 결론 / 다음

MCD는 "미학습 도시 × 유사-계열-이지만-FOV-다른 센서" 이중 스트레스 테스트가
됨. 핸드헬드 roll/pitch는 스캔 자체에 내재 — 결과 해석 시 보행 플랫폼임을
명시할 것. 다음: pd2026 test 캐시 빌드에 포함 (완료 후 test 일지에 결과 기록).
