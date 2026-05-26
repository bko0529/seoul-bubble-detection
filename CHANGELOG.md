# CHANGELOG

## [Unreleased]

### Added
- **ECOS 거시경제 데이터 수집** (`scripts/collect/run_collect_ecos.py`)
  - 한국은행 ECOS API를 통해 기준금리·M2·CPI·모기지금리 등 수집
  - 출력: `data/raw/ecos/macro_ecos_2013_2025.csv`
- **피처 엔지니어링 Step 2~5** (`scripts/feature/run_feature_engineering_steps2_5.py`)
  - 거시지표 병합, YoY/MoM 파생변수, 전세가율·PIR·소득대비전세율 계산
  - 버블 Pseudo-label(0=정상/1=과열/2=버블) 생성 로직 포함
  - 출력: `data/processed/features_sido_monthly.csv`, `data/processed/features_sgg_monthly.csv`
- **LSTM Autoencoder 이상 탐지** (`scripts/feature/run_lstm_ae.py`)
  - 시도별(서울·경기·인천) 정상 시퀀스로 학습 후 재구성 오차 기반 버블 탐지
  - Optuna HPO (15 trials), Early Stopping(patience=20), 95th percentile 임계값
  - 출력: `data/processed/lstm_ae_scores.csv`, `models/lstm_ae_best.pth`
  - 시각화: `notebooks/fig_ae_01~05_*.png` (학습곡선·오차분포·혼동행렬·피처기여도)
- **HMM 레짐 탐지** (`scripts/feature/run_hmm.py`)
  - GaussianHMM + BIC 기준 최적 상태 수 선택(2~4), 시도별 독립 학습
  - 상태 매핑: 정상/과열/버블 (YoY·전세가율·PIR 복합 점수 기반)
  - HMM vs LSTM-AE AND 앙상블 성능 비교
  - 출력: `data/processed/hmm_regime.csv`
  - 시각화: `notebooks/fig_hmm_01~05_*.png` (BIC·전이행렬·레짐시계열·혼동행렬·피처분포)
- **노트북 업데이트**
  - `notebooks/03_lstm_autoencoder.ipynb` — LSTM-AE 실험 노트북
  - `notebooks/04_hmm_regime.ipynb` — HMM 레짐 탐지 노트북
- **프로젝트 스펙 문서** (`docs/PROJECT_SPEC.md`) — 전체 파이프라인 설계 문서

### Changed
- `HMM_FEATURES`: `recon_error`(LSTM-AE 출력) 포함 7개 피처로 확정
- LSTM-AE 입력 피처: 16개 (거시+가격+자금+심리+공급)
- 버블 판정 기준: bubble_label==2 (정상=0, 과열=1, 버블=2) 3단계

---

## [0.3.0] — 2026-05-22

### Added
- 부동산 정책 이벤트 CSV 추가 (`data/raw/policy/policy_events.csv`)
  - 2013~2025년 주요 규제 이벤트 27개, 컬럼 한글화
- 행정구역 표준화 — 경기 시 단위 통일, 인천 남구→미추홀구

---

## [0.2.0] — 2025-05-22

### Added
- 수도권(서울·경기·인천) 확장
- 매매 전처리 완료 (2013~2025, 3,340,896건)
- 전월세 전처리 완료 (2013~2025, 수도권 6,918,409건)
- 매매 × 전월세 피처 조인 (`features_sudogwon.parquet`, 45,996행 × 22컬럼)
- 버블 Pseudo-label 생성 (정상/과열/버블)
- 날짜 범위 검증 (39개 파일 전수 확인)
- 전월세 파일명 표준화 스크립트

### Changed
- 프로젝트명: seoul → sudogwon
- REGION_CONFIG 멀티지역 파싱 구조 도입
- scripts/ 폴더 구조 정리 (explore/, preprocess/)

---

## [0.1.0] — 2025-05-22

### Added
- 프로젝트 초기 구조 설정
- PostgreSQL 스키마 설계
- 브랜치 전략 수립 (main / dev-kangwook / dev-sihwan)
