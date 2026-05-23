# 수도권 아파트 버블 조기경보 시스템

수도권(서울·경기·인천) 아파트 가격 버블 조기경보 MLOps 파이프라인

## 프로젝트 개요

국토부 실거래가, 한국은행, 통계청, 뉴스 감성 데이터를 결합하여  
LSTM-Autoencoder + HMM + Granger + CUSUM 앙상블로 버블 상태를 조기 탐지합니다.

## 팀 구성

| 역할 | 브랜치 |
|---|---|
| 강욱 | `dev-kangwook` |
| 팀원 | `dev-teammate` |

## 브랜치 전략

```
main              ← 최종 완성본만
├── dev-kangwook  ← 강욱 작업 브랜치
└── dev-teammate  ← 팀원 작업 브랜치
```

- `main` 직접 푸시 금지
- 각자 브랜치에서 작업 → PR → 상대방 확인 → merge
- 매일 저녁 한 번 merge 루틴

## 폴더 구조

```
sudogwon-bubble-detection/
├── data/
│   ├── raw/          ← 각 폴더 README.md 참조해서 직접 다운로드
│   └── processed/    ← 전처리 완료 데이터 (로컬에만)
├── notebooks/        ← EDA ~ 앙상블 분석 노트북 (7개)
├── src/              ← 전처리 / Feature / 모델 / 시각화 코드
├── api/              ← FastAPI 서빙
├── n8n/workflows/    ← n8n 자동화 워크플로우
└── presentation/     ← 발표 자료
```

## 데이터 수집

각 `data/raw/*/README.md` 파일에 수집 방법 명시되어 있습니다.  
데이터 파일(CSV, xlsx)은 `.gitignore`에 의해 깃에서 제외됩니다.

## 환경 설정

```bash
cp .env.example .env
# .env 에 실제 API 키 입력

pip install -r requirements.txt
```

## 커밋 메시지 규칙

```
init     : 초기 설정
data     : 전처리·데이터 관련
feat     : 새 기능
fix      : 버그 수정
refactor : 코드 개선
docs     : 문서 수정
test     : 테스트
```

## 역할별 담당

**강욱:** `schema.sql`, `preprocessing.py`, `lstm_autoencoder.py`, `ensemble.py`, `api/main.py`, `docker-compose.yml`

**팀원:** `feature_engineering.py`, `hmm_model.py`, `granger_analysis.py`, `cusum_detector.py`, `shap_analysis.py`, `n8n/workflows/`
