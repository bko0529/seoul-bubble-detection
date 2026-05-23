# scripts/

일회성 실행 스크립트 모음. `src/`와 달리 파이프라인에 포함되지 않는 작업용 스크립트.

## preprocess/
| 파일 | 설명 | 실행 순서 |
|------|------|-----------|
| `rename_jeonse.py` | 전월세 파일명 표준화 (1회성) | ① |
| `run_preprocess_2013.py` | 2013년 매매 개별 전처리 (테스트용) | — |
| `run_preprocess_jeonse.py` | 전월세 전체 전처리 (2013~2025) | ② |
| `run_feature_engineering.py` | 피처 테이블 생성 (매매 + 전월세 조인) | ③ |
| `summary_all.py` | 전처리 결과 요약 출력 | — |
| `reprocess_2013.py` | 2013년 재전처리 스크립트 | — |

## explore/
| 파일 | 설명 |
|------|------|
| `check_columns.py` | 매매 CSV 컬럼 확인 |
| `check_date_range.py` | 날짜 범위 검증 |
| `check_jeonse_cols.py` | 전월세 CSV 컬럼 확인 |
| `check_jeonse_columns.py` | 전월세 컬럼 빠른 확인 |

## 실행 순서 (최초 세팅)
```bash
python scripts/preprocess/rename_jeonse.py
python scripts/preprocess/run_preprocess_jeonse.py
python scripts/preprocess/run_feature_engineering.py
```
