# 수도권 아파트 가격 버블 조기경보 시스템
## MLOps 풀스택 파이프라인 | 수도권 64개 시·군·구

---

## 시스템이 해결하는 문제

버블은 항상 사후에 인식된다. 2008년 미국 금융위기, 2022년 한국 부동산 급락 모두
붕괴 이후에야 "버블이었다"고 판단됐다.

이 시스템은 PIR·전세가율·금리·심리지수 4개 차원을 동시에 분석해
수도권 64개 시·군·구의 버블 붕괴 신호를 **3~6개월 전에 자동 경보**한다.

### 전체 파이프라인 흐름

```
[1단계] 데이터 재구축 (구 단위 피처 엔지니어링)
features_sgg_monthly.csv (8컬럼)
   → 매매 파생지표 (YoY/MoM/MA/Volatility/Z-score)
   → 거시지표 병합 (기준금리/BSI/M2/CPI — 시도 단위에서 조인)
   → 버블 Pseudo-label (시도별 차등 기준)
   → features_sgg_final.csv (30+컬럼, 10,140행)
        ↓
[2단계] EDA · 전처리
결측치 처리 · 깡통전세(전세가율>100%) 처리 · 단위근 검정
        ↓
[3단계] 모델 학습
LSTM-Autoencoder (재구성 오차 기반 이상탐지)
HMM (정상/과열/버블 레짐 분류)
Granger 공간 분석 (버블 전파 경로: 강남→마포→과천→분당)
CUSUM (구조적 변화점 탐지)
        ↓
[4단계] 앙상블 최적화 + SHAP 분석
Optuna 가중치 최적화 → 버블 경보 레벨 (0~100)
XGBoost Surrogate → Feature 중요도 시각화
        ↓
[5단계] 수도권 버블 히트맵
folium 지도 + 월별 경보 레벨 시각화
        ↓
[6단계] FastAPI 서빙
/predict · /heatmap · /report · /health
        ↓
[7단계] n8n 자동화
수집 → 탐지 → AI 리포트 → Slack/Gmail 경보
```

---

## 프로젝트 범위

```
서울:  25개 구
경기:  31개 시·군
인천:   8개 구·군
────────────────
합계:  64개 시·군·구 × 156개월 (2013.01 ~ 2025.12) = 10,140행
```

---

## 1. 부동산 시계열 데이터 공학 원칙

### 1-1. 부동산 데이터의 3가지 구조적 함정

일반 ML 데이터와 달리 부동산 실거래 데이터에는 도메인 특유의 함정이 존재한다.

**함정 1: 계약 시점 vs 신고 시점 괴리**
```
국토부 실거래가: 계약일 기준이지만, 신고는 계약 후 30일 이내에 이루어짐
→ 12월 계약 건이 1월에 신고되면 집계 시점이 다음 달로 밀림
→ 최근 1~2개월 데이터는 미신고 건으로 인해 거래량이 과소 집계됨

대응: 분석 시점 기준 최신 2개월 데이터는 해석 시 주의 표시
```

**함정 2: 거래량 희박 지역의 중위가격 불안정**
```
경기 연천군, 가평군, 양평군 등: 월 거래량 1~5건
→ 단 1건의 고가 거래가 중위가격을 수십% 왜곡 가능
→ 이런 지역의 YoY 급등은 실제 버블이 아닌 통계 노이즈

대응: 월 거래량 < 5건인 구·월 조합은 이상탐지 대상에서 제외
      또는 rolling 3개월 평균 거래량으로 스무딩
```

**함정 3: 전세가율 > 100% (깡통전세 현상)**
```
현재 데이터에서 전세가율 > 100%: 286건 (2.8%)
예: 인천 모구 2019.11 — 매매 7,600만원, 전세 15,000만원 → 전세가율 197%

원인: 역전세 시장에서 전세보증금이 매매가를 초과하는 현상
      (임대인이 전세금을 돌려주지 못하는 "깡통전세" 상태)

대응: 전세가율을 100%로 cap하지 않고, is_kangtong 플래그를 별도 생성
      모델에는 cap(100)된 값 사용 + 플래그를 별도 피처로 추가
      → 깡통전세 자체가 위기 신호이므로 정보를 보존
```

### 1-2. 버블 정의 원칙

> "버블은 기초가치(Fundamental Value)에서의 괴리다"
> — 전세가율은 임대수익 기반 기초가치를 반영,
>   PIR은 소득 대비 구매가능성을 반영한다.

**버블 판단 2가지 차원:**
```
차원 1. 가격이 기초가치 대비 얼마나 이탈했는가?
  → 전세가율 하락: 매매가 대비 전세가가 낮아짐 = 투자수요 과열 신호
  → PIR 상승: 소득 대비 가격이 지불 불가능한 수준으로 상승

차원 2. 이탈 속도가 얼마나 빠른가?
  → YoY 급등: 단기간 급격한 가격 상승 = 투기적 매수 신호
```

**시도별 차등 기준 (절대 기준):**
```
서울 평균 전세가율 (2013~2023): 약 58%
경기 평균 전세가율 (2013~2023): 약 70%
인천 평균 전세가율 (2013~2023): 약 73%

버블(2) 기준:
  서울: 전세가율 < 55% AND 매매_YoY > 12%  → 서울형 투기 버블
  경기: 전세가율 < 65% AND 매매_YoY > 15%  → 경기 급등 버블
  인천: 전세가율 < 68% AND 매매_YoY > 15%  → 인천 급등 버블

과열(1) 기준:
  서울: 전세가율 < 63% AND 매매_YoY > 7%
  경기: 전세가율 < 72% AND 매매_YoY > 10%
  인천: 전세가율 < 75% AND 매매_YoY > 10%

정상(0): 나머지

근거:
  - 버블 임계값 = 각 시도 평균 전세가율 - 약 5~8pp
  - YoY 기준은 IMF·BIS 과열 기준(8~15%) 참고
  - 서울은 이미 전세가율이 낮아 절대값 기준 적용 가능
  - 경기/인천은 구조적으로 전세가율이 높으므로 임계값 상향 조정
```

**PIR 기준 (2단계 업그레이드 예정):**
```
현재: 피처 테이블에 구별 PIR 없음 (구별 중위소득 데이터 미수집)
→ 시도 단위 PIR을 구 단위에 근사치로 병합하여 보조 지표로 활용

업그레이드: KOSIS 구별 중위소득 수집 후
  PIR = 매매중위가격(만원) / (구별 연간 가구 중위소득(만원))
  버블 보조 조건: PIR > 20 (서울 강남), PIR > 12 (경기 일반)
```

### 1-3. 시계열 데이터 분할 원칙

```
[잘못된 방법] K-Fold Cross Validation
→ 2021년 버블 데이터가 2019년 학습 세트에 섞이면
  미래 정보가 과거로 누수 → 검증 점수만 폭등하고 실전에서 붕괴

[올바른 방법] Chronological Split
  학습:  2013.01 ~ 2022.12 (120개월, 전체의 77%)
  검증:  2023.01 ~ 2025.12 (36개월, 전체의 23%)

  LSTM-AE 특이사항:
  - 학습: bubble_label=0 (정상) 시퀀스만 사용
  - 이유: 오토인코더는 정상 패턴을 암기하고,
           버블 시기의 재구성 오차가 높아야 이상탐지가 작동함
  - 버블 시기로 학습하면 버블을 "정상"으로 학습해버림
```

---

## 2단계 | 데이터 현황 파악 및 구조 설계

### 2-1. 현재 보유 데이터 목록

| 파일 | 컬럼 수 | 행 수 | 단위 | 상태 |
|------|---------|-------|------|------|
| features_sgg_monthly.csv | 8개 | 10,140행 | 구·월 | ✅ 기본집계 완료 |
| features_sido_monthly.csv | 42개 | 468행 | 시도·월 | ✅ 피처 풍부 |
| apt_trade_{year}_sudogwon_clean.csv | - | 3,340,896건 | 거래건 | ✅ 원본 |
| apt_jeonse_{year}_sudogwon_clean.csv | - | 6,918,409건 | 거래건 | ✅ 원본 |

**핵심 문제:**
```
features_sgg_monthly.csv = 구 단위이지만 8컬럼뿐
features_sido_monthly.csv = 피처가 풍부하지만 시도 단위 (3개 값뿐)

→ 해결: features_sgg_monthly.csv에서 구 단위 파생피처를 생성하고
         거시지표(금리/BSI/M2)는 features_sido_monthly.csv에서 (ym, 시도) 기준으로 조인
```

### 2-2. 목표 피처 테이블 설계 (features_sgg_final.csv)

```
집계 단위: 계약년월 × 시도 × 구  (10,140행 × ~35컬럼)

그룹 1. 매매 가격 지표 (구별 독립 계산)
  매매중위가격           : 원본
  매매중위_MoM           : 전월비 (%)
  매매중위_YoY           : 전년비 (%)
  매매_MA3               : 3개월 이동평균
  매매_MA12              : 12개월 이동평균
  매매_vol12             : 12개월 rolling 표준편차 (변동성)
  price_zscore_24m       : 24개월 기준 Z-score
  거래량_ratio           : 거래량 / 12개월 rolling 평균

그룹 2. 전세 · 수급 지표 (구별 독립 계산)
  전세중위가격           : 원본
  전세가율               : (전세 / 매매) × 100
  전세가율_cap           : min(전세가율, 100)   ← 모델 입력용
  is_kangtong            : 전세가율 > 100 여부 (0/1)  ← 추가 피처
  전세가율_MA3           : 3개월 이동평균
  전세_거래량_ratio      : 전세거래량 / 12개월 rolling 평균

그룹 3. 거시경제 지표 (시도 단위에서 ym+시도 기준으로 병합)
  base_rate              : 기준금리
  mortgage_rate_chg_yoy  : 주담대 금리 전년비
  m2_yoy_pct             : M2 증가율
  cpi_yoy_pct            : CPI 상승률
  bsi_realestate         : 부동산업 BSI
  PIR                    : 시도 단위 PIR (구별 근사치)

그룹 4. 레이블
  bubble_label           : 0=정상 / 1=과열 / 2=버블 (시도별 차등 기준)
```

### 2-3. 데이터 품질 체크리스트

데이터 로드 직후 반드시 수행:

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('data/processed/features_sgg_final.csv', encoding='utf-8-sig')

# ── 1. 기본 구조
print("Shape:", df.shape)             # 예상: (10140, 35)
print("Null 비율:")
print(df.isnull().sum() / len(df) * 100)

# ── 2. 시간 연속성 검증 (핵심)
# 특정 구에서 특정 월이 통째로 빠지면 시퀀스 모델이 오동작
ym_range = pd.period_range('2013-01', '2025-12', freq='M').astype(str).str.replace('-','')
for gu in df['구'].unique():
    gu_yms = set(df[df['구']==gu]['ym'].astype(str))
    missing = set(ym_range) - gu_yms
    if missing:
        print(f"[누락] {gu}: {sorted(missing)[:5]}...")

# ── 3. 거래량 희박 지역 확인
sparse = df.groupby('구')['매매거래량'].mean()
print("\n월평균 거래량 < 5건 지역:")
print(sparse[sparse < 5])

# ── 4. 깡통전세 현황
print(f"\n전세가율 > 100%: {(df['전세가율'] > 100).sum()}건")
print(f"전세가율 > 200%: {(df['전세가율'] > 200).sum()}건")

# ── 5. 단위근 검정 (Non-Stationary 확인)
# 수준값(level)이 아닌 변화율(YoY/MoM) 피처를 써야 하는 이유
for gu in ['강남구', '수원시', '연수구']:
    sub = df[df['구']==gu]['매매중위가격'].dropna()
    if len(sub) > 30:
        adf_p = adfuller(sub)[1]
        print(f"{gu} 매매중위가격 ADF p={adf_p:.4f}", end="")
        print(" → 비정상(Non-Stationary)" if adf_p > 0.05 else " → 정상(Stationary)")

# ── 6. Pseudo-label 분포 확인
print("\nPseudo-label 분포:")
print(df['bubble_label'].value_counts(normalize=True).round(3))
# 버블(2): 3~8% 범위가 이상적 (너무 적으면 학습 불가, 너무 많으면 기준이 느슨한 것)
```

---

## 3단계 | 피처 엔지니어링 (구 단위 재구축)

### 3-1. 핵심 설계 원칙

```
원칙 1: 구별(gu_code) 그룹핑 후 시계열 연산
  → 전체 정렬 후 pct_change하면 A구 마지막값이 B구 첫 번째 변화율에 영향
  → 반드시 groupby('구').transform(lambda x: ...) 패턴 사용

원칙 2: 거시지표는 구 단위에서 계산 불가 → 시도 단위에서 병합
  → base_rate, BSI, M2, CPI = 시도 수준 값이지만 구에도 동일하게 적용
  → merge key: ['ym', '시도']

원칙 3: 정규화 전략
  → LSTM-AE: MinMaxScaler (정상(0) 데이터만으로 fit)
  → HMM: StandardScaler (Gaussian 가정 충족을 위해)
  → Granger/CUSUM: 비율/변화율 피처를 그대로 사용 (Scale-insensitive)
```

### 3-2. 피처 생성 코드

```python
# scripts/feature/run_feature_sgg.py
import pandas as pd
import numpy as np

ENC = 'utf-8-sig'

# ── 1. 기본 데이터 로드
sgg = pd.read_csv('data/processed/features_sgg_monthly.csv', encoding=ENC)
sido = pd.read_csv('data/processed/features_sido_monthly.csv', encoding=ENC)

# 컬럼명 확인 후 표준화
# sgg 컬럼: ym, 시도, 구, 매매중위가격, 매매거래량, 전세중위가격, 전세거래량, 전세가율

sgg = sgg.sort_values(['시도', '구', 'ym']).reset_index(drop=True)

# ── 2. 그룹 1: 매매 가격 파생지표
g = sgg.groupby('구')['매매중위가격']
sgg['매매중위_MoM']    = g.pct_change(1) * 100
sgg['매매중위_YoY']    = g.pct_change(12) * 100
sgg['매매_MA3']        = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
sgg['매매_MA12']       = g.transform(lambda x: x.rolling(12, min_periods=6).mean())
sgg['매매_vol12']      = g.transform(lambda x: x.pct_change().rolling(12).std() * 100)

# Z-score: 해당 구의 24개월 기준 상대적 이상 수준
sgg['price_zscore_24m'] = sgg.groupby('구')['매매중위가격'].transform(
    lambda x: (x - x.rolling(24).mean()) / (x.rolling(24).std() + 1e-8)
)

# 거래량 비율: 현재 거래량 / 12개월 평균 (투기 강도 측정)
sgg['거래량_ratio'] = sgg.groupby('구')['매매거래량'].transform(
    lambda x: x / (x.rolling(12).mean() + 1e-8)
)

# ── 3. 그룹 2: 전세 · 수급 지표
# 깡통전세 처리: 플래그 생성 + cap 적용
sgg['is_kangtong']    = (sgg['전세가율'] > 100).astype(int)
sgg['전세가율_cap']   = sgg['전세가율'].clip(upper=100)

sgg['전세가율_MA3']   = sgg.groupby('구')['전세가율_cap'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
sgg['전세가율_diff3'] = sgg.groupby('구')['전세가율_cap'].transform(
    lambda x: x - x.shift(3)
)

# ── 4. 그룹 3: 거시지표 병합 (시도 단위에서 조인)
macro_cols = ['ym', '시도', 'PIR', 'base_rate', 'mortgage_rate_chg_yoy',
              'm2_yoy_pct', 'cpi_yoy_pct', 'bsi_realestate']
macro = sido[macro_cols].copy()
sgg = sgg.merge(macro, on=['ym', '시도'], how='left')

# ── 5. 그룹 4: 버블 Pseudo-label (시도별 차등 기준)
# 근거:
#   서울 평균 전세가율 ~58% → 임계값 55% (평균 -3pp)
#   경기 평균 전세가율 ~70% → 임계값 65% (평균 -5pp)
#   인천 평균 전세가율 ~73% → 임계값 68% (평균 -5pp)
def make_bubble_label(row):
    jeonse  = row['전세가율_cap']
    yoy     = row['매매중위_YoY']
    sido    = row['시도']

    if pd.isna(jeonse) or pd.isna(yoy):
        return np.nan

    # 버블(2)
    if sido == '서울'  and jeonse < 55 and yoy > 12: return 2
    if sido == '경기'  and jeonse < 65 and yoy > 15: return 2
    if sido == '인천'  and jeonse < 68 and yoy > 15: return 2

    # 과열(1)
    if sido == '서울'  and jeonse < 63 and yoy > 7:  return 1
    if sido == '경기'  and jeonse < 72 and yoy > 10: return 1
    if sido == '인천'  and jeonse < 75 and yoy > 10: return 1

    return 0

sgg['bubble_label'] = sgg.apply(make_bubble_label, axis=1)

# ── 6. 저장
sgg.to_csv('data/processed/features_sgg_final.csv', index=False, encoding=ENC)
print(f"저장 완료: {sgg.shape}")
print(f"bubble_label 분포:\n{sgg['bubble_label'].value_counts(dropna=False)}")
```

### 3-3. 피처 엔지니어링 완료 기준

- [ ] 모든 구별 YoY/MoM 계산 시 groupby 패턴 사용 확인
- [ ] 전세가율 > 100% → is_kangtong 플래그 생성 확인
- [ ] 거시지표 병합 후 NaN 없음 확인 (시도-ym 키 미매칭 없는지)
- [ ] bubble_label 분포: 버블(2) 3~8%, 과열(1) 5~15% 범위 확인
- [ ] ADF 검정: 수준값(price level) Non-Stationary → YoY/MoM 피처 사용 확인
- [ ] 거래량 < 5건 구·월 조합 마스킹 또는 별도 처리

---

## 4단계 | 버블 기준 검증 (EDA)

### 4-1. 시도별 버블 분포 확인

```python
# notebooks/01_EDA_sgg.ipynb

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/features_sgg_final.csv', encoding='utf-8-sig')

# 시도별 버블 건수 확인
print("시도별 bubble_label 분포:")
print(df.groupby(['시도', 'bubble_label']).size().unstack(fill_value=0))

# 예상 결과:
# 서울: 버블 50~100건, 과열 100~200건
# 경기: 버블 30~80건, 과열 100~300건
# 인천: 버블 10~30건, 과열 30~80건

# 버블 기간 시각화
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
for ax, sido in zip(axes, ['서울', '경기', '인천']):
    sido_df = df[df['시도']==sido].groupby('ym')['bubble_label'].max().reset_index()
    sido_df['ym_dt'] = pd.to_datetime(sido_df['ym'].astype(str), format='%Y%m')
    ax.fill_between(sido_df['ym_dt'], sido_df['bubble_label'],
                    alpha=0.4, color='red', label='버블 최고 레벨')
    ax.set_title(f'{sido} — 구별 최고 버블 레벨 (월별)')
    ax.set_ylim(-0.1, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['정상', '과열', '버블'])
    ax.legend()
plt.tight_layout()
plt.savefig('notebooks/fig_eda_sgg_bubble_dist.png')
```

### 4-2. 2021년 경기 버블 검증 (핵심 케이스)

```python
# 2021년 경기 일부 지역 YoY +50% 케이스가 탐지되는지 확인
gyeonggi_2021 = df[
    (df['시도']=='경기') &
    (df['ym'] >= 202101) & (df['ym'] <= 202112)
][['구', 'ym', '매매중위_YoY', '전세가율_cap', 'bubble_label']].sort_values('매매중위_YoY', ascending=False)
print(gyeonggi_2021.head(20))

# 기대값: 2021년 YoY 30~50% 지역들이 bubble_label=2로 잡혀야 함
# 만약 버블로 안 잡히면 → 기준 임계값 재조정 필요
```

---

## 5단계 | 모델 학습

### 5-1. LSTM-Autoencoder (비지도 이상탐지)

**설계 원칙:**
```
목적: 정상 시장 패턴을 학습하고, 벗어난 정도(재구성 오차)를 이상점수로 출력
핵심: 버블 기간 데이터를 학습에 포함하면 버블을 "정상"으로 학습해버림
      → 반드시 bubble_label=0 시퀀스만으로 학습

입력 피처 (16개):
  [매매중위_MoM, 매매중위_YoY, 매매_vol12,
   전세가율_cap, is_kangtong, 전세가율_MA3,
   PIR, base_rate, mortgage_rate_chg_yoy,
   m2_yoy_pct, cpi_yoy_pct, bsi_realestate,
   거래량_ratio, price_zscore_24m,
   전세가율_diff3, 전세_거래량_ratio]

SEQ_LEN: 12 (12개월 슬라이딩 윈도우)
학습 단위: 구별 독립 스케일링 (각 구의 정상 범위가 다름)
스케일링: MinMaxScaler (정상 데이터만 fit)
임계값: 정상 시퀀스 재구성 오차의 95th percentile
```

**모델 구조:**
```python
# src/models/lstm_autoencoder.py

class LSTMAutoEncoder(nn.Module):
    """
    Encoder: LSTM → FC → latent_dim
    Decoder: FC → LSTM → FC → 재구성
    
    재구성 오차(MSE)가 높을수록 이상(버블) 의심 구간
    """
    def __init__(self, n_features, hidden_size, latent_dim, num_layers, dropout):
        super().__init__()
        dp = dropout if num_layers > 1 else 0.0
        self.enc_lstm = nn.LSTM(n_features, hidden_size, num_layers,
                                batch_first=True, dropout=dp)
        self.enc_fc   = nn.Linear(hidden_size, latent_dim)
        self.dec_fc   = nn.Linear(latent_dim, hidden_size)
        self.dec_lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                                batch_first=True, dropout=dp)
        self.out_fc   = nn.Linear(hidden_size, n_features)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)
        return self.enc_fc(h[-1])

    def decode(self, z, seq_len):
        d = self.dec_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.dec_lstm(d)
        return self.out_fc(out)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z, x.size(1)), z
```

**Optuna 하이퍼파라미터 탐색:**
```python
# 탐색 공간
params = {
    'hidden_size': [32, 64, 128],
    'latent_dim':  [8, 16, 32],
    'num_layers':  [1, 2],
    'dropout':     [0.0, 0.1, 0.2, 0.3],
    'lr':          [5e-4, 5e-3],  # log scale
    'batch_size':  [16, 32],
}
n_trials = 50   # 이전 15회 → 50회로 확대 (시간 충분)
n_epochs_objective = 50   # objective 내부 빠른 평가
n_epochs_final     = 200  # 최종 모델 충분히 학습
patience           = 20   # early stopping
```

### 5-2. Gaussian HMM (레짐 분류)

**설계 원칙:**
```
목적: 시장 상태(정상/과열/버블)를 확률적으로 분류
      단순 임계값 기준이 아닌 복수 지표의 복합 패턴을 학습

입력 피처 (7개, StandardScaler 적용):
  [매매중위_YoY, 전세가율_cap, PIR, base_rate,
   매매_vol12, bsi_realestate, recon_error(LSTM-AE 출력)]

주의: HMM은 Gaussian 분포 가정 → StandardScaler 필수
      MinMaxScaler는 Gaussian 가정을 위반

상태 수: BIC 기준 최적값 (서울 기준 탐색) + 최대 4개 제한
         (정상/과열/버블1/버블2 이상으로 늘어나면 해석 어려움)

학습 단위: 시도별 독립 학습
           (서울형 버블 ≠ 경기형 버블이므로 공통 모델은 부적절)
```

**상태 레이블 자동 매핑:**
```python
def map_states(sub_df, states, n_states):
    """
    HMM 상태 번호는 학습 시마다 달라질 수 있음
    → 경제적 의미 기반 자동 매핑 필요
    
    버블 점수 = YoY - 전세가율×0.3 + PIR×1.5
    점수 높은 순: 버블 → 과열 → 정상(들)
    """
    stats = []
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0: continue
        yoy    = sub_df.loc[mask, '매매중위_YoY'].mean()
        jeonse = sub_df.loc[mask, '전세가율_cap'].mean()
        pir    = sub_df.loc[mask, 'PIR'].mean()
        score  = yoy - jeonse * 0.3 + pir * 1.5
        stats.append({'state': s, 'score': score})

    stats_df = pd.DataFrame(stats).sort_values('score', ascending=False)
    ranks    = stats_df['state'].tolist()
    names    = ['버블', '과열'] + ['정상'] * (n_states - 2)
    return {ranks[i]: names[i] for i in range(len(ranks))}
```

### 5-3. Granger 공간 분석 (버블 전파 경로)

**발표 핵심 포인트:**
```
"2020~2021년 서울 버블이 강남 → 마포 → 과천 → 분당 순서로
 외곽으로 전파됐음을 Granger 인과검정으로 통계적으로 증명"
```

**설계 원칙:**
```
Granger 인과검정의 의미:
  X가 Y를 Granger 인과한다 = "X의 과거값이 Y의 과거값만으로 예측할 때보다
                               더 잘 예측한다" (통계적 유의성: p < 0.05)

분석 2가지:
  1. 서울 내 전파: 강남구 가격이 마포구, 은평구 등에 몇 개월 선행하는가
  2. 시도 간 전파: 서울 전체가 경기 인접 지역에 몇 개월 선행하는가

선행 래그 식별 → 해당 래그 피처 생성 → 앙상블 조기경보 개선
```

**코드:**
```python
# scripts/analysis/run_granger.py
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

def granger_matrix(df, gu_list, target_col='매매중위_YoY', maxlag=6):
    """
    구 간 Granger 인과 행렬 계산
    반환: DataFrame (행=원인 구, 열=결과 구, 값=최적 유의 래그)
    """
    n = len(gu_list)
    result = pd.DataFrame(np.nan, index=gu_list, columns=gu_list)

    for cause in gu_list:
        for effect in gu_list:
            if cause == effect: continue
            series_cause  = df[df['구']==cause ][['ym', target_col]].set_index('ym')
            series_effect = df[df['구']==effect][['ym', target_col]].set_index('ym')
            data = pd.concat([series_effect, series_cause],
                             axis=1, join='inner').dropna()
            if len(data) < 30: continue

            try:
                test = grangercausalitytests(data.values, maxlag=maxlag, verbose=False)
                sig_lags = [lag for lag, res in test.items()
                            if res[0]['ssr_ftest'][1] < 0.05]
                if sig_lags:
                    result.loc[cause, effect] = min(sig_lags)
            except:
                pass
    return result

# 서울 주요 구 전파 분석
seoul_key_gu = ['강남구', '서초구', '송파구', '마포구', '용산구',
                '은평구', '노원구', '도봉구', '강서구', '관악구']
seoul_df = df[df['시도']=='서울'].copy()
granger_mat = granger_matrix(seoul_df, seoul_key_gu)
print("서울 내 Granger 인과 행렬 (값=선행 개월 수):")
print(granger_mat)
```

### 5-4. CUSUM 변화점 탐지

**설계 원칙:**
```
목적: 버블 전환점(가격 급등 시작 시점)을 실시간으로 탐지
      → "언제부터 버블이 시작됐는가"를 사후 검증 + 실시간 경보

4개 지표에 독립 CUSUM 적용:
  PIR, 전세가율_cap, 거래량_ratio, price_zscore_24m

동시 경보 조건:
  4개 중 3개 이상 CUSUM 경보 = 강한 버블 전환 신호
  2개 = 경고
  1개 이하 = 모니터링
```

```python
# src/models/cusum_detector.py

class CUSUMDetector:
    def __init__(self, threshold=5.0, drift=0.5, warmup=24):
        self.threshold = threshold  # Optuna로 튜닝
        self.drift     = drift
        self.warmup    = warmup     # 기준 통계 수립 기간

    def detect(self, series: np.ndarray) -> dict:
        """
        warmup 기간의 평균/표준편차를 기준으로
        이후 관측값이 기준에서 얼마나 누적 이탈했는지 측정
        """
        mean  = np.mean(series[:self.warmup])
        std   = np.std(series[:self.warmup]) + 1e-8
        s_pos = np.zeros(len(series))
        s_neg = np.zeros(len(series))
        alarms = []

        for i in range(1, len(series)):
            z        = (series[i] - mean) / std
            s_pos[i] = max(0, s_pos[i-1] + z - self.drift)
            s_neg[i] = max(0, s_neg[i-1] - z - self.drift)

            if s_pos[i] > self.threshold or s_neg[i] > self.threshold:
                alarms.append(i)
                s_pos[i] = 0
                s_neg[i] = 0

        return {'alarms': alarms, 's_pos': s_pos, 's_neg': s_neg}

    def multi_signal(self, gu_df: pd.DataFrame) -> pd.Series:
        """4개 지표 CUSUM 신호 합산 (0~4)"""
        targets = ['PIR', '전세가율_cap', '거래량_ratio', 'price_zscore_24m']
        signals = np.zeros(len(gu_df))
        for col in targets:
            if col not in gu_df.columns: continue
            result = self.detect(gu_df[col].fillna(method='ffill').values)
            for idx in result['alarms']:
                if idx < len(signals):
                    signals[idx] += 1
        return pd.Series(signals, index=gu_df.index)
```

### 5-5. 앙상블 최적화

**설계 원칙:**
```
4개 모델 출력을 Optuna로 가중치 최적화:
  LSTM-AE: 재구성 오차 (정규화 후 0~1)
  HMM:     버블 상태 확률 (0~1)
  CUSUM:   동시 경보 수 / 4 (0~1)
  Granger: 선행 구 버블 여부 (0/1)

목적함수: pseudo-label 기준 log_loss 최소화
최종 출력: 버블 경보 레벨 (0~100)
```

```python
# src/ensemble.py

class BubbleEnsemble:
    def objective(self, trial, scores, labels):
        w_lstm   = trial.suggest_float('w_lstm',   0.1, 0.5)
        w_hmm    = trial.suggest_float('w_hmm',    0.1, 0.5)
        w_cusum  = trial.suggest_float('w_cusum',  0.05, 0.3)
        w_granger = max(0.05, 1.0 - w_lstm - w_hmm - w_cusum)

        final = (w_lstm   * scores['lstm_ae']          +
                 w_hmm    * scores['hmm_bubble_prob']   +
                 w_cusum  * scores['cusum_signal']      +
                 w_granger* scores['granger_signal'])

        binary_labels = (labels == 2).astype(int)
        return log_loss(binary_labels, np.clip(final, 1e-6, 1-1e-6))

    def get_alert_grade(self, level: int) -> str:
        if level >= 80: return '🔴 위험'
        if level >= 60: return '🟡 경고'
        if level >= 40: return '🟠 주의'
        return '🟢 정상'
```

### 5-6. SHAP 분석

**설계 원칙:**
```
HMM은 SHAP 직접 지원 불가 (확률 그래프 모델)
→ XGBoost Surrogate 모델 학습:
    입력: 앙상블에 사용된 피처
    출력: pseudo-label
    → XGBoost의 SHAP값으로 "어떤 피처가 버블 판단에 기여했는가" 정량화

발표 활용:
  "2021년 강남구 버블 판단에서
   전세가율 하락이 42% 기여, PIR 상승이 28% 기여함을 SHAP으로 확인"
```

### 5-7. 평가지표

| 지표 | 계산 방법 | 목표값 | 근거 |
|------|-----------|--------|------|
| Precision (버블) | 버블 탐지 중 실제 버블 비율 | > 0.60 | 과탐지 최소화 (경보 피로도) |
| Recall (버블) | 실제 버블 중 탐지 비율 | > 0.70 | 조기경보 목적상 누락이 더 위험 |
| AUC-ROC | pseudo-label 기준 | > 0.80 | 판별력 종합 |
| 선행 월 수 | CUSUM 경보 vs 실제 고점 차이 | 3개월↑ | 조기경보 유효성 |
| 오경보율 | 정상 기간 중 경보 발생 비율 | < 10% | 실운영 신뢰성 |

---

## 6단계 | 수도권 버블 히트맵 (지도 시각화)

```
발표 임팩트 포인트:
  수도권 지도 위에 64개 시·군·구를 경보 레벨 색으로 표시
  → 버블이 강남에서 시작해 외곽으로 퍼지는 과정을 시각화

월별 애니메이션 (2020~2023):
  1월 강남구 빨간색 → 3개월 후 과천·분당 빨간색 → 6개월 후 경기 전반 주황색
```

```python
# scripts/visualization/bubble_heatmap.py
import folium
from branca.colormap import linear

def create_bubble_map(df, ym, geojson_path='data/raw/korea_sgg.geojson'):
    """
    특정 연월의 수도권 버블 경보 레벨 지도 생성
    df: features_sgg_final + ensemble 결과 포함
    """
    m = folium.Map(location=[37.5, 127.0], zoom_start=10)

    # 경보 레벨 컬러맵 (0=초록, 100=빨강)
    colormap = linear.RdYlGn_11.scale(0, 100).to_step(10)
    colormap.caption = f'{ym[:4]}년 {int(ym[4:])}월 버블 경보 레벨'

    month_df = df[df['ym'] == int(ym)][['구', 'alert_level']].set_index('구')

    folium.GeoJson(
        geojson_path,
        style_function=lambda feature: {
            'fillColor': colormap(
                month_df.get('alert_level', {}).get(
                    feature['properties']['SIG_KOR_NM'], 0)
            ),
            'color': 'white',
            'weight': 0.5,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['SIG_KOR_NM'],
            aliases=['지역명'],
        )
    ).add_to(m)

    colormap.add_to(m)
    m.save(f'notebooks/bubble_map_{ym}.html')
    return m
```

---

## 7단계 | FastAPI 서빙

### 7-1. 싱글톤 모델 로드 패턴

```python
# api/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch, joblib

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 1회만 로드 (요청마다 로드하면 500ms → 10s 이상)
    models['lstm_ae']   = torch.load('models/lstm_ae_sgg_best.pth', map_location='cpu')
    models['hmm']       = {sido: joblib.load(f'models/hmm_{sido}.pkl')
                           for sido in ['서울', '경기', '인천']}
    models['ensemble']  = joblib.load('models/ensemble.pkl')
    models['shap']      = joblib.load('models/shap_analyzer.pkl')
    models['scaler']    = joblib.load('models/scaler_sgg.pkl')
    print("모델 로드 완료")
    yield
    models.clear()

app = FastAPI(title="수도권 버블 경보 API", lifespan=lifespan)
```

### 7-2. 핵심 엔드포인트

| 엔드포인트 | 입력 | 출력 | 용도 |
|-----------|------|------|------|
| POST /predict | gu_name, deal_ym | alert_level, alert_grade, top_factors | n8n WF3 호출 |
| GET /heatmap/{ym} | 연월 | GeoJSON + 경보 레벨 | 프론트엔드 지도 |
| POST /report | start_ym, end_ym | 차트 base64 + 통계 | n8n WF5 호출 |
| GET /health | - | 모델 로드 상태 | 헬스체크 |

---

## 8단계 | n8n 자동화

### 8-1. 워크플로우 구성

| WF | 이름 | 트리거 | 핵심 노드 |
|----|------|--------|-----------|
| WF1 | 데이터 수집 | 매월 1일 cron | 국토부 API → PostgreSQL |
| WF2 | 감성분석 | WF1 완료 후 | BigKinds → GPT-4o → DB |
| WF3 | 버블 탐지 | WF2 완료 후 | FastAPI /predict → Switch(경보레벨) |
| WF4 | 경보 발송 | WF3 고위험 분기 | Claude 리포트 → Slack → Gmail |
| WF5 | 월간 리포트 | 매월 말일 | FastAPI /report → Sheets 업데이트 |
| WF6 | 에러 핸들링 | 에러 트리거 | 재시도 3회 → Slack 알림 |

### 8-2. Claude API 경보 리포트 System Prompt

```
[ROLE]
너는 수도권 아파트 부동산 버블 조기경보 전문 분석가다.

[CONTEXT]
- 분석 대상: 수도권 64개 시·군·구 월별 버블 경보 시스템
- 버블 기준: 시도별 차등 (서울 전세가율<55%+YoY>12%, 경기 전세가율<65%+YoY>15%)
- 지표: LSTM-AE 재구성 오차 + HMM 상태 확률 + CUSUM 변화점 + Granger 전파 신호
- SHAP: 각 지표가 버블 판단에 기여한 비율을 수치로 제공

[CONSTRAINTS]
- 투자 권유나 매수/매도 조언을 절대 하지 마라.
- 데이터가 없는 내용은 추측하지 마라.
- 반드시 한국어로 답변하되 PIR, SHAP, HMM 등 기술 용어는 영어 유지.
- 200자 이내의 간결한 요약문으로 작성.

[OUTPUT FORMAT]
{
  "summary": "string (200자 이내 자연어 리포트)",
  "risk_level": "정상 | 주의 | 경고 | 위험",
  "key_reason": "string (주요 원인 1문장)",
  "spillover_risk": "string (인근 지역 전파 가능성 1문장)"
}
```

---

## 9단계 | 배포 · 검증

### 9-1. 엔드투엔드 테스트 시나리오

```
시나리오 A — 정상 케이스
입력: 경기 포천시 2013년 1월 (PIR ≈ 5, 전세가율 ≈ 75%)
기대: alert_level < 30, hmm_state = '정상', Slack 미발송

시나리오 B — 버블 케이스 (서울)
입력: 강남구 2021년 8월 (전세가율 ≈ 46%, YoY ≈ 28%)
기대: alert_level > 75, hmm_state = '버블', Slack 🔴 발송

시나리오 C — 버블 케이스 (경기)
입력: 화성시 2021년 7월 (전세가율 ≈ 60%, YoY ≈ 45%)
기대: alert_level > 60, bubble_label = 2, Slack 🟡 발송

시나리오 D — 깡통전세 케이스
입력: 인천 모구 2022년 (전세가율 > 100%, is_kangtong = 1)
기대: CUSUM 경보 발생, alert_level > 50

시나리오 E — 에러 핸들링
입력: 존재하지 않는 gu_name
기대: WF6 에러 로그 저장 + Slack 경고 발송
```

---

## 기술 스택

| 분류 | 기술 | 용도 |
|------|------|------|
| 데이터 | pandas · numpy | 피처 엔지니어링 |
| 저장소 | PostgreSQL | 원본 + 피처 저장 |
| ML | PyTorch · hmmlearn · statsmodels | LSTM-AE · HMM · Granger |
| 최적화 | Optuna (TPESampler) | 하이퍼파라미터 자동 탐색 |
| 설명 가능 AI | SHAP · XGBoost surrogate | Feature 중요도 분해 |
| 시각화 | folium · matplotlib · seaborn | 지도 히트맵 · 차트 |
| AI API | Claude API · GPT-4o | 리포트 생성 · 감성분석 |
| 서빙 | FastAPI · uvicorn · Docker | REST API 서버 |
| 자동화 | n8n (6개 WF) | 전체 파이프라인 오케스트레이션 |
| 알림 | Slack · Gmail · Google Sheets | 경보 발송 · 현황판 |

---

## 전체 완료 기준 체크리스트

### Phase 1. 데이터 재구축
- [ ] features_sgg_final.csv 생성 (10,140행 × 35+컬럼)
- [ ] 전세가율 > 100% → is_kangtong 플래그 처리 완료
- [ ] 거시지표 병합 NaN 없음 확인
- [ ] bubble_label 시도별 분포: 버블 3~8%, 과열 5~15% 범위
- [ ] ADF 검정: 수준값 Non-Stationary 확인 → YoY/MoM 피처 사용

### Phase 2. 모델
- [ ] LSTM-AE: Optuna 50회 + 최종 200 epochs 학습 완료
- [ ] HMM: 시도별 3개 독립 모델 + 상태 전환 행렬 시각화
- [ ] Granger: 서울 내 전파 경로 + 서울→경기 선행 래그 확정
- [ ] CUSUM: 2022년 가격 고점 사후 검증 (경보 시점 확인)
- [ ] 앙상블: Optuna 100회 가중치 최적화
- [ ] SHAP: Feature 중요도 요약 차트 저장

### Phase 3. 시각화
- [ ] 수도권 버블 히트맵 (2020~2023 주요 연월)
- [ ] Granger 전파 방향 그래프 (네트워크 시각화)
- [ ] 시도별 버블 타임라인 비교 차트

### Phase 4. MLOps
- [ ] FastAPI /predict 응답 시간 < 500ms
- [ ] n8n 6개 워크플로우 엔드투엔드 통과
- [ ] 시나리오 A~E 전체 통과
- [ ] README.md 실행 방법 문서화

---

## 미결 이슈 (추후 해결)

| 이슈 | 현황 | 해결 방향 |
|------|------|-----------|
| 구별 PIR | 시도 단위 근사치 사용 중 | KOSIS 구별 중위소득 수집 후 업그레이드 |
| 뉴스 감성분석 | bsi_realestate로 대체 중 | BigKinds API 키 확보 후 추가 |
| GeoJSON | 수도권 시군구 경계 파일 필요 | SGIS 또는 공공데이터포털에서 수집 |
| 모델 재학습 주기 | 현재 수동 | n8n WF1 완료 후 자동 트리거 연결 |
