"""
Feature Engineering: 매매 + 전월세 CSV 조인 → 피처 테이블 생성
집계 단위: 계약년월(deal_ym) × 시도(sido) × 구(gu)
출력: data/processed/features_sudogwon.parquet
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np

os.chdir("c:/Users/bko05/Desktop/seoul-bubble-detection")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 데이터 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 매매 데이터 로드 중...")
trade_frames = []
for year in range(2013, 2026):
    path = f"data/processed/apt_trade_{year}_sudogwon_clean.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str,
                         usecols=["계약년월", "시도", "구", "거래금액(만원)", "㎡당단가(만원)"])
        trade_frames.append(df)
df_trade = pd.concat(trade_frames, ignore_index=True)
df_trade["거래금액(만원)"]  = pd.to_numeric(df_trade["거래금액(만원)"],  errors="coerce")
df_trade["㎡당단가(만원)"] = pd.to_numeric(df_trade["㎡당단가(만원)"], errors="coerce")
df_trade = df_trade.dropna(subset=["거래금액(만원)", "㎡당단가(만원)"])
print(f"   매매 총 {len(df_trade):,}건")

print("▶ 전월세 데이터 로드 중...")
jeonse_frames = []
for year in range(2013, 2026):
    path = f"data/processed/apt_jeonse_{year}_sudogwon_clean.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str,
                         usecols=["계약년월", "시도", "구", "전월세구분",
                                  "보증금(만원)", "전용면적(㎡)"])
        jeonse_frames.append(df)
df_jeonse = pd.concat(jeonse_frames, ignore_index=True)
df_jeonse["보증금(만원)"]  = pd.to_numeric(df_jeonse["보증금(만원)"],  errors="coerce")
df_jeonse["전용면적(㎡)"] = pd.to_numeric(df_jeonse["전용면적(㎡)"], errors="coerce")
df_jeonse = df_jeonse.dropna(subset=["보증금(만원)", "전용면적(㎡)"])
df_jeonse["보증금_per_m2"] = df_jeonse["보증금(만원)"] / df_jeonse["전용면적(㎡)"]
print(f"   전월세 총 {len(df_jeonse):,}건")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 매매 집계: 월 × 시도 × 구
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 매매 집계 중...")
KEY = ["계약년월", "시도", "구"]

agg_trade = df_trade.groupby(KEY).agg(
    median_price      = ("거래금액(만원)",  "median"),
    median_price_m2   = ("㎡당단가(만원)", "median"),
    trade_volume      = ("거래금액(만원)",  "count"),
).reset_index()
print(f"   집계 결과: {len(agg_trade):,}행")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 전세 집계: 월 × 시도 × 구
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 전세 집계 중...")
df_jeonse_only = df_jeonse[df_jeonse["전월세구분"] == "전세"]
agg_jeonse = df_jeonse_only.groupby(KEY).agg(
    median_jeonse_m2  = ("보증금_per_m2", "median"),
    jeonse_volume     = ("보증금(만원)",  "count"),
).reset_index()
print(f"   집계 결과: {len(agg_jeonse):,}행")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 월세전환율 집계: 월 × 시도 × 구
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 월세전환율 집계 중...")
agg_wolse = df_jeonse.groupby(KEY).apply(
    lambda x: pd.Series({
        "total_rent_volume": len(x),
        "wolse_volume":      (x["전월세구분"] == "월세").sum(),
    })
).reset_index()
agg_wolse["wolse_ratio"] = agg_wolse["wolse_volume"] / agg_wolse["total_rent_volume"]
print(f"   집계 결과: {len(agg_wolse):,}행")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 조인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 조인 중...")
df = agg_trade \
    .merge(agg_jeonse, on=KEY, how="left") \
    .merge(agg_wolse[KEY + ["wolse_ratio", "total_rent_volume"]], on=KEY, how="left")

# 전세가율 (㎡당단가 기준)
df["jeonse_rate"] = df["median_jeonse_m2"] / df["median_price_m2"]

# 거래량 비율 (매매 / 전세)
df["trade_to_jeonse_ratio"] = df["trade_volume"] / (df["jeonse_volume"] + 1)

df = df.sort_values(["시도", "구", "계약년월"]).reset_index(drop=True)
print(f"   조인 결과: {len(df):,}행  ({df[['시도','구']].drop_duplicates().shape[0]}개 지역)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 시계열 파생 피처 (구별 그룹)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 시계열 파생 피처 생성 중...")
g = df.groupby(["시도", "구"])

# 가격 변화율
df["price_mom"]    = g["median_price_m2"].pct_change(1)          # 전월비
df["price_yoy"]    = g["median_price_m2"].pct_change(12)         # 전년비
df["price_ma3"]    = g["median_price_m2"].transform(
    lambda x: x.rolling(3,  min_periods=1).mean())
df["price_ma12"]   = g["median_price_m2"].transform(
    lambda x: x.rolling(12, min_periods=6).mean())
df["price_vol"]    = g["price_mom"].transform(
    lambda x: x.rolling(12, min_periods=3).std())                 # 가격 변동성
df["price_zscore"] = g["median_price_m2"].transform(
    lambda x: (x - x.rolling(24, min_periods=12).mean())
              / (x.rolling(24, min_periods=12).std() + 1e-8))    # 표준화 점수

# 전세가율 변화
df["jeonse_rate_ma3"]  = g["jeonse_rate"].transform(
    lambda x: x.rolling(3, min_periods=1).mean())
df["jeonse_rate_diff"] = g["jeonse_rate"].diff(3)                # 3개월 변화

# 월세전환율 변화
df["wolse_ratio_ma3"]  = g["wolse_ratio"].transform(
    lambda x: x.rolling(3, min_periods=1).mean())

print("   완료")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 버블 Pseudo-label
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 버블 레이블 생성 중...")
conditions = [
    # 버블(2): 전세가율 50% 미만 AND 연간 15% 이상 상승
    (df["jeonse_rate"] < 0.50) & (df["price_yoy"] > 0.15),
    # 과열(1): 전세가율 60% 미만 AND 연간 8% 이상 상승
    (df["jeonse_rate"] < 0.60) & (df["price_yoy"] > 0.08),
]
df["bubble_label"] = np.select(conditions, [2, 1], default=0)

lc = df["bubble_label"].value_counts().sort_index()
print(f"   정상(0): {lc.get(0,0):,}행")
print(f"   과열(1): {lc.get(1,0):,}행")
print(f"   버블(2): {lc.get(2,0):,}행")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
out_path = "data/processed/features_sudogwon.parquet"
df.to_parquet(out_path, index=False)
print(f"\n✅ 저장 완료: {out_path}")
print(f"   shape  : {df.shape}")
print(f"   컬럼   : {df.columns.tolist()}")

# ── 샘플 출력
print("\n[피처 테이블 샘플 - 서울 강남구]")
sample = df[(df["시도"] == "서울") & (df["구"] == "강남구")].tail(6)
show_cols = ["계약년월", "median_price_m2", "jeonse_rate",
             "wolse_ratio", "price_yoy", "price_zscore", "bubble_label"]
print(sample[show_cols].to_string(index=False))
