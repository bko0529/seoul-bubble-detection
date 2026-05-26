"""
Feature Engineering Steps 2~5
입력:
  - data/processed/features_sgg_monthly.csv
  - data/raw/ecos/macro_ecos_2013_2025.csv
  - data/raw/bok/housing_permit_sudogwon_2013_2025.csv
  - data/raw/bok/housing_completion_sudogwon_2013_2025.csv
  - data/raw/bok/household_debt_per_borrower_sudogwon_2013_2025.csv
  - data/raw/bok/household_debt_share_sudogwon_2013_2025.csv
  - data/raw/bok/unsold_apt_sudogwon_2013_2025.csv
  - data/raw/kosis/kosis_income_median.csv
  - data/raw/naver/search_trend_2016_2025_monthly.csv
출력:
  - data/processed/features_sido_monthly.csv
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import numpy as np

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
ENC = "utf-8-sig"

print("=" * 60)
print("  Feature Engineering Steps 2~5")
print("=" * 60)


# ═══════════════════════════════════════════════════════════
# 0. 데이터 로드
# ═══════════════════════════════════════════════════════════
print("\n[0] 데이터 로드")

sgg        = pd.read_csv("data/processed/features_sgg_monthly.csv",                       encoding=ENC)
ecos       = pd.read_csv("data/raw/ecos/macro_ecos_2013_2025.csv")
permit     = pd.read_csv("data/raw/bok/housing_permit_sudogwon_2013_2025.csv",            encoding=ENC)
completion = pd.read_csv("data/raw/bok/housing_completion_sudogwon_2013_2025.csv",        encoding=ENC)
debt       = pd.read_csv("data/raw/bok/household_debt_per_borrower_sudogwon_2013_2025.csv", encoding=ENC)
debt_share = pd.read_csv("data/raw/bok/household_debt_share_sudogwon_2013_2025.csv",      encoding=ENC)
unsold     = pd.read_csv("data/raw/bok/unsold_apt_sudogwon_2013_2025.csv",                encoding=ENC)
income     = pd.read_csv("data/raw/kosis/kosis_income_median.csv",                        encoding=ENC)
naver      = pd.read_csv("data/raw/naver/search_trend_2016_2025_monthly.csv",             encoding=ENC)

# ym 정수화
for df_ in [sgg, permit, completion, debt, debt_share, unsold, naver]:
    df_["ym"] = df_["ym"].astype(int)
ecos["ym"] = ecos["ym"].astype(int)

print(f"  sgg:        {sgg.shape}")
print(f"  ecos:       {ecos.shape}")
print(f"  permit:     {permit.shape}")
print(f"  completion: {completion.shape}")
print(f"  debt:       {debt.shape}")
print(f"  debt_share: {debt_share.shape}")
print(f"  unsold:     {unsold.shape}")
print(f"  income:     {income.shape}")
print(f"  naver:      {naver.shape}  (2016+만 존재)")


# ═══════════════════════════════════════════════════════════
# STEP 2: 거시/금융 지표 파생 피처
# ═══════════════════════════════════════════════════════════
print("\n[Step 2] 거시/금융 지표 파생 피처")

macro = ecos.sort_values("ym").reset_index(drop=True).copy()

macro["base_rate_chg_yoy"]     = macro["base_rate"]     - macro["base_rate"].shift(12)
macro["mortgage_rate_chg_yoy"] = macro["mortgage_rate"] - macro["mortgage_rate"].shift(12)
macro["rate_spread"]           = macro["mortgage_rate"] - macro["base_rate"]
macro["m2_yoy_pct"]            = macro["m2_bil_krw"].pct_change(12) * 100
macro["cpi_yoy_pct"]           = macro["cpi"].pct_change(12) * 100
macro["bsi_yoy_chg"]           = macro["bsi_realestate"] - macro["bsi_realestate"].shift(12)

print(f"  완료: {macro.shape[1]}개 컬럼")


# ═══════════════════════════════════════════════════════════
# STEP 3: 공급/레버리지/미분양 파생 피처
# ═══════════════════════════════════════════════════════════
print("\n[Step 3] 공급/레버리지/미분양 파생 피처")

# 인허가·준공: rolling 12개월 합산 YoY% (12월 스파이크 자동 희석)
def rolling12_yoy(series: pd.Series) -> pd.Series:
    roll = series.rolling(12, min_periods=12).sum()
    return roll.pct_change(12) * 100

for region in ["서울", "경기", "인천"]:
    permit[f"{region}_인허가_yoy"]   = rolling12_yoy(permit[f"{region}_인허가"])
    completion[f"{region}_준공_yoy"] = rolling12_yoy(completion[f"{region}_준공"])

# 미분양 YoY%
for region in ["서울", "경기", "인천"]:
    unsold[f"{region}_미분양_yoy"] = unsold[f"{region}_미분양"].pct_change(12) * 100

# 차주당대출 YoY%
debt["서울_차주당대출_yoy"]     = debt["서울_차주당대출_십만원"].pct_change(12) * 100
debt["인천경기_차주당대출_yoy"] = debt["인천경기_차주당대출_십만원"].pct_change(12) * 100

# 가계대출비중 YoY 변화 (pp 차이)
debt_share["수도권_가계대출비중_yoy"] = debt_share["수도권_가계대출비중"] - debt_share["수도권_가계대출비중"].shift(12)
debt_share["서울_가계대출비중_yoy"]   = debt_share["서울_가계대출비중"]   - debt_share["서울_가계대출비중"].shift(12)

print(f"  완료")


# ═══════════════════════════════════════════════════════════
# STEP 4: 시도별 가격 집계 + PIR
# ═══════════════════════════════════════════════════════════
print("\n[Step 4] 시도별 가격 집계 + PIR")

# 4-1. SGG → 시도 집계
sido_price = (
    sgg.groupby(["ym", "시도"])
    .agg(
        매매중위가격 = ("매매중위가격", "median"),
        매매거래건수 = ("매매거래건수", "sum"),
        전세중위가격 = ("전세중위가격", "median"),
        전세거래건수 = ("전세거래건수", "sum"),
        전세가율     = ("전세가율",     "median"),
    )
    .reset_index()
    .sort_values(["시도", "ym"])
    .reset_index(drop=True)
)

# 4-2. 시도별 가격 파생 피처
price_parts = []
for region, grp in sido_price.groupby("시도", sort=False):
    grp = grp.sort_values("ym").copy()
    p   = grp["매매중위가격"]
    mom = p.pct_change(1) * 100
    grp["매매중위_MoM"]  = mom
    grp["매매중위_YoY"]  = p.pct_change(12) * 100
    grp["매매_MA3"]      = p.rolling(3,  min_periods=1).mean()
    grp["매매_MA12"]     = p.rolling(12, min_periods=6).mean()
    grp["매매_vol12"]    = mom.rolling(12).std()
    grp["전세가율_MA3"]  = grp["전세가율"].rolling(3, min_periods=1).mean()
    price_parts.append(grp)

sido_price = pd.concat(price_parts, ignore_index=True)
print(f"  가격집계 완료: {sido_price.shape}")

# 4-3. 가구소득 연간 → 월별 (해당 연도 값 모든 월 적용)
income_map = {
    "서울": dict(zip(income["year"], income["서울_가구소득_만원"])),
    "인천": dict(zip(income["year"], income["인천_가구소득_만원"])),
    "경기": dict(zip(income["year"], income["경기_가구소득_만원"])),
}
all_ym = sorted(sgg["ym"].unique().astype(int))

income_rows = []
for region, yr_map in income_map.items():
    for ym in all_ym:
        income_rows.append({
            "ym": ym,
            "시도": region,
            "가구소득_만원": yr_map.get(ym // 100, np.nan),
        })
income_monthly = pd.DataFrame(income_rows)

# 4-4. PIR 계산
sido_full = pd.merge(sido_price, income_monthly, on=["ym", "시도"], how="left")
sido_full["PIR"]          = (sido_full["매매중위가격"] / sido_full["가구소득_만원"]).round(2)
sido_full["소득대비전세율"] = (sido_full["전세중위가격"] / sido_full["가구소득_만원"]).round(2)

print(f"  PIR 계산 완료: {sido_full.shape}")
print("  서울 PIR 최신 3개월:")
preview = sido_full[sido_full["시도"]=="서울"].tail(3)[["ym","매매중위가격","가구소득_만원","PIR"]]
print(preview.to_string(index=False))


# ═══════════════════════════════════════════════════════════
# STEP 5: 전체 합치기
# ═══════════════════════════════════════════════════════════
print("\n[Step 5] 전체 합치기")

df = sido_full.copy()

# 5-1. 거시지표 (전국 → 모든 시도 동일 적용)
macro_cols = [
    "ym",
    "base_rate", "mortgage_rate", "rate_spread",
    "base_rate_chg_yoy", "mortgage_rate_chg_yoy",
    "m2_bil_krw", "m2_yoy_pct",
    "cpi", "cpi_yoy_pct",
    "bsi_realestate", "bsi_yoy_chg",
]
df = pd.merge(df, macro[macro_cols], on="ym", how="left")

# 5-2. 공급지표 → 시도별 long 변환 후 merge
def to_long(src, region, col_raw, col_yoy, new_raw, new_yoy):
    t = src[["ym", col_raw, col_yoy]].copy()
    t["시도"] = region
    return t.rename(columns={col_raw: new_raw, col_yoy: new_yoy})

permit_long = pd.concat([
    to_long(permit, "서울", "서울_인허가", "서울_인허가_yoy", "인허가", "인허가_yoy"),
    to_long(permit, "경기", "경기_인허가", "경기_인허가_yoy", "인허가", "인허가_yoy"),
    to_long(permit, "인천", "인천_인허가", "인천_인허가_yoy", "인허가", "인허가_yoy"),
], ignore_index=True)

comp_long = pd.concat([
    to_long(completion, "서울", "서울_준공", "서울_준공_yoy", "준공", "준공_yoy"),
    to_long(completion, "경기", "경기_준공", "경기_준공_yoy", "준공", "준공_yoy"),
    to_long(completion, "인천", "인천_준공", "인천_준공_yoy", "준공", "준공_yoy"),
], ignore_index=True)

unsold_long = pd.concat([
    to_long(unsold, "서울", "서울_미분양", "서울_미분양_yoy", "미분양", "미분양_yoy"),
    to_long(unsold, "경기", "경기_미분양", "경기_미분양_yoy", "미분양", "미분양_yoy"),
    to_long(unsold, "인천", "인천_미분양", "인천_미분양_yoy", "미분양", "미분양_yoy"),
], ignore_index=True)

df = pd.merge(df, permit_long,  on=["ym", "시도"], how="left")
df = pd.merge(df, comp_long,    on=["ym", "시도"], how="left")
df = pd.merge(df, unsold_long,  on=["ym", "시도"], how="left")

# 5-3. 차주당대출 (서울 / 인천경기 분리)
debt_long = pd.concat([
    to_long(debt, "서울", "서울_차주당대출_십만원",     "서울_차주당대출_yoy",     "차주당대출_십만원", "차주당대출_yoy"),
    to_long(debt, "경기", "인천경기_차주당대출_십만원", "인천경기_차주당대출_yoy", "차주당대출_십만원", "차주당대출_yoy"),
    to_long(debt, "인천", "인천경기_차주당대출_십만원", "인천경기_차주당대출_yoy", "차주당대출_십만원", "차주당대출_yoy"),
], ignore_index=True)
df = pd.merge(df, debt_long, on=["ym", "시도"], how="left")

# 5-4. 가계대출비중 (서울 개별 / 경기·인천은 인천경기)
share_seoul = debt_share[["ym", "서울_가계대출비중", "서울_가계대출비중_yoy"]].copy()
share_seoul["시도"] = "서울"
share_seoul = share_seoul.rename(columns={"서울_가계대출비중": "가계대출비중", "서울_가계대출비중_yoy": "가계대출비중_yoy"})

share_gyeonggi = debt_share[["ym", "인천경기_가계대출비중"]].copy()
share_gyeonggi["시도"] = "경기"
share_gyeonggi["가계대출비중_yoy"] = debt_share["인천경기_가계대출비중"] - debt_share["인천경기_가계대출비중"].shift(12)
share_gyeonggi = share_gyeonggi.rename(columns={"인천경기_가계대출비중": "가계대출비중"})

share_incheon = debt_share[["ym", "인천경기_가계대출비중"]].copy()
share_incheon["시도"] = "인천"
share_incheon["가계대출비중_yoy"] = debt_share["인천경기_가계대출비중"] - debt_share["인천경기_가계대출비중"].shift(12)
share_incheon = share_incheon.rename(columns={"인천경기_가계대출비중": "가계대출비중"})

share_long = pd.concat([share_seoul, share_gyeonggi, share_incheon], ignore_index=True)
df = pd.merge(df, share_long, on=["ym", "시도"], how="left")

# 5-5. 네이버 검색트렌드 (2013~2015 = 0, 플래그 추가)
all_ym_df = pd.DataFrame({"ym": sorted(df["ym"].unique())})
naver_full = pd.merge(all_ym_df, naver, on="ym", how="left")
naver_full["아파트매매_검색"]  = naver_full["아파트매매_검색"].fillna(0)
naver_full["부동산버블_검색"]  = naver_full["부동산버블_검색"].fillna(0)
naver_full["집값급등_검색"]    = naver_full["집값급등_검색"].fillna(0)
naver_full["검색트렌드_유효"]  = (naver_full["ym"] >= 201601).astype(int)

df = pd.merge(df, naver_full, on="ym", how="left")

# 5-6. 버블 Pseudo-label — 시도별 차등 기준 (sgg와 동일 기준 적용)
# ┌──────┬──────────────────────────────┬──────────────────────────────┐
# │ 시도 │ 버블(2)                       │ 과열(1)                       │
# ├──────┼──────────────────────────────┼──────────────────────────────┤
# │ 서울 │ 전세가율 < 55 AND YoY > 12   │ 전세가율 < 63 AND YoY >  7   │
# │ 경기 │ 전세가율 < 65 AND YoY > 15   │ 전세가율 < 72 AND YoY > 10   │
# │ 인천 │ 전세가율 < 68 AND YoY > 15   │ 전세가율 < 75 AND YoY > 10   │
# └──────┴──────────────────────────────┴──────────────────────────────┘
CRITERIA = {
    "서울": {"bubble": (55, 12), "overheat": (63,  7)},
    "경기": {"bubble": (65, 15), "overheat": (72, 10)},
    "인천": {"bubble": (68, 15), "overheat": (75, 10)},
}
label_arr = np.zeros(len(df), dtype=float)
for sido_name, crit in CRITERIA.items():
    mask = (df["시도"] == sido_name).values
    jr   = df.loc[mask, "전세가율"].values
    yoy  = df.loc[mask, "매매중위_YoY"].values
    b_j, b_y = crit["bubble"]
    o_j, o_y = crit["overheat"]
    lbl = np.zeros(mask.sum(), dtype=float)
    lbl[(jr < o_j) & (yoy > o_y)] = 1   # 과열
    lbl[(jr < b_j) & (yoy > b_y)] = 2   # 버블 (과열 위 덮어씀)
    label_arr[mask] = lbl
df["bubble_label"] = label_arr
df.loc[df["매매중위_YoY"].isna(), "bubble_label"] = np.nan

# 5-7. 컬럼 순서 정리
col_order = [
    # 키
    "ym", "시도",
    # 가격
    "매매중위가격", "매매거래건수", "전세중위가격", "전세거래건수",
    "전세가율", "전세가율_MA3",
    "매매중위_MoM", "매매중위_YoY", "매매_MA3", "매매_MA12", "매매_vol12",
    # 소득/PIR
    "가구소득_만원", "PIR", "소득대비전세율",
    # 거시
    "base_rate", "mortgage_rate", "rate_spread",
    "base_rate_chg_yoy", "mortgage_rate_chg_yoy",
    "m2_bil_krw", "m2_yoy_pct",
    "cpi", "cpi_yoy_pct",
    "bsi_realestate", "bsi_yoy_chg",
    # 공급
    "인허가", "인허가_yoy", "준공", "준공_yoy", "미분양", "미분양_yoy",
    # 레버리지
    "차주당대출_십만원", "차주당대출_yoy",
    "가계대출비중", "가계대출비중_yoy",
    # 검색트렌드
    "아파트매매_검색", "부동산버블_검색", "집값급등_검색", "검색트렌드_유효",
    # 레이블
    "bubble_label",
]
col_order = [c for c in col_order if c in df.columns]
df = df[col_order].sort_values(["시도", "ym"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# 저장
# ═══════════════════════════════════════════════════════════
out_path = "data/processed/features_sido_monthly.csv"
df.to_csv(out_path, index=False, encoding=ENC)

print(f"\n✅ 저장: {out_path}")
print(f"   shape: {df.shape}")
print(f"   시도: {sorted(df['시도'].unique())}  기간: {df['ym'].min()}~{df['ym'].max()}")
print(f"   컬럼 {len(df.columns)}개:\n   {col_order}")

print("\n[결측치 현황]")
nulls = df.isnull().sum()
null_info = nulls[nulls > 0]
print(null_info.to_string() if len(null_info) else "  없음")

print("\n[bubble_label 분포]")
lbl = df["bubble_label"].value_counts(dropna=False).sort_index()
for k, v in lbl.items():
    name = {0.0: "정상", 1.0: "과열", 2.0: "버블", np.nan: "NaN(초기)"}.get(k, str(k))
    print(f"  {name}: {v}행 ({v/len(df)*100:.1f}%)")

print("\n[서울 최신 6개월 미리보기]")
preview_cols = ["ym", "매매중위가격", "PIR", "전세가율", "base_rate",
                "미분양", "가계대출비중", "아파트매매_검색", "bubble_label"]
print(df[df["시도"]=="서울"].tail(6)[preview_cols].to_string(index=False))

print("\n" + "=" * 60)
print("  Feature Engineering Steps 2~5 완료")
print("=" * 60)
