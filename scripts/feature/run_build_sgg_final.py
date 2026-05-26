"""
features_sgg_final.csv 생성 스크립트
─────────────────────────────────────
입력:
  data/processed/features_sgg_monthly.csv   (10,140행 × 8컬럼  — 구별 기본집계)
  data/processed/features_sido_monthly.csv  (  468행 × 42컬럼 — 시도별 거시/공급 피처)

출력:
  data/processed/features_sgg_final.csv     (10,140행 × ~30컬럼)

컬럼 구성:
  그룹1. 매매 파생지표   — MoM/YoY/MA3/MA12/vol12/zscore_24m/거래량_ratio
  그룹2. 전세 파생지표   — 전세가율_cap/is_kangtong/전세가율_MA3/전세거래량_ratio
  그룹3. 거시지표 조인   — (ym+시도) 기준으로 features_sido_monthly에서 병합
  그룹4. 버블 레이블     — 시도별 차등 기준 (정상0/과열1/버블2)
"""
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

ENC = 'utf-8-sig'

print("=" * 60)
print("  features_sgg_final.csv 생성")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════
print("\n[1] 데이터 로드")

sgg  = pd.read_csv("data/processed/features_sgg_monthly.csv",  encoding=ENC)
sido = pd.read_csv("data/processed/features_sido_monthly.csv", encoding=ENC)

sgg["ym"]  = sgg["ym"].astype(int)
sido["ym"] = sido["ym"].astype(int)

print(f"  sgg  : {sgg.shape}  |  구 수={sgg['구'].nunique()}  기간={sgg['ym'].min()}~{sgg['ym'].max()}")
print(f"  sido : {sido.shape}  |  컬럼={sido.shape[1]}")


# ══════════════════════════════════════════════════════════════
# 2. 구별 파생 피처 계산
# ══════════════════════════════════════════════════════════════
print("\n[2] 구별 파생 피처 계산 ...")

parts = []
for (sido_name, gu), grp in sgg.groupby(["시도", "구"], sort=False):
    grp = grp.sort_values("ym").copy().reset_index(drop=True)

    p   = grp["매매중위가격"]
    mom = p.pct_change(1) * 100

    # ── 그룹1. 매매 파생
    grp["매매중위_MoM"]     = mom.round(2)
    grp["매매중위_YoY"]     = (p.pct_change(12) * 100).round(2)
    grp["매매_MA3"]         = p.rolling(3,  min_periods=1).mean().round(0)
    grp["매매_MA12"]        = p.rolling(12, min_periods=6).mean().round(0)
    grp["매매_vol12"]       = mom.rolling(12, min_periods=6).std().round(3)
    grp["price_zscore_24m"] = (
        (p - p.rolling(24, min_periods=12).mean()) /
         p.rolling(24, min_periods=12).std()
    ).round(3)

    # 거래량 / 12개월 rolling 평균
    tr_roll = grp["매매거래건수"].rolling(12, min_periods=6).mean().replace(0, np.nan)
    grp["거래량_ratio"] = (grp["매매거래건수"] / tr_roll).round(3)

    # ── 그룹2. 전세 파생
    grp["전세가율_cap"]    = grp["전세가율"].clip(upper=100).round(2)
    grp["is_kangtong"]     = (grp["전세가율"] > 100).astype(int)
    grp["전세가율_MA3"]    = grp["전세가율"].rolling(3, min_periods=1).mean().round(2)

    jt_roll = grp["전세거래건수"].rolling(12, min_periods=6).mean().replace(0, np.nan)
    grp["전세거래량_ratio"] = (grp["전세거래건수"] / jt_roll).round(3)

    parts.append(grp)

df = pd.concat(parts, ignore_index=True)
print(f"  완료: {df.shape}")


# ══════════════════════════════════════════════════════════════
# 3. 거시지표 조인 (시도 단위 → 구 단위 근사치)
# ══════════════════════════════════════════════════════════════
print("\n[3] 거시지표 조인 (ym + 시도 기준)")

MACRO_COLS = [
    "ym", "시도",
    # 금리
    "base_rate", "mortgage_rate", "rate_spread", "mortgage_rate_chg_yoy",
    # 유동성/물가
    "m2_yoy_pct", "cpi_yoy_pct",
    # 심리
    "bsi_realestate",
    # 소득/PIR
    "PIR", "소득대비전세율",
    # 공급/레버리지 (시도 단위 근사)
    "미분양_yoy", "인허가_yoy", "가계대출비중_yoy",
]
macro_sel = [c for c in MACRO_COLS if c in sido.columns]
macro     = sido[macro_sel].copy()

before = len(df)
df = pd.merge(df, macro, on=["ym", "시도"], how="left")
print(f"  조인 전 {before}행 → 후 {len(df)}행  |  추가 컬럼={len(df.columns)-len(parts[0].columns)}")


# ══════════════════════════════════════════════════════════════
# 4. 버블 Pseudo-label (시도별 차등 기준)
# ══════════════════════════════════════════════════════════════
print("\n[4] 버블 Pseudo-label 생성 (시도별 차등 기준)")

# 기준값: (전세가율_임계, YoY_임계)
CRITERIA = {
    # (버블 전세가율, 버블 YoY, 과열 전세가율, 과열 YoY)
    "서울": (55, 12, 63,  7),
    "경기": (65, 15, 72, 10),
    "인천": (68, 15, 75, 10),
}

label_arr = np.zeros(len(df), dtype=float)

for sido_name, (b_j, b_y, o_j, o_y) in CRITERIA.items():
    mask = (df["시도"] == sido_name).values
    jr   = df.loc[mask, "전세가율"].values
    yoy  = df.loc[mask, "매매중위_YoY"].values

    lbl = np.zeros(mask.sum(), dtype=float)
    lbl[(jr < o_j) & (yoy > o_y)] = 1   # 과열
    lbl[(jr < b_j) & (yoy > b_y)] = 2   # 버블 (과열 위에 덮어씀)
    label_arr[mask] = lbl

df["bubble_label"] = label_arr
# YoY 결측 구간(초기 12개월) → NaN
df.loc[df["매매중위_YoY"].isna(), "bubble_label"] = np.nan

# 분포 출력
lbl_dist = df["bubble_label"].value_counts(dropna=False).sort_index()
name_map  = {0.0: "정상", 1.0: "과열", 2.0: "버블", np.nan: "NaN(초기)"}
for k, v in lbl_dist.items():
    ratio = v / len(df) * 100
    print(f"  {name_map.get(k, str(k)):<10}: {v:5}행  ({ratio:.1f}%)")


# ══════════════════════════════════════════════════════════════
# 5. 컬럼 정렬 & 저장
# ══════════════════════════════════════════════════════════════
print("\n[5] 컬럼 정렬 & 저장")

COL_ORDER = [
    # 키
    "ym", "시도", "구",
    # 매매 가격
    "매매중위가격", "매매중위_MoM", "매매중위_YoY",
    "매매_MA3", "매매_MA12", "매매_vol12", "price_zscore_24m",
    "매매거래건수", "거래량_ratio",
    # 전세 · 수급
    "전세중위가격", "전세가율", "전세가율_cap", "is_kangtong",
    "전세가율_MA3", "전세거래건수", "전세거래량_ratio",
    # 거시 (시도 근사)
    "base_rate", "mortgage_rate", "rate_spread", "mortgage_rate_chg_yoy",
    "m2_yoy_pct", "cpi_yoy_pct",
    "bsi_realestate",
    "PIR", "소득대비전세율",
    # 공급 · 레버리지 (시도 근사)
    "미분양_yoy", "인허가_yoy", "가계대출비중_yoy",
    # 레이블
    "bubble_label",
]
col_order = [c for c in COL_ORDER if c in df.columns]
df = df[col_order].sort_values(["시도", "구", "ym"]).reset_index(drop=True)

out_path = "data/processed/features_sgg_final.csv"
df.to_csv(out_path, index=False, encoding=ENC)

print(f"\n✅ 저장 완료: {out_path}")
print(f"   shape : {df.shape}")
print(f"   기간  : {df['ym'].min()} ~ {df['ym'].max()}")
print(f"   구 수 : {df['구'].nunique()}개")
print(f"   컬럼  : {len(df.columns)}개")
print(f"   {col_order}")

# ── 결측치 현황
print("\n[결측치 현황]")
nulls     = df.isnull().sum()
null_info = nulls[nulls > 0]
if len(null_info):
    for col, cnt in null_info.items():
        print(f"  {col:<30}: {cnt}행  ({cnt/len(df)*100:.1f}%)")
else:
    print("  없음")

# ── 미리보기: 서울 강남구 최신 6개월
print("\n[서울 강남구 최신 6개월 미리보기]")
preview_cols = [
    "ym", "매매중위가격", "매매중위_YoY", "price_zscore_24m",
    "전세가율_cap", "is_kangtong", "base_rate", "PIR", "bubble_label"
]
preview = df[(df["시도"] == "서울") & (df["구"] == "강남구")].tail(6)
print(preview[[c for c in preview_cols if c in preview.columns]].to_string(index=False))

# ── 시도별 버블 분포 요약
print("\n[시도별 버블 분포]")
for sido_name in ["서울", "경기", "인천"]:
    sub = df[df["시도"] == sido_name]
    b   = int((sub["bubble_label"] == 2).sum())
    o   = int((sub["bubble_label"] == 1).sum())
    n   = int((sub["bubble_label"] == 0).sum())
    total = len(sub)
    print(f"  {sido_name}: 버블={b}({b/total*100:.1f}%)  과열={o}({o/total*100:.1f}%)  정상={n}({n/total*100:.1f}%)")

print("\n" + "=" * 60)
print("  완료")
print("=" * 60)
