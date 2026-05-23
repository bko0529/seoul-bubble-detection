import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np

os.chdir("c:/Users/bko05/Desktop/seoul-bubble-detection")

# ── 설정 ──────────────────────────────────────────────
REGION_CONFIG = {
    "seoul":    {"sido": "서울", "gu_idx": 1, "dong_idx": 2},
    "incheon":  {"sido": "인천", "gu_idx": 1, "dong_idx": 2},
    "gyeonggi": {"sido": "경기", "gu_idx": 2, "dong_idx": 3},
}

KO_COLUMNS = {
    "deal_date":        "계약일",
    "deal_ym":          "계약년월",
    "sido":             "시도",
    "gu":               "구",
    "dong":             "동",
    "apt_name":         "단지명",
    "area_m2":          "전용면적(㎡)",
    "contract_type":    "전월세구분",
    "deposit_10k":      "보증금(만원)",
    "monthly_rent_10k": "월세금(만원)",
}


def preprocess_jeonse(csv_path, region):
    cfg = REGION_CONFIG[region]

    df = pd.read_csv(csv_path, skiprows=15, encoding="cp949", dtype=str)
    df.columns = df.columns.str.strip()

    # 아파트만 필터 (주택유형 컬럼 존재 시)
    if "주택유형" in df.columns:
        df = df[df["주택유형"].str.strip() == "아파트"]

    # 전월세구분
    df["contract_type"] = df["전월세구분"].str.strip()

    # 보증금 파싱
    df["deposit_10k"] = pd.to_numeric(
        df["보증금(만원)"].str.replace(",", "").str.strip(), errors="coerce"
    )
    df = df[df["deposit_10k"].notna() & (df["deposit_10k"] > 0)]

    # 월세금 파싱 (전세는 0)
    df["monthly_rent_10k"] = pd.to_numeric(
        df["월세금(만원)"].str.replace(",", "").str.strip(), errors="coerce"
    ).fillna(0)

    # 날짜 파싱
    df["deal_ym"]   = df["계약년월"].str.strip()
    df["deal_date"] = pd.to_datetime(
        df["계약년월"].str.strip() + df["계약일"].str.strip().str.zfill(2),
        format="%Y%m%d", errors="coerce"
    )
    df = df[df["deal_date"].notna()]

    # 면적 파싱
    df["area_m2"] = pd.to_numeric(df["전용면적(㎡)"], errors="coerce")
    df["area_m2"] = df["area_m2"].where(df["area_m2"] > 0, np.nan)
    df = df[df["area_m2"].notna()]

    # 주소 파싱
    split_addr  = df["시군구"].str.strip().str.split()
    df["sido"]  = cfg["sido"]
    df["gu"]    = split_addr.str.get(cfg["gu_idx"]).str.strip()
    df["dong"]  = split_addr.str.get(cfg["dong_idx"]).str.strip()
    df = df[df["gu"].notna()]

    # 단지명
    df["apt_name"] = df["단지명"].str.strip()

    # 타입 정리
    df["deposit_10k"]      = df["deposit_10k"].astype(int)
    df["monthly_rent_10k"] = df["monthly_rent_10k"].astype(int)

    result = df[[
        "deal_date", "deal_ym", "sido", "gu", "dong", "apt_name",
        "area_m2", "contract_type", "deposit_10k", "monthly_rent_10k"
    ]].reset_index(drop=True)

    jeonse_cnt = (result["contract_type"] == "전세").sum()
    wolse_cnt  = (result["contract_type"] == "월세").sum()
    print(f"    {cfg['sido']:3s}: {len(result):>7,}건  (전세 {jeonse_cnt:,} / 월세 {wolse_cnt:,})")
    return result


# ── 실행 ──────────────────────────────────────────────
summary_rows = []
total_all    = 0

for year in range(2013, 2026):
    print(f"\n[{year}년 전월세 전처리]")
    frames = []
    ok = True

    for region in ["seoul", "gyeonggi", "incheon"]:
        path = f"data/raw/molit/apt_jeonse_{year}_{region}.csv"
        if not os.path.exists(path):
            print(f"  ⚠️  {path} 없음 → 스킵")
            ok = False
            continue
        frames.append(preprocess_jeonse(path, region))

    if not frames:
        continue

    df_year = pd.concat(frames, ignore_index=True)

    jeonse_n = (df_year["contract_type"] == "전세").sum()
    wolse_n  = (df_year["contract_type"] == "월세").sum()
    total_all += len(df_year)

    # 저장 (한글 컬럼명)
    out_path = f"data/processed/apt_jeonse_{year}_sudogwon_clean.csv"
    df_year.rename(columns=KO_COLUMNS).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )

    sido_cnt = df_year["sido"].value_counts()
    summary_rows.append({
        "연도":   year,
        "합계":   len(df_year),
        "전세":   jeonse_n,
        "월세":   wolse_n,
        "서울":   sido_cnt.get("서울", 0),
        "경기":   sido_cnt.get("경기", 0),
        "인천":   sido_cnt.get("인천", 0),
        "중위보증금(만원)": int(df_year["deposit_10k"].median()),
    })
    print(f"  ✅ 저장 완료: {out_path}  ({len(df_year):,}건)")

# ── 전체 요약 ─────────────────────────────────────────
print()
print("=" * 75)
print("  수도권 전월세 전처리 전체 요약 (2013~2025)")
print("=" * 75)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
print("=" * 75)
print(f"  총 합계: {total_all:,}건")
print("=" * 75)
