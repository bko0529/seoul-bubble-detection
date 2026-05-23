"""
수도권 매매 데이터 전처리 스크립트
서울(seoul) + 경기(gyeonggi) + 인천(incheon) 2013~2025
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

COL_MAP = {
    "시군구":        "sigungu_raw",
    "단지명":        "apt_name",
    "전용면적(㎡)":  "area_m2",
    "전용면적":      "area_m2",
    "거래금액(만원)": "price_raw",
    "거래금액":      "price_raw",
    "계약년월":      "ym",
    "계약일":        "day",
    "해제사유발생일": "cancel_date",
}

KO_COLUMNS = {
    "deal_date":    "계약일",
    "deal_ym":      "계약년월",
    "sido":         "시도",
    "gu":           "구",
    "dong":         "동",
    "apt_name":     "단지명",
    "area_m2":      "전용면적(㎡)",
    "price_10k":    "거래금액(만원)",
    "price_per_m2": "㎡당단가(만원)",
}

# 시도별 주소 파싱 인덱스
# 서울/인천: "서울특별시 강남구 대치동"     → gu=str[1], dong=str[2]
# 경기:      "경기도 수원시 권선구 호매실동" → gu=str[2], dong=str[3]
REGION_CONFIG = {
    "seoul":    {"sido": "서울",  "gu_idx": 1, "dong_idx": 2},
    "incheon":  {"sido": "인천",  "gu_idx": 1, "dong_idx": 2},
    "gyeonggi": {"sido": "경기",  "gu_idx": 2, "dong_idx": 3},
}


def preprocess(csv_path: str, region: str) -> pd.DataFrame:
    cfg   = REGION_CONFIG[region]
    fname = os.path.basename(csv_path)

    df = pd.read_csv(csv_path, skiprows=15, encoding="cp949", dtype=str)
    df.columns = df.columns.str.strip()
    total = len(df)

    print(f"\n{'─'*52}")
    print(f"📂 {fname}  [{cfg['sido']}]")
    print(f"{'─'*52}")
    print(f"원본 행 수       : {total:>8,}건")

    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

    required = {"sigungu_raw", "apt_name", "area_m2", "price_raw", "ym", "day"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # ① 계약취소 제거
    before = len(df)
    if "cancel_date" in df.columns:
        df = df[df["cancel_date"].str.strip() == "-"]
    print(f"① 계약취소 제거  : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ② 거래금액 파싱
    before = len(df)
    df["price_10k"] = pd.to_numeric(
        df["price_raw"].str.replace(",", "").str.strip(), errors="coerce"
    )
    df = df[df["price_10k"].notna() & (df["price_10k"] > 0)]
    print(f"② 금액 파싱 실패 : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ③ 날짜 파싱
    before = len(df)
    df["deal_ym"]   = df["ym"].str.strip()
    df["deal_date"] = pd.to_datetime(
        df["ym"].str.strip() + df["day"].str.strip().str.zfill(2),
        format="%Y%m%d", errors="coerce"
    )
    df = df[df["deal_date"].notna()]
    print(f"③ 날짜 파싱 실패 : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ④ 면적 이상값
    before = len(df)
    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df["area_m2"] = df["area_m2"].where(df["area_m2"] > 0, np.nan)
    df = df[df["area_m2"].notna()]
    print(f"④ 면적 이상값    : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ⑤ 주소 파싱 (시도별 인덱스 다름)
    before = len(df)
    split_addr  = df["sigungu_raw"].str.strip().str.split()
    df["sido"]  = cfg["sido"]
    df["gu"]    = split_addr.str.get(cfg["gu_idx"]).str.strip()
    df["dong"]  = split_addr.str.get(cfg["dong_idx"]).str.strip()
    df = df[df["gu"].notna()]
    print(f"⑤ 주소 파싱 실패 : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    df["price_10k"]    = df["price_10k"].astype(int)
    df["price_per_m2"] = (df["price_10k"] / df["area_m2"]).round(2)

    remove_rate = (total - len(df)) / total * 100
    print(f"{'─'*52}")
    print(f"✅ 최종: {len(df):,}건  (제거율 {remove_rate:.2f}%)")
    print(f"{'─'*52}")

    return df[["deal_date","deal_ym","sido","gu","dong","apt_name",
               "area_m2","price_10k","price_per_m2"]].reset_index(drop=True)


if __name__ == "__main__":
    os.chdir("c:/Users/bko05/Desktop/seoul-bubble-detection")
    os.makedirs("data/processed", exist_ok=True)

    REGIONS = ["seoul", "gyeonggi", "incheon"]
    total_all    = 0
    summary_rows = []

    for year in range(2013, 2026):
        year_frames = []

        for region in REGIONS:
            csv_path = f"data/raw/molit/apt_trade_{year}_{region}.csv"
            if not os.path.exists(csv_path):
                print(f"⚠️  apt_trade_{year}_{region}.csv 없음 → 스킵")
                continue
            df = preprocess(csv_path, region)
            year_frames.append(df)

        if not year_frames:
            continue

        # 연도별 수도권 합산 저장
        df_year = pd.concat(year_frames, ignore_index=True)
        df_year.rename(columns=KO_COLUMNS).to_csv(
            f"data/processed/apt_trade_{year}_sudogwon_clean.csv",
            index=False, encoding="utf-8-sig"
        )

        total_all += len(df_year)
        sido_counts = df_year["sido"].value_counts().to_dict()
        summary_rows.append({
            "year":          year,
            "total":         len(df_year),
            "seoul":         sido_counts.get("서울", 0),
            "gyeonggi":      sido_counts.get("경기", 0),
            "incheon":       sido_counts.get("인천", 0),
            "median_price":  int(df_year["price_10k"].median()),
            "median_per_m2": round(df_year["price_per_m2"].median(), 1),
        })

    print(f"\n{'='*60}")
    print(f"수도권 전체 완료: {total_all:,}건 (2013~2025)")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.columns = ["연도","합계","서울","경기","인천","중위가격(만원)","중위단가(만원/m2)"]
    print(summary_df.to_string(index=False))
    print(f"{'='*60}")
