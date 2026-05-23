import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np

COL_MAP = {
    "시군구": "sigungu_raw", "단지명": "apt_name",
    "전용면적(㎡)": "area_m2", "전용면적": "area_m2",
    "거래금액(만원)": "price_raw", "거래금액": "price_raw",
    "계약년월": "ym", "계약일": "day", "해제사유발생일": "cancel_date",
}
KO_COLUMNS = {
    "deal_date": "계약일", "deal_ym": "계약년월", "sido": "시도",
    "gu": "구", "dong": "동", "apt_name": "단지명",
    "area_m2": "전용면적(㎡)", "price_10k": "거래금액(만원)", "price_per_m2": "㎡당단가(만원)",
}
REGION_CONFIG = {
    "seoul":    {"sido": "서울", "gu_idx": 1, "dong_idx": 2},
    "incheon":  {"sido": "인천", "gu_idx": 1, "dong_idx": 2},
    "gyeonggi": {"sido": "경기", "gu_idx": 2, "dong_idx": 3},
}


def preprocess(csv_path, region):
    cfg = REGION_CONFIG[region]
    df = pd.read_csv(csv_path, skiprows=15, encoding="cp949", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
    if "cancel_date" in df.columns:
        df = df[df["cancel_date"].str.strip() == "-"]
    df["price_10k"] = pd.to_numeric(
        df["price_raw"].str.replace(",", "").str.strip(), errors="coerce"
    )
    df = df[df["price_10k"].notna() & (df["price_10k"] > 0)]
    df["deal_ym"]   = df["ym"].str.strip()
    df["deal_date"] = pd.to_datetime(
        df["ym"].str.strip() + df["day"].str.strip().str.zfill(2),
        format="%Y%m%d", errors="coerce"
    )
    df = df[df["deal_date"].notna()]
    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df["area_m2"] = df["area_m2"].where(df["area_m2"] > 0, np.nan)
    df = df[df["area_m2"].notna()]
    split_addr  = df["sigungu_raw"].str.strip().str.split()
    df["sido"]  = cfg["sido"]
    df["gu"]    = split_addr.str.get(cfg["gu_idx"]).str.strip()
    df["dong"]  = split_addr.str.get(cfg["dong_idx"]).str.strip()
    df = df[df["gu"].notna()]
    df["price_10k"]    = df["price_10k"].astype(int)
    df["price_per_m2"] = (df["price_10k"] / df["area_m2"]).round(2)
    result = df[["deal_date","deal_ym","sido","gu","dong","apt_name",
                 "area_m2","price_10k","price_per_m2"]].reset_index(drop=True)
    print(f"  {cfg['sido']}: {len(result):,}건")
    return result


os.chdir("c:/Users/bko05/Desktop/seoul-bubble-detection")

print("2013년 수도권 전처리 시작...")
frames = []
for region in ["seoul", "gyeonggi", "incheon"]:
    path = f"data/raw/molit/apt_trade_2013_{region}.csv"
    frames.append(preprocess(path, region))

df_2013 = pd.concat(frames, ignore_index=True)
sido_counts = df_2013["sido"].value_counts()

print()
print(f"서울: {sido_counts.get('서울', 0):,}건")
print(f"경기: {sido_counts.get('경기', 0):,}건")
print(f"인천: {sido_counts.get('인천', 0):,}건")
print(f"합계: {len(df_2013):,}건")
print(f"중위가격: {int(df_2013['price_10k'].median()):,}만원")
print(f"중위단가: {df_2013['price_per_m2'].median():.1f}만원/m2")

df_2013.rename(columns=KO_COLUMNS).to_csv(
    "data/processed/apt_trade_2013_sudogwon_clean.csv",
    index=False, encoding="utf-8-sig"
)
print()
print("저장 완료: apt_trade_2013_sudogwon_clean.csv")
