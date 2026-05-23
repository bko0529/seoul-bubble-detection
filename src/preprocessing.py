"""
전처리 모듈: 국토부 CSV → PostgreSQL 적재

[사용 컬럼]
시군구, 단지명, 전용면적(㎡), 거래금액(만원), 계약년월, 계약일, 해제사유발생일
→ 층·건축년도·본번·부번·도로명 등은 버블 탐지에 불필요하여 제외

[전처리 단계]
① 계약취소 제거  - 해제사유발생일 != "-" 행 제거
② 거래금액 파싱  - 쉼표 제거 + 숫자 변환 (실패 행 제거)
③ 날짜 파싱      - 계약년월+계약일 합쳐서 DATE 변환 (실패 행 제거)
④ 면적 이상값    - 0㎡ 이하 제거 (price_per_m2 inf 방지)
⑤ 주소 파싱      - 시군구 → gu(구) / dong(동) 분리 (실패 행 제거)
"""
import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",   "localhost"),
    "dbname":   os.getenv("DB_NAME",   "bubble_detection"),
    "user":     os.getenv("DB_USER",   "postgres"),
    "password": os.getenv("DB_PASS",   ""),
}

# 필요한 컬럼만 매핑 (구형/신형 컬럼명 모두 대응)
COL_MAP = {
    "시군구":         "sigungu_raw",
    "단지명":         "apt_name",
    "전용면적(㎡)":   "area_m2",
    "전용면적":       "area_m2",       # 구형 fallback
    "거래금액(만원)":  "price_raw",
    "거래금액":       "price_raw",      # 구형 fallback
    "계약년월":       "ym",
    "계약일":         "day",
    "해제사유발생일":  "cancel_date",
}


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# ──────────────────────────────────────────────
# 국토부 아파트 매매 CSV 전처리
# ──────────────────────────────────────────────
def load_molit_trade(csv_path: str) -> pd.DataFrame:
    """
    국토부 매매 CSV → 정제된 DataFrame

    Returns:
        columns: deal_date, deal_ym, gu, dong, apt_name,
                 area_m2, price_10k, price_per_m2
    """
    fname = os.path.basename(csv_path)
    df = pd.read_csv(csv_path, skiprows=15, encoding="cp949", dtype=str)
    df.columns = df.columns.str.strip()
    total = len(df)

    print(f"\n{'─'*48}")
    print(f"📂 {fname}")
    print(f"{'─'*48}")
    print(f"원본 행 수       : {total:>8,}건")

    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

    # 필수 컬럼 체크
    required = {"sigungu_raw", "apt_name", "area_m2", "price_raw", "ym", "day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{fname}] 필수 컬럼 누락: {missing}\n실제 컬럼: {df.columns.tolist()}")

    # ── ① 계약취소 제거 ──────────────────────────────────────────
    before = len(df)
    if "cancel_date" in df.columns:
        df = df[df["cancel_date"].str.strip() == "-"]
    print(f"① 계약취소 제거  : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ── ② 거래금액 파싱 ──────────────────────────────────────────
    before = len(df)
    df["price_10k"] = pd.to_numeric(
        df["price_raw"].str.replace(",", "").str.strip(),
        errors="coerce"
    )
    df = df[df["price_10k"].notna() & (df["price_10k"] > 0)]
    print(f"② 금액 파싱 실패 : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ── ③ 날짜 파싱 ──────────────────────────────────────────────
    before = len(df)
    df["deal_ym"]   = df["ym"].str.strip()
    df["deal_date"] = pd.to_datetime(
        df["ym"].str.strip() + df["day"].str.strip().str.zfill(2),
        format="%Y%m%d",
        errors="coerce"
    )
    df = df[df["deal_date"].notna()]
    print(f"③ 날짜 파싱 실패 : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ── ④ 면적 이상값 처리 ───────────────────────────────────────
    before = len(df)
    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df["area_m2"] = df["area_m2"].where(df["area_m2"] > 0, np.nan)
    df = df[df["area_m2"].notna()]
    print(f"④ 면적 이상값    : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ── ⑤ 주소 파싱 ──────────────────────────────────────────────
    before = len(df)
    split_addr = df["sigungu_raw"].str.strip().str.split()
    df["gu"]   = split_addr.str.get(1).str.strip()   # 강남구
    df["dong"] = split_addr.str.get(2).str.strip()   # 대치동
    df = df[df["gu"].notna()]
    print(f"⑤ 주소 파싱 실패 : {before - len(df):>8,}건 제거 → {len(df):,}건 남음")

    # ── m²당 단가 계산 ────────────────────────────────────────────
    df["price_10k"]    = df["price_10k"].astype(int)
    df["price_per_m2"] = (df["price_10k"] / df["area_m2"]).round(2)

    # ── 품질 요약 ─────────────────────────────────────────────────
    remove_rate = (total - len(df)) / total * 100
    print(f"{'─'*48}")
    print(f"✅ 최종: {len(df):,}건  (제거율 {remove_rate:.2f}%)")
    print(f"price_per_m2 범위: {df['price_per_m2'].min():.1f} ~ {df['price_per_m2'].max():.1f} 만원/m²")
    print(f"{'─'*48}")

    return df[[
        "deal_date", "deal_ym",
        "gu", "dong", "apt_name",
        "area_m2", "price_10k", "price_per_m2",
    ]].reset_index(drop=True)


def insert_trade(df: pd.DataFrame):
    """apt_trade 테이블에 upsert"""
    rows = [
        (
            row.deal_date,
            row.deal_ym,
            row.gu,
            row.dong     if pd.notna(row.dong)     else None,
            row.apt_name if pd.notna(row.apt_name) else None,
            float(row.area_m2),
            int(row.price_10k),
            float(row.price_per_m2),
        )
        for row in df.itertuples()
    ]
    sql = """
        INSERT INTO apt_trade
            (deal_date, deal_ym, gu, dong, apt_name,
             area_m2, price_10k, price_per_m2)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
    print(f"✅ apt_trade {len(rows):,}건 DB 적재 완료")


# ──────────────────────────────────────────────
# 한국은행 엑셀 전처리
# ──────────────────────────────────────────────
def load_bok_excel(excel_dir: str) -> pd.DataFrame:
    """BOK 지표 파일들을 월별 하나의 DataFrame으로 병합"""
    files = {
        "base_rate": "base_rate.xlsx",
        "mortgage_rate": "mortgage_rate.xlsx",
        "household_debt": "household_debt.xlsx",
        "m2": "m2.xlsx",
        "bsi_realestate": "bsi_realestate.xlsx",
    }
    frames = {}
    for col, fname in files.items():
        path = os.path.join(excel_dir, fname)
        if not os.path.exists(path):
            print(f"⚠️  {fname} 없음 → 스킵")
            continue
        df = pd.read_excel(path, skiprows=2)
        df.columns = ["ref_date", col]
        df["ref_date"] = pd.to_datetime(df["ref_date"], errors="coerce")
        df = df.dropna(subset=["ref_date"])
        frames[col] = df.set_index("ref_date")[col]

    result = pd.DataFrame(frames)
    result.index.name = "ref_date"
    result = result.reset_index()
    result["ref_date"] = result["ref_date"].dt.to_period("M").dt.to_timestamp()
    return result


def insert_macro(df: pd.DataFrame):
    """macro_indicators 테이블에 upsert"""
    sql = """
        INSERT INTO macro_indicators
            (ref_date, base_rate, mortgage_rate, household_debt, m2, bsi_realestate)
        VALUES %s
        ON CONFLICT (ref_date) DO UPDATE SET
            base_rate = EXCLUDED.base_rate,
            mortgage_rate = EXCLUDED.mortgage_rate,
            household_debt = EXCLUDED.household_debt,
            m2 = EXCLUDED.m2,
            bsi_realestate = EXCLUDED.bsi_realestate
    """
    rows = [
        (
            row.ref_date,
            row.get("base_rate"),
            row.get("mortgage_rate"),
            row.get("household_debt"),
            row.get("m2"),
            row.get("bsi_realestate"),
        )
        for row in df.itertuples()
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()
    print(f"✅ macro_indicators {len(rows)}건 적재 완료")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    RAW = "data/raw"

    # 국토부 매매 2013~2025
    molit_dir = os.path.join(RAW, "molit")
    for year in range(2013, 2026):
        path = os.path.join(molit_dir, f"apt_trade_{year}.csv")
        if os.path.exists(path):
            df = load_molit_trade(path)
            insert_trade(df)

    # 한국은행 지표
    bok_df = load_bok_excel(os.path.join(RAW, "bok"))
    if bok_df is not None and len(bok_df):
        insert_macro(bok_df)
