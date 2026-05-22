"""
전처리 모듈: 국토부 CSV → PostgreSQL 적재
"""
import os
import re
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "dbname": os.getenv("DB_NAME", "bubble_detection"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", ""),
}


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# ──────────────────────────────────────────────
# 국토부 아파트 매매 CSV 전처리
# ──────────────────────────────────────────────
def load_molit_trade(csv_path: str) -> pd.DataFrame:
    """국토부 매매 CSV 읽기 (skiprows=15)"""
    df = pd.read_csv(csv_path, skiprows=15, encoding="cp949", dtype=str)
    df.columns = df.columns.str.strip()

    # 컬럼 매핑 (연도별 컬럼명 다를 수 있음)
    col_map = {
        "시군구": "gu",
        "법정동": "dong",
        "아파트": "apt_name",
        "전용면적(㎡)": "area_m2",
        "거래금액(만원)": "price_raw",
        "계약년월": "ym",
        "계약일": "day",
        "층": "floor",
        "건축년도": "build_year",
        "해제사유발생일": "cancel_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 계약취소 제거
    if "cancel_date" in df.columns:
        df = df[df["cancel_date"].str.strip() == "-"]

    # 거래금액 정제
    df["price_10k"] = (
        df["price_raw"].str.replace(",", "").str.strip().astype(float).astype(int)
    )

    # 날짜 파싱
    df["deal_date"] = pd.to_datetime(
        df["ym"].astype(str) + df["day"].astype(str).str.zfill(2), format="%Y%m%d"
    )

    # 수치형 변환
    df["area_m2"] = pd.to_numeric(df.get("area_m2"), errors="coerce")
    df["floor"] = pd.to_numeric(df.get("floor"), errors="coerce")
    df["build_year"] = pd.to_numeric(df.get("build_year"), errors="coerce")

    return df[["deal_date", "gu", "dong", "apt_name", "area_m2", "price_10k", "floor", "build_year"]]


def insert_trade(df: pd.DataFrame):
    """apt_trade 테이블에 upsert"""
    rows = [
        (
            row.deal_date,
            row.gu,
            row.dong if pd.notna(row.dong) else None,
            row.apt_name if pd.notna(row.apt_name) else None,
            row.area_m2 if pd.notna(row.area_m2) else None,
            row.price_10k,
            int(row.floor) if pd.notna(row.floor) else None,
            int(row.build_year) if pd.notna(row.build_year) else None,
        )
        for row in df.itertuples()
    ]
    sql = """
        INSERT INTO apt_trade
            (deal_date, gu, dong, apt_name, area_m2, price_10k, floor, build_year)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
    print(f"✅ apt_trade {len(rows)}건 적재 완료")


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
