"""
Feature Engineering: 23개 지표 생성
"""
import os
import pandas as pd
import numpy as np
import psycopg2
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


def build_monthly_features() -> pd.DataFrame:
    """
    월별 구별 Feature 23개 생성

    가격 관련 (8)
    ─ median_price, price_mom, price_yoy, price_3m, price_6m,
      price_vol_3m, price_z_score, upper_band

    거래 관련 (4)
    ─ trade_volume, vol_mom, vol_yoy, vol_ma3

    전세/PIR (4)
    ─ jeonse_rate, jeonse_mom, pir, pir_ma3

    거시 (4)
    ─ rate_spread, debt_growth, m2_growth, cpi_growth

    감성/검색 (3)
    ─ news_sentiment, article_count, naver_search_idx
    """
    sql_trade = """
        SELECT
            DATE_TRUNC('month', deal_date) AS ref_date,
            gu,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_10k / NULLIF(area_m2, 0)) AS median_price,
            COUNT(*) AS trade_volume
        FROM apt_trade
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    sql_jeonse = """
        SELECT
            DATE_TRUNC('month', deal_date) AS ref_date,
            gu,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY deposit_10k / NULLIF(area_m2, 0)) AS median_jeonse
        FROM apt_jeonse
        WHERE contract_type = '전세'
        GROUP BY 1, 2
    """
    sql_macro = "SELECT * FROM macro_indicators ORDER BY ref_date"

    with get_conn() as conn:
        df_trade  = pd.read_sql(sql_trade,   conn, parse_dates=["ref_date"])
        df_jeonse = pd.read_sql(sql_jeonse,  conn, parse_dates=["ref_date"])
        df_macro  = pd.read_sql(sql_macro,   conn, parse_dates=["ref_date"])

    # 병합
    df = df_trade.merge(df_jeonse, on=["ref_date", "gu"], how="left")
    df = df.merge(df_macro, on="ref_date", how="left")
    df = df.sort_values(["gu", "ref_date"])

    # ── 가격 피처
    df["jeonse_rate"]  = df["median_jeonse"] / df["median_price"]
    df["price_mom"]    = df.groupby("gu")["median_price"].pct_change(1)
    df["price_yoy"]    = df.groupby("gu")["median_price"].pct_change(12)
    df["price_3m"]     = df.groupby("gu")["median_price"].pct_change(3)
    df["price_6m"]     = df.groupby("gu")["median_price"].pct_change(6)
    df["price_vol_3m"] = df.groupby("gu")["price_mom"].transform(
        lambda x: x.rolling(3).std()
    )
    df["price_z_score"] = df.groupby("gu")["median_price"].transform(
        lambda x: (x - x.rolling(24).mean()) / x.rolling(24).std()
    )
    df["upper_band"] = df.groupby("gu")["median_price"].transform(
        lambda x: x.rolling(24).mean() + 2 * x.rolling(24).std()
    )

    # ── 거래 피처
    df["vol_mom"]  = df.groupby("gu")["trade_volume"].pct_change(1)
    df["vol_yoy"]  = df.groupby("gu")["trade_volume"].pct_change(12)
    df["vol_ma3"]  = df.groupby("gu")["trade_volume"].transform(
        lambda x: x.rolling(3).mean()
    )

    # ── PIR
    df["pir"]    = df["median_price"] * 12 / df.get("income_median", np.nan)
    df["pir_ma3"] = df.groupby("gu")["pir"].transform(lambda x: x.rolling(3).mean())

    # ── 거시 피처
    df["rate_spread"]  = df["mortgage_rate"] - df["base_rate"]
    df["debt_growth"]  = df["household_debt"].pct_change(12)
    df["m2_growth"]    = df["m2"].pct_change(12)
    df["cpi_growth"]   = df["cpi"].pct_change(12)

    return df


if __name__ == "__main__":
    df = build_monthly_features()
    print(df.shape)
    print(df.columns.tolist())
    df.to_parquet("data/processed/features.parquet", index=False)
    print("✅ features.parquet 저장 완료")
