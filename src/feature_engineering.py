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


def build_monthly_features(min_trade: int = 5) -> pd.DataFrame:
    """
    월별 동별 Feature 생성 (구→동 단위로 변경)

    분석 단위: gu + dong + deal_ym
    예상 행 수: ~300개 동 × 140개월 = 42,000행
                (버블 label=2: ~2,500행으로 증가)

    min_trade: 월 최소 거래 건수 (이 미만인 동/월 제외)
               거래가 적으면 중위가격이 불안정하기 때문
    """

    # ── 매매: 동별 월별 중위 m²당 단가 + 거래량
    sql_trade = """
        SELECT
            deal_ym,
            gu,
            dong,
            PERCENTILE_CONT(0.5) WITHIN GROUP
                (ORDER BY price_per_m2) AS median_price,
            COUNT(*)                   AS trade_volume
        FROM apt_trade
        WHERE price_per_m2 IS NOT NULL
        GROUP BY deal_ym, gu, dong
        ORDER BY deal_ym, gu, dong
    """

    # ── 전세: 동별 월별 중위 m²당 전세가
    sql_jeonse = """
        SELECT
            deal_ym,
            gu,
            dong,
            PERCENTILE_CONT(0.5) WITHIN GROUP
                (ORDER BY deposit_10k / NULLIF(area_m2, 0)) AS median_jeonse
        FROM apt_jeonse
        WHERE contract_type = '전세'
          AND deposit_10k IS NOT NULL
          AND area_m2     IS NOT NULL
        GROUP BY deal_ym, gu, dong
    """

    # ── 거시경제 (월별 전국 단일값)
    sql_macro = "SELECT deal_ym, base_rate, mortgage_rate, household_debt, m2, cpi, bsi_realestate FROM macro_indicators ORDER BY deal_ym"

    with get_conn() as conn:
        df_trade  = pd.read_sql(sql_trade,  conn)
        df_jeonse = pd.read_sql(sql_jeonse, conn)
        df_macro  = pd.read_sql(sql_macro,  conn)

    # ── 거래량 적은 동/월 제외 (중위가격 불안정 방지)
    before = len(df_trade)
    df_trade = df_trade[df_trade["trade_volume"] >= min_trade]
    print(f"거래량 {min_trade}건 미만 제외: {before - len(df_trade)}행 제거")
    print(f"남은 행: {len(df_trade)}행 ({df_trade[['gu','dong']].drop_duplicates().shape[0]}개 동)")

    # ── 병합
    df = df_trade.merge(df_jeonse, on=["deal_ym", "gu", "dong"], how="left")
    df = df.merge(df_macro,  on="deal_ym", how="left")
    df = df.sort_values(["gu", "dong", "deal_ym"]).reset_index(drop=True)

    # ── 그룹 기준: 동별 시계열
    g = df.groupby(["gu", "dong"])

    # ── 가격 파생 피처
    df["jeonse_rate"] = df["median_jeonse"] / df["median_price"]
    df["price_mom"]   = g["median_price"].pct_change(1)
    df["price_yoy"]   = g["median_price"].pct_change(12)
    df["price_ma3"]   = g["median_price"].transform(lambda x: x.rolling(3,  min_periods=1).mean())
    df["price_ma12"]  = g["median_price"].transform(lambda x: x.rolling(12, min_periods=6).mean())
    df["price_vol"]   = g["price_mom"].transform(lambda x: x.rolling(12).std())
    df["price_zscore"] = g["median_price"].transform(
        lambda x: (x - x.rolling(24).mean()) / (x.rolling(24).std() + 1e-8)
    )

    # ── 거시 파생 피처
    df["rate_spread"] = df["mortgage_rate"] - df["base_rate"]
    df["debt_growth"] = df["household_debt"].pct_change(12)
    df["cpi_growth"]  = df["cpi"].pct_change(12)

    # ── 버블 Pseudo-label 생성
    #    PIR은 소득 데이터 없으면 NaN → 전세가율 + 가격상승률만으로 부분 레이블
    conditions = [
        # 버블: 전세가율 50% 미만 AND 연간 15% 이상 상승
        (df["jeonse_rate"] < 0.50) & (df["price_yoy"] > 0.15),
        # 과열: 전세가율 60% 미만 AND 연간 8% 이상 상승
        (df["jeonse_rate"] < 0.60) & (df["price_yoy"] > 0.08),
    ]
    df["bubble_label"] = np.select(conditions, [2, 1], default=0)

    label_counts = df["bubble_label"].value_counts()
    print(f"\n버블 레이블 분포:")
    print(f"  정상(0): {label_counts.get(0,0)}행")
    print(f"  과열(1): {label_counts.get(1,0)}행")
    print(f"  버블(2): {label_counts.get(2,0)}행")

    return df


if __name__ == "__main__":
    df = build_monthly_features(min_trade=5)
    print(f"\n최종 shape: {df.shape}")
    print(df.columns.tolist())
    df.to_parquet("data/processed/features.parquet", index=False)
    print("✅ features.parquet 저장 완료")
