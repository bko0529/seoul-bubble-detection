"""
Granger 인과성 분석: 선행지표 탐색
어떤 지표가 아파트 가격을 선행하는지 탐지
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


def check_stationarity(series: pd.Series, name: str = "") -> bool:
    """ADF 검정으로 정상성 확인"""
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05
    print(f"{'✅' if is_stationary else '⚠️ '} {name or series.name}: "
          f"ADF p={p_value:.4f} ({'정상' if is_stationary else '비정상'})")
    return is_stationary


def make_stationary(series: pd.Series, max_diff: int = 2) -> pd.Series:
    """차분으로 정상화"""
    for d in range(1, max_diff + 1):
        diffed = series.diff(d).dropna()
        if adfuller(diffed)[1] < 0.05:
            print(f"  {series.name}: {d}차 차분 후 정상화")
            return diffed
    return series.diff(max_diff).dropna()


def run_granger(
    target: pd.Series,
    candidates: dict[str, pd.Series],
    max_lag: int = 6,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    target      : 종속변수 (아파트 가격 변화율)
    candidates  : {이름: 시계열} 선행지표 후보
    max_lag     : 최대 시차 (개월)
    alpha       : 유의수준

    Returns
    -------
    DataFrame: 유의한 선행지표와 최적 lag
    """
    results = []
    for name, series in candidates.items():
        df = pd.concat([target, series], axis=1).dropna()
        if len(df) < max_lag + 10:
            continue
        try:
            test_result = grangercausalitytests(df, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                p_val = test_result[lag][0]["ssr_ftest"][1]
                if p_val < alpha:
                    results.append({
                        "indicator": name,
                        "lag_months": lag,
                        "p_value": p_val,
                        "significant": True,
                    })
                    break
        except Exception as e:
            print(f"  ⚠️  {name}: {e}")

    return pd.DataFrame(results).sort_values("p_value") if results else pd.DataFrame()


def summarize_leading_indicators(df_features: pd.DataFrame) -> pd.DataFrame:
    """Feature DataFrame에서 선행지표 자동 탐색"""
    target = df_features["price_mom"].dropna()
    candidates = {
        "base_rate": df_features["base_rate"],
        "mortgage_rate": df_features["mortgage_rate"],
        "household_debt_growth": df_features["debt_growth"],
        "m2_growth": df_features["m2_growth"],
        "trade_volume": df_features["trade_volume"],
        "jeonse_rate": df_features["jeonse_rate"],
        "news_sentiment": df_features.get("news_sentiment", pd.Series(dtype=float)),
    }
    # 정상성 확보
    candidates_stat = {
        k: (v if check_stationarity(v, k) else make_stationary(v))
        for k, v in candidates.items()
        if not v.dropna().empty
    }
    return run_granger(target, candidates_stat)
