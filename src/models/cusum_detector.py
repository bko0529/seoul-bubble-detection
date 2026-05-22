"""
CUSUM: 구조적 변화점 탐지
가격 상승 추세의 급격한 변화를 탐지
"""
import numpy as np
import pandas as pd


def cusum_detect(
    series: pd.Series,
    k: float = 0.5,
    h: float = 4.0,
) -> pd.DataFrame:
    """
    CUSUM (Cumulative Sum) 제어 차트

    Parameters
    ----------
    series : 정규화된 시계열 (예: price_z_score)
    k      : 허용 편차 (슬랙 파라미터, 보통 0.5σ)
    h      : 결정 임계값 (보통 4~5σ)

    Returns
    -------
    DataFrame: cusum_pos, cusum_neg, alert 컬럼 포함
    """
    s_pos = np.zeros(len(series))
    s_neg = np.zeros(len(series))
    alert = np.zeros(len(series), dtype=bool)

    values = series.fillna(0).values
    mu = values.mean()
    sigma = values.std() if values.std() > 0 else 1.0

    for t in range(1, len(values)):
        z = (values[t] - mu) / sigma
        s_pos[t] = max(0, s_pos[t - 1] + z - k)
        s_neg[t] = max(0, s_neg[t - 1] - z - k)
        alert[t] = (s_pos[t] > h) or (s_neg[t] > h)

    return pd.DataFrame({
        "cusum_pos": s_pos,
        "cusum_neg": s_neg,
        "cusum_alert": alert,
    }, index=series.index)


def detect_changepoints(series: pd.Series, **kwargs) -> list[pd.Timestamp]:
    """변화점 타임스탬프 리스트 반환"""
    result = cusum_detect(series, **kwargs)
    alerts = result[result["cusum_alert"]]
    # 연속된 알림 중 첫 번째만 추출
    changepoints = []
    in_alert = False
    for idx, row in alerts.iterrows():
        if not in_alert:
            changepoints.append(idx)
            in_alert = True
        elif not result.loc[idx, "cusum_alert"]:
            in_alert = False
    return changepoints


def rolling_cusum(
    df: pd.DataFrame,
    feature_col: str = "price_z_score",
    window: int = 24,
    **kwargs,
) -> pd.Series:
    """롤링 윈도우 CUSUM 알림 횟수"""
    alerts = cusum_detect(df[feature_col], **kwargs)["cusum_alert"]
    return alerts.rolling(window).sum()
