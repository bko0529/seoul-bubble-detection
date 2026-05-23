"""
수도권 버블 히트맵 시각화
- 동별 버블 경보 레벨을 지도 위에 히트맵으로 표시
- 월별 슬라이드 애니메이션 지원
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_bubble_heatmap(df: pd.DataFrame, deal_ym: str, save_path: str = None):
    """
    수도권 동별 버블 경보 레벨 히트맵

    Args:
        df: bubble_features 테이블 (gu, dong, alert_level 포함)
        deal_ym: 표시할 연월 (예: '202101')
        save_path: 저장 경로 (None이면 화면 출력)
    """
    target = df[df["deal_ym"] == deal_ym].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    # 경보 레벨별 색상
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bubble", ["#4CAF50", "#FFC107", "#FF5722", "#B71C1C"]
    )

    scatter = ax.scatter(
        x=range(len(target)),
        y=target["alert_level"],
        c=target["alert_level"],
        cmap=cmap, vmin=0, vmax=100,
        s=80, alpha=0.8
    )

    plt.colorbar(scatter, ax=ax, label="버블 경보 레벨 (0~100)")
    ax.set_title(f"수도권 동별 버블 경보 레벨 — {deal_ym[:4]}년 {deal_ym[4:]}월",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("동 (지역)")
    ax.set_ylabel("경보 레벨")
    ax.axhline(y=80, color="red",    linestyle="--", alpha=0.5, label="위험 (80)")
    ax.axhline(y=60, color="orange", linestyle="--", alpha=0.5, label="경고 (60)")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 히트맵 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_price_trend(df: pd.DataFrame, regions: list, save_path: str = None):
    """
    수도권 주요 지역 m²당 가격 추이

    Args:
        df: 전처리 완료 데이터 (gu, deal_ym, price_per_m2 포함)
        regions: 표시할 구 목록 (예: ['강남구', '수원시', '부평구'])
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10.colors
    for i, region in enumerate(regions):
        region_df = (
            df[df["gu"] == region]
            .groupby("deal_ym")["price_per_m2"]
            .median()
            .reset_index()
        )
        ax.plot(
            region_df["deal_ym"],
            region_df["price_per_m2"],
            label=region,
            color=colors[i % len(colors)],
            linewidth=2
        )

    ax.set_title("수도권 주요 지역 ㎡당 가격 추이 (2013~2025)", fontsize=14)
    ax.set_xlabel("연월")
    ax.set_ylabel("중위 단가 (만원/㎡)")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 가격 추이 차트 저장: {save_path}")
    else:
        plt.show()

    plt.close()
