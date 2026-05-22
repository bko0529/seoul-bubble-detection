"""
SHAP 분석: 앙상블 모델 Feature 중요도 시각화
"""
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def run_shap_analysis(model, X: pd.DataFrame, output_dir: str = "reports"):
    """
    Parameters
    ----------
    model   : XGBoost / sklearn 호환 모델
    X       : Feature DataFrame
    output_dir : 저장 경로
    """
    os.makedirs(output_dir, exist_ok=True)

    # 한글 폰트 설정
    try:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
    except Exception:
        pass

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150)
    plt.close()

    # 2. Bar plot (평균 |SHAP|)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_importance.png"), dpi=150)
    plt.close()

    # 3. Top 5 dependence plots
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5_idx = mean_abs.argsort()[-5:][::-1]
    for idx in top5_idx:
        feat = X.columns[idx]
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.tight_layout()
        fname = f"shap_dep_{feat}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()

    # 4. 중요도 CSV 저장
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(os.path.join(output_dir, "shap_importance.csv"), index=False)

    print(f"✅ SHAP 분석 완료 → {output_dir}/")
    return importance_df
