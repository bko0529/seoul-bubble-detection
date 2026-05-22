"""
Optuna 앙상블: LSTM + HMM + CUSUM + Granger 점수 최적 가중치 탐색
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score


def ensemble_score(
    lstm_scores: np.ndarray,
    hmm_probs: np.ndarray,
    cusum_alerts: np.ndarray,
    granger_lead: np.ndarray,
    w_lstm: float,
    w_hmm: float,
    w_cusum: float,
    w_granger: float,
) -> np.ndarray:
    """가중 앙상블 점수 계산 (0~1)"""
    total = w_lstm + w_hmm + w_cusum + w_granger
    score = (
        w_lstm    * normalize(lstm_scores)
        + w_hmm     * hmm_probs
        + w_cusum   * cusum_alerts.astype(float)
        + w_granger * normalize(granger_lead)
    ) / total
    return np.clip(score, 0, 1)


def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def optimize_weights(
    lstm_scores: np.ndarray,
    hmm_probs: np.ndarray,
    cusum_alerts: np.ndarray,
    granger_lead: np.ndarray,
    y_true: np.ndarray,
    n_trials: int = 200,
) -> dict:
    """
    Optuna로 AUC 최대화하는 가중치 탐색

    Parameters
    ----------
    y_true : 실제 버블 레이블 (0/1) — 과거 데이터 기준
    """

    def objective(trial):
        w_lstm    = trial.suggest_float("w_lstm",    0.0, 1.0)
        w_hmm     = trial.suggest_float("w_hmm",     0.0, 1.0)
        w_cusum   = trial.suggest_float("w_cusum",   0.0, 1.0)
        w_granger = trial.suggest_float("w_granger", 0.0, 1.0)

        if w_lstm + w_hmm + w_cusum + w_granger < 1e-6:
            return 0.0

        scores = ensemble_score(
            lstm_scores, hmm_probs, cusum_alerts, granger_lead,
            w_lstm, w_hmm, w_cusum, w_granger
        )
        try:
            return roc_auc_score(y_true, scores)
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"✅ 최적 AUC: {study.best_value:.4f}")
    print(f"   가중치: lstm={best['w_lstm']:.3f}, hmm={best['w_hmm']:.3f}, "
          f"cusum={best['w_cusum']:.3f}, granger={best['w_granger']:.3f}")
    return best


def alert_level(score: float) -> str:
    """점수 → 경보 단계"""
    if score >= 0.8:
        return "위험"
    elif score >= 0.6:
        return "경고"
    elif score >= 0.4:
        return "주의"
    return "정상"
