"""
HMM: 버블(bubble) / 과열(overheat) / 정상(normal) 상태 전환 모델
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm


STATE_LABELS = {0: "정상", 1: "과열", 2: "버블"}


def fit_hmm(X: np.ndarray, n_components: int = 3,
            n_iter: int = 500, random_state: int = 42) -> hmm.GaussianHMM:
    """
    Parameters
    ----------
    X : shape (T, n_features) — 정규화된 Feature 시계열
    """
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)
    return model


def predict_states(model: hmm.GaussianHMM, X: np.ndarray) -> pd.DataFrame:
    """상태 시퀀스 + 확률 반환"""
    states = model.predict(X)
    probs = model.predict_proba(X)

    # 버블 상태 = 평균 가격 상승률이 가장 높은 상태로 자동 매핑
    means = model.means_[:, 0]  # 첫 번째 feature 기준
    sorted_idx = np.argsort(means)  # 낮은→높은 순
    remap = {sorted_idx[0]: 0, sorted_idx[1]: 1, sorted_idx[2]: 2}
    mapped = np.array([remap[s] for s in states])

    df = pd.DataFrame(probs, columns=[f"prob_{i}" for i in range(model.n_components)])
    df["hmm_state"] = [STATE_LABELS[s] for s in mapped]
    df["bubble_prob"] = df["prob_2"]  # 버블 상태 확률
    return df


def decode_viterbi(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Viterbi 알고리즘으로 최적 상태 시퀀스"""
    log_prob, states = model.decode(X, algorithm="viterbi")
    return states, log_prob
