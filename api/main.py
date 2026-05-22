"""
FastAPI 서빙: /predict  /report  /health
"""
import os
from datetime import date
from typing import Optional

import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="서울 아파트 버블 조기경보 API",
    description="LSTM + HMM + CUSUM + Granger 앙상블 버블 탐지",
    version="1.0.0",
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "dbname": os.getenv("DB_NAME", "bubble_detection"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", ""),
}


# ──────────────────────────────────────────────
# 스키마
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    gu: Optional[str] = None          # 구 이름 (없으면 서울 전체)
    ref_date: Optional[date] = None   # 기준월 (없으면 최신)


class PredictResponse(BaseModel):
    ref_date: str
    gu: str
    ensemble_score: float
    alert_level: str
    hmm_state: str
    bubble_prob: float
    lstm_score: float
    cusum_alert: bool


class ReportResponse(BaseModel):
    generated_at: str
    alerts: list[PredictResponse]
    summary: str


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    """헬스체크"""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """특정 구/월의 버블 점수 반환"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB 연결 실패: {e}")

    with conn:
        with conn.cursor() as cur:
            if req.ref_date and req.gu:
                cur.execute(
                    """
                    SELECT signal_date, gu, ensemble_score, alert_level,
                           hmm_state, hmm_prob, lstm_score, cusum_alert
                    FROM bubble_signals
                    WHERE signal_date = %s AND gu = %s
                    ORDER BY signal_date DESC LIMIT 1
                    """,
                    (req.ref_date, req.gu),
                )
            else:
                cur.execute(
                    """
                    SELECT signal_date, gu, ensemble_score, alert_level,
                           hmm_state, hmm_prob, lstm_score, cusum_alert
                    FROM bubble_signals
                    WHERE gu = %s
                    ORDER BY signal_date DESC LIMIT 1
                    """,
                    (req.gu or "서울 전체",),
                )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="데이터 없음")

    return PredictResponse(
        ref_date=str(row[0]),
        gu=row[1],
        ensemble_score=float(row[2]),
        alert_level=row[3],
        hmm_state=row[4],
        bubble_prob=float(row[5]),
        lstm_score=float(row[6]),
        cusum_alert=bool(row[7]),
    )


@app.get("/report", response_model=ReportResponse)
def report():
    """최신 버블 경보 전체 리포트"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB 연결 실패: {e}")

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (gu)
                    signal_date, gu, ensemble_score, alert_level,
                    hmm_state, hmm_prob, lstm_score, cusum_alert
                FROM bubble_signals
                ORDER BY gu, signal_date DESC
                """
            )
            rows = cur.fetchall()

    alerts = [
        PredictResponse(
            ref_date=str(r[0]), gu=r[1],
            ensemble_score=float(r[2]), alert_level=r[3],
            hmm_state=r[4], bubble_prob=float(r[5]),
            lstm_score=float(r[6]), cusum_alert=bool(r[7]),
        )
        for r in rows
    ]

    danger_count = sum(1 for a in alerts if a.alert_level in ("경고", "위험"))
    summary = (
        f"총 {len(alerts)}개 구 분석 완료. "
        f"경고 이상 {danger_count}개 구 탐지."
    )

    return ReportResponse(
        generated_at=str(date.today()),
        alerts=alerts,
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
