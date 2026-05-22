-- ============================================================
-- 서울 아파트 버블 조기경보 시스템 PostgreSQL 스키마
-- ============================================================

-- 1. 아파트 매매 실거래가
CREATE TABLE apt_trade (
    id              SERIAL PRIMARY KEY,
    deal_date       DATE NOT NULL,
    gu             VARCHAR(20) NOT NULL,
    dong           VARCHAR(30),
    apt_name       VARCHAR(100),
    area_m2        NUMERIC(8,2),
    price_10k      INTEGER NOT NULL,   -- 단위: 만원
    floor          INTEGER,
    build_year     INTEGER,
    created_at     TIMESTAMP DEFAULT NOW()
);

-- 2. 아파트 전월세
CREATE TABLE apt_jeonse (
    id              SERIAL PRIMARY KEY,
    deal_date       DATE NOT NULL,
    contract_type  VARCHAR(10) NOT NULL,  -- 전세/월세
    gu             VARCHAR(20) NOT NULL,
    dong           VARCHAR(30),
    apt_name       VARCHAR(100),
    area_m2        NUMERIC(8,2),
    deposit_10k    INTEGER,               -- 보증금 (만원)
    monthly_10k    INTEGER,               -- 월세 (만원)
    floor          INTEGER,
    created_at     TIMESTAMP DEFAULT NOW()
);

-- 3. 거시경제 지표
CREATE TABLE macro_indicators (
    id              SERIAL PRIMARY KEY,
    ref_date        DATE NOT NULL,        -- 기준월 (월 첫째날)
    base_rate       NUMERIC(5,3),         -- 기준금리 (%)
    mortgage_rate   NUMERIC(5,3),         -- 주담대금리 (%)
    household_debt  BIGINT,               -- 가계대출 잔액 (억원)
    m2              BIGINT,               -- M2 광의통화 (십억원)
    cpi             NUMERIC(7,2),         -- 소비자물가지수
    bsi_realestate  NUMERIC(6,1),         -- 부동산업 BSI
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (ref_date)
);

-- 4. 월별 구별 집계 (Feature 테이블)
CREATE TABLE monthly_gu_agg (
    id              SERIAL PRIMARY KEY,
    ref_date        DATE NOT NULL,
    gu             VARCHAR(20) NOT NULL,
    median_price    NUMERIC(12,2),        -- 중위 매매가 (만원/m2)
    jeonse_rate     NUMERIC(5,3),         -- 전세가율
    pir             NUMERIC(6,3),         -- PIR (가격/소득)
    trade_volume    INTEGER,              -- 거래량
    price_mom       NUMERIC(6,3),         -- 전월대비 변화율
    price_yoy       NUMERIC(6,3),         -- 전년대비 변화율
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (ref_date, gu)
);

-- 5. 버블 탐지 결과
CREATE TABLE bubble_signals (
    id              SERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL,
    gu             VARCHAR(20),           -- NULL이면 서울 전체
    lstm_score      NUMERIC(6,4),         -- LSTM 이상 점수 (0~1)
    hmm_state       VARCHAR(20),          -- 정상/과열/버블
    hmm_prob        NUMERIC(6,4),         -- 버블 확률
    cusum_alert     BOOLEAN DEFAULT FALSE,
    ensemble_score  NUMERIC(6,4),         -- 앙상블 종합 점수
    alert_level     VARCHAR(10),          -- 정상/주의/경고/위험
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 6. 뉴스 감성 지표
CREATE TABLE news_sentiment (
    id              SERIAL PRIMARY KEY,
    ref_date        DATE NOT NULL,
    article_count   INTEGER,              -- 월별 기사 수
    sentiment_score NUMERIC(6,4),         -- 감성 점수 (-1~1)
    keyword_freq    JSONB,                -- {"버블":12, "급등":34}
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (ref_date)
);

-- 인덱스
CREATE INDEX idx_apt_trade_date_gu ON apt_trade (deal_date, gu);
CREATE INDEX idx_apt_jeonse_date_gu ON apt_jeonse (deal_date, gu);
CREATE INDEX idx_macro_date ON macro_indicators (ref_date);
CREATE INDEX idx_monthly_agg_date_gu ON monthly_gu_agg (ref_date, gu);
CREATE INDEX idx_bubble_signals_date ON bubble_signals (signal_date);
