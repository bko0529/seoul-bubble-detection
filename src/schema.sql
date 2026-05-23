-- ============================================================
-- 서울 아파트 버블 조기경보 시스템 PostgreSQL 스키마
-- ============================================================

-- 1. 아파트 매매 실거래가
CREATE TABLE apt_trade (
    id              SERIAL PRIMARY KEY,
    deal_date       DATE NOT NULL,
    deal_ym         VARCHAR(6) NOT NULL,   -- YYYYMM (인덱스용)
    gu              VARCHAR(20) NOT NULL,  -- 강남구
    dong            VARCHAR(30) NOT NULL,  -- 대치동
    apt_name        VARCHAR(100),
    area_m2         NUMERIC(8,2),
    price_10k       INTEGER NOT NULL,      -- 단위: 만원
    price_per_m2    NUMERIC(10,2),         -- m²당 단가 (만원/m²)
    floor           INTEGER,
    build_year      INTEGER,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (deal_date, gu, dong, apt_name, area_m2, floor, price_10k)
);

-- 2. 아파트 전월세
CREATE TABLE apt_jeonse (
    id              SERIAL PRIMARY KEY,
    deal_date       DATE NOT NULL,
    deal_ym         VARCHAR(6) NOT NULL,   -- YYYYMM
    contract_type   VARCHAR(10) NOT NULL,  -- 전세/월세
    gu              VARCHAR(20) NOT NULL,
    dong            VARCHAR(30) NOT NULL,
    apt_name        VARCHAR(100),
    area_m2         NUMERIC(8,2),
    deposit_10k     INTEGER,               -- 보증금 (만원)
    monthly_10k     INTEGER,               -- 월세 (만원)
    floor           INTEGER,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 3. 거시경제 지표 (월별 전국 단일값)
CREATE TABLE macro_indicators (
    id              SERIAL PRIMARY KEY,
    ref_date        DATE NOT NULL,        -- 기준월 (월 첫째날)
    deal_ym         VARCHAR(6) NOT NULL,  -- YYYYMM
    base_rate       NUMERIC(5,3),         -- 기준금리 (%)
    mortgage_rate   NUMERIC(5,3),         -- 주담대금리 (%)
    household_debt  BIGINT,               -- 가계대출 잔액 (억원)
    m2              BIGINT,               -- M2 광의통화 (십억원)
    cpi             NUMERIC(7,2),         -- 소비자물가지수
    bsi_realestate  NUMERIC(6,1),         -- 부동산업 BSI
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (deal_ym)
);

-- 4. 월별 동별 집계 (Feature 테이블 — 구→동 단위로 변경)
--    25개 구 × 140개월 = 3,500행  →  ~300개 동 × 140개월 = 42,000행
CREATE TABLE monthly_dong_agg (
    id              SERIAL PRIMARY KEY,
    deal_ym         VARCHAR(6) NOT NULL,  -- YYYYMM
    gu              VARCHAR(20) NOT NULL, -- 강남구 (상위 필터링용)
    dong            VARCHAR(30) NOT NULL, -- 대치동
    -- 거래량 (거래 적은 동 필터링 기준)
    trade_volume    INTEGER,              -- 월 거래 건수
    -- 가격 지표
    median_price    NUMERIC(12,2),        -- 중위 매매가 (만원/m²)
    price_mom       NUMERIC(6,4),         -- 전월 대비 변화율
    price_yoy       NUMERIC(6,4),         -- 전년 대비 변화율
    price_ma3       NUMERIC(12,2),        -- 3개월 이동평균
    price_ma12      NUMERIC(12,2),        -- 12개월 이동평균
    price_zscore    NUMERIC(6,4),         -- Z-score (24개월 기준)
    price_vol       NUMERIC(6,4),         -- 변동성 (12개월 표준편차)
    -- 전세 지표
    median_jeonse   NUMERIC(12,2),        -- 중위 전세가 (만원/m²)
    jeonse_rate     NUMERIC(5,4),         -- 전세가율 (0~1)
    -- 거시경제 (macro_indicators 에서 JOIN)
    base_rate       NUMERIC(5,3),
    household_debt  BIGINT,
    cpi             NUMERIC(7,2),
    bsi_realestate  NUMERIC(6,1),
    -- PIR (통계청 소득 데이터 연결 후 계산)
    pir             NUMERIC(6,2),         -- 가격/연소득
    -- 버블 레이블
    bubble_label    SMALLINT,             -- 0:정상 1:과열 2:버블
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (deal_ym, gu, dong)
);

-- 5. 버블 탐지 결과
CREATE TABLE bubble_signals (
    id              SERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL,
    deal_ym         VARCHAR(6) NOT NULL,
    gu              VARCHAR(20),
    dong            VARCHAR(30),          -- 동 단위 탐지 결과
    lstm_score      NUMERIC(6,4),         -- LSTM 이상 점수 (0~1)
    hmm_state       VARCHAR(10),          -- 정상/과열/버블
    hmm_prob        NUMERIC(6,4),         -- 버블 확률
    cusum_alert     BOOLEAN DEFAULT FALSE,
    ensemble_score  NUMERIC(6,4),         -- 앙상블 종합 점수
    alert_level     VARCHAR(10),          -- 정상/주의/경고/위험
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 6. 뉴스 감성 지표 (월별 서울 전체 단일값)
CREATE TABLE news_sentiment (
    id              SERIAL PRIMARY KEY,
    deal_ym         VARCHAR(6) NOT NULL,
    article_count   INTEGER,              -- 월별 기사 수
    sentiment_score NUMERIC(6,4),         -- 감성 점수 (-1~1)
    keyword_freq    JSONB,                -- {"버블":12, "급등":34}
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (deal_ym)
);

-- 인덱스
CREATE INDEX idx_apt_trade_ym      ON apt_trade (deal_ym);
CREATE INDEX idx_apt_trade_dong    ON apt_trade (gu, dong, deal_ym);
CREATE INDEX idx_apt_jeonse_dong   ON apt_jeonse (gu, dong, deal_ym);
CREATE INDEX idx_macro_ym          ON macro_indicators (deal_ym);
CREATE INDEX idx_monthly_dong_ym   ON monthly_dong_agg (deal_ym);
CREATE INDEX idx_monthly_dong_key  ON monthly_dong_agg (gu, dong);
CREATE INDEX idx_bubble_ym         ON bubble_signals (deal_ym);
CREATE INDEX idx_bubble_dong       ON bubble_signals (gu, dong);
