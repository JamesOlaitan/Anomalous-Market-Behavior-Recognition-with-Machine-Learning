-- Database schema for anomaly detection project

-- Raw prices table
CREATE TABLE IF NOT EXISTS raw_prices (
    date DATE,
    symbol VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT,
    PRIMARY KEY (date, symbol)
);

-- Raw VIX table
CREATE TABLE IF NOT EXISTS raw_vix (
    date DATE PRIMARY KEY,
    vix DOUBLE
);

-- Features table
CREATE TABLE IF NOT EXISTS features (
    date DATE,
    symbol VARCHAR,
    returns DOUBLE,
    log_returns DOUBLE,
    volatility DOUBLE,
    rolling_corr DOUBLE,
    corr_zscore DOUBLE,
    vol_zscore DOUBLE,
    vix DOUBLE,
    vix_delta DOUBLE,
    PRIMARY KEY (date, symbol)
);

-- Labels table
CREATE TABLE IF NOT EXISTS labels (
    date DATE,
    symbol VARCHAR,
    label INTEGER,
    PRIMARY KEY (date, symbol)
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    date DATE,
    symbol VARCHAR,
    p_anom DOUBLE,
    post_normal DOUBLE,
    post_anomalous DOUBLE,
    state VARCHAR,
    PRIMARY KEY (date, symbol)
);

