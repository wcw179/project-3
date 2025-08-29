import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'data' / 'm5_trading.db'
SPLIT_DIR = ROOT / 'artifacts' / 'data' / 'splits' / 'EURUSDm'
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_COMPONENTS = ROOT / 'artifacts' / 'models' / 'xgb_mfe' / 'EURUSDm' / 'model_components.pkl'
FEATURE_NAMES_JSON = ROOT / 'artifacts' / 'analysis' / 'EURUSDm' / 'feature_names.json'

SYMBOL = 'EURUSDm'
TIME_COL = 'time'
N_MFE = 20  # bars for MFE horizon (assumption)
VALIDATION_FRACTION = 0.2  # last 20% as validation


def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(period).mean()
    roll_down = pd.Series(down, index=close.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(d_period).mean()
    return k, d

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-12)

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    return tr.rolling(period).mean()

def bollinger(close: pd.Series, period=20, n_std=2.0):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = upper - lower
    pos = (close - lower) / (width + 1e-12)
    return ma, upper, lower, width, pos

def macd_signal_hist(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd = ema_fast - ema_slow
    sig = ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def obv(close: pd.Series, volume: pd.Series):
    dirn = np.sign(close.diff().fillna(0))
    return (dirn * volume.fillna(0)).cumsum()

def rolling_slope(s: pd.Series, window: int = 10):
    # simple slope = (s - s.shift(window)) / window
    return (s - s.shift(window)) / (window + 1e-12)

def price_efficiency(close: pd.Series, period=20):
    num = (close - close.shift(period)).abs()
    den = close.diff().abs().rolling(period).sum()
    return num / (den + 1e-12)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    ema20_ratio = ema20 / close
    ema50_ratio = ema50 / close
    ema200_ratio = ema200 / close

    ema20_slope = rolling_slope(ema20, 5) / (ema20.abs() + 1e-12)
    ema50_slope = rolling_slope(ema50, 5) / (ema50.abs() + 1e-12)

    ema20_above_50 = (ema20 > ema50).astype(int)
    ema50_above_200 = (ema50 > ema200).astype(int)
    ema_bullish_alignment = ((ema20 > ema50) & (ema50 > ema200)).astype(int)

    rsi14 = rsi(close, 14)
    rsi_oversold = (rsi14 < 30).astype(int)
    rsi_overbought = (rsi14 > 70).astype(int)

    macd, macd_sig, macd_hist = macd_signal_hist(close)
    macd_signal_ = macd_sig
    macd_histogram = macd_hist
    macd_bullish = (macd > macd_sig).astype(int)

    k, d = stoch(high, low, close)
    stoch_k = k
    stoch_d = d
    stoch_oversold = (k < 20).astype(int)
    stoch_overbought = (k > 80).astype(int)

    wr = williams_r(high, low, close)

    atr14 = atr(high, low, close, 14)
    atr_close_ratio = atr14 / close

    ma20, bb_up, bb_lo, bb_width, bb_pos = bollinger(close)
    bb_width_pct = bb_width / close
    bb_position = bb_pos
    # Squeeze: width below its rolling 20th percentile
    wq = bb_width.rolling(200).quantile(0.2)
    bb_squeeze = (bb_width < wq).astype(int)
    bb_breakout_up = ((close.shift(1) <= bb_up.shift(1)) & (close > bb_up)).astype(int)
    bb_breakout_down = ((close.shift(1) >= bb_lo.shift(1)) & (close < bb_lo)).astype(int)

    # VWAP distance using rolling 50-bar VWAP
    typical = (high + low + close) / 3.0
    vol = volume.fillna(0)
    vwap50 = (typical * vol).rolling(50).sum() / (vol.rolling(50).sum() + 1e-12)
    vwap_distance = (close - vwap50) / close

    # Volume regime (z-score buckets) & spike
    vol_mean = volume.rolling(200).mean()
    vol_std = volume.rolling(200).std()
    vol_z = (volume - vol_mean) / (vol_std + 1e-12)
    volume_regime = pd.cut(vol_z, bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0,1,2]).astype(float).fillna(1).astype(int)
    volume_spike = (vol_z > 2.0).astype(int)

    # OBV slope
    obv_series = obv(close, volume)
    obv_slope = rolling_slope(obv_series, 10)

    # Candle features
    hl_range_pct = (high - low) / close
    body = (close - open_).abs()
    body_size_pct = body / close
    upper_shadow_pct = (high - close.where(close>open_, open_)) / close
    lower_shadow_pct = (close.where(close<open_, open_) - low) / close

    # Returns features
    ret = close.pct_change()
    returns_5bar_mean = ret.rolling(5).mean()
    returns_5bar_std = ret.rolling(5).std()
    returns_20bar_mean = ret.rolling(20).mean()
    returns_20bar_std = ret.rolling(20).std()
    returns_skew = ret.rolling(20).skew()
    returns_kurt = ret.rolling(20).kurt()

    # Patterns (basic
    def engulfing():
        prev_body = (close.shift(1) - open_.shift(1))
        curr_body = (close - open_)
        bull = ((prev_body < 0) & (curr_body > 0) & (close >= open_.shift(1)) & (open_ <= close.shift(1))).astype(int)
        bear = ((prev_body > 0) & (curr_body < 0) & (open_ >= close.shift(1)) & (close <= open_.shift(1))).astype(int)
        return bull, bear
    pat_bull, pat_bear = engulfing()

    def marubozu(th=0.1):
        rng = (high - low)
        up_sh = (high - close.where(close>open_, open_))
        lo_sh = (close.where(close<open_, open_) - low)
        maru_bull = ((close>open_) & (up_sh<=th*rng) & (lo_sh<=th*rng)).astype(int)
        maru_bear = ((close<open_) & (up_sh<=th*rng) & (lo_sh<=th*rng)).astype(int)
        return maru_bull, maru_bear
    maru_bull, maru_bear = marubozu()

    # Other named patterns set to 0 by default (placeholders)
    zeros = pd.Series(0, index=df.index, dtype=int)
    pattern_cols = {
        'pattern_three_candles_bull': zeros,
        'pattern_three_candles_bear': zeros,
        'pattern_double_trouble_bull': zeros,
        'pattern_double_trouble_bear': zeros,
        'pattern_tasuki_bull': zeros,
        'pattern_tasuki_bear': zeros,
        'pattern_hikkake_bull': zeros,
        'pattern_hikkake_bear': zeros,
        'pattern_quintuplets_bull': zeros,
        'pattern_quintuplets_bear': zeros,
        'pattern_bottle_bull': zeros,
        'pattern_bottle_bear': zeros,
        'pattern_slingshot_bull': zeros,
        'pattern_slingshot_bear': zeros,
        'pattern_h_pattern_bull': zeros,
        'pattern_h_pattern_bear': zeros,
    }

    # Time/session features
    ts = pd.to_datetime(df['time'], utc=True)
    hour_of_day = ts.dt.hour
    day_of_week = ts.dt.weekday
    is_london_session = ((hour_of_day>=7) & (hour_of_day<16)).astype(int)
    is_ny_session = ((hour_of_day>=12) & (hour_of_day<21)).astype(int)
    is_overlap_session = ((hour_of_day>=12) & (hour_of_day<16)).astype(int)

    # Others
    spread_proxy = (high - low) / close
    price_eff = price_efficiency(close, 20)
    gap_up = (open_ > close.shift(1)).astype(int)
    gap_down = (open_ < close.shift(1)).astype(int)
    direction_flag = np.sign(ret).fillna(0).astype(int)

    out = pd.DataFrame({
        'ema20_ratio': ema20_ratio,
        'ema50_ratio': ema50_ratio,
        'ema200_ratio': ema200_ratio,
        'ema20_slope': ema20_slope,
        'ema50_slope': ema50_slope,
        'ema20_above_50': ema20_above_50,
        'ema50_above_200': ema50_above_200,
        'ema_bullish_alignment': ema_bullish_alignment,
        'rsi14': rsi14,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'macd_signal': macd_signal_,
        'macd_histogram': macd_histogram,
        'macd_bullish': macd_bullish,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'stoch_oversold': stoch_oversold,
        'stoch_overbought': stoch_overbought,
        'williams_r': wr,
        'atr_close_ratio': atr_close_ratio,
        'bb_width_pct': bb_width_pct,
        'bb_position': bb_position,
        'bb_squeeze': bb_squeeze,
        'bb_breakout_up': bb_breakout_up,
        'bb_breakout_down': bb_breakout_down,
        'vwap_distance': vwap_distance,
        'volume_regime': volume_regime,
        'volume_spike': volume_spike,
        'obv_slope': obv_slope,
        'hl_range_pct': hl_range_pct,
        'body_size_pct': body_size_pct,
        'upper_shadow_pct': upper_shadow_pct,
        'lower_shadow_pct': lower_shadow_pct,
        'returns_5bar_mean': returns_5bar_mean,
        'returns_5bar_std': returns_5bar_std,
        'returns_20bar_mean': returns_20bar_mean,
        'returns_20bar_std': returns_20bar_std,
        'returns_skew': returns_skew,
        'returns_kurt': returns_kurt,
        'pattern_bullish_engulfing': pat_bull,
        'pattern_bearish_engulfing': pat_bear,
        'pattern_marubozu_bull': maru_bull,
        'pattern_marubozu_bear': maru_bear,
        **pattern_cols,
        'total_bullish_patterns': (
            pat_bull + maru_bull + sum(pattern_cols[k] for k in pattern_cols if k.endswith('_bull'))
        ),
        'total_bearish_patterns': (
            pat_bear + maru_bear + sum(pattern_cols[k] for k in pattern_cols if k.endswith('_bear'))
        ),
        'pattern_strength': lambda: 0
    })

    # Fix pattern_strength after creation
    out['pattern_strength'] = out['total_bullish_patterns'] - out['total_bearish_patterns']

    out['hour_of_day'] = hour_of_day
    out['day_of_week'] = day_of_week
    out['is_london_session'] = is_london_session
    out['is_ny_session'] = is_ny_session
    out['is_overlap_session'] = is_overlap_session
    out['spread_proxy'] = spread_proxy
    out['price_efficiency'] = price_eff
    out['gap_up'] = gap_up
    out['gap_down'] = gap_down
    out['direction_flag'] = direction_flag

    return out


def load_bars(symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    q = "SELECT symbol, time, open, high, low, close, volume FROM bars WHERE symbol=? ORDER BY time ASC"
    df = pd.read_sql_query(q, con, params=(symbol,))
    con.close()
    return df


def build_target_mfe(df: pd.DataFrame, horizon: int = N_MFE) -> pd.Series:
    # MFE over next horizon: max(high[t+1..t+h]) - close[t] (in pips)
    high = df['high']
    future_max = high[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    mfe = future_max - df['close']
    pips = mfe * 10000.0
    # Set last horizon rows to NaN
    pips.iloc[-horizon:] = np.nan
    return pips


def main():
    # Load feature names and scaler
    comps = pickle.load(open(MODEL_COMPONENTS, 'rb'))
    feat_names = comps.get('feature_names') or comps.get('features')
    scaler = comps.get('scaler')
    if not feat_names:
        # Try reading from analysis dump
        if FEATURE_NAMES_JSON.exists():
            feat_names = json.loads(FEATURE_NAMES_JSON.read_text())
        else:
            raise SystemExit('feature_names not found in model_components or analysis dump.')

    # Load bars and compute features/target
    bars = load_bars(SYMBOL)
    if bars.empty:
        raise SystemExit(f'No bars found for {SYMBOL} in {DB_PATH}')

    feats = compute_features(bars)
    # Align to feat_names, fill missing with 0
    for col in feat_names:
        if col not in feats.columns:
            feats[col] = 0
    feats = feats[feat_names]

    # Build target
    y = build_target_mfe(bars, N_MFE)

    # Drop NaNs
    data = pd.concat([bars[[TIME_COL]], feats, y.rename('y')], axis=1).dropna()

    # Time-based split
    times = pd.to_datetime(data[TIME_COL])
    cutoff = times.quantile(1 - VALIDATION_FRACTION)
    train_mask = times < cutoff
    val_mask = times >= cutoff

    X_train = data.loc[train_mask, feat_names].copy()
    y_train = data.loc[train_mask, 'y'].astype(float).values
    X_val = data.loc[val_mask, feat_names].copy()
    y_val = data.loc[val_mask, 'y'].astype(float).values

    # Scale if scaler present
    if scaler is not None:
        try:
            X_train_scaled = scaler.transform(X_train.values)
            X_val_scaled = scaler.transform(X_val.values)
        except Exception:
            # Fit a new scaler if mismatch
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X_train.values)
            X_train_scaled = sc.transform(X_train.values)
            X_val_scaled = sc.transform(X_val.values)
            scaler = sc
    else:
        X_train_scaled = X_train.values
        X_val_scaled = X_val.values

    # Save splits
    out = {
        'feature_names': feat_names,
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val
    }
    with open(SPLIT_DIR / 'splits.pkl', 'wb') as f:
        pickle.dump(out, f)

    # Also save as .npy for large arrays
    np.save(SPLIT_DIR / 'X_train.npy', X_train_scaled)
    np.save(SPLIT_DIR / 'y_train.npy', y_train)
    np.save(SPLIT_DIR / 'X_val.npy', X_val_scaled)
    np.save(SPLIT_DIR / 'y_val.npy', y_val)

    print('Saved splits to', SPLIT_DIR)
    print('Train size:', X_train.shape, 'Val size:', X_val.shape)

if __name__ == '__main__':
    main()

