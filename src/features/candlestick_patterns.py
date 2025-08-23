"""
Candlestick pattern detection for M5 Multi-Symbol Trend Bot
Implements 20 specific bullish and bearish patterns with enhanced logic.
"""

import pandas as pd
import numpy as np

def _df_has_ohlc(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in ["open", "high", "low", "close"])


# --- Core pattern functions (adapted to columns: open, high, low, close) ---

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    return (
        (df["close"] > df["open"]) &
        (prev["close"] < prev["open"]) &
        (df["close"] >= prev["open"]) &
        (df["open"] <= prev["close"])
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    return (
        (df["close"] < df["open"]) &
        (prev["close"] > prev["open"]) &
        (df["close"] <= prev["open"]) &
        (df["open"] >= prev["close"])
    )


def marubozu_bull(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] > df["open"]) &
        (df["high"] == df["close"]) &
        (df["low"] == df["open"])
    )


def marubozu_bear(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] < df["open"]) &
        (df["high"] == df["open"]) &
        (df["low"] == df["close"])
    )


def three_candles_bull(df: pd.DataFrame, body_threshold: float = 0.001) -> pd.Series:
    cond_body = (
        (df["close"] - df["open"] > body_threshold) &
        (df["close"].shift(1) - df["open"].shift(1) > body_threshold) &
        (df["close"].shift(2) - df["open"].shift(2) > body_threshold)
    )
    cond_ascending = (
        (df["close"] > df["close"].shift(1)) &
        (df["close"].shift(1) > df["close"].shift(2)) &
        (df["close"].shift(2) > df["close"].shift(3))
    )
    return cond_body & cond_ascending


def three_candles_bear(df: pd.DataFrame, body_threshold: float = 0.001) -> pd.Series:
    cond_body = (
        (df["open"] - df["close"] > body_threshold) &
        (df["open"].shift(1) - df["close"].shift(1) > body_threshold) &
        (df["open"].shift(2) - df["close"].shift(2) > body_threshold)
    )
    cond_descending = (
        (df["close"] < df["close"].shift(1)) &
        (df["close"].shift(1) < df["close"].shift(2)) &
        (df["close"].shift(2) < df["close"].shift(3))
    )
    return cond_body & cond_descending


# --- ATR for patterns ---

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def three_candles_bull(df: pd.DataFrame) -> pd.Series:
    """Three White Soldiers"""
    c1_close, c2_close, c3_close = df['close'].shift(2), df['close'].shift(1), df['close']
    c1_open, c2_open, c3_open = df['open'].shift(2), df['open'].shift(1), df['open']
    return (c1_close > c1_open) & (c2_close > c2_open) & (c3_close > c3_open) & \
           (c2_open > c1_open) & (c2_open < c1_close) & \
           (c3_open > c2_open) & (c3_open < c2_close)

def three_candles_bear(df: pd.DataFrame) -> pd.Series:
    """Three Black Crows"""
    c1_close, c2_close, c3_close = df['close'].shift(2), df['close'].shift(1), df['close']
    c1_open, c2_open, c3_open = df['open'].shift(2), df['open'].shift(1), df['open']
    return (c1_close < c1_open) & (c2_close < c2_open) & (c3_close < c3_open) & \
           (c2_open < c1_open) & (c2_open > c1_close) & \
           (c3_open < c2_open) & (c3_open > c2_close)

def double_trouble_bull(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    atr_vals = calculate_atr(df, period=atr_period)

    cond_bullish = (df["close"] > df["open"]) & (df["close"].shift(1) > df["open"].shift(1))
    cond_higher_close = df["close"] > df["close"].shift(1)
    cond_wide_range = (df["high"] - df["low"]) > (2 * atr_vals)
    cond_larger_body = (df["close"] - df["open"]) > (df["close"].shift(1) - df["open"].shift(1))

    return cond_bullish & cond_higher_close & cond_wide_range & cond_larger_body


def double_trouble_bear(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    atr_vals = calculate_atr(df, period=atr_period)

    cond_bearish = (df["close"] < df["open"]) & (df["close"].shift(1) < df["open"].shift(1))
    cond_lower_close = df["close"] < df["close"].shift(1)
    cond_wide_range = (df["high"] - df["low"]) > (2 * atr_vals)
    cond_larger_body = (df["open"] - df["close"]) > (df["open"].shift(1) - df["close"].shift(1))

    return cond_bearish & cond_lower_close & cond_wide_range & cond_larger_body


# --- Tasuki ---

def tasuki_bull(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] < df["open"]) &
        (df["close"] < df["open"].shift(1)) &
        (df["close"] > df["close"].shift(2)) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["open"].shift(1) > df["close"].shift(2)) &
        (df["close"].shift(2) > df["open"].shift(2))
    )


def tasuki_bear(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] > df["open"]) &
        (df["close"] > df["open"].shift(1)) &
        (df["close"] < df["close"].shift(2)) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["open"].shift(1) < df["close"].shift(2)) &
        (df["close"].shift(2) < df["open"].shift(2))
    )


# --- Hikkake ---

def hikkake_bull(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] > df["high"].shift(3)) &
        (df["close"] > df["close"].shift(4)) &
        (df["low"].shift(1) < df["open"]) &
        (df["close"].shift(1) < df["close"]) &
        (df["high"].shift(1) <= df["high"].shift(3)) &
        (df["low"].shift(2) < df["open"]) &
        (df["close"].shift(2) < df["close"]) &
        (df["high"].shift(2) <= df["high"].shift(3)) &
        (df["high"].shift(3) < df["high"].shift(4)) &
        (df["low"].shift(3) > df["low"].shift(4)) &
        (df["close"].shift(4) > df["open"].shift(4))
    )


def hikkake_bear(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] < df["low"].shift(3)) &
        (df["close"] < df["close"].shift(4)) &
        (df["high"].shift(1) > df["open"]) &
        (df["close"].shift(1) > df["close"]) &
        (df["low"].shift(1) >= df["low"].shift(3)) &
        (df["high"].shift(2) > df["open"]) &
        (df["close"].shift(2) > df["close"]) &
        (df["low"].shift(2) >= df["low"].shift(3)) &
        (df["low"].shift(3) > df["low"].shift(4)) &
        (df["high"].shift(3) < df["high"].shift(4)) &
        (df["close"].shift(4) < df["open"].shift(4))
    )


# --- Quintuplets ---

def quintuplets_bull(df: pd.DataFrame, body: float = 0.0) -> pd.Series:
    conds = []
    for i in range(5):
        c = (
            (df["close"].shift(i) > df["open"].shift(i)) &
            (df["close"].shift(i) - df["open"].shift(i) < body) &
            (df["close"].shift(i) > df["close"].shift(i+1) if i < 4 else True)
        )
        conds.append(c)
    return pd.Series(np.logical_and.reduce(conds), index=df.index)


def quintuplets_bear(df: pd.DataFrame, body: float = 0.0) -> pd.Series:
    conds = []
    for i in range(5):
        c = (
            (df["close"].shift(i) < df["open"].shift(i)) &
            (df["open"].shift(i) - df["close"].shift(i) < body) &
            (df["close"].shift(i) < df["close"].shift(i+1) if i < 4 else True)
        )
        conds.append(c)
    return pd.Series(np.logical_and.reduce(conds), index=df.index)


# --- Bottle ---

def bottle_bull(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] > df["open"]) &
        (df["open"] == df["low"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["open"] < df["close"].shift(1))
    )


def bottle_bear(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] < df["open"]) &
        (df["open"] == df["high"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["open"] > df["close"].shift(1))
    )


# --- Slingshot ---

def slingshot_bull(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] > df["high"].shift(1)) &
        (df["close"] > df["high"].shift(2)) &
        (df["low"] <= df["high"].shift(3)) &
        (df["close"] > df["open"]) &
        (df["close"].shift(1) >= df["high"].shift(3)) &
        (df["close"].shift(2) > df["open"].shift(2)) &
        (df["close"].shift(2) > df["high"].shift(3)) &
        (df["high"].shift(1) <= df["high"].shift(2))
    )


def slingshot_bear(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] < df["low"].shift(1)) &
        (df["close"] < df["low"].shift(2)) &
        (df["high"] >= df["low"].shift(3)) &
        (df["close"] < df["open"]) &
        (df["high"].shift(1) <= df["high"].shift(3)) &
        (df["close"].shift(2) < df["open"].shift(2)) &
        (df["close"].shift(2) < df["low"].shift(3)) &
        (df["low"].shift(1) >= df["low"].shift(2))
    )


# --- H Pattern ---

def h_pattern_bull(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] > df["open"]) &
        (df["close"] > df["close"].shift(1)) &
        (df["low"] > df["low"].shift(1)) &
        (df["close"].shift(1) == df["open"].shift(1)) &
        (df["close"].shift(2) > df["open"].shift(2)) &
        (df["high"].shift(2) < df["high"].shift(1))
    )


def h_pattern_bear(df: pd.DataFrame) -> pd.Series:
    return (
        (df["close"] < df["open"]) &
        (df["close"] < df["close"].shift(1)) &
        (df["low"] < df["low"].shift(1)) &
        (df["close"].shift(1) == df["open"].shift(1)) &
        (df["close"].shift(2) < df["open"].shift(2)) &
        (df["low"].shift(2) > df["low"].shift(1))
    )


def extract_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all candlestick patterns to a DataFrame of 0/1 ints (columns per pattern)."""
    assert _df_has_ohlc(df), "DataFrame must have open, high, low, close columns"
    patterns = {
        "bullish_engulfing": bullish_engulfing(df),
        "bearish_engulfing": bearish_engulfing(df),
        "marubozu_bull": marubozu_bull(df),
        "marubozu_bear": marubozu_bear(df),
        "three_candles_bull": three_candles_bull(df),
        "three_candles_bear": three_candles_bear(df),
        "double_trouble_bull": double_trouble_bull(df),
        "double_trouble_bear": double_trouble_bear(df),
        "tasuki_bull": tasuki_bull(df),
        "tasuki_bear": tasuki_bear(df),
        "hikkake_bull": hikkake_bull(df),
        "hikkake_bear": hikkake_bear(df),
        "quintuplets_bull": quintuplets_bull(df),
        "quintuplets_bear": quintuplets_bear(df),
        "bottle_bull": bottle_bull(df),
        "bottle_bear": bottle_bear(df),
        "slingshot_bull": slingshot_bull(df),
        "slingshot_bear": slingshot_bear(df),
        "h_pattern_bull": h_pattern_bull(df),
        "h_pattern_bear": h_pattern_bear(df),
    }
    out = pd.DataFrame(patterns, index=df.index)
    out = out.fillna(False).astype(int)
    return out

