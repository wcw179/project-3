"""
Enhanced Feature Pipeline for Black-Swan Hunter Trading Bot
Generates XGB tabular features and LSTM sequential features with ATR normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.features.technical_indicators import TechnicalIndicators
from src.features.candlestick_patterns import extract_candlestick_patterns

logger = logging.getLogger(__name__)

class BlackSwanFeaturePipeline:
    """Enhanced feature pipeline for dual-model Black Swan detection"""
    
    def __init__(self):
        self.ti = TechnicalIndicators()
        self.sequence_length = 60  # 5 hours of M5 data
        self.forecast_horizon = 288  # ~8.3 hours ahead
        
    def calculate_atr_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR and normalize price-based features for R-multiple consistency"""
        df = df.copy()
        
        # Calculate ATR(14) for normalization
        atr14 = self.ti.calculate_atr(df['high'], df['low'], df['close'], period=14)
        df['atr14'] = atr14
        
        # ATR-normalized OHLC for LSTM
        df['open_atr_norm'] = (df['open'] - df['close'].shift(1)) / atr14
        df['high_atr_norm'] = (df['high'] - df['close']) / atr14
        df['low_atr_norm'] = (df['low'] - df['close']) / atr14
        df['close_atr_norm'] = (df['close'] - df['close'].shift(1)) / atr14
        
        # Log returns for LSTM
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # HL range percentage
        df['hl_range_pct'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def generate_xgb_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate enhanced tabular features for XGBoost MFE regressor (~40+ features)"""
        df = self.calculate_atr_normalized_features(df)
        features = pd.DataFrame(index=df.index)

        # Trend Features - EMA ratios and slopes
        emas = self.ti.calculate_ema(df['close'], [20, 50, 200])
        features['ema20_ratio'] = df['close'] / emas['ema_20']
        features['ema50_ratio'] = df['close'] / emas['ema_50']
        features['ema200_ratio'] = df['close'] / emas['ema_200']
        features['ema20_slope'] = (emas['ema_20'] - emas['ema_20'].shift(5)) / (5 * df['atr14'])
        features['ema50_slope'] = (emas['ema_50'] - emas['ema_50'].shift(5)) / (5 * df['atr14'])

        # EMA cross signals
        features['ema20_above_50'] = (emas['ema_20'] > emas['ema_50']).astype(int)
        features['ema50_above_200'] = (emas['ema_50'] > emas['ema_200']).astype(int)
        features['ema_bullish_alignment'] = ((emas['ema_20'] > emas['ema_50']) &
                                           (emas['ema_50'] > emas['ema_200'])).astype(int)

        # Momentum Features using talib
        import talib

        # Convert to numpy arrays for talib
        close_arr = np.array(df['close'].values, dtype=np.float64)
        high_arr = np.array(df['high'].values, dtype=np.float64)
        low_arr = np.array(df['low'].values, dtype=np.float64)
        open_arr = np.array(df['open'].values, dtype=np.float64)

        features['rsi14'] = talib.RSI(close_arr, timeperiod=14)
        features['rsi_oversold'] = (features['rsi14'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi14'] > 70).astype(int)

        # MACD
        macd_line, macd_signal, macd_hist = talib.MACD(close_arr)
        features['macd_signal'] = pd.Series(macd_signal, index=df.index) / df['atr14']
        features['macd_histogram'] = pd.Series(macd_hist, index=df.index) / df['atr14']
        features['macd_bullish'] = (macd_line > macd_signal).astype(int)

        # Stochastic
        slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr)
        features['stoch_k'] = pd.Series(slowk, index=df.index)
        features['stoch_d'] = pd.Series(slowd, index=df.index)
        features['stoch_oversold'] = (pd.Series(slowk, index=df.index) < 20).astype(int)
        features['stoch_overbought'] = (pd.Series(slowk, index=df.index) > 80).astype(int)

        # Williams %R
        features['williams_r'] = pd.Series(talib.WILLR(high_arr, low_arr, close_arr), index=df.index)

        # Volatility Features
        features['atr_close_ratio'] = df['atr14'] / df['close']

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_arr, timeperiod=20)
        bb_upper = pd.Series(bb_upper, index=df.index)
        bb_middle = pd.Series(bb_middle, index=df.index)
        bb_lower = pd.Series(bb_lower, index=df.index)

        features['bb_width_pct'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_squeeze'] = (features['bb_width_pct'] < features['bb_width_pct'].rolling(20).quantile(0.2)).astype(int)
        features['bb_breakout_up'] = (df['close'] > bb_upper).astype(int)
        features['bb_breakout_down'] = (df['close'] < bb_lower).astype(int)

        # Volume Features (if available)
        if 'volume' in df.columns and not df['volume'].isna().all():
            # Convert volume to numpy array
            volume_arr = np.array(df['volume'].values, dtype=np.float64)

            # VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            features['vwap_distance'] = (df['close'] - vwap) / df['atr14']

            # Volume indicators
            vol_ma20 = df['volume'].rolling(20).mean()
            features['volume_regime'] = (df['volume'] / vol_ma20).fillna(1.0)
            features['volume_spike'] = (df['volume'] > vol_ma20 * 2).astype(int)

            # On Balance Volume
            obv = talib.OBV(close_arr, volume_arr)
            features['obv_slope'] = pd.Series(obv, index=df.index).pct_change(5)
        else:
            features['vwap_distance'] = 0.0
            features['volume_regime'] = 1.0
            features['volume_spike'] = 0
            features['obv_slope'] = 0.0

        # Price Action Features
        features['hl_range_pct'] = df['hl_range_pct']
        features['body_size_pct'] = abs(df['close'] - df['open']) / df['close']
        features['upper_shadow_pct'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        features['lower_shadow_pct'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']

        # Return statistics
        features['returns_5bar_mean'] = df['log_returns'].rolling(5).mean()
        features['returns_5bar_std'] = df['log_returns'].rolling(5).std()
        features['returns_20bar_mean'] = df['log_returns'].rolling(20).mean()
        features['returns_20bar_std'] = df['log_returns'].rolling(20).std()
        features['returns_skew'] = df['log_returns'].rolling(20).skew()
        features['returns_kurt'] = df['log_returns'].rolling(20).kurt()

        # Pattern Recognition Features - Enhanced with 36 patterns
        patterns = extract_candlestick_patterns(df)
        for pattern_name in patterns.columns:
            features[f'pattern_{pattern_name}'] = patterns[pattern_name]

        # Pattern strength indicators
        pattern_cols = [col for col in features.columns if col.startswith('pattern_') or col.startswith('talib_')]
        features['total_bullish_patterns'] = features[[col for col in pattern_cols if 'bull' in col or 'white' in col or 'hammer' in col or 'morning' in col or 'piercing' in col or 'harami' in col]].sum(axis=1)
        features['total_bearish_patterns'] = features[[col for col in pattern_cols if 'bear' in col or 'black' in col or 'shooting' in col or 'evening' in col or 'dark' in col or 'hanging' in col]].sum(axis=1)
        features['pattern_strength'] = features['total_bullish_patterns'] - features['total_bearish_patterns']

        # Context Features - handle datetime index safely
        try:
            # Try to access datetime attributes
            hour_values = pd.to_datetime(df.index).hour
            dow_values = pd.to_datetime(df.index).dayofweek

            features['hour_of_day'] = hour_values
            features['day_of_week'] = dow_values
            features['is_london_session'] = ((hour_values >= 8) & (hour_values < 17)).astype(int)
            features['is_ny_session'] = ((hour_values >= 13) & (hour_values < 22)).astype(int)
            features['is_overlap_session'] = ((hour_values >= 13) & (hour_values < 17)).astype(int)
        except:
            # Fallback for non-datetime index
            features['hour_of_day'] = 12  # Default to noon
            features['day_of_week'] = 2   # Default to Tuesday
            features['is_london_session'] = 1
            features['is_ny_session'] = 1
            features['is_overlap_session'] = 1

        # Spread proxy and market microstructure
        features['spread_proxy'] = df['atr14'] / df['close'] * 10000  # in pips
        features['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])

        # Gap detection
        features['gap_up'] = ((df['open'] > df['close'].shift(1)) &
                             (df['low'] > df['high'].shift(1))).astype(int)
        features['gap_down'] = ((df['open'] < df['close'].shift(1)) &
                               (df['high'] < df['low'].shift(1))).astype(int)

        # Add symbol identifier
        features['symbol'] = symbol

        return features.fillna(0)
    
    def generate_lstm_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate enhanced sequential features for LSTM tail classifier (~15 features)"""
        df = self.calculate_atr_normalized_features(df)
        features = pd.DataFrame(index=df.index)

        # Core sequential features
        features['log_returns'] = df['log_returns']
        features['open_atr_norm'] = df['open_atr_norm']
        features['high_atr_norm'] = df['high_atr_norm']
        features['low_atr_norm'] = df['low_atr_norm']
        features['close_atr_norm'] = df['close_atr_norm']

        # Technical indicators (normalized for LSTM)
        import talib
        close_arr = np.array(df['close'].values, dtype=np.float64)
        high_arr = np.array(df['high'].values, dtype=np.float64)
        low_arr = np.array(df['low'].values, dtype=np.float64)

        features['rsi14'] = talib.RSI(close_arr, timeperiod=14) / 100.0  # Normalize to [0,1]
        features['atr_normalized'] = df['atr14'] / df['close']
        features['hl_range_pct'] = df['hl_range_pct']

        # EMA deviations
        emas = self.ti.calculate_ema(df['close'], [20, 50])
        features['ema20_deviation'] = (df['close'] / emas['ema_20']) - 1
        features['ema50_deviation'] = (df['close'] / emas['ema_50']) - 1

        # Momentum indicators (normalized)
        _, _, macd_hist = talib.MACD(close_arr)
        features['macd_normalized'] = pd.Series(macd_hist, index=df.index) / df['atr14']

        # Stochastic (already 0-100, normalize to 0-1)
        slowk, _ = talib.STOCH(high_arr, low_arr, close_arr)
        features['stoch_k_norm'] = slowk / 100.0

        # Volume feature (normalized)
        if 'volume' in df.columns and not df['volume'].isna().all():
            vol_ma20 = df['volume'].rolling(20).mean()
            features['volume_normalized'] = (df['volume'] / vol_ma20).fillna(1.0)
        else:
            features['volume_normalized'] = 1.0
        
        # Add symbol identifier
        features['symbol'] = symbol
        
        return features.fillna(0)
    
    def prepare_lstm_sequences(self, features: pd.DataFrame, labels: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequential data for LSTM training/inference"""
        if len(features) < self.sequence_length:
            logger.warning(f"Insufficient data for sequences: {len(features)} < {self.sequence_length}")
            return np.array([]), None if labels is None else np.array([])
        
        # Remove symbol column for numerical processing
        feature_cols = [col for col in features.columns if col != 'symbol']
        feature_data = features[feature_cols].values
        
        # Create sequences
        X_sequences = []
        y_sequences = [] if labels is not None else None

        for i in range(self.sequence_length, len(feature_data)):
            X_sequences.append(feature_data[i-self.sequence_length:i])
            if labels is not None and y_sequences is not None:
                y_sequences.append(labels.iloc[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y_sequences is not None else None
        
        return X_sequences, y_sequences
    
    def validate_features(self, features: pd.DataFrame, feature_type: str) -> bool:
        """Validate feature quality and completeness"""
        if features.empty:
            logger.error(f"Empty {feature_type} features DataFrame")
            return False
        
        # Check for excessive NaN values
        nan_pct = features.isna().sum() / len(features)
        problematic_cols = nan_pct[nan_pct > 0.1].index.tolist()
        if problematic_cols:
            logger.warning(f"{feature_type} features with >10% NaN: {problematic_cols}")
        
        # Check for infinite values
        numeric_features = features.select_dtypes(include=['number'])
        inf_cols = []
        for col in numeric_features.columns:
            if np.isinf(numeric_features[col]).any():
                inf_cols.append(col)
        if inf_cols:
            logger.error(f"{feature_type} features with infinite values: {inf_cols}")
            return False

        # Check feature count expectations (updated for massively enhanced features)
        expected_counts = {'xgb': (80, 150), 'lstm': (12, 20)}
        if feature_type in expected_counts:
            min_count, max_count = expected_counts[feature_type]
            actual_count = len([col for col in features.columns if col != 'symbol'])
            if not (min_count <= actual_count <= max_count):
                logger.warning(f"{feature_type} feature count {actual_count} outside expected range [{min_count}, {max_count}]")
        
        logger.info(f"{feature_type} features validation passed: {features.shape}")
        return True

# Legacy compatibility
class FeaturePipeline(BlackSwanFeaturePipeline):
    """Legacy compatibility wrapper"""
    pass
