"""
Technical indicators for M5 Multi-Symbol Trend Bot
Implements EMA, ATR, volatility measures, and pattern detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import talib

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicator calculations"""
    
    def __init__(self):
        pass
    
    def calculate_ema(self, prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Calculate EMA for multiple periods"""
        emas = pd.DataFrame(index=prices.index)
        
        for period in periods:
            emas[f'ema_{period}'] = talib.EMA(prices.values, timeperiod=period)
        
        return emas
    
    def calculate_ema_states(self, prices: pd.Series, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate EMA states: cross, separation, slope"""
        emas = self.calculate_ema(prices, periods)
        states = pd.DataFrame(index=prices.index)
        
        # EMA crosses
        if len(periods) >= 2:
            states['ema_cross_20_50'] = np.where(
                emas[f'ema_{periods[0]}'] > emas[f'ema_{periods[1]}'], 1, -1
            )
        
        if len(periods) >= 3:
            states['ema_cross_50_200'] = np.where(
                emas[f'ema_{periods[1]}'] > emas[f'ema_{periods[2]}'], 1, -1
            )
            
            states['ema_cross_20_200'] = np.where(
                emas[f'ema_{periods[0]}'] > emas[f'ema_{periods[2]}'], 1, -1
            )
        
        # EMA separations (normalized)
        for i, period in enumerate(periods[:-1]):
            next_period = periods[i + 1]
            separation = (emas[f'ema_{period}'] - emas[f'ema_{next_period}']) / emas[f'ema_{next_period}']
            states[f'ema_separation_{period}_{next_period}'] = separation
        
        # EMA slopes (rate of change)
        for period in periods:
            ema_col = f'ema_{period}'
            slope = emas[ema_col].pct_change(5)  # 5-bar slope
            states[f'ema_slope_{period}'] = slope
        
        # Overall trend alignment
        if len(periods) >= 3:
            states['ema_alignment'] = np.where(
                (emas[f'ema_{periods[0]}'] > emas[f'ema_{periods[1]}']) & 
                (emas[f'ema_{periods[1]}'] > emas[f'ema_{periods[2]}']) &
                (prices > emas[f'ema_{periods[0]}']), 1,  # Strong uptrend
                np.where(
                    (emas[f'ema_{periods[0]}'] < emas[f'ema_{periods[1]}']) & 
                    (emas[f'ema_{periods[1]}'] < emas[f'ema_{periods[2]}']) &
                    (prices < emas[f'ema_{periods[0]}']), -1,  # Strong downtrend
                    0  # Mixed/sideways
                )
            )
        
        return states
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        return pd.Series(
            talib.ATR(high.values, low.values, close.values, timeperiod=period),
            index=close.index,
            name=f'atr_{period}'
        )
    
    def calculate_volatility_stats(self, ohlc: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """Calculate comprehensive volatility statistics"""
        vol_stats = pd.DataFrame(index=ohlc.index)
        
        # ATR
        vol_stats['atr'] = self.calculate_atr(ohlc['high'], ohlc['low'], ohlc['close'], atr_period)
        
        # True Range
        tr = np.maximum(
            ohlc['high'] - ohlc['low'],
            np.maximum(
                np.abs(ohlc['high'] - ohlc['close'].shift(1)),
                np.abs(ohlc['low'] - ohlc['close'].shift(1))
            )
        )
        vol_stats['true_range'] = tr
        
        # Realized volatility (rolling std of returns)
        returns = ohlc['close'].pct_change()
        vol_stats['realized_vol_5'] = returns.rolling(5).std()
        vol_stats['realized_vol_20'] = returns.rolling(20).std()
        vol_stats['realized_vol_60'] = returns.rolling(60).std()
        
        # High-Low range as percentage of close
        vol_stats['hl_range_pct'] = (ohlc['high'] - ohlc['low']) / ohlc['close']
        
        # Intraday range momentum
        vol_stats['range_momentum'] = vol_stats['hl_range_pct'].rolling(5).mean()
        
        # ATR percentile (where current ATR stands in recent history)
        vol_stats['atr_percentile'] = vol_stats['atr'].rolling(100).rank(pct=True)
        
        # Volatility regime (low/medium/high based on ATR percentiles)
        vol_stats['vol_regime'] = np.where(
            vol_stats['atr_percentile'] > 0.8, 2,  # High vol
            np.where(vol_stats['atr_percentile'] < 0.2, 0, 1)  # Low vol, else medium
        )
        
        return vol_stats
    
    def detect_black_swan_patterns(self, ohlc: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Detect 20 specific candlestick patterns in 20-candle sequences"""
        from .candlestick_patterns import (
            bullish_engulfing, bearish_engulfing, marubozu_bull, marubozu_bear,
            three_candles_bull, three_candles_bear, double_trouble_bull, double_trouble_bear,
            tasuki_bull, tasuki_bear, hikkake_bull, hikkake_bear,
            quintuplets_bull, quintuplets_bear, bottle_bull, bottle_bear,
            slingshot_bull, slingshot_bear, h_pattern_bull, h_pattern_bear
        )

        patterns = pd.DataFrame(index=ohlc.index)

        # 20 specific candlestick patterns
        patterns['bullish_engulfing'] = bullish_engulfing(ohlc)
        patterns['bearish_engulfing'] = bearish_engulfing(ohlc)
        patterns['marubozu_bull'] = marubozu_bull(ohlc)
        patterns['marubozu_bear'] = marubozu_bear(ohlc)
        patterns['three_candles_bull'] = three_candles_bull(ohlc)
        patterns['three_candles_bear'] = three_candles_bear(ohlc)
        patterns['double_trouble_bull'] = double_trouble_bull(ohlc)
        patterns['double_trouble_bear'] = double_trouble_bear(ohlc)
        patterns['tasuki_bull'] = tasuki_bull(ohlc)
        patterns['tasuki_bear'] = tasuki_bear(ohlc)
        patterns['hikkake_bull'] = hikkake_bull(ohlc)
        patterns['hikkake_bear'] = hikkake_bear(ohlc)
        patterns['quintuplets_bull'] = quintuplets_bull(ohlc)
        patterns['quintuplets_bear'] = quintuplets_bear(ohlc)
        patterns['bottle_bull'] = bottle_bull(ohlc)
        patterns['bottle_bear'] = bottle_bear(ohlc)
        patterns['slingshot_bull'] = slingshot_bull(ohlc)
        patterns['slingshot_bear'] = slingshot_bear(ohlc)
        patterns['h_pattern_bull'] = h_pattern_bull(ohlc)
        patterns['h_pattern_bear'] = h_pattern_bear(ohlc)

        # Convert boolean patterns to int
        pattern_cols = [col for col in patterns.columns]
        for col in pattern_cols:
            patterns[col] = patterns[col].astype(int)

        # Composite black swan score (0-1)
        patterns['black_swan_score'] = patterns[pattern_cols].mean(axis=1)

        # Binary black swan flag (threshold at 0.3)
        patterns['black_swan_flag'] = patterns['black_swan_score'] > 0.3

        return patterns
    
    def create_pattern_embeddings(self, patterns: pd.DataFrame, embedding_dim: int = 8) -> pd.DataFrame:
        """Create embeddings for black swan patterns"""
        # Simple approach: use PCA-like dimensionality reduction
        from sklearn.decomposition import PCA
        
        pattern_cols = [col for col in patterns.columns if col not in ['black_swan_score', 'black_swan_flag']]
        pattern_data = patterns[pattern_cols].fillna(0).astype(float)
        
        if len(pattern_data) < embedding_dim:
            # Not enough data for embeddings
            embeddings = pd.DataFrame(
                np.zeros((len(patterns), embedding_dim)),
                index=patterns.index,
                columns=[f'pattern_emb_{i}' for i in range(embedding_dim)]
            )
        else:
            # Fit PCA
            pca = PCA(n_components=embedding_dim)
            embeddings_array = pca.fit_transform(pattern_data)
            
            embeddings = pd.DataFrame(
                embeddings_array,
                index=patterns.index,
                columns=[f'pattern_emb_{i}' for i in range(embedding_dim)]
            )
        
        return embeddings
    
    def calculate_market_stats(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Calculate market statistics for XGB features"""
        stats = pd.DataFrame(index=ohlc.index)
        
        # Price-based stats
        returns = ohlc['close'].pct_change()
        
        # Rolling statistics
        for window in [5, 20, 60]:
            stats[f'return_mean_{window}'] = returns.rolling(window).mean()
            stats[f'return_std_{window}'] = returns.rolling(window).std()
            stats[f'return_skew_{window}'] = returns.rolling(window).skew()
            stats[f'return_kurt_{window}'] = returns.rolling(window).kurt()
        
        # Price momentum
        for period in [5, 10, 20]:
            stats[f'momentum_{period}'] = ohlc['close'].pct_change(period)
        
        # RSI
        stats['rsi'] = pd.Series(
            talib.RSI(ohlc['close'].values, timeperiod=14),
            index=ohlc.index
        )
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            ohlc['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        stats['bb_position'] = (ohlc['close'] - bb_lower) / (bb_upper - bb_lower)
        stats['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(ohlc['close'].values)
        stats['macd'] = pd.Series(macd, index=ohlc.index)
        stats['macd_signal'] = pd.Series(macd_signal, index=ohlc.index)
        stats['macd_hist'] = pd.Series(macd_hist, index=ohlc.index)
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            ohlc['high'].values, ohlc['low'].values, ohlc['close'].values
        )
        stats['stoch_k'] = pd.Series(slowk, index=ohlc.index)
        stats['stoch_d'] = pd.Series(slowd, index=ohlc.index)
        
        # Williams %R
        stats['williams_r'] = pd.Series(
            talib.WILLR(ohlc['high'].values, ohlc['low'].values, ohlc['close'].values),
            index=ohlc.index
        )
        
        # Commodity Channel Index
        stats['cci'] = pd.Series(
            talib.CCI(ohlc['high'].values, ohlc['low'].values, ohlc['close'].values),
            index=ohlc.index
        )
        
        # Average Directional Index
        stats['adx'] = pd.Series(
            talib.ADX(ohlc['high'].values, ohlc['low'].values, ohlc['close'].values),
            index=ohlc.index
        )
        
        return stats
    
    def calculate_spread_proxy(self, ohlc: pd.DataFrame) -> pd.Series:
        """Calculate spread proxy from OHLC data"""
        # Simple spread proxy: (high - low) / close
        spread_proxy = (ohlc['high'] - ohlc['low']) / ohlc['close']
        
        # Smooth it
        spread_proxy_smooth = spread_proxy.rolling(5).mean()
        
        return spread_proxy_smooth.fillna(spread_proxy)
    
    def normalize_features(self, features: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normalize features"""
        if method == 'zscore':
            return features.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
        elif method == 'minmax':
            return features.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x)
        elif method == 'robust':
            return features.apply(lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)) 
                                 if x.quantile(0.75) > x.quantile(0.25) else x)
        else:
            return features
    
    def create_lstm_sequence_features(self, features: pd.DataFrame, sequence_length: int = 60) -> np.ndarray:
        """Create sequences for LSTM input"""
        if len(features) < sequence_length:
            return np.array([])
        
        sequences = []
        for i in range(sequence_length, len(features)):
            sequence = features.iloc[i-sequence_length:i].values
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def get_feature_importance_scores(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Calculate feature importance using correlation and mutual information"""
        from sklearn.feature_selection import mutual_info_regression
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features[valid_idx]
        target_clean = target[valid_idx]
        
        if len(features_clean) == 0:
            return pd.Series(index=features.columns, dtype=float)
        
        # Correlation-based importance
        correlations = features_clean.corrwith(target_clean).abs()
        
        # Mutual information-based importance
        try:
            mi_scores = mutual_info_regression(features_clean.fillna(0), target_clean)
            mi_series = pd.Series(mi_scores, index=features.columns)
        except:
            mi_series = pd.Series(0, index=features.columns)
        
        # Combined importance (average of normalized scores)
        corr_norm = correlations / correlations.max() if correlations.max() > 0 else correlations
        mi_norm = mi_series / mi_series.max() if mi_series.max() > 0 else mi_series
        
        importance = (corr_norm + mi_norm) / 2
        
        return importance.fillna(0)
