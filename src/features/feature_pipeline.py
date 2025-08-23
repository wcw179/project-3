"""
Feature engineering pipeline for M5 Multi-Symbol Trend Bot
Orchestrates LSTM and XGB feature generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json

from src.features.technical_indicators import TechnicalIndicators
from src.features.hierarchical_extremes import HierarchicalExtremes
from src.data.database import TradingDatabase

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Main feature engineering pipeline"""
    
    def __init__(self):
        # This component should not be aware of the database.
        # It is the responsibility of the calling script to handle data persistence.
        self.db = None
        self.indicators = TechnicalIndicators()
        self.he = HierarchicalExtremes(levels=3, atr_lookback=1440)
        
    def generate_lstm_features(self, ohlc: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate features for LSTM model (sequence inputs, no HE)"""
        lstm_features = pd.DataFrame(index=ohlc.index)
        
        # EMA states (cross, separation, slope)
        ema_states = self.indicators.calculate_ema_states(ohlc['close'])
        lstm_features = pd.concat([lstm_features, ema_states], axis=1)
        
        # Black swan patterns (binary flags and embeddings)
        patterns = self.indicators.detect_black_swan_patterns(ohlc)
        pattern_embeddings = self.indicators.create_pattern_embeddings(patterns)
        
        # Add binary pattern flags
        pattern_cols = [col for col in patterns.columns if col not in ['black_swan_score']]
        lstm_features = pd.concat([lstm_features, patterns[pattern_cols]], axis=1)
        
        # Add pattern embeddings
        lstm_features = pd.concat([lstm_features, pattern_embeddings], axis=1)
        
        # Volatility/ATR stats
        vol_stats = self.indicators.calculate_volatility_stats(ohlc)
        lstm_features = pd.concat([lstm_features, vol_stats], axis=1)
        
        # Additional market stats for context
        market_stats = self.indicators.calculate_market_stats(ohlc)
        
        # Select relevant stats for LSTM (avoid too many features)
        lstm_market_cols = [
            'return_mean_5', 'return_std_5', 'return_mean_20', 'return_std_20',
            'momentum_5', 'momentum_10', 'rsi', 'bb_position', 'bb_width',
            'macd_hist', 'stoch_k', 'adx'
        ]
        
        for col in lstm_market_cols:
            if col in market_stats.columns:
                lstm_features[col] = market_stats[col]
        
        # Normalize features for LSTM
        lstm_features = self.indicators.normalize_features(lstm_features, method='zscore')
        
        # Fill NaN values
        lstm_features = lstm_features.ffill().fillna(0)
        
        logger.info(f"Generated {len(lstm_features.columns)} LSTM features for {symbol}")
        return lstm_features
    
    def generate_xgb_features(self, ohlc: pd.DataFrame, symbol: str, 
                             lstm_probabilities: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate features for XGBoost model (execution features)"""
        xgb_features = pd.DataFrame(index=ohlc.index)
        
        # LSTM probabilities (if available)
        if lstm_probabilities is not None:
            xgb_features = pd.concat([xgb_features, lstm_probabilities], axis=1)
        else:
            # Placeholder columns
            xgb_features['lstm_p_up'] = 0.33
            xgb_features['lstm_p_down'] = 0.33
            xgb_features['lstm_p_neutral'] = 0.34
        
        # EMA states & pattern flags
        ema_states = self.indicators.calculate_ema_states(ohlc['close'])
        xgb_features = pd.concat([xgb_features, ema_states], axis=1)
        
        patterns = self.indicators.detect_black_swan_patterns(ohlc)
        pattern_flags = patterns[[col for col in patterns.columns if col != 'black_swan_score']]
        xgb_features = pd.concat([xgb_features, pattern_flags], axis=1)
        
        # Market stats (comprehensive)
        market_stats = self.indicators.calculate_market_stats(ohlc)
        xgb_features = pd.concat([xgb_features, market_stats], axis=1)
        
        # Volatility stats
        vol_stats = self.indicators.calculate_volatility_stats(ohlc)
        xgb_features = pd.concat([xgb_features, vol_stats], axis=1)
        
        # Spread proxy
        spread_proxy = self.indicators.calculate_spread_proxy(ohlc)
        xgb_features['spread_proxy'] = spread_proxy
        
        # HierarchicalExtremes features (for execution context, not fed to LSTM)
        hierarchical_levels = self.he.identify_hierarchical_levels(ohlc)
        
        # Extract HE features for each bar
        he_features_list = []
        current_atr = vol_stats['atr'].ffill()
        
        for timestamp in ohlc.index:
            # Ensure current_price is a scalar, handling duplicate timestamps by taking the first value
            current_price_val = ohlc.loc[timestamp, 'close']
            if isinstance(current_price_val, pd.Series):
                current_price_val = current_price_val.iloc[0]

            current_atr_val = current_atr.loc[timestamp] if timestamp in current_atr.index else None
            if isinstance(current_atr_val, pd.Series):
                current_atr_val = current_atr_val.iloc[0]

            he_features = self.he.get_level_features(hierarchical_levels, current_price_val, current_atr_val)
            he_features['timestamp'] = timestamp
            he_features_list.append(he_features)
        
        he_df = pd.DataFrame(he_features_list).set_index('timestamp')
        xgb_features = pd.concat([xgb_features, he_df], axis=1)
        
        # Additional execution-specific features
        xgb_features['price_vs_vwap'] = self._calculate_vwap_ratio(ohlc)
        xgb_features['volume_profile'] = self._calculate_volume_profile(ohlc)
        xgb_features['time_of_day'] = ohlc.index.hour + ohlc.index.minute / 60
        xgb_features['day_of_week'] = ohlc.index.dayofweek
        
        # Fill NaN values
        xgb_features = xgb_features.ffill().fillna(0)
        
        logger.info(f"Generated {len(xgb_features.columns)} XGB features for {symbol}")
        return xgb_features
    
    def _calculate_vwap_ratio(self, ohlc: pd.DataFrame) -> pd.Series:
        """Calculate price vs VWAP ratio"""
        if 'volume' not in ohlc.columns or ohlc['volume'].sum() == 0:
            return pd.Series(1.0, index=ohlc.index)
        
        typical_price = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
        vwap = (typical_price * ohlc['volume']).rolling(20).sum() / ohlc['volume'].rolling(20).sum()
        
        return ohlc['close'] / vwap
    
    def _calculate_volume_profile(self, ohlc: pd.DataFrame) -> pd.Series:
        """Calculate volume profile indicator"""
        if 'volume' not in ohlc.columns:
            return pd.Series(0.5, index=ohlc.index)
        
        vol_ma = ohlc['volume'].rolling(20).mean()
        return ohlc['volume'] / vol_ma
    
