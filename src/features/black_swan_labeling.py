"""
MFE Labeling and Tail Event Classification for Black-Swan Hunter Trading Bot
Implements regression targets for XGB and multi-class classification for LSTM
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

logger = logging.getLogger(__name__)

class BlackSwanLabeling:
    """MFE labeling and tail event classification for Black Swan detection"""
    
    def __init__(self, forecast_horizon: int = 100, max_mfe_clip: float = 50.0):
        self.ti = TechnicalIndicators()
        self.forecast_horizon = forecast_horizon  # Forward-looking window
        self.max_mfe_clip = max_mfe_clip  # Clip MFE values to handle outliers
        
    def calculate_mfe_labels(self, df: pd.DataFrame, direction: str = 'long') -> pd.Series:
        """
        Calculate Maximum Favorable Excursion (MFE) in risk multiples (R)
        
        Args:
            df: OHLCV DataFrame with ATR14 column
            direction: 'long' or 'short' for trade direction
            
        Returns:
            Series of MFE values in R-multiples
        """
        if 'atr14' not in df.columns:
            atr14 = self.ti.calculate_atr(df['high'], df['low'], df['close'], period=14)
            df = df.copy()
            df['atr14'] = atr14
        
        mfe_values = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df) - self.forecast_horizon):
            current_close = df['close'].iloc[i]
            current_atr = df['atr14'].iloc[i]
            
            if pd.isna(current_atr) or current_atr <= 0:
                mfe_values.iloc[i] = 0.0
                continue
            
            # Look ahead for MFE calculation
            future_window = df.iloc[i+1:i+1+self.forecast_horizon]
            
            if direction == 'long':
                # For long trades: MFE = max(high - entry_close) / ATR
                max_favorable = future_window['high'].max()
                mfe = (max_favorable - current_close) / current_atr
            else:  # short
                # For short trades: MFE = max(entry_close - low) / ATR
                min_favorable = future_window['low'].min()
                mfe = (current_close - min_favorable) / current_atr
            
            # Clip to handle outliers
            mfe_values.iloc[i] = np.clip(mfe, 0, self.max_mfe_clip)
        
        # Fill remaining values with 0
        mfe_values.fillna(0.0, inplace=True)
        
        return mfe_values
    
    def calculate_tail_event_labels(self, mfe_values: pd.Series) -> pd.Series:
        """
        Convert MFE values to multi-class tail event labels
        
        Classes:
        - 0: MFE < 5R (normal moves)
        - 1: 5R ≤ MFE < 10R (moderate tail)
        - 2: 10R ≤ MFE < 20R (strong tail)
        - 3: MFE ≥ 20R (extreme tail)
        """
        tail_labels = pd.Series(index=mfe_values.index, dtype=int)
        
        # Define thresholds
        tail_labels[mfe_values < 5.0] = 0
        tail_labels[(mfe_values >= 5.0) & (mfe_values < 10.0)] = 1
        tail_labels[(mfe_values >= 10.0) & (mfe_values < 20.0)] = 2
        tail_labels[mfe_values >= 20.0] = 3
        
        return tail_labels
    
    def calculate_class_weights(self, tail_labels: pd.Series) -> Dict[int, float]:
        """Calculate class weights for handling severe imbalance in tail events"""
        class_counts = tail_labels.value_counts().sort_index()
        total_samples = len(tail_labels)
        
        # Calculate inverse frequency weights
        class_weights = {}
        for class_id in range(4):
            if class_id in class_counts.index:
                weight = total_samples / (4 * class_counts[class_id])
                class_weights[class_id] = weight
            else:
                class_weights[class_id] = 1.0
        
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        logger.info(f"Class weights: {class_weights}")
        
        return class_weights
    
    def generate_labels_for_symbol(self, df: pd.DataFrame, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Generate both XGB regression labels and LSTM classification labels
        
        Returns:
            Dictionary with 'xgb_long', 'xgb_short', 'lstm_long', 'lstm_short' DataFrames
        """
        # Ensure ATR is calculated
        if 'atr14' not in df.columns:
            atr14 = self.ti.calculate_atr(df['high'], df['low'], df['close'], period=14)
            df = df.copy()
            df['atr14'] = atr14
        
        results = {}
        
        # Generate labels for both directions
        for direction in ['long', 'short']:
            # XGB regression labels (MFE in R-multiples)
            mfe_values = self.calculate_mfe_labels(df, direction)
            
            xgb_labels = pd.DataFrame({
                'symbol': symbol,
                'direction': direction,
                'mfe_target': mfe_values,
                'atr14': df['atr14']
            }, index=df.index)
            
            results[f'xgb_{direction}'] = xgb_labels
            
            # LSTM classification labels (tail events)
            tail_labels = self.calculate_tail_event_labels(mfe_values)
            
            lstm_labels = pd.DataFrame({
                'symbol': symbol,
                'direction': direction,
                'tail_class': tail_labels,
                'mfe_source': mfe_values,
                'atr14': df['atr14']
            }, index=df.index)
            
            results[f'lstm_{direction}'] = lstm_labels
        
        return results
    
    def validate_labels(self, labels_dict: Dict[str, pd.DataFrame]) -> bool:
        """Validate label quality and distribution"""
        validation_passed = True
        
        for label_type, labels_df in labels_dict.items():
            if labels_df.empty:
                logger.error(f"Empty labels DataFrame for {label_type}")
                validation_passed = False
                continue
            
            if 'mfe_target' in labels_df.columns:
                # Validate XGB regression labels
                mfe_values = labels_df['mfe_target']
                
                # Check for reasonable range
                if mfe_values.min() < 0:
                    logger.warning(f"{label_type}: Negative MFE values found")
                
                if mfe_values.max() > self.max_mfe_clip:
                    logger.warning(f"{label_type}: MFE values exceed clip threshold")
                
                # Check distribution
                percentiles = mfe_values.quantile([0.5, 0.9, 0.95, 0.99])
                logger.info(f"{label_type} MFE percentiles: {percentiles.to_dict()}")
                
            elif 'tail_class' in labels_df.columns:
                # Validate LSTM classification labels
                tail_classes = labels_df['tail_class']
                class_dist = tail_classes.value_counts().sort_index()
                
                # Check class balance
                class_0_pct = class_dist.get(0, 0) / len(tail_classes) * 100
                extreme_tail_pct = class_dist.get(3, 0) / len(tail_classes) * 100
                
                logger.info(f"{label_type} class distribution: {class_dist.to_dict()}")
                logger.info(f"{label_type} normal events: {class_0_pct:.1f}%, extreme tails: {extreme_tail_pct:.3f}%")
                
                if extreme_tail_pct < 0.1:
                    logger.warning(f"{label_type}: Very few extreme tail events ({extreme_tail_pct:.3f}%)")
        
        return validation_passed
    
    def create_training_datasets(self, labels_dict: Dict[str, pd.DataFrame], 
                                features_dict: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Combine features and labels into training-ready datasets
        
        Returns:
            Dictionary with (features, labels) tuples for each model type
        """
        datasets = {}
        
        # XGB datasets (combine long and short)
        if 'xgb_long' in labels_dict and 'xgb_short' in labels_dict:
            xgb_labels_combined = pd.concat([
                labels_dict['xgb_long'],
                labels_dict['xgb_short']
            ]).sort_index()
            
            if 'xgb' in features_dict:
                # Align features with labels
                common_idx = features_dict['xgb'].index.intersection(xgb_labels_combined.index)
                aligned_features = features_dict['xgb'].loc[common_idx]
                aligned_labels = xgb_labels_combined.loc[common_idx]['mfe_target']
                
                datasets['xgb'] = (aligned_features, aligned_labels)
        
        # LSTM datasets (combine long and short)
        if 'lstm_long' in labels_dict and 'lstm_short' in labels_dict:
            lstm_labels_combined = pd.concat([
                labels_dict['lstm_long'],
                labels_dict['lstm_short']
            ]).sort_index()
            
            if 'lstm' in features_dict:
                # Align features with labels
                common_idx = features_dict['lstm'].index.intersection(lstm_labels_combined.index)
                aligned_features = features_dict['lstm'].loc[common_idx]
                aligned_labels = lstm_labels_combined.loc[common_idx]['tail_class']
                
                datasets['lstm'] = (aligned_features, aligned_labels)
        
        return datasets
