"""
Triple-Barrier labeling system for M5 Multi-Symbol Trend Bot
Implements multiple RR presets and AFML meta-labeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class TripleBarrierLabeling:
    """Triple-barrier labeling with multiple risk-reward presets"""
    
    def __init__(self, rr_presets: List[str] = ['1:2', '1:3', '1:4']):
        self.rr_presets = rr_presets
        self.rr_ratios = {preset: self._parse_rr_ratio(preset) for preset in rr_presets}
        
    def _parse_rr_ratio(self, rr_preset: str) -> Tuple[float, float]:
        """Parse RR ratio string to (risk, reward) tuple"""
        parts = rr_preset.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid RR preset format: {rr_preset}")
        
        risk = float(parts[0])
        reward = float(parts[1])
        return risk, reward
    
    def calculate_barriers(self, entry_price: float, atr: float, rr_preset: str, 
                          direction: int = 1) -> Dict[str, float]:
        """Calculate PT/SL barriers for given RR preset"""
        risk, reward = self.rr_ratios[rr_preset]
        
        # Base stop loss distance (1 ATR for risk unit)
        sl_distance = atr * risk
        pt_distance = atr * reward
        
        if direction == 1:  # Long position
            stop_loss = entry_price - sl_distance
            profit_target = entry_price + pt_distance
        else:  # Short position
            stop_loss = entry_price + sl_distance
            profit_target = entry_price - pt_distance
        
        return {
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'sl_distance': sl_distance,
            'pt_distance': pt_distance
        }
    
    def apply_triple_barrier(self, prices: pd.Series, entry_idx: int, barriers: Dict[str, float],
                           max_horizon: int = 100) -> Dict:
        """Apply triple-barrier method to a single trade"""
        entry_price = prices.iloc[entry_idx]
        entry_time = prices.index[entry_idx]
        
        # Get price series from entry point
        future_prices = prices.iloc[entry_idx + 1:entry_idx + 1 + max_horizon]
        
        if len(future_prices) == 0:
            return {
                'label': 0,  # Neutral/timeout
                'exit_time': None,
                'exit_price': entry_price,
                'return': 0.0,
                'barrier_hit': 'timeout',
                'holding_period': 0
            }
        
        stop_loss = barriers['stop_loss']
        profit_target = barriers['profit_target']
        
        # Check each future price point
        for i, (timestamp, price) in enumerate(future_prices.items()):
            # Check profit target hit
            if (entry_price < profit_target and price >= profit_target) or \
               (entry_price > profit_target and price <= profit_target):
                return {
                    'label': 1,  # Profit target hit
                    'exit_time': timestamp,
                    'exit_price': price,
                    'return': (price - entry_price) / entry_price,
                    'barrier_hit': 'profit_target',
                    'holding_period': i + 1
                }
            
            # Check stop loss hit
            if (entry_price > stop_loss and price <= stop_loss) or \
               (entry_price < stop_loss and price >= stop_loss):
                return {
                    'label': -1,  # Stop loss hit
                    'exit_time': timestamp,
                    'exit_price': price,
                    'return': (price - entry_price) / entry_price,
                    'barrier_hit': 'stop_loss',
                    'holding_period': i + 1
                }
        
        # Timeout - no barrier hit within horizon
        final_price = future_prices.iloc[-1]
        return {
            'label': 0,  # Neutral/timeout
            'exit_time': future_prices.index[-1],
            'exit_price': final_price,
            'return': (final_price - entry_price) / entry_price,
            'barrier_hit': 'timeout',
            'holding_period': len(future_prices)
        }
    
    def generate_labels_single_symbol(self, ohlc: pd.DataFrame, atr: pd.Series,
                                     rr_preset: str = '1:2', 
                                     max_horizon: int = 100) -> pd.DataFrame:
        """Generate triple-barrier labels for single symbol"""
        labels_data = []
        prices = ohlc['close']
        
        # Generate labels for each valid entry point
        for i in range(len(prices) - max_horizon):
            entry_time = prices.index[i]
            entry_price = prices.iloc[i]
            current_atr = atr.iloc[i] if i < len(atr) else atr.iloc[-1]
            
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # Generate labels for both directions
            for direction in [1, -1]:  # Long and short
                barriers = self.calculate_barriers(entry_price, current_atr, rr_preset, direction)
                result = self.apply_triple_barrier(prices, i, barriers, max_horizon)
                
                # Adjust label based on direction
                if direction == -1:  # Short position
                    result['label'] = -result['label'] if result['label'] != 0 else 0
                    result['return'] = -result['return']
                
                labels_data.append({
                    'timestamp': entry_time,
                    'direction': direction,
                    'rr_preset': rr_preset,
                    'label': result['label'],
                    'exit_time': result['exit_time'],
                    'exit_price': result['exit_price'],
                    'return': result['return'],
                    'barrier_hit': result['barrier_hit'],
                    'holding_period': result['holding_period'],
                    'entry_price': entry_price,
                    'barriers': barriers
                })
        
        labels_df = pd.DataFrame(labels_data)
        if not labels_df.empty:
            labels_df.set_index('timestamp', inplace=True)
        
        return labels_df
    
    def generate_primary_labels(self, ohlc: pd.DataFrame, atr: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate primary labels (y_base) for all RR presets"""
        primary_labels = {}
        
        for rr_preset in self.rr_presets:
            logger.info(f"Generating primary labels for RR preset: {rr_preset}")
            labels_df = self.generate_labels_single_symbol(ohlc, atr, rr_preset)
            primary_labels[rr_preset] = labels_df
        
        return primary_labels
    
    def calculate_sample_weights(self, labels_df: pd.DataFrame, 
                               decay_factor: float = 0.95,
                               uniqueness_threshold: float = 0.1) -> pd.Series:
        """Calculate sample weights using time decay and uniqueness"""
        weights = pd.Series(1.0, index=labels_df.index)
        
        # Time decay weighting (more recent samples get higher weight)
        if len(labels_df) > 1:
            time_diff = (labels_df.index.max() - labels_df.index).total_seconds() / 3600  # Hours
            max_time_diff = time_diff.max()
            if max_time_diff > 0:
                time_weights = decay_factor ** (time_diff / max_time_diff * 100)
                weights *= time_weights
        
        # Uniqueness weighting (reduce weight for overlapping events)
        if 'holding_period' in labels_df.columns:
            for i, (timestamp, row) in enumerate(labels_df.iterrows()):
                holding_period = row['holding_period']
                
                # Find overlapping events
                overlap_start = timestamp
                overlap_end = timestamp + pd.Timedelta(minutes=5 * holding_period)
                
                overlapping_events = labels_df[
                    (labels_df.index >= overlap_start) & 
                    (labels_df.index <= overlap_end)
                ]
                
                overlap_count = len(overlapping_events)
                if overlap_count > 1:
                    # Reduce weight based on overlap
                    uniqueness_weight = 1.0 / overlap_count
                    weights.loc[timestamp] *= uniqueness_weight
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def generate_meta_labels(self, primary_labels: pd.DataFrame, 
                           lstm_probabilities: pd.DataFrame,
                           threshold: float = 0.55) -> pd.DataFrame:
        """Generate meta-labels (y_meta) for AFML"""
        meta_labels_data = []
        
        # Align timestamps, ensuring uniqueness for iteration
        common_timestamps = primary_labels.index.intersection(lstm_probabilities.index).unique()

        for timestamp in common_timestamps:
            primary_rows = primary_labels.loc[timestamp]
            if isinstance(primary_rows, pd.Series):
                primary_rows = primary_rows.to_frame().T

            lstm_row = lstm_probabilities.loc[timestamp]

            # Get LSTM prediction confidence and direction
            max_prob = max(lstm_row.get('p_up', 0), lstm_row.get('p_down', 0))
            predicted_direction = 1 if lstm_row.get('p_up', 0) > lstm_row.get('p_down', 0) else -1

            # Select the primary label that corresponds to the LSTM's predicted direction
            relevant_primary_row = primary_rows[primary_rows['direction'] == predicted_direction]

            if relevant_primary_row.empty:
                continue

            # Take the first row if multiple exist for the same direction (unlikely but safe)
            actual_label_info = relevant_primary_row.iloc[0]
            actual_label = actual_label_info['label']

            # Meta-label is 1 if the LSTM was confident and the resulting trade was profitable (label == 1)
            should_trade = (max_prob >= threshold and actual_label == 1)

            meta_labels_data.append({
                'timestamp': timestamp,
                'meta_label': 1 if should_trade else 0,
                'lstm_confidence': max_prob,
                'lstm_direction': predicted_direction,
                'primary_label': actual_label,
                'primary_return': actual_label_info.get('return', 0)
            })
        
        meta_labels_df = pd.DataFrame(meta_labels_data)
        if not meta_labels_df.empty:
            meta_labels_df.set_index('timestamp', inplace=True)
        
        return meta_labels_df
    
    def create_balanced_dataset(self, labels_df: pd.DataFrame, 
                               target_balance: Dict[int, float] = None) -> pd.DataFrame:
        """Create balanced dataset by undersampling"""
        if target_balance is None:
            target_balance = {-1: 0.33, 0: 0.34, 1: 0.33}  # Equal balance
        
        balanced_data = []
        
        for label_value, target_ratio in target_balance.items():
            label_subset = labels_df[labels_df['label'] == label_value]
            
            if len(label_subset) == 0:
                continue
            
            # Calculate target count
            total_target_size = int(len(labels_df) * 0.8)  # Use 80% of original size
            target_count = int(total_target_size * target_ratio)
            
            # Sample subset
            if len(label_subset) > target_count:
                # Undersample
                sampled_subset = label_subset.sample(n=target_count, random_state=42)
            else:
                # Use all available samples
                sampled_subset = label_subset
            
            balanced_data.append(sampled_subset)
        
        balanced_df = pd.concat(balanced_data).sort_index()
        
        logger.info(f"Created balanced dataset: {len(labels_df)} -> {len(balanced_df)} samples")
        return balanced_df
    
    def remove_overlapping_events(self, labels_df: pd.DataFrame, 
                                 min_separation_bars: int = 5) -> pd.DataFrame:
        """Remove overlapping events to reduce data leakage"""
        if labels_df.empty or 'holding_period' not in labels_df.columns:
            return labels_df
        
        # Sort by timestamp
        sorted_labels = labels_df.sort_index()
        
        # Keep track of non-overlapping events
        selected_events = []
        last_end_time = None
        
        for timestamp, row in sorted_labels.iterrows():
            holding_period = row['holding_period']
            event_end_time = timestamp + pd.Timedelta(minutes=5 * holding_period)
            
            # Check if this event overlaps with the last selected event
            if last_end_time is None or timestamp >= last_end_time + pd.Timedelta(minutes=5 * min_separation_bars):
                selected_events.append(timestamp)
                last_end_time = event_end_time
        
        filtered_df = sorted_labels.loc[selected_events]
        
        logger.info(f"Removed overlapping events: {len(labels_df)} -> {len(filtered_df)} samples")
        return filtered_df
    
    def validate_labels(self, labels_df: pd.DataFrame, rr_preset: str) -> Dict:
        """Validate label quality"""
        if labels_df.empty:
            return {'status': 'empty', 'rr_preset': rr_preset}
        
        validation = {
            'rr_preset': rr_preset,
            'total_samples': len(labels_df),
            'label_distribution': labels_df['label'].value_counts().to_dict(),
            'avg_return': labels_df['return'].mean(),
            'win_rate': (labels_df['label'] == 1).mean(),
            'avg_holding_period': labels_df['holding_period'].mean(),
            'issues': []
        }
        
        # Check for class imbalance
        label_counts = labels_df['label'].value_counts()
        if len(label_counts) > 1:
            max_count = label_counts.max()
            min_count = label_counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 10:
                validation['issues'].append(f"High class imbalance: {imbalance_ratio:.1f}")
        
        # Check for unrealistic returns
        extreme_returns = labels_df['return'].abs() > 0.1  # 10% returns
        if extreme_returns.any():
            validation['issues'].append(f"Extreme returns in {extreme_returns.sum()} samples")
        
        # Check for very short holding periods
        short_periods = labels_df['holding_period'] < 3
        if short_periods.any():
            validation['issues'].append(f"Very short holding periods in {short_periods.sum()} samples")
        
        validation['quality_score'] = max(0, 100 - len(validation['issues']) * 20)
        
        return validation
    
    def export_labels_summary(self, all_labels: Dict[str, pd.DataFrame], symbol: str) -> Dict:
        """Export comprehensive labels summary"""
        summary = {
            'symbol': symbol,
            'rr_presets': {},
            'overall_stats': {
                'total_samples': 0,
                'date_range': {'start': None, 'end': None}
            }
        }
        
        all_timestamps = []
        
        for rr_preset, labels_df in all_labels.items():
            if labels_df.empty:
                continue
            
            preset_summary = self.validate_labels(labels_df, rr_preset)
            summary['rr_presets'][rr_preset] = preset_summary
            summary['overall_stats']['total_samples'] += len(labels_df)
            
            all_timestamps.extend(labels_df.index.tolist())
        
        if all_timestamps:
            summary['overall_stats']['date_range']['start'] = min(all_timestamps).isoformat()
            summary['overall_stats']['date_range']['end'] = max(all_timestamps).isoformat()
        
        return summary
