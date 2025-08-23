"""
Data preprocessing pipeline for M5 Multi-Symbol Trend Bot
Handles data cleaning, alignment, gap filling, and winsorization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing pipeline for OHLCV data"""
    
    def __init__(self, winsorize_percentiles: Tuple[float, float] = (0.5, 99.5)):
        self.winsorize_percentiles = winsorize_percentiles
        
    def align_to_grid(self, df: pd.DataFrame, timeframe: str = '5T') -> pd.DataFrame:
        """Align data to regular time grid"""
        if df.empty:
            return df
            
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Create regular time grid
        start_time = df.index.min().floor(timeframe)
        end_time = df.index.max().ceil(timeframe)
        
        # Generate complete time range
        time_grid = pd.date_range(start=start_time, end=end_time, freq=timeframe)
        
        # Reindex to grid
        df_aligned = df.reindex(time_grid)
        
        logger.info(f"Aligned data to {timeframe} grid: {len(df)} -> {len(df_aligned)} bars")
        return df_aligned
    
    def fill_gaps(self, df: pd.DataFrame, max_gap_bars: int = 2) -> pd.DataFrame:
        """Fill gaps using forward fill for gaps <= max_gap_bars"""
        if df.empty:
            return df
            
        df_filled = df.copy()
        
        # Identify gaps
        missing_mask = df_filled.isnull().any(axis=1)
        
        if not missing_mask.any():
            return df_filled
        
        # Group consecutive missing values
        missing_groups = (missing_mask != missing_mask.shift()).cumsum()
        
        for group_id in missing_groups[missing_mask].unique():
            group_mask = (missing_groups == group_id) & missing_mask
            gap_size = group_mask.sum()
            
            if gap_size <= max_gap_bars:
                # Forward fill small gaps
                df_filled.loc[group_mask] = df_filled.fillna(method='ffill').loc[group_mask]
            else:
                # Log large gaps but don't fill
                gap_start = df_filled.index[group_mask][0]
                gap_end = df_filled.index[group_mask][-1]
                logger.warning(f"Large gap detected ({gap_size} bars): {gap_start} to {gap_end}")
        
        filled_count = missing_mask.sum() - df_filled.isnull().any(axis=1).sum()
        logger.info(f"Filled {filled_count} missing bars (max gap: {max_gap_bars})")
        
        return df_filled
    
    def remove_market_breaks(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove market break periods based on symbol type"""
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # Define market hours based on symbol type
        if symbol.startswith('EUR') or symbol.startswith('GBP') or symbol.startswith('USD'):
            # Forex - remove weekends
            df_clean = df_clean[df_clean.index.dayofweek < 5]  # Monday=0, Sunday=6
            
        elif symbol.startswith('US') or symbol.startswith('SPX') or symbol.startswith('NAS'):
            # US indices - remove weekends and major holidays
            df_clean = df_clean[df_clean.index.dayofweek < 5]
            
            # Remove major US holidays (simplified)
            holidays = [
                '2023-01-01', '2023-07-04', '2023-12-25',  # Add more as needed
                '2024-01-01', '2024-07-04', '2024-12-25',
            ]
            holiday_dates = pd.to_datetime(holidays)
            df_clean = df_clean[~df_clean.index.date.isin(holiday_dates.date)]
            
        removed_count = len(df) - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} bars during market breaks")
            
        return df_clean
    
    def winsorize_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize returns at specified percentiles"""
        if df.empty:
            return df
            
        df_winsorized = df.copy()
        
        # Calculate returns
        returns = df_winsorized['close'].pct_change()
        
        # Winsorize returns
        lower_bound = np.percentile(returns.dropna(), self.winsorize_percentiles[0])
        upper_bound = np.percentile(returns.dropna(), self.winsorize_percentiles[1])
        
        # Count extreme values
        extreme_count = ((returns < lower_bound) | (returns > upper_bound)).sum()
        
        # Apply winsorization by adjusting prices
        winsorized_returns = np.clip(returns, lower_bound, upper_bound)
        
        # Reconstruct prices from winsorized returns
        if not winsorized_returns.isna().all():
            first_valid_idx = winsorized_returns.first_valid_index()
            if first_valid_idx is not None:
                # Start with original first price
                winsorized_prices = [df_winsorized.loc[first_valid_idx, 'close']]
                
                for i, ret in enumerate(winsorized_returns.iloc[1:], 1):
                    if pd.notna(ret):
                        new_price = winsorized_prices[-1] * (1 + ret)
                        winsorized_prices.append(new_price)
                    else:
                        winsorized_prices.append(df_winsorized.iloc[i]['close'])
                
                # Update close prices
                df_winsorized.loc[winsorized_returns.index, 'close'] = winsorized_prices
                
                # Adjust OHLC to maintain consistency
                for idx in df_winsorized.index[1:]:
                    if pd.notna(winsorized_returns.loc[idx]):
                        original_close = df.loc[idx, 'close']
                        new_close = df_winsorized.loc[idx, 'close']
                        adjustment_factor = new_close / original_close if original_close != 0 else 1
                        
                        # Adjust other OHLC values proportionally
                        df_winsorized.loc[idx, 'open'] *= adjustment_factor
                        df_winsorized.loc[idx, 'high'] *= adjustment_factor
                        df_winsorized.loc[idx, 'low'] *= adjustment_factor
        
        if extreme_count > 0:
            logger.info(f"Winsorized {extreme_count} extreme returns at [{self.winsorize_percentiles[0]}%, {self.winsorize_percentiles[1]}%]")
        
        return df_winsorized
    
    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data consistency"""
        if df.empty:
            return df
            
        df_valid = df.copy()
        invalid_count = 0
        
        # Check OHLC relationships
        invalid_mask = (
            (df_valid['high'] < df_valid['low']) |
            (df_valid['high'] < df_valid['open']) |
            (df_valid['high'] < df_valid['close']) |
            (df_valid['low'] > df_valid['open']) |
            (df_valid['low'] > df_valid['close']) |
            (df_valid['open'] <= 0) |
            (df_valid['high'] <= 0) |
            (df_valid['low'] <= 0) |
            (df_valid['close'] <= 0) |
            (df_valid['volume'] < 0)
        )
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(f"Found {invalid_count} invalid OHLCV bars - removing")
            df_valid = df_valid[~invalid_mask]
        
        return df_valid
    
    def process_symbol_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Complete preprocessing pipeline for a single symbol"""
        logger.info(f"Processing {symbol} data: {len(df)} bars")
        
        if df.empty:
            logger.warning(f"Empty dataframe for {symbol}")
            return df
        
        # Step 1: Validate OHLCV
        df_processed = self.validate_ohlcv(df)
        
        # Step 2: Align to grid
        df_processed = self.align_to_grid(df_processed)
        
        # Step 3: Fill small gaps
        df_processed = self.fill_gaps(df_processed)
        
        # Step 4: Remove market breaks
        df_processed = self.remove_market_breaks(df_processed, symbol)
        
        # Step 5: Winsorize returns
        df_processed = self.winsorize_returns(df_processed)
        
        # Final validation
        df_processed = self.validate_ohlcv(df_processed)
        
        logger.info(f"Completed processing {symbol}: {len(df)} -> {len(df_processed)} bars")
        return df_processed
    
    def process_multi_symbol_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process multiple symbols with alignment"""
        processed_data = {}
        
        for symbol, df in data_dict.items():
            processed_data[symbol] = self.process_symbol_data(df, symbol)
        
        # Align all symbols to common time grid
        if len(processed_data) > 1:
            processed_data = self._align_multi_symbol(processed_data)
        
        return processed_data
    
    def _align_multi_symbol(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align multiple symbols to common time grid"""
        # Find common time range
        all_indices = [df.index for df in data_dict.values() if not df.empty]
        
        if not all_indices:
            return data_dict
        
        # Get intersection of all time indices
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        
        if len(common_index) == 0:
            logger.warning("No common time periods found across symbols")
            return data_dict
        
        # Align all symbols to common index
        aligned_data = {}
        for symbol, df in data_dict.items():
            if not df.empty:
                aligned_data[symbol] = df.reindex(common_index)
            else:
                aligned_data[symbol] = df
        
        logger.info(f"Aligned {len(data_dict)} symbols to common grid: {len(common_index)} bars")
        return aligned_data
    
    def get_data_quality_report(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate data quality report"""
        if df.empty:
            return {"symbol": symbol, "status": "empty"}
        
        report = {
            "symbol": symbol,
            "total_bars": len(df),
            "date_range": {
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat()
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict()
            },
            "price_stats": {
                "close_mean": float(df['close'].mean()),
                "close_std": float(df['close'].std()),
                "volume_mean": float(df['volume'].mean()),
                "max_gap_hours": self._calculate_max_gap(df)
            },
            "data_quality_score": self._calculate_quality_score(df)
        }
        
        return report
    
    def _calculate_max_gap(self, df: pd.DataFrame) -> float:
        """Calculate maximum time gap in hours"""
        if len(df) < 2:
            return 0.0
        
        time_diffs = df.index.to_series().diff().dropna()
        max_gap = time_diffs.max()
        return max_gap.total_seconds() / 3600 if pd.notna(max_gap) else 0.0
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-100)"""
        if df.empty:
            return 0.0
        
        # Factors for quality score
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        consistency = 1 - (self.validate_ohlcv(df).shape[0] / len(df))  # Inverse of invalid ratio
        
        # Simple weighted average
        quality_score = (completeness * 0.7 + consistency * 0.3) * 100
        return min(100.0, max(0.0, quality_score))
