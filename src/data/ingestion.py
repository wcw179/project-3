"""
Data ingestion module for M5 Multi-Symbol Trend Bot
Handles MT5 data retrieval and database storage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from src.data.database import TradingDatabase
from src.data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class DataIngestion:
    """Data ingestion pipeline for MT5 and database operations"""
    
    def __init__(self, db_path: str = "data/trading_system.db"):
        self.db = TradingDatabase(db_path)
        self.preprocessor = DataPreprocessor()
        self.mt5_initialized = False
        
    def initialize_mt5(self, login: Optional[int] = None, password: Optional[str] = None, 
                       server: Optional[str] = None) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            if login and password and server:
                if not mt5.login(login, password, server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
                    
            self.mt5_initialized = True
            logger.info("MT5 initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def shutdown_mt5(self):
        """Shutdown MT5 connection"""
        if self.mt5_initialized:
            mt5.shutdown()
            self.mt5_initialized = False
            logger.info("MT5 connection closed")
    
    def get_mt5_symbols(self) -> List[str]:
        """Get available symbols from MT5"""
        if not self.mt5_initialized:
            logger.error("MT5 not initialized")
            return []
        
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                logger.error("Failed to get symbols from MT5")
                return []
            
            symbol_names = [symbol.name for symbol in symbols if symbol.visible]
            logger.info(f"Retrieved {len(symbol_names)} symbols from MT5")
            return symbol_names
            
        except Exception as e:
            logger.error(f"Error getting MT5 symbols: {e}")
            return []
    
    def fetch_mt5_data(self, symbol: str, timeframe: int, start_date: datetime, 
                       end_date: Optional[datetime] = None, count: Optional[int] = None) -> pd.DataFrame:
        """Fetch OHLCV data from MT5"""
        if not self.mt5_initialized:
            logger.error("MT5 not initialized")
            return pd.DataFrame()
        
        try:
            # Convert timeframe
            mt5_timeframe = self._get_mt5_timeframe(timeframe)
            
            # Fetch data
            if end_date:
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            elif count:
                rates = mt5.copy_rates_from(symbol, mt5_timeframe, start_date, count)
            else:
                rates = mt5.copy_rates_from(symbol, mt5_timeframe, start_date, 10000)  # Default
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to standard format
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'real_volume': 'real_volume'
            }, inplace=True)
            
            # Ensure volume column exists
            if 'volume' not in df.columns and 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            elif 'volume' not in df.columns:
                df['volume'] = 0
            
            logger.info(f"Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching MT5 data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_mt5_timeframe(self, minutes: int) -> int:
        """Convert minutes to MT5 timeframe constant"""
        timeframe_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1
        }
        
        return timeframe_map.get(minutes, mt5.TIMEFRAME_M5)
    
    def ingest_symbol_data(self, symbol: str, start_date: datetime, 
                          end_date: Optional[datetime] = None, 
                          preprocess: bool = True) -> bool:
        """Ingest and store data for a single symbol"""
        try:
            # Fetch raw data from MT5
            raw_data = self.fetch_mt5_data(symbol, 5, start_date, end_date)
            
            if raw_data.empty:
                logger.warning(f"No data fetched for {symbol}")
                return False
            
            # Preprocess if requested
            if preprocess:
                processed_data = self.preprocessor.process_symbol_data(raw_data, symbol)
            else:
                processed_data = raw_data
            
            # Store in database
            inserted_count = self.db.insert_bars_batch(symbol, processed_data)
            
            logger.info(f"Ingested {inserted_count} bars for {symbol}")
            return inserted_count > 0
            
        except Exception as e:
            logger.error(f"Error ingesting data for {symbol}: {e}")
            return False
    
    def ingest_multi_symbol_data(self, symbols: List[str], start_date: datetime,
                                end_date: Optional[datetime] = None,
                                preprocess: bool = True) -> Dict[str, bool]:
        """Ingest data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            logger.info(f"Ingesting data for {symbol}")
            results[symbol] = self.ingest_symbol_data(symbol, start_date, end_date, preprocess)
        
        successful = sum(results.values())
        logger.info(f"Successfully ingested data for {successful}/{len(symbols)} symbols")
        
        return results
    
    def update_symbol_data(self, symbol: str, lookback_days: int = 7) -> bool:
        """Update existing symbol data with recent bars"""
        try:
            # Get last stored timestamp
            existing_data = self.db.get_ohlcv_data(symbol)
            
            if existing_data.empty:
                logger.info(f"No existing data for {symbol}, performing full ingest")
                start_date = datetime.now() - timedelta(days=lookback_days)
                return self.ingest_symbol_data(symbol, start_date)
            
            # Start from last timestamp
            last_timestamp = existing_data.index.max()
            start_date = last_timestamp - timedelta(hours=1)  # Small overlap
            
            # Fetch new data
            new_data = self.fetch_mt5_data(symbol, 5, start_date)
            
            if new_data.empty:
                logger.info(f"No new data available for {symbol}")
                return True
            
            # Remove overlap
            new_data = new_data[new_data.index > last_timestamp]
            
            if new_data.empty:
                logger.info(f"No new bars for {symbol}")
                return True
            
            # Preprocess new data
            processed_data = self.preprocessor.process_symbol_data(new_data, symbol)
            
            # Store in database
            inserted_count = self.db.insert_bars_batch(symbol, processed_data)
            
            logger.info(f"Updated {symbol} with {inserted_count} new bars")
            return inserted_count > 0
            
        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {e}")
            return False
    
    def get_data_summary(self, symbols: Optional[List[str]] = None) -> Dict:
        """Get summary of stored data"""
        summary = {
            "symbols": {},
            "total_symbols": 0,
            "total_bars": 0,
            "date_range": {"start": None, "end": None}
        }
        
        try:
            # Get all symbols if none specified
            if symbols is None:
                with self.db.get_connection() as conn:
                    cursor = conn.execute("SELECT DISTINCT symbol FROM bars")
                    symbols = [row[0] for row in cursor.fetchall()]
            
            all_start_dates = []
            all_end_dates = []
            
            for symbol in symbols:
                data = self.db.get_ohlcv_data(symbol)
                
                if not data.empty:
                    symbol_summary = {
                        "bars": len(data),
                        "start_date": data.index.min().isoformat(),
                        "end_date": data.index.max().isoformat(),
                        "quality_report": self.preprocessor.get_data_quality_report(data, symbol)
                    }
                    
                    summary["symbols"][symbol] = symbol_summary
                    summary["total_bars"] += len(data)
                    
                    all_start_dates.append(data.index.min())
                    all_end_dates.append(data.index.max())
            
            summary["total_symbols"] = len(summary["symbols"])
            
            if all_start_dates:
                summary["date_range"]["start"] = min(all_start_dates).isoformat()
                summary["date_range"]["end"] = max(all_end_dates).isoformat()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return summary
    
    def validate_data_integrity(self, symbol: str) -> Dict:
        """Validate data integrity for a symbol"""
        try:
            data = self.db.get_ohlcv_data(symbol)
            
            if data.empty:
                return {"symbol": symbol, "status": "no_data"}
            
            validation_results = {
                "symbol": symbol,
                "total_bars": len(data),
                "issues": []
            }
            
            # Check for gaps
            expected_freq = pd.Timedelta(minutes=5)
            time_diffs = data.index.to_series().diff().dropna()
            large_gaps = time_diffs[time_diffs > expected_freq * 3]  # Gaps > 15 minutes
            
            if not large_gaps.empty:
                validation_results["issues"].append({
                    "type": "large_gaps",
                    "count": len(large_gaps),
                    "max_gap_hours": large_gaps.max().total_seconds() / 3600
                })
            
            # Check for invalid OHLCV
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            ).sum()
            
            if invalid_ohlc > 0:
                validation_results["issues"].append({
                    "type": "invalid_ohlc",
                    "count": invalid_ohlc
                })
            
            # Check for zero/negative prices
            zero_prices = (
                (data['open'] <= 0) |
                (data['high'] <= 0) |
                (data['low'] <= 0) |
                (data['close'] <= 0)
            ).sum()
            
            if zero_prices > 0:
                validation_results["issues"].append({
                    "type": "zero_negative_prices",
                    "count": zero_prices
                })
            
            validation_results["status"] = "valid" if not validation_results["issues"] else "issues_found"
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}
    
    def cleanup_old_data(self, symbol: str, keep_days: int = 365) -> int:
        """Remove old data beyond keep_days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM bars
                    WHERE symbol = ? AND time < ?
                """, (symbol, cutoff_date))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old bars for {symbol}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up data for {symbol}: {e}")
            return 0
