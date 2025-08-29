"""
Generate features and labels for EURUSDm using Black Swan system components
Follows system rules and database schema for proper data persistence
"""

import logging
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Setup project environment
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.database import TradingDatabase
from src.features.black_swan_pipeline import BlackSwanFeaturePipeline
from src.features.black_swan_labeling import BlackSwanLabeling

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("black_swan_generator")

DB_PATH = ROOT / 'data' / 'trading_system.db'
SYMBOL = 'EURUSDm'
CHUNK_SIZE = 20000
LOOKBACK = 200
FORECAST_HORIZON = 100

def clear_existing_data(db: TradingDatabase, symbol: str):
    """Clear existing features and labels for the symbol"""
    logger.info(f"Clearing existing features and labels for {symbol}")
    
    with db.get_connection() as conn:
        # Clear features
        conn.execute("DELETE FROM features WHERE symbol = ?", (symbol,))
        # Clear labels  
        conn.execute("DELETE FROM labels WHERE symbol = ?", (symbol,))
        conn.commit()
    
    logger.info(f"Cleared existing data for {symbol}")

def process_chunk(db: TradingDatabase, bars_chunk: pd.DataFrame, symbol: str, 
                 feature_pipeline: BlackSwanFeaturePipeline, 
                 labeling_pipeline: BlackSwanLabeling,
                 chunk_start_idx: int = 0):
    """Process a single chunk of bars data"""
    
    if bars_chunk.empty:
        logger.warning("Empty bars chunk, skipping")
        return 0, 0
    
    logger.info(f"Processing chunk with {len(bars_chunk)} bars")
    
    try:
        # Generate features
        logger.info("Generating XGB features...")
        xgb_features = feature_pipeline.generate_xgb_features(bars_chunk, symbol)
        
        logger.info("Generating LSTM features...")
        lstm_features = feature_pipeline.generate_lstm_features(bars_chunk, symbol)
        
        # Validate features
        if not feature_pipeline.validate_features(xgb_features, 'xgb'):
            logger.error("XGB feature validation failed")
            return 0, 0
            
        if not feature_pipeline.validate_features(lstm_features, 'lstm'):
            logger.error("LSTM feature validation failed")
            return 0, 0
        
        # Generate labels
        logger.info("Generating Black Swan labels...")
        labels_dict = labeling_pipeline.generate_labels_for_symbol(bars_chunk, symbol)
        
        # Validate labels
        if not labeling_pipeline.validate_labels(labels_dict):
            logger.error("Label validation failed")
            return 0, 0
        
        # Determine valid timestamps (skip lookback period for first chunk)
        valid_timestamps = bars_chunk.index[LOOKBACK if chunk_start_idx == 0 else 0:]
        
        # Store features
        features_inserted = 0
        
        # Store XGB features
        valid_xgb = xgb_features.loc[xgb_features.index.intersection(valid_timestamps)]
        for timestamp, row in valid_xgb.iterrows():
            features_dict = {k: v for k, v in row.to_dict().items() if k != 'symbol' and pd.notna(v)}
            if features_dict:
                success = db.insert_features(symbol, timestamp.to_pydatetime(), 'xgb', features_dict)
                if success:
                    features_inserted += 1
        
        # Store LSTM features
        valid_lstm = lstm_features.loc[lstm_features.index.intersection(valid_timestamps)]
        for timestamp, row in valid_lstm.iterrows():
            features_dict = {k: v for k, v in row.to_dict().items() if k != 'symbol' and pd.notna(v)}
            if features_dict:
                success = db.insert_features(symbol, timestamp.to_pydatetime(), 'lstm', features_dict)
                if success:
                    features_inserted += 1
        
        # Store labels following TradingDatabase schema
        labels_inserted = 0
        
        # Process XGB labels (regression targets)
        for direction in ['long', 'short']:
            xgb_key = f'xgb_{direction}'
            if xgb_key in labels_dict:
                xgb_labels = labels_dict[xgb_key]
                valid_xgb_labels = xgb_labels.loc[xgb_labels.index.intersection(valid_timestamps)]
                
                for timestamp, row in valid_xgb_labels.iterrows():
                    # Convert MFE target to integer label for database
                    mfe_target = row['mfe_target']
                    
                    # Create barrier metadata
                    barrier_meta = {
                        'direction': direction,
                        'mfe_target': float(mfe_target),
                        'atr14': float(row['atr14']),
                        'label_type': 'xgb_regression'
                    }
                    
                    # Convert MFE to discrete label (0: <2R, 1: 2-5R, 2: 5-10R, 3: >10R)
                    if mfe_target < 2.0:
                        label_value = 0
                    elif mfe_target < 5.0:
                        label_value = 1
                    elif mfe_target < 10.0:
                        label_value = 2
                    else:
                        label_value = 3
                    
                    success = db.insert_labels(
                        symbol=symbol,
                        time=timestamp.to_pydatetime(),
                        label_type=f'xgb_{direction}',
                        rr_preset='mfe_regression',
                        label_value=label_value,
                        barrier_meta=barrier_meta
                    )
                    if success:
                        labels_inserted += 1
        
        # Process LSTM labels (classification)
        for direction in ['long', 'short']:
            lstm_key = f'lstm_{direction}'
            if lstm_key in labels_dict:
                lstm_labels = labels_dict[lstm_key]
                valid_lstm_labels = lstm_labels.loc[lstm_labels.index.intersection(valid_timestamps)]
                
                for timestamp, row in valid_lstm_labels.iterrows():
                    # Use tail_class directly as label_value
                    tail_class = int(row['tail_class'])
                    
                    # Create barrier metadata
                    barrier_meta = {
                        'direction': direction,
                        'tail_class': tail_class,
                        'mfe_source': float(row['mfe_source']),
                        'atr14': float(row['atr14']),
                        'label_type': 'lstm_classification'
                    }
                    
                    success = db.insert_labels(
                        symbol=symbol,
                        time=timestamp.to_pydatetime(),
                        label_type=f'lstm_{direction}',
                        rr_preset='tail_classification',
                        label_value=tail_class,
                        barrier_meta=barrier_meta
                    )
                    if success:
                        labels_inserted += 1
        
        logger.info(f"Chunk processed: {features_inserted} features, {labels_inserted} labels inserted")
        return features_inserted, labels_inserted
        
    except Exception as e:
        logger.error(f"Error processing chunk: {e}", exc_info=True)
        return 0, 0

def main():
    """Main execution function"""
    logger.info(f"Starting Black Swan feature and label generation for {SYMBOL}")
    
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return
    
    # Initialize components
    db = TradingDatabase(str(DB_PATH))
    feature_pipeline = BlackSwanFeaturePipeline()
    labeling_pipeline = BlackSwanLabeling(forecast_horizon=FORECAST_HORIZON)
    
    # Check if bars exist for symbol
    with db.get_connection() as conn:
        total_bars = conn.execute("SELECT COUNT(*) FROM bars WHERE symbol = ?", (SYMBOL,)).fetchone()[0]
    
    if total_bars == 0:
        logger.error(f"No bars found for symbol {SYMBOL}")
        return
    
    logger.info(f"Found {total_bars} bars for {SYMBOL}")
    
    # Clear existing data
    clear_existing_data(db, SYMBOL)
    
    # Process in chunks
    total_features_inserted = 0
    total_labels_inserted = 0
    
    for offset in range(0, total_bars, CHUNK_SIZE):
        logger.info(f"Processing chunk: offset={offset}, chunk_size={CHUNK_SIZE}")
        
        # Calculate read parameters with lookback
        read_offset = max(0, offset - LOOKBACK)
        read_limit = CHUNK_SIZE + (offset - read_offset)
        
        # Load bars chunk
        with db.get_connection() as conn:
            query = f"""
                SELECT time, open, high, low, close, volume 
                FROM bars 
                WHERE symbol = ? 
                ORDER BY time 
                LIMIT {read_limit} OFFSET {read_offset}
            """
            bars_chunk = pd.read_sql_query(query, conn, params=[SYMBOL], parse_dates=['time'])
        
        if bars_chunk.empty:
            continue
        
        # Prepare data
        bars_chunk.rename(columns={'time': 'timestamp'}, inplace=True)
        bars_chunk.set_index('timestamp', inplace=True)
        bars_chunk.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        bars_chunk = bars_chunk[~bars_chunk.index.duplicated(keep='first')]
        
        if bars_chunk.empty:
            continue
        
        # Process chunk
        chunk_features, chunk_labels = process_chunk(
            db, bars_chunk, SYMBOL, feature_pipeline, labeling_pipeline, offset
        )
        
        total_features_inserted += chunk_features
        total_labels_inserted += chunk_labels
        
        # Clean up memory
        del bars_chunk
        gc.collect()
    
    logger.info(f"Generation complete for {SYMBOL}:")
    logger.info(f"  Total features inserted: {total_features_inserted}")
    logger.info(f"  Total labels inserted: {total_labels_inserted}")
    
    # Final verification
    with db.get_connection() as conn:
        features_count = conn.execute("SELECT COUNT(*) FROM features WHERE symbol = ?", (SYMBOL,)).fetchone()[0]
        labels_count = conn.execute("SELECT COUNT(*) FROM labels WHERE symbol = ?", (SYMBOL,)).fetchone()[0]
    
    logger.info(f"Final verification - Features: {features_count}, Labels: {labels_count}")

if __name__ == '__main__':
    main()
