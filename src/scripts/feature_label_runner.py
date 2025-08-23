"""
Feature and Label Generation Runner (Robust Version)

This script processes bar data from the SQLite database in manageable chunks
to prevent memory overload. It generates and persists features and labels
for all specified symbols, adhering to the system's data science rules.

Key Features:
- Processes data in chunks to handle large datasets.
- Uses a lookback period to ensure accurate rolling feature calculation.
- Cleans and validates data within each chunk.
- Correctly aligns timestamps to prevent KeyErrors during label persistence.
"""

import json
import sqlite3
import gc
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
import sys

# --- Setup Project Environment ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# --- COLAB/DRIVE PATH CONFIGURATION ---
# If running on Google Colab, set your project's base directory and the path to your database file.
# 1. Set the project's root folder in your Google Drive.
BASE_DIR = Path('/content/drive/MyDrive/M5_Trading_Bot')
# 2. Set the full path to your m5_trading.db file.
DB_PATH = Path('/content/drive/MyDrive/trading_bot_data/m5_trading.db')

# If running locally, you can use the following default paths:
# BASE_DIR = ROOT
# DB_PATH = BASE_DIR / 'data' / 'm5_trading.db'
# ---

from src.data.database import M5Database
from src.features.feature_pipeline import FeaturePipeline
from src.features.labeling import TripleBarrierLabeling
from src.features.technical_indicators import TechnicalIndicators

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("feature_label_runner_v2")

CONFIG_PATH = BASE_DIR / 'config.json'

# --- Core Functions ---

def get_run_config(conn: sqlite3.Connection) -> tuple:
    """Load symbols and date range from config file, with fallback to DB."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
        symbols = cfg.get('symbols', [])
        start = cfg.get('data', {}).get('start_date')
        end = cfg.get('data', {}).get('end_date')
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) if end else None
        if symbols:
            logger.info(f"Loaded config: Symbols={symbols}, Start={start_dt}, End={end_dt}")
            return symbols, start_dt, end_dt

    logger.info("Config file not found or empty. Falling back to full DB scan.")
    rows = conn.cursor().execute("SELECT symbol, MIN(time), MAX(time) FROM bars GROUP BY symbol").fetchall()
    symbols = [r[0] for r in rows]
    if rows:
        start_dt = pd.to_datetime([r[1] for r in rows]).min()
        end_dt = pd.to_datetime([r[2] for r in rows]).max()
    else:
        start_dt, end_dt = None, None
    logger.info(f"DB Fallback: Symbols={symbols}, Start={start_dt}, End={end_dt}")
    return symbols, start_dt, end_dt

def run_symbol_processing(symbol: str, start_dt: datetime, end_dt: datetime):
    """Main processing pipeline for a single symbol."""
    logger.info(f"--- Starting processing for symbol: {symbol} ---")
    db = M5Database(str(DB_PATH))
    conn = db.get_connection()

    count_query = "SELECT COUNT(*) FROM bars WHERE symbol = ?"
    params = [symbol]
    if start_dt: count_query += " AND time >= ?"; params.append(start_dt.isoformat(sep=' '))
    if end_dt: count_query += " AND time <= ?"; params.append(end_dt.isoformat(sep=' '))
    total_bars = conn.cursor().execute(count_query, tuple(params)).fetchone()[0]

    if total_bars == 0:
        logger.warning(f"No bars found for {symbol}. Skipping.")
        conn.close()
        return

    logger.info(f"Total bars to process for {symbol}: {total_bars}")

    chunk_size = 10000
    lookback = 200

    fp = FeaturePipeline()
    ti = TechnicalIndicators()
    tbl = TripleBarrierLabeling(rr_presets=['1:2', '1:3', '1:4'])

    for offset in range(0, total_bars, chunk_size):
        logger.info(f"Processing chunk for {symbol}: offset={offset}, size={chunk_size}")

        read_offset = max(0, offset - lookback)
        read_limit = chunk_size + (offset - read_offset)

        query = f"SELECT * FROM bars WHERE symbol = ? ORDER BY time LIMIT {read_limit} OFFSET {read_offset}"
        bars_chunk = pd.read_sql_query(query, conn, params=[symbol], parse_dates=['time'])

        if bars_chunk.empty: continue

        bars_chunk.rename(columns={'time': 'timestamp'}, inplace=True)
        bars_chunk.set_index('timestamp', inplace=True)
        bars_chunk.dropna(axis=0, how='any', subset=['open', 'high', 'low', 'close'], inplace=True)
        bars_chunk = bars_chunk[~bars_chunk.index.duplicated(keep='first')]
        if pd.isna(bars_chunk.index).any(): bars_chunk = bars_chunk.loc[bars_chunk.index.dropna()]

        if bars_chunk.empty: continue

        try:
            lstm_features = fp.generate_lstm_features(bars_chunk, symbol)
            xgb_features = fp.generate_xgb_features(bars_chunk, symbol)
            atr14 = ti.calculate_atr(bars_chunk['high'], bars_chunk['low'], bars_chunk['close'], period=14)
        except Exception as e:
            logger.error(f"Feature generation failed for chunk at offset {offset}: {e}", exc_info=True)
            continue

        labels_by_preset = {}
        for preset in tbl.rr_presets:
            labels_df = tbl.generate_labels_single_symbol(bars_chunk, atr14, rr_preset=preset, max_horizon=100)
            labels_by_preset[preset] = labels_df

        chunk_timestamps = bars_chunk.index[lookback if offset > 0 else 0:]

        feature_timestamps = lstm_features.index.union(xgb_features.index).intersection(chunk_timestamps).dropna()
        for ts in feature_timestamps:
            lstm_row = lstm_features.loc[ts].to_dict() if ts in lstm_features.index else {}
            xgb_row = xgb_features.loc[ts].to_dict() if ts in xgb_features.index else {}
            db.upsert_feature_subset(symbol, ts.to_pydatetime(), 'lstm', lstm_row)
            db.upsert_feature_subset(symbol, ts.to_pydatetime(), 'xgb', xgb_row)

        for preset, df in labels_by_preset.items():
            valid_labels = df.loc[df.index.intersection(chunk_timestamps)]
            for ts, row in valid_labels.iterrows():
                meta = row.to_dict()
                if 'exit_time' in meta and isinstance(meta.get('exit_time'), pd.Timestamp):
                    meta['exit_time'] = meta['exit_time'].isoformat()
                db.upsert_labels_meta(symbol, ts.to_pydatetime(), meta.get('label', 0), meta)

        logger.info(f"Successfully processed and persisted chunk at offset {offset}.")

        del bars_chunk, lstm_features, xgb_features, atr14, labels_by_preset
        gc.collect()

    conn.close()
    logger.info(f"--- Finished processing for symbol: {symbol} ---")

# --- Main Execution ---
if __name__ == '__main__':
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    db_connection = sqlite3.connect(DB_PATH)
    try:
        symbols_to_run, start_date, end_date = get_run_config(db_connection)
        if not symbols_to_run:
            raise ValueError("No symbols found to process.")

        for symbol in symbols_to_run:
            run_symbol_processing(symbol, start_date, end_date)

        logger.info("All symbols processed successfully.")

    except Exception as e:
        logger.error(f"An unhandled error occurred: {e}", exc_info=True)
    finally:
        db_connection.close()

