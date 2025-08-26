"""
Regenerate features for TradingDatabase using the FeaturePipeline.

This script reads bar data, computes fresh LSTM and XGB feature sets, and saves
them to the `features` table, ensuring data is standardized and up-to-date.
It processes symbols in parallel for efficiency.

Usage:
  python src/scripts/regenerate_features_trading_db.py --symbols XAUUSDm,USDJPYm
"""
import argparse
import json

import logging
import multiprocessing
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# --- Setup Project Environment ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.database import TradingDatabase
from src.features.feature_pipeline import FeaturePipeline

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("regenerate_features_trading_db")

DEFAULT_DB = ROOT / 'data' / 'trading_system.db'


def regenerate_features_for_symbol(db_path: str, symbol: str, chunk_size: int = 50000, lookback: int = 200) -> str:
    """Worker function: computes features and saves them to a temporary file, returning the file path."""
    db = TradingDatabase(db_path)
    fp = FeaturePipeline()
    logger.info(f"[Worker] Starting feature computation for symbol: {symbol}")

    with db.get_connection() as conn:
        total_bars = conn.execute("SELECT COUNT(*) FROM bars WHERE symbol = ?", (symbol,)).fetchone()[0]
    if total_bars == 0:
        logger.warning(f"[Worker] No bars for {symbol}. Skipping.")
        return ""

    all_features_for_symbol = []
    for offset in range(0, total_bars, chunk_size):
        read_offset = max(0, offset - lookback)
        read_limit = chunk_size + (offset - read_offset)
        with db.get_connection() as conn:
            q = f"SELECT time, open, high, low, close, volume FROM bars WHERE symbol = ? ORDER BY time LIMIT {read_limit} OFFSET {read_offset}"
            bars_chunk = pd.read_sql_query(q, conn, params=[symbol], parse_dates=['time'])

        if bars_chunk.empty: continue
        bars_chunk.rename(columns={'time': 'timestamp'}, inplace=True)
        bars_chunk.set_index('timestamp', inplace=True)
        bars_chunk.dropna(axis=0, how='any', subset=['open', 'high', 'low', 'close'], inplace=True)
        bars_chunk = bars_chunk[~bars_chunk.index.duplicated(keep='first')]
        if bars_chunk.empty: continue

        try:
            lstm_features = fp.generate_lstm_features(bars_chunk, symbol)
            xgb_features = fp.generate_xgb_features(bars_chunk, symbol)
        except Exception as e:
            logger.error(f"[Worker] Feature generation failed for {symbol} at offset {offset}: {e}", exc_info=True)
            continue

        valid_timestamps = bars_chunk.index[lookback if offset > 0 else 0:]
        valid_lstm = lstm_features.loc[lstm_features.index.intersection(valid_timestamps)]
        for ts, row in valid_lstm.iterrows():
            features_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
            if features_dict:
                all_features_for_symbol.append((symbol, ts.to_pydatetime().isoformat(), 'lstm', json.dumps(features_dict)))

        valid_xgb = xgb_features.loc[xgb_features.index.intersection(valid_timestamps)]
        for ts, row in valid_xgb.iterrows():
            features_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
            if features_dict:
                all_features_for_symbol.append((symbol, ts.to_pydatetime().isoformat(), 'xgb', json.dumps(features_dict)))

    temp_dir = ROOT / 'artifacts' / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / f"features_{symbol}.json"
    with temp_file_path.open('w') as f:
        json.dump(all_features_for_symbol, f)
    logger.info(f"[Worker] Finished for {symbol}. Saved {len(all_features_for_symbol)} features to {temp_file_path}")
    return str(temp_file_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(DEFAULT_DB))
    ap.add_argument('--symbols', type=str, required=True, help='Comma-separated list of symbols to process')
    ap.add_argument('--chunk', type=int, default=50000)
    ap.add_argument('--lookback', type=int, default=200)
    args = ap.parse_args()

    symbols_to_run = [s.strip() for s in args.symbols.split(',') if s.strip()]
    if not symbols_to_run:
        raise ValueError("No symbols provided to process.")

    db_writer = TradingDatabase(args.db)
    total_inserted = 0
    max_workers = multiprocessing.cpu_count()
    logger.info(f"[Main] Starting parallel feature generation with up to {max_workers} workers for symbols: {symbols_to_run}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = { executor.submit(regenerate_features_for_symbol, args.db, sym, args.chunk, args.lookback): sym for sym in symbols_to_run }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                temp_file_path_str = future.result()
                if temp_file_path_str:
                    temp_file_path = Path(temp_file_path_str)
                    logger.info(f"[Main] Reading features for {symbol} from {temp_file_path}...")
                    with temp_file_path.open('r') as f:
                        features_to_insert = json.load(f)

                    if features_to_insert:
                        # Timestamps are already ISO strings from the worker
                        logger.info(f"[Main] Writing {len(features_to_insert)} feature sets for {symbol} to database...")
                        inserted_count = db_writer.insert_features_batch(features_to_insert)
                        total_inserted += inserted_count
                        logger.info(f"[Main] Completed {symbol}, inserted {inserted_count} feature sets.")

                    temp_file_path.unlink() # Clean up the temp file
                else:
                    logger.info(f"[Main] Completed {symbol}, no new feature sets to insert.")
            except Exception as e:
                logger.error(f"[Main] Symbol {symbol} failed with error: {e}", exc_info=True)

    logger.info(f"[Main] All symbols processed. Total feature sets inserted: {total_inserted}")


if __name__ == '__main__':
    main()

