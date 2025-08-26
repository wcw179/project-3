"""
Re-label TradingDatabase bars with triple-barrier for multiple RR presets.
Saves per-direction base labels using label_type in {'y_base_long','y_base_short'}.

Usage:
  python src/scripts/relabel_trading_db.py [--db data/trading_system.db] [--horizon 100]

Notes:
- Requires bars to be populated in TradingDatabase (e.g., via migration).
- ATR period = 14 by default.
- Processes each symbol in chunks with lookback for rolling indicators.
"""
import argparse
import json
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import pandas as pd

from src.data.database import TradingDatabase
from src.features.technical_indicators import TechnicalIndicators
from src.features.labeling import TripleBarrierLabeling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("relabel_trading_db")

DEFAULT_DB = Path(__file__).resolve().parents[2] / 'data' / 'trading_system.db'


def _short_outcome_to_y(label_value: int, barrier_hit: str) -> int:
    """Convert labeling output for short direction to y in {1,0,-1} where 1=profit, -1=loss."""
    if barrier_hit == 'profit_target':
        return 1
    if barrier_hit == 'stop_loss':
        return -1
    return 0


def _long_outcome_to_y(label_value: int, barrier_hit: str) -> int:
    """For long, labeling already uses 1=profit, -1=loss, 0=timeout."""
    if barrier_hit == 'profit_target':
        return 1
    if barrier_hit == 'stop_loss':
        return -1
    return 0


def relabel_symbol(db_path: str, symbol: str, rr_presets, horizon: int = 100, chunk_size: int = 50000, lookback: int = 200) -> str:
    """Worker function: computes labels and saves them to a temporary file, returning the file path."""
    db = TradingDatabase(db_path)
    logger.info(f"[Worker] Starting label computation for symbol={symbol}")

    with db.get_connection() as conn:
        total_bars = conn.execute("SELECT COUNT(*) FROM bars WHERE symbol = ?", (symbol,)).fetchone()[0]
    if total_bars == 0:
        logger.warning(f"[Worker] No bars for {symbol}")
        return ""

    ti = TechnicalIndicators()
    tbl = TripleBarrierLabeling(rr_presets=rr_presets)
    all_labels_for_symbol = []

    for offset in range(0, total_bars, chunk_size):
        read_offset = max(0, offset - lookback)
        read_limit = chunk_size + (offset - read_offset)
        with db.get_connection() as conn:
            q = f"SELECT time, open, high, low, close, IFNULL(volume,0) as volume FROM bars WHERE symbol = ? ORDER BY time LIMIT {read_limit} OFFSET {read_offset}"
            bars = pd.read_sql_query(q, conn, params=[symbol], parse_dates=['time'])
        if bars.empty: continue

        bars.rename(columns={'time': 'timestamp'}, inplace=True)
        bars.set_index('timestamp', inplace=True)
        bars.dropna(subset=['open','high','low','close'], inplace=True)
        bars = bars[~bars.index.duplicated(keep='first')]
        if bars.empty: continue

        atr14 = ti.calculate_atr(bars['high'], bars['low'], bars['close'], period=14)
        valid_index = bars.index[lookback if offset > 0 else 0:]

        for preset in rr_presets:
            labels_df = tbl.generate_labels_single_symbol(bars, atr14, rr_preset=preset, max_horizon=horizon)
            if labels_df.empty: continue
            labels_df = labels_df.loc[labels_df.index.intersection(valid_index)]
            if labels_df.empty: continue

            longs = labels_df[labels_df['direction'] == 1]
            shorts = labels_df[labels_df['direction'] == -1]

            for ts, row in longs.iterrows():
                meta = row.to_dict()
                if isinstance(meta.get('exit_time'), pd.Timestamp): meta['exit_time'] = meta['exit_time'].isoformat()
                y = _long_outcome_to_y(row['label'], str(row.get('barrier_hit','')))
                all_labels_for_symbol.append((symbol, ts.to_pydatetime().isoformat(), 'y_base_long', preset, int(y), json.dumps(meta), None))

            for ts, row in shorts.iterrows():
                meta = row.to_dict()
                if isinstance(meta.get('exit_time'), pd.Timestamp): meta['exit_time'] = meta['exit_time'].isoformat()
                y = _short_outcome_to_y(row['label'], str(row.get('barrier_hit','')))
                all_labels_for_symbol.append((symbol, ts.to_pydatetime().isoformat(), 'y_base_short', preset, int(y), json.dumps(meta), None))

    temp_dir = ROOT / 'artifacts' / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / f"labels_{symbol}.json"
    with temp_file_path.open('w') as f:
        json.dump(all_labels_for_symbol, f)
    logger.info(f"[Worker] Finished for {symbol}. Saved {len(all_labels_for_symbol)} labels to {temp_file_path}")
    return str(temp_file_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(DEFAULT_DB))
    ap.add_argument('--horizon', type=int, default=100)
    ap.add_argument('--chunk', type=int, default=50000)
    ap.add_argument('--lookback', type=int, default=200)
    ap.add_argument('--presets', type=str, default='1:2,1:3,1:4')
    ap.add_argument('--symbols', type=str, help='Optional comma-separated list of symbols to process (default: all)')
    args = ap.parse_args()

    db_writer = TradingDatabase(args.db)
    with db_writer.get_connection() as conn:
        if args.symbols:
            symbols_to_run = [s.strip() for s in args.symbols.split(',') if s.strip()]
        else:
            symbols_to_run = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM bars").fetchall()]

    if not symbols_to_run:
        raise RuntimeError("No symbols found to process.")

    rr_presets = [s.strip() for s in args.presets.split(',') if s.strip()]
    total_inserted = 0
    max_workers = multiprocessing.cpu_count()
    logger.info(f"[Main] Starting parallel relabeling with up to {max_workers} workers for {len(symbols_to_run)} symbols.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = { executor.submit(relabel_symbol, args.db, sym, rr_presets, args.horizon, args.chunk, args.lookback): sym for sym in symbols_to_run }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                temp_file_path_str = future.result()
                if temp_file_path_str:
                    temp_file_path = Path(temp_file_path_str)
                    logger.info(f"[Main] Reading labels for {symbol} from {temp_file_path}...")
                    with temp_file_path.open('r') as f:
                        labels_to_insert = json.load(f)

                    if labels_to_insert:
                        # Timestamps are already ISO strings from the worker
                        logger.info(f"[Main] Writing {len(labels_to_insert)} labels for {symbol} to database...")
                        inserted_count = db_writer.insert_labels_batch(labels_to_insert)
                        total_inserted += inserted_count
                        logger.info(f"[Main] Completed {symbol}, inserted {inserted_count} labels.")

                    temp_file_path.unlink()
                else:
                    logger.info(f"[Main] Completed {symbol}, no new labels to insert.")
            except Exception as e:
                logger.error(f"[Main] Symbol {symbol} failed with error: {e}", exc_info=True)

    logger.info(f"[Main] All done. Inserted total labels: {total_inserted}")


if __name__ == '__main__':
    main()

