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
from pathlib import Path
from datetime import datetime

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


def relabel_symbol(db: TradingDatabase, symbol: str, rr_presets, horizon: int = 100, chunk_size: int = 50000, lookback: int = 200):
    logger.info(f"Re-labeling symbol={symbol} horizon={horizon} presets={rr_presets}")

    # Determine total bars
    with db.get_connection() as conn:
        total_bars = conn.execute("SELECT COUNT(*) FROM bars WHERE symbol = ?", (symbol,)).fetchone()[0]
    if total_bars == 0:
        logger.warning(f"No bars for {symbol}")
        return 0

    ti = TechnicalIndicators()
    tbl = TripleBarrierLabeling(rr_presets=rr_presets)

    inserted = 0

    for offset in range(0, total_bars, chunk_size):
        read_offset = max(0, offset - lookback)
        read_limit = chunk_size + (offset - read_offset)
        with db.get_connection() as conn:
            q = f"SELECT time, open, high, low, close, IFNULL(volume,0) as volume FROM bars WHERE symbol = ? ORDER BY time LIMIT {read_limit} OFFSET {read_offset}"
            bars = pd.read_sql_query(q, conn, params=[symbol], parse_dates=['time'])
        if bars.empty:
            continue
        bars.rename(columns={'time': 'timestamp'}, inplace=True)
        bars.set_index('timestamp', inplace=True)
        bars.dropna(subset=['open','high','low','close'], inplace=True)
        bars = bars[~bars.index.duplicated(keep='first')]
        if bars.empty:
            continue

        # Indicators
        atr14 = ti.calculate_atr(bars['high'], bars['low'], bars['close'], period=14)

        # Valid portion to persist (skip warm-up)
        valid_index = bars.index[lookback if offset > 0 else 0:]

        for preset in rr_presets:
            labels_df = tbl.generate_labels_single_symbol(bars, atr14, rr_preset=preset, max_horizon=horizon)
            if labels_df.empty:
                continue
            # Filter to valid window
            labels_df = labels_df.loc[labels_df.index.intersection(valid_index)]
            if labels_df.empty:
                continue

            # Persist two directions separately
            longs = labels_df[labels_df['direction'] == 1]
            shorts = labels_df[labels_df['direction'] == -1]

            for ts, row in longs.iterrows():
                meta = row.to_dict()
                if isinstance(meta.get('exit_time'), pd.Timestamp):
                    meta['exit_time'] = meta['exit_time'].isoformat()
                y = _long_outcome_to_y(row['label'], row.get('barrier_hit',''))
                ok = db.insert_labels(symbol=symbol, time=ts.to_pydatetime(), label_type='y_base_long',
                                      rr_preset=preset, label_value=int(y), barrier_meta=meta,
                                      sample_weight=None)
                if ok: inserted += 1

            for ts, row in shorts.iterrows():
                meta = row.to_dict()
                if isinstance(meta.get('exit_time'), pd.Timestamp):
                    meta['exit_time'] = meta['exit_time'].isoformat()
                y = _short_outcome_to_y(row['label'], row.get('barrier_hit',''))
                ok = db.insert_labels(symbol=symbol, time=ts.to_pydatetime(), label_type='y_base_short',
                                      rr_preset=preset, label_value=int(y), barrier_meta=meta,
                                      sample_weight=None)
                if ok: inserted += 1

        logger.info(f"Chunk offset {offset}: inserted so far {inserted}")

    logger.info(f"Finished {symbol}: inserted {inserted} labels")
    return inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(DEFAULT_DB))
    ap.add_argument('--horizon', type=int, default=100)
    ap.add_argument('--chunk', type=int, default=50000)
    ap.add_argument('--lookback', type=int, default=200)
    ap.add_argument('--presets', type=str, default='1:2,1:3,1:4')
    args = ap.parse_args()

    db = TradingDatabase(args.db)

    with db.get_connection() as conn:
        symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM bars").fetchall()]
    if not symbols:
        raise RuntimeError("No symbols in bars table. Run migration first.")

    rr_presets = [s.strip() for s in args.presets.split(',') if s.strip()]

    total_inserted = 0
    for sym in symbols:
        total_inserted += relabel_symbol(db, sym, rr_presets, args.horizon, args.chunk, args.lookback)

    logger.info(f"All done. Inserted total labels: {total_inserted}")


if __name__ == '__main__':
    main()

