"""
Migrate data from legacy m5_trading.db to TradingDatabase (trading_system.db)
- Bars: full copy
- Features: split combined JSON into rows per feature_set ('lstm'/'xgb')
- Labels: migrate existing legacy labels to y_base when rr_preset available in meta

Note: Legacy DB likely lost many RR presets due to primary key overwrite. This script
migrates what's available; for "đạt chuẩn" multi-preset, run relabel_trading_db.py after this.
"""
from pathlib import Path
import sqlite3
import json
import sys
import logging
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.database import TradingDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("migrate_to_trading_db")

SRC_DB = ROOT / 'data' / 'm5_trading.db'
DST_DB = ROOT / 'data' / 'trading_system.db'

CHUNK = 50000


def migrate_bars(src_conn: sqlite3.Connection, dst_db: TradingDatabase):
    logger.info("Migrating bars...")
    cur = src_conn.cursor()
    syms = [r[0] for r in cur.execute("SELECT DISTINCT symbol FROM bars").fetchall()]
    total = 0
    for sym in syms:
        logger.info(f"Bars: migrating symbol {sym}...")
        offset = 0
        while True:
            rows = cur.execute(
                "SELECT time, open, high, low, close, IFNULL(volume,0) as volume "
                "FROM bars WHERE symbol = ? ORDER BY time LIMIT ? OFFSET ?",
                (sym, CHUNK, offset)
            ).fetchall()
            if not rows:
                break
            df = pd.DataFrame(rows, columns=['time','open','high','low','close','volume'])
            # Robust datetime parsing across mixed formats / timezone-aware strings
            try:
                df['time'] = pd.to_datetime(df['time'], format='mixed', errors='ignore')
            except Exception:
                pass
            df = df.set_index('time')
            inserted = dst_db.insert_bars_batch(sym, df)
            total += inserted
            offset += CHUNK
            logger.info(f"Bars: {sym} +{inserted} (offset {offset})")
    logger.info(f"Bars migration done. Inserted {total} rows.")


def migrate_features(src_conn: sqlite3.Connection, dst_db: TradingDatabase):
    logger.info("Migrating features...")
    cur = src_conn.cursor()
    offset = 0
    total_rows = cur.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    migrated = 0
    while True:
        rows = cur.execute(
            "SELECT symbol, time, feat FROM features ORDER BY time LIMIT ? OFFSET ?",
            (CHUNK, offset)
        ).fetchall()
        if not rows:
            break
        for sym, ts, feat_str in rows:
            try:
                d = json.loads(feat_str) if feat_str else {}
            except Exception:
                d = {}
            for subset in ('lstm','xgb'):
                data = d.get(subset)
                if isinstance(data, dict) and len(data)>0:
                    dst_db.insert_features(sym, ts, subset, data)
                    migrated += 1
        offset += CHUNK
        logger.info(f"Features: migrated ~{migrated}/{total_rows*2} rows (counting subsets)")
    logger.info(f"Features migration done. Inserted subset rows: {migrated}.")


def migrate_labels(src_conn: sqlite3.Connection, dst_db: TradingDatabase):
    logger.info("Migrating labels (legacy -> y_base when rr_preset present)...")
    cur = src_conn.cursor()
    offset = 0
    total = cur.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    migrated = 0
    skipped_no_rr = 0
    while True:
        rows = cur.execute(
            "SELECT symbol, time, y, meta FROM labels ORDER BY time LIMIT ? OFFSET ?",
            (CHUNK, offset)
        ).fetchall()
        if not rows:
            break
        for sym, ts, y, meta_str in rows:
            rr = None
            meta = None
            if meta_str:
                try:
                    meta = json.loads(meta_str)
                    rr = meta.get('rr_preset')
                except Exception:
                    meta = None
            if not rr:
                skipped_no_rr += 1
                continue
            dst_db.insert_labels(sym, ts, label_type='y_base', rr_preset=rr,
                                 label_value=int(y), barrier_meta=meta, sample_weight=None)
            migrated += 1
        offset += CHUNK
        logger.info(f"Labels: migrated {migrated}/{total} (skipped_no_rr={skipped_no_rr})")
    logger.info(f"Labels migration done. Inserted rows: {migrated}. Skipped (no rr_preset): {skipped_no_rr}.")


def main():
    if not SRC_DB.exists():
        raise FileNotFoundError(f"Source DB not found: {SRC_DB}")
    DST_DB.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(SRC_DB))
    src_conn.row_factory = sqlite3.Row

    dst_db = TradingDatabase(str(DST_DB))

    migrate_bars(src_conn, dst_db)
    migrate_features(src_conn, dst_db)
    migrate_labels(src_conn, dst_db)

    src_conn.close()
    logger.info(f"Migration complete. New DB: {DST_DB}")


if __name__ == '__main__':
    main()

