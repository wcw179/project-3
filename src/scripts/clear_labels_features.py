import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.database import TradingDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("clear_labels_features")

DEFAULT_DB = ROOT / 'data' / 'trading_system.db'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(DEFAULT_DB))
    args = ap.parse_args()

    db = TradingDatabase(args.db)
    logger.info(f"Opening database: {args.db}")

    with db.get_connection() as conn:
        cur = conn.cursor()
        # Check existence
        tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
        missing = [t for t in ('features','labels') if t not in tables]
        if missing:
            logger.warning(f"Missing tables: {missing}")
        # Counts before
        cnt_features_before = cur.execute("SELECT COUNT(*) FROM features").fetchone()[0] if 'features' in tables else 0
        cnt_labels_before = cur.execute("SELECT COUNT(*) FROM labels").fetchone()[0] if 'labels' in tables else 0
        logger.info(f"Before delete: features={cnt_features_before}, labels={cnt_labels_before}")
        # Delete
        if 'features' in tables:
            cur.execute("DELETE FROM features;")
        if 'labels' in tables:
            cur.execute("DELETE FROM labels;")
        conn.commit()
        # Vacuum
        cur.execute("VACUUM;")
        # Counts after
        cnt_features_after = cur.execute("SELECT COUNT(*) FROM features").fetchone()[0] if 'features' in tables else 0
        cnt_labels_after = cur.execute("SELECT COUNT(*) FROM labels").fetchone()[0] if 'labels' in tables else 0
        logger.info(f"After delete: features={cnt_features_after}, labels={cnt_labels_after}")

    logger.info("Completed clearing features and labels.")

if __name__ == '__main__':
    main()

