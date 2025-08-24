import argparse
import sqlite3
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(Path(__file__).resolve().parents[2] / 'data' / 'trading_system.db'))
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB_NOT_FOUND: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("=== DB OVERVIEW ===")
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("Tables:", tables)

    for t in ['bars','features','labels']:
        try:
            cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
            schema = ", ".join([f"{c['name']}:{c['type']}" for c in cols])
            count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"Schema {t}: {schema}")
            print(f"Rows {t}: {count}")
        except Exception as e:
            print(f"Schema/Count {t}: ERROR {e}")

    print("\n=== LABELS (TradingDatabase) ===")
    try:
        lbl_types = [r[0] for r in cur.execute("SELECT DISTINCT label_type FROM labels").fetchall()]
        print("label_types:", lbl_types)
        rr_counts = {}
        for lt in lbl_types:
            rr_counts[lt] = {}
            rows = cur.execute("SELECT rr_preset, COUNT(*) FROM labels WHERE label_type=? GROUP BY rr_preset", (lt,)).fetchall()
            for rr, c in rows:
                rr_counts[lt][rr] = c
        print("rr_preset counts by label_type:", rr_counts)

        # y distribution by label_type
        y_dist = {}
        rows = cur.execute("SELECT label_type, label_value, COUNT(*) FROM labels GROUP BY label_type, label_value").fetchall()
        for lt, y, c in rows:
            y_dist.setdefault(lt, {})[y] = c
        print("label_value distribution by label_type:", y_dist)

        # Integrity: labels must have matching bars
        missing = cur.execute(
            "SELECT COUNT(*) FROM labels l LEFT JOIN bars b ON b.symbol=l.symbol AND b.time=l.time WHERE b.time IS NULL"
        ).fetchone()[0]
        print("Labels without matching bar:", missing)

    except Exception as e:
        print("Labels analysis ERROR:", e)

    print("\n=== FEATURES COVERAGE ===")
    try:
        total_features = cur.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        like_lstm = cur.execute("SELECT COUNT(*) FROM features WHERE json_extract(features, '$') IS NOT NULL AND feature_set='lstm'").fetchone()[0]
        like_xgb = cur.execute("SELECT COUNT(*) FROM features WHERE json_extract(features, '$') IS NOT NULL AND feature_set='xgb'").fetchone()[0]
        print({
            'total_feature_rows': total_features,
            'lstm_rows': like_lstm,
            'xgb_rows': like_xgb,
        })
        # Integrity: features must have matching bars
        missingf = cur.execute(
            "SELECT COUNT(*) FROM features f LEFT JOIN bars b ON b.symbol=f.symbol AND b.time=f.time WHERE b.time IS NULL"
        ).fetchone()[0]
        print("Features without matching bar:", missingf)
    except Exception as e:
        print("Features analysis ERROR:", e)

    conn.close()


if __name__ == '__main__':
    main()

