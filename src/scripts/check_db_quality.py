import sqlite3, json, sys
from pathlib import Path
from datetime import datetime

DB_PATH = r"c:\Users\wcw17\Pictures\project 2\Project-3\project-3\data\m5_trading.db"

def main():
    p = Path(DB_PATH)
    if not p.exists():
        print(f"DB_NOT_FOUND: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
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

    # Features quality
    print("\n=== FEATURES QUALITY (sample up to 100k) ===")
    feat_stats = {
        'total': 0,
        'json_ok': 0,
        'has_lstm': 0,
        'has_xgb': 0,
        'has_both': 0,
        'json_errors': 0,
    }
    try:
        rows = cur.execute("SELECT symbol, time, feat FROM features ORDER BY time LIMIT 100000").fetchall()
        feat_stats['total'] = len(rows)
        for r in rows:
            try:
                d = json.loads(r['feat']) if r['feat'] is not None else {}
                feat_stats['json_ok'] += 1
            except Exception:
                d = {}
                feat_stats['json_errors'] += 1
            if isinstance(d, dict):
                has_l = isinstance(d.get('lstm'), dict) and len(d.get('lstm'))>0
                has_x = isinstance(d.get('xgb'), dict) and len(d.get('xgb'))>0
                if has_l: feat_stats['has_lstm'] += 1
                if has_x: feat_stats['has_xgb'] += 1
                if has_l and has_x: feat_stats['has_both'] += 1
        print(feat_stats)
    except Exception as e:
        print(f"Features stats ERROR: {e}")

    # Labels quality
    print("\n=== LABELS QUALITY (sample up to 100k) ===")
    lab_stats = {
        'total': 0,
        'json_ok': 0,
        'json_errors': 0,
        'y_counts': {},
        'rr_preset_counts': {},
        'missing_exit_time': 0,
        'short_holding_lt3': 0,
        'extreme_return_gt_10pct': 0,
    }
    try:
        rows = cur.execute("SELECT symbol, time, y, meta FROM labels ORDER BY time LIMIT 100000").fetchall()
        lab_stats['total'] = len(rows)
        for r in rows:
            y = r['y']
            lab_stats['y_counts'][y] = lab_stats['y_counts'].get(y, 0) + 1
            meta = {}
            try:
                meta = json.loads(r['meta']) if r['meta'] is not None else {}
                lab_stats['json_ok'] += 1
            except Exception:
                lab_stats['json_errors'] += 1
            rr = meta.get('rr_preset')
            if rr:
                lab_stats['rr_preset_counts'][rr] = lab_stats['rr_preset_counts'].get(rr,0)+1
            if meta.get('exit_time') in (None, ''):
                lab_stats['missing_exit_time'] += 1
            hp = meta.get('holding_period')
            if isinstance(hp, (int,float)) and hp < 3:
                lab_stats['short_holding_lt3'] += 1
            ret = meta.get('return')
            try:
                if abs(float(ret))>0.1:
                    lab_stats['extreme_return_gt_10pct'] += 1
            except Exception:
                pass
        print(lab_stats)
    except Exception as e:
        print(f"Labels stats ERROR: {e}")

    # Referential integrity
    print("\n=== REFERENTIAL INTEGRITY ===")
    try:
        q = (
            "SELECT COUNT(*) FROM labels l LEFT JOIN bars b "
            "ON b.symbol = l.symbol AND b.time = l.time WHERE b.time IS NULL"
        )
        missing = cur.execute(q).fetchone()[0]
        print(f"Labels without matching bar: {missing}")
    except Exception as e:
        print(f"Labels vs bars join ERROR: {e}")

    try:
        q = (
            "SELECT COUNT(*) FROM features f LEFT JOIN bars b "
            "ON b.symbol = f.symbol AND b.time = f.time WHERE b.time IS NULL"
        )
        missing = cur.execute(q).fetchone()[0]
        print(f"Features without matching bar: {missing}")
    except Exception as e:
        print(f"Features vs bars join ERROR: {e}")

    conn.close()

if __name__ == '__main__':
    main()

