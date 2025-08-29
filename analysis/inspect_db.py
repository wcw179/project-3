import sqlite3
from pathlib import Path

def inspect_db(path: Path):
    print(f"\n--- Inspecting {path} ---")
    if not path.exists():
        print("File does not exist.")
        return
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("Tables:", tables)
    for t in tables:
        try:
            cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
            schema = ', '.join([f"{c[1]} {c[2]}" for c in cols])
            print(f"  Table {t}: {schema}")
            cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  -> rows: {cnt}")
            # Print a small sample
            sample = cur.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
            print(f"  -> sample(3): {sample}")
        except Exception as e:
            print(f"  Error inspecting {t}: {e}")
    con.close()

if __name__ == "__main__":
    for p in [Path('data/m5_trading.db'), Path('data/trading_system.db')]:
        inspect_db(p)

