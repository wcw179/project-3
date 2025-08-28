import sqlite3
from pathlib import Path

DB_PATH = Path('data/trading_system.db')

if not DB_PATH.exists():
    print('DB_NOT_FOUND', DB_PATH)
    raise SystemExit(1)

con = sqlite3.connect(str(DB_PATH))
cur = con.cursor()

for tbl in ('features', 'labels'):
    try:
        cur.execute(f'DROP TABLE IF EXISTS {tbl}')
        print(f'Dropped table if existed: {tbl}')
    except sqlite3.Error as e:
        print(f'Error dropping {tbl}:', e)

con.commit()

# Show remaining tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print('Remaining tables:', [r[0] for r in cur.fetchall()])

con.close()

