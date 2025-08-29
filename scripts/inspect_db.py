import sqlite3
from pathlib import Path

db_path = Path('data/trading_system.db')
if not db_path.exists():
    print('DB_NOT_FOUND', db_path)
    raise SystemExit(1)

con = sqlite3.connect(str(db_path))
cur = con.cursor()

print('--- Listing objects (tables/views) ---')
cur.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name")
rows = cur.fetchall()
for name, typ, sql in rows:
    print(f'{typ.upper():>5}: {name}')
    if typ == 'table':
        cur.execute(f'PRAGMA table_info("{name}")')
        cols = cur.fetchall()
        print('      Columns:', ', '.join([c[1] for c in cols]))

print('\n--- Searching for columns exactly named futures/label ---')
hits = []
for name, typ, _ in rows:
    if typ != 'table':
        continue
    cur.execute(f'PRAGMA table_info("{name}")')
    for col in cur.fetchall():
        if col[1].lower() in ('futures', 'label'):
            hits.append((name, col[1]))
print('HITS:', hits)

print('\n--- Searching for tables exactly named futures/label ---')
table_names = [n for n,t,_ in rows if t=='table']
print('TABLE_MATCHES:', [n for n in table_names if n.lower() in ('futures','label')])

con.close()

