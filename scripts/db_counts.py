import sqlite3
from pathlib import Path
import sys

DB = Path('data/trading_system.db')
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else 'EURUSDm'

if not DB.exists():
    print('DB_NOT_FOUND', DB)
    raise SystemExit(1)

con = sqlite3.connect(str(DB))
cur = con.cursor()

def count(q, params=()):
    try:
        cur.execute(q, params)
        return cur.fetchone()[0]
    except Exception as e:
        return f'ERR:{e}'

print('symbol:', SYMBOL)
print('bars_count:', count("SELECT COUNT(*) FROM bars WHERE symbol=?", (SYMBOL,)))
print('features_count:', count("SELECT COUNT(*) FROM features WHERE symbol=?", (SYMBOL,)))
print('labels_count:', count("SELECT COUNT(*) FROM labels WHERE symbol=?", (SYMBOL,)))

# Show a sample timestamp overlap counts
try:
    cur.execute("SELECT COUNT(*) FROM features f LEFT JOIN bars b ON b.symbol=f.symbol AND b.time=f.time WHERE f.symbol=? AND b.time IS NULL", (SYMBOL,))
    print('features_without_matching_bar:', cur.fetchone()[0])
except Exception as e:
    print('features_join_check_error:', e)

try:
    cur.execute("SELECT COUNT(*) FROM labels l LEFT JOIN bars b ON b.symbol=l.symbol AND b.time=l.time WHERE l.symbol=? AND b.time IS NULL", (SYMBOL,))
    print('labels_without_matching_bar:', cur.fetchone()[0])
except Exception as e:
    print('labels_join_check_error:', e)

con.close()

