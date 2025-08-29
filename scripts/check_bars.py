import sqlite3
from pathlib import Path
import sys

DB = Path('data/trading_system.db')
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else 'EURUSDm'

print('DB exists:', DB.exists(), 'path:', DB)

con = sqlite3.connect(str(DB))
cur = con.cursor()
try:
    cur.execute("SELECT COUNT(*) FROM bars WHERE symbol=?", (SYMBOL,))
    c = cur.fetchone()[0]
    print('bars_count', SYMBOL, c)
except Exception as e:
    print('bars_count_error', repr(e))
con.close()

