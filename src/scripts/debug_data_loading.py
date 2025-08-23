"""
Data Loading Debugger

This script is designed to diagnose the root cause of the data insufficiency
error by inspecting the DataFrames at each step of the loading and alignment process.
It does not train any models.
"""

import json
import sqlite3
import pandas as pd
import logging
from pathlib import Path
import sys

# --- Setup Project Environment ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_debugger")

DB_PATH = ROOT / 'data' / 'm5_trading.db'
DATA_SUBSET = 100000
SYMBOL = "XAUUSDm"
PRESET = '1:2'

# --- Main Debugging Workflow ---
def debug_data_loading():
    logger.info(f"--- Starting Data Loading Debug for {SYMBOL} ---")

    # Step 1: Load raw aligned data from DB
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
        SELECT f.time, f.feat, l.meta FROM 
        (SELECT time, feat, symbol FROM features WHERE symbol = ?) f
        INNER JOIN 
        (SELECT time, meta, symbol FROM labels WHERE symbol = ?) l
        ON f.time = l.time 
        ORDER BY f.time DESC LIMIT {DATA_SUBSET}
        """
        df_reversed = pd.read_sql_query(query, conn, params=(SYMBOL, SYMBOL), parse_dates=['time'])
        df = df_reversed.sort_values(by='time', ascending=True)
    
    if df.empty:
        logger.error("FAIL: No aligned data found from the database query.")
        return

    logger.info(f"PASS: Loaded raw aligned data. Shape: {df.shape}")
    print("\nRaw DataFrame head:\n", df.head())

    df.set_index('time', inplace=True)
    df.index.name = 'timestamp'

    # Step 2: Unpack features
    lstm_features = pd.DataFrame([json.loads(f).get('lstm', {}) for f in df['feat']], index=df.index)
    logger.info(f"Unpacked LSTM features. Shape: {lstm_features.shape}")
    print("\nLSTM Features head:\n", lstm_features.head())

    # Step 3: Unpack primary labels
    primary_labels_data = []
    for idx, row in df.iterrows():
        meta = json.loads(row['meta'])
        if PRESET in meta:
            for direction in ['long', 'short']:
                if direction in meta[PRESET]:
                    label_info = meta[PRESET][direction]
                    primary_labels_data.append({
                        'timestamp': idx, 'direction': 1 if direction == 'long' else -1,
                        'label': label_info.get('label', 0)
                    })
    primary_labels = pd.DataFrame(primary_labels_data).set_index('timestamp')
    logger.info(f"Unpacked primary labels. Shape: {primary_labels.shape}")
    print("\nPrimary Labels head:\n", primary_labels.head())

    # Step 4: Prepare final training data for LSTM
    y_base = primary_labels['label']
    logger.info(f"Created y_base (labels series). Shape: {y_base.shape}")
    
    # This is the critical alignment step that was failing
    X_lstm = lstm_features.loc[lstm_features.index.intersection(y_base.index)]
    logger.info(f"Aligned X_lstm with y_base index. Shape: {X_lstm.shape}")
    
    y_base_aligned = y_base.loc[X_lstm.index]
    logger.info(f"Aligned y_base with X_lstm index. Shape: {y_base_aligned.shape}")

    if X_lstm.shape[0] < 60:
        logger.error(f"FAIL: Final dataset has {X_lstm.shape[0]} rows, which is less than the required 60 for the LSTM sequence.")
    else:
        logger.info("PASS: Final dataset appears sufficient for LSTM training.")

if __name__ == '__main__':
    debug_data_loading()

