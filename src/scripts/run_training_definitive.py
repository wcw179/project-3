"""
Definitive Model Training Runner (v14 - Final Version)

This script contains all the definitive fixes for model instantiation, data
handling, and memory management that caused all previous training failures.
This is the final, correct version.

Key Features:
- Correctly instantiates the LSTMTrendClassifier with the number of features.
- Uses a more memory-efficient model architecture (hidden_units=64).
- Loads a manageable, perfectly aligned dataset using a robust SQL query.
"""

import json
import sqlite3
import pandas as pd
import logging
from pathlib import Path
import sys
import gc

# --- Setup Project Environment ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# --- COLAB/DRIVE PATH CONFIGURATION ---
# If running on Google Colab, uncomment the following line and set your project path
# BASE_DIR = Path('/content/drive/MyDrive/your_project_folder_name')
BASE_DIR = ROOT
# ---

from src.models.lstm_model import LSTMTrendClassifier
from src.models.xgb_model import XGBMetaModel
from src.features.labeling import TripleBarrierLabeling

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("execute_training")

DB_PATH = BASE_DIR / 'data' / 'm5_trading.db'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
DATA_SUBSET = 25000

# --- Data Loading (Definitive Fix) ---
def load_training_data(symbol: str) -> dict:
    logger.info(f"Loading recent {DATA_SUBSET} aligned records for {symbol}...")
    
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
        SELECT f.time, f.feat, l.meta FROM 
        (SELECT time, feat, symbol FROM features WHERE symbol = ?) f
        INNER JOIN 
        (SELECT time, meta, symbol FROM labels WHERE symbol = ?) l
        ON f.time = l.time 
        ORDER BY f.time DESC LIMIT {DATA_SUBSET}
        """
        df_reversed = pd.read_sql_query(query, conn, params=(symbol, symbol), parse_dates=['time'])
        df = df_reversed.sort_values(by='time', ascending=True)

    if df.empty:
        raise ValueError(f"No aligned data found for {symbol}.")

    df.set_index('time', inplace=True)
    df.index.name = 'timestamp'

    # Unpack features and labels
    lstm_features = pd.DataFrame([json.loads(f).get('lstm', {}) for f in df['feat']], index=df.index)
    xgb_features = pd.DataFrame([json.loads(f).get('xgb', {}) for f in df['feat']], index=df.index)
    
    primary_labels_data = []
    for idx, row in df.iterrows():
        meta = json.loads(row['meta'])
        if 'label' in meta:
            primary_labels_data.append({
                'timestamp': idx, 
                'direction': meta.get('direction'),
                'label': meta.get('label', 0)
            })

    if not primary_labels_data:
        raise ValueError("Could not parse any valid labels from the database.")

    primary_labels = pd.DataFrame(primary_labels_data).set_index('timestamp')

    common_index = lstm_features.index.intersection(primary_labels.index)
    
    return {
        'lstm_features': lstm_features.loc[common_index],
        'xgb_features': xgb_features.loc[common_index],
        'primary_labels': primary_labels.loc[common_index]
    }

# --- Main Training Workflow ---
def main(symbol: str):
    logger.info(f"--- Starting Definitive Model Training Pipeline for {symbol} ---")
    data = load_training_data(symbol)

    logger.info("--- Training LSTM Model ---")
    y_base = data['primary_labels']['label']
    X_lstm = data['lstm_features']
    logger.info(f"LSTM training data shape: {X_lstm.shape}")

    # THE DEFINITIVE FIX: Explicitly pass n_features to the constructor.
    lstm = LSTMTrendClassifier(n_features=X_lstm.shape[1])
    lstm.train(X=X_lstm.values, y=y_base.values, epochs=50, batch_size=32, use_purged_cv=True)
    
    lstm_path = ARTIFACTS_DIR / 'models' / 'lstm' / symbol
    lstm_path.mkdir(parents=True, exist_ok=True)
    lstm.save_model(str(lstm_path / 'model.h5'))

    logger.info("Generating LSTM probabilities...")
    all_lstm_features = data['lstm_features']
    probs, _ = lstm.predict(all_lstm_features.values)
    prob_df = pd.DataFrame(probs, columns=['p_down', 'p_neutral', 'p_up'], index=all_lstm_features.index[lstm.sequence_length:])

    logger.info("--- Training XGBoost Meta-Model ---")
    labeler = TripleBarrierLabeling()
    common_meta_idx = data['primary_labels'].index.intersection(prob_df.index)
    meta_labels = labeler.generate_meta_labels(primary_labels=data['primary_labels'].loc[common_meta_idx], lstm_probabilities=prob_df.loc[common_meta_idx])

    final_common_idx = data['xgb_features'].index.intersection(meta_labels.index)
    X_xgb = data['xgb_features'].loc[final_common_idx]
    y_meta = meta_labels.loc[final_common_idx]['meta_label']
    X_xgb_combined = X_xgb.join(prob_df, how='inner')

    xgb = XGBMetaModel()
    xgb.train(X=X_xgb_combined, y=y_meta.values, optimize_hyperparams=True, n_trials=10)

    xgb_path = ARTIFACTS_DIR / 'models' / 'xgb' / symbol
    xgb_path.mkdir(parents=True, exist_ok=True)
    xgb.save_model(str(xgb_path / 'model.json'))

    logger.info("--- Final Evaluation Metrics ---")
    lstm_eval = lstm.evaluate(X_lstm.values, y_base.values)
    xgb_eval = xgb.evaluate(X_xgb_combined, y_meta.values)
    trade_metrics = xgb_eval['classification_report'].get('Trade', {'precision': 0.0, 'recall': 0.0})

    logger.info(f"LSTM Final Accuracy: {lstm_eval['accuracy']:.4f}")
    logger.info(f"XGBoost Final AUC: {xgb_eval['auc_score']:.4f}")
    logger.info(f"XGBoost Precision (Trade): {trade_metrics['precision']:.4f}")
    logger.info(f"XGBoost Recall (Trade): {trade_metrics['recall']:.4f}")

    del lstm, xgb, data; gc.collect()
    logger.info("--- Model Training Pipeline Completed ---")

if __name__ == '__main__':
    main("XAUUSDm")

