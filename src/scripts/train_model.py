import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import gc

# --- Setup Project Environment ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.database import TradingDatabase
from src.models.lstm_model import LSTMTrendClassifier
from src.models.xgb_model import XGBMetaModel
from src.features.labeling import TripleBarrierLabeling

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_model")

DEFAULT_DB = ROOT / 'data' / 'trading_system.db'
ARTIFACTS_DIR = ROOT / 'artifacts'

# --- Data Loading (Corrected) ---
def load_training_data(db_path: str, symbol: str, feature_set: str, rr_preset: str, limit: int = 50000):
    """
    Loads and aligns features and labels from the trading_system.db for a specific symbol.
    Crucially, it loads BOTH long and short labels to create a unified target.
    """
    logger.info(f"Loading data for {symbol} with feature_set='{feature_set}', rr_preset='{rr_preset}'")
    db = TradingDatabase(db_path)

    with db.get_connection() as conn:
        query = f"""
        SELECT f.time, f.features, l.label_type, l.label_value, l.barrier_meta
        FROM features f
        INNER JOIN labels l ON f.symbol = l.symbol AND f.time = l.time
        WHERE f.symbol = ? AND f.feature_set = ? AND l.rr_preset = ? AND l.label_type IN ('y_base_long', 'y_base_short')
        ORDER BY f.time DESC
        LIMIT ?
        """
        # Fetch more to ensure we have enough data for both directions before limiting
        params = (symbol, feature_set, rr_preset, limit * 2)
        df_reversed = pd.read_sql_query(query, conn, params=params, parse_dates=['time'])
        df = df_reversed.sort_values(by='time', ascending=True)

    if df.empty:
        raise ValueError("No aligned data found for the given parameters.")

    # Unpack features
    features_df = pd.DataFrame([json.loads(f) for f in df['features']], index=df['time'])
    features_df = features_df[~features_df.index.duplicated(keep='first')]

    # Pivot labels to get long and short outcomes in the same row
    labels_pivot = df.pivot(index='time', columns='label_type', values=['label_value', 'barrier_meta'])
    labels_pivot.columns = ['_'.join(col).strip() for col in labels_pivot.columns.values]

    # Align features and labels
    aligned_idx = features_df.index.intersection(labels_pivot.index)
    features_df = features_df.loc[aligned_idx]
    labels_pivot = labels_pivot.loc[aligned_idx]

    # Take the most recent 'limit' samples
    features_df = features_df.tail(limit)
    labels_pivot = labels_pivot.tail(limit)

    logger.info(f"Loaded and aligned {len(features_df)} records.")
    return features_df, labels_pivot

# --- Main Training Workflow (Corrected) ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(DEFAULT_DB))
    ap.add_argument('--symbol', type=str, required=True, help='Symbol to train, e.g., EURUSDm')
    ap.add_argument('--model', type=str, required=True, choices=['lstm', 'xgb'], help='Model to train: lstm or xgb')
    ap.add_argument('--rr-preset', type=str, default='1:2', help='Risk-reward preset for the labels')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    # This argument is now unused, but kept for compatibility to avoid breaking calls
    # ap.add_argument('--label-type', type=str, default='y_base_long')

    logger.info(f"--- Starting Training Pipeline for {args.symbol} ({args.model}) ---")

    if args.model == 'lstm':
        X, labels_df = load_training_data(args.db, args.symbol, 'lstm', args.rr_preset)

        # Create a unified target variable: 1 for up, -1 for down, 0 for neutral
        y = pd.Series(0, index=labels_df.index, dtype=int)
        y[labels_df['label_value_y_base_long'] == 1] = 1
        y[labels_df['label_value_y_base_short'] == 1] = -1 # Note: short label is 1 when price goes down

        logger.info(f"LSTM training data shape: {X.shape}, Unified labels shape: {y.shape}")
        logger.info(f"Unified label distribution:\n{y.value_counts()}")

        model = LSTMTrendClassifier(n_features=X.shape[1])
        model.train(X=X.values, y=y.to_numpy(), epochs=args.epochs, batch_size=args.batch_size, use_purged_cv=True)

        model_path = ARTIFACTS_DIR / 'models' / 'lstm' / args.symbol
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path / f'model_unified_{args.rr_preset}.h5'))
        logger.info(f"LSTM model saved to {model_path}")

        eval_metrics = model.evaluate(X.values, y.to_numpy())
        logger.info(f"LSTM Final Accuracy: {eval_metrics['accuracy']:.4f}")

    elif args.model == 'xgb':
        X_xgb_base, labels_xgb = load_training_data(args.db, args.symbol, 'xgb', args.rr_preset)
        X_lstm_base, _ = load_training_data(args.db, args.symbol, 'lstm', args.rr_preset)

        lstm_model_path = ARTIFACTS_DIR / 'models' / 'lstm' / args.symbol / f'model_unified_{args.rr_preset}.h5'
        if not lstm_model_path.exists():
            logger.error(f"Unified LSTM model not found at {lstm_model_path}. Please train the LSTM model first.")
            return

        lstm_model = LSTMTrendClassifier(n_features=X_lstm_base.shape[1])
        lstm_model.load_model(str(lstm_model_path))

        logger.info("Generating LSTM probabilities as features for XGBoost...")
        probs, _ = lstm_model.predict(X_lstm_base.values)
        prob_df = pd.DataFrame(probs, columns=['p_down', 'p_neutral', 'p_up'], index=X_lstm_base.index[lstm_model.sequence_length:])

        # Construct the primary_labels DataFrame needed for meta-labeling
        long_meta = pd.DataFrame({'direction': 1, 'label': labels_xgb['label_value_y_base_long']}, index=labels_xgb.index)
        short_meta = pd.DataFrame({'direction': -1, 'label': labels_xgb['label_value_y_base_short']}, index=labels_xgb.index)
        primary_labels_for_meta = pd.concat([long_meta, short_meta]).sort_index()

        labeler = TripleBarrierLabeling()
        common_meta_idx = primary_labels_for_meta.index.intersection(prob_df.index)
        meta_labels = labeler.generate_meta_labels(primary_labels=primary_labels_for_meta.loc[common_meta_idx], lstm_probabilities=prob_df.loc[common_meta_idx])

        final_common_idx = X_xgb_base.index.intersection(meta_labels.index)
        X_xgb_final = X_xgb_base.loc[final_common_idx]
        y_meta = meta_labels.loc[final_common_idx]['meta_label']

        X_xgb_combined = X_xgb_final.join(prob_df, how='inner')

        logger.info(f"XGBoost training data shape: {X_xgb_combined.shape}, Meta-labels shape: {y_meta.shape}")

        xgb_model = XGBMetaModel()
        xgb_model.train(X=X_xgb_combined, y=y_meta.to_numpy(), optimize_hyperparams=True)

        model_path = ARTIFACTS_DIR / 'models' / 'xgb' / args.symbol
        model_path.mkdir(parents=True, exist_ok=True)
        xgb_model.save_model(str(model_path / f'model_unified_{args.rr_preset}.json'))
        logger.info(f"XGBoost model saved to {model_path}")

        eval_metrics = xgb_model.evaluate(X_xgb_combined, y_meta.to_numpy())
        trade_metrics = eval_metrics['classification_report'].get('Trade', {'precision': 0.0, 'recall': 0.0})
        logger.info(f"XGBoost Final AUC: {eval_metrics['auc_score']:.4f}")
        logger.info(f"XGBoost Precision (Trade): {trade_metrics['precision']:.4f}")
        logger.info(f"XGBoost Recall (Trade): {trade_metrics['recall']:.4f}")

    gc.collect()
    logger.info(f"--- Training Pipeline for {args.symbol} ({args.model}) Completed ---")

if __name__ == '__main__':
    main()

