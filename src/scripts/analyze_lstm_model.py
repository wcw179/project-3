import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.database import TradingDatabase
from src.models.lstm_model import LSTMTrendClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze_lstm")

DEFAULT_DB = ROOT / 'data' / 'trading_system.db'
ARTIFACTS_DIR = ROOT / 'artifacts'


def load_unified_data(db_path: str, symbol: str, limit: int = 50000, rr_preset: str = '1:2'):
    db = TradingDatabase(db_path)
    with db.get_connection() as conn:
        query = f"""
        SELECT f.time, f.features, l.label_type, l.label_value
        FROM features f
        INNER JOIN labels l ON f.symbol = l.symbol AND f.time = l.time
        WHERE f.symbol = ? AND f.feature_set = 'lstm' AND l.rr_preset = ? AND l.label_type IN ('y_base_long','y_base_short')
        ORDER BY f.time DESC
        LIMIT ?
        """
        params = (symbol, rr_preset, limit * 2)
        df_rev = pd.read_sql_query(query, conn, params=params, parse_dates=['time'])
        df = df_rev.sort_values(by='time', ascending=True)
    if df.empty:
        raise ValueError("No data found to analyze.")
    X = pd.DataFrame([json.loads(f) for f in df['features']], index=df['time'])
    X = X[~X.index.duplicated(keep='first')]
    labels_pivot = df.pivot(index='time', columns='label_type', values='label_value')
    aligned_idx = X.index.intersection(labels_pivot.index)
    X = X.loc[aligned_idx]
    labels_pivot = labels_pivot.loc[aligned_idx]
    X = X.tail(limit)
    labels_pivot = labels_pivot.tail(limit)
    # unified target: 1 up, -1 down, 0 neutral
    y = pd.Series(0, index=labels_pivot.index, dtype=int)
    if 'y_base_long' in labels_pivot.columns:
        y[labels_pivot['y_base_long'] == 1] = 1
    if 'y_base_short' in labels_pivot.columns:
        y[labels_pivot['y_base_short'] == 1] = -1
    return X, y


def find_model_path(symbol: str, rr_preset: str = '1:2') -> Path:
    base = ARTIFACTS_DIR / 'models' / 'lstm' / symbol
    if not base.exists():
        raise FileNotFoundError(f"Model directory not found: {base}")

    logger.info(f"Scanning model directory: {base}")
    children = list(base.iterdir())
    for c in children:
        logger.info(f"- found: {c} (dir={c.is_dir()})")

    # Prefer unified models
    unified_candidates = sorted([p for p in children if p.name.startswith('model_unified')], key=lambda p: p.stat().st_mtime, reverse=True)
    if unified_candidates:
        return unified_candidates[0]

    # Fallback to old naming
    legacy_candidates = sorted([p for p in children if p.name.startswith('model_y_base_long')], key=lambda p: p.stat().st_mtime, reverse=True)
    if legacy_candidates:
        return legacy_candidates[0]

    raise FileNotFoundError(f"No suitable LSTM model found in {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(DEFAULT_DB))
    ap.add_argument('--symbol', type=str, required=True)
    ap.add_argument('--limit', type=int, default=50000)
    ap.add_argument('--rr-preset', type=str, default='1:2')
    args = ap.parse_args()

    logger.info(f"Analyzing LSTM model for {args.symbol} with rr_preset={args.rr_preset}")
    X, y = load_unified_data(args.db, args.symbol, limit=args.limit, rr_preset=args.rr_preset)
    logger.info(f"Data loaded: X shape={X.shape}, label distribution=\n{y.value_counts()}")

    model_path = find_model_path(args.symbol, rr_preset=args.rr_preset)
    logger.info(f"Using model at: {model_path}")

    model = LSTMTrendClassifier(n_features=X.shape[1])
    model.load_model(str(model_path))

    eval_metrics = model.evaluate(X.values, y.to_numpy())

    acc = eval_metrics['accuracy']
    cls_report = eval_metrics['classification_report']
    conf_mat = eval_metrics['confusion_matrix']

    logger.info("===== LSTM EURUSDm Analysis Summary =====")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Samples: {len(y)} | Accuracy: {acc:.4f}")
    logger.info("Class distribution (true):\n" + y.value_counts().to_string())
    for cls_name in ['Down', 'Neutral', 'Up']:
        if cls_name in cls_report:
            m = cls_report[cls_name]
            logger.info(f"{cls_name:7s} | precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1-score']:.3f}")
    logger.info("Confusion matrix [rows=true, cols=pred]:")
    for row in conf_mat:
        logger.info(str(row))

if __name__ == '__main__':
    main()

