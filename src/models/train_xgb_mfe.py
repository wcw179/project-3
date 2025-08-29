"""
XGBoost MFE Regressor Training Pipeline for Black-Swan Hunter
Implements purged time series CV and MFE prediction in risk multiples
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import gc

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.database import TradingDatabase
from src.features.black_swan_pipeline import BlackSwanFeaturePipeline
from src.features.black_swan_labeling import BlackSwanLabeling
from src.models.xgb_mfe_model import XGBMFERegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_xgb_mfe")

DEFAULT_DB = ROOT / 'data' / 'trading_system.db'
ARTIFACTS_DIR = ROOT / 'artifacts'

class PurgedTimeSeriesSplit:
    """Purged Time Series Split for financial data to prevent data leakage"""

    def __init__(self, n_splits: int = 5, embargo_bars: int = 3):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars

    def split(self, X, y=None, groups=None):
        """Generate train/test splits with purging and embargo"""
        n_samples = len(X)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test set boundaries
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = list(range(test_start, test_end))

            # Training set with embargo
            train_end = max(0, test_start - self.embargo_bars)
            train_indices = list(range(0, train_end))

            # Add training data after test set (if available)
            if test_end < n_samples:
                train_start_after = min(n_samples, test_end + self.embargo_bars)
                train_indices.extend(list(range(train_start_after, n_samples)))

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

def load_training_data(db_path: str, symbol: str, limit: int = 200000):
    """Load and prepare training data for XGB MFE regressor"""
    logger.info(f"Loading training data for {symbol}")

    db = TradingDatabase(db_path)

    # Load OHLCV data
    with db.get_connection() as conn:
        query = """
        SELECT time, open, high, low, close, IFNULL(volume, 0) as volume
        FROM bars
        WHERE symbol = ?
        ORDER BY time DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[symbol, limit], parse_dates=['time'])
        df = df.sort_values('time').set_index('time')

    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}")

    logger.info(f"Loaded {len(df)} bars for {symbol}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default=str(DEFAULT_DB))
    ap.add_argument('--symbol', type=str, required=True, help='Symbol to train on')
    ap.add_argument('--limit', type=int, default=200000, help='Max bars to load')
    ap.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    ap.add_argument('--embargo-bars', type=int, default=3, help='Embargo bars for purged CV')
    ap.add_argument('--optimize-hyperparams', action='store_true', help='Run hyperparameter optimization')
    ap.add_argument('--n-trials', type=int, default=50, help='Optuna trials for hyperparameter optimization')
    args = ap.parse_args()

    logger.info(f"--- Starting XGB MFE Training for {args.symbol} ---")

    # Load data
    df = load_training_data(args.db, args.symbol, args.limit)

    # Initialize pipelines
    feature_pipeline = BlackSwanFeaturePipeline()
    labeling_pipeline = BlackSwanLabeling(forecast_horizon=100)

    # Generate features
    logger.info("Generating XGB features...")
    xgb_features = feature_pipeline.generate_xgb_features(df, args.symbol)

    # Validate features
    if not feature_pipeline.validate_features(xgb_features, 'xgb'):
        logger.error("Feature validation failed")
        return

    # Generate labels
    logger.info("Generating MFE labels...")
    labels_dict = labeling_pipeline.generate_labels_for_symbol(df, args.symbol)

    # Validate labels
    if not labeling_pipeline.validate_labels(labels_dict):
        logger.error("Label validation failed")
        return
<<<<<<< HEAD

=======
    
>>>>>>> ba74014bc27d7d2a54cd990e89509a76b3a012d5
    # Build training set with direction-aware duplication and alignment
    X_parts = []
    y_parts = []
    for dir_key, dir_flag in [('xgb_long', 1), ('xgb_short', -1)]:
        lbl_df = labels_dict.get(dir_key, pd.DataFrame())
        if lbl_df.empty:
            logger.warning(f"No labels for {dir_key}, skipping")
            continue
        common_idx = xgb_features.index.intersection(lbl_df.index)
        X_dir = xgb_features.loc[common_idx].copy()
        # Remove non-numeric/non-feature column and add direction flag
        feature_cols = [c for c in X_dir.columns if c != 'symbol']
        X_dir = X_dir[feature_cols]
        X_dir['direction_flag'] = dir_flag
        y_dir = lbl_df.loc[common_idx]['mfe_target']
        # Keep index to allow later sorting/consistency
        X_dir.index = common_idx
        y_dir.index = common_idx
        X_parts.append(X_dir)
        y_parts.append(y_dir)

    if not X_parts:
        logger.error("No training data assembled. Exiting.")
        return

    X_all = pd.concat(X_parts).sort_index()
    y_all = pd.concat(y_parts).sort_index()

    # Ensure aligned order
    common_idx2 = X_all.index.intersection(y_all.index)
    X_all = X_all.loc[common_idx2]
    y_all = y_all.loc[common_idx2]

    X_numeric = X_all
    y = y_all

    logger.info(f"Training data shape: X={X_numeric.shape}, y={y.shape}")
    logger.info(f"MFE target statistics: mean={y.mean():.3f}, std={y.std():.3f}, max={y.max():.3f}")

    # Initialize model
    model = XGBMFERegressor(
        objective='reg:squarederror',
        random_state=42
    )

    # Setup cross-validation
    cv_splitter = PurgedTimeSeriesSplit(
        n_splits=args.cv_folds,
        embargo_bars=args.embargo_bars
    )

    # Train model with cross-validation
    training_results = model.train(
        X=X_numeric,
        y=y.values,
        cv_splitter=cv_splitter,
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials
    )

    # Save model
    model_dir = ARTIFACTS_DIR / 'models' / 'xgb_mfe' / args.symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model.json'

    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training results
    results_path = model_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for key, value in training_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.generic):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value

        json.dump(serializable_results, f, indent=2)

    logger.info(f"Training results saved to {results_path}")

    # Save the training data for future analysis
    data_path = model_dir / 'training_data.pkl'
    try:
        training_data_df = X_all.copy()
        training_data_df['mfe_target'] = y_all
        training_data_df.to_pickle(data_path)
        logger.info(f"Training data saved to {data_path}")
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")

    # Print summary
    cv_rmse = training_results.get('cv_rmse_mean', 0)
    cv_r2 = training_results.get('cv_r2_mean', 0)
    feature_importance = training_results.get('feature_importance', {})

    logger.info(f"Cross-validation RMSE: {cv_rmse:.4f}")
    logger.info(f"Cross-validation R²: {cv_r2:.4f}")

    if feature_importance:
        logger.info("Top 10 most important features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")

    # Cleanup
    gc.collect()
    logger.info(f"--- XGB MFE Training for {args.symbol} Completed ---")

if __name__ == '__main__':
    main()
