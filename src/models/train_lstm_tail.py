"""
LSTM Tail Classifier Training Pipeline for Black-Swan Hunter
Implements Focal Loss and walk-forward validation for tail event prediction
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
from src.models.lstm_tail_model import LSTMTailClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_lstm_tail")

DEFAULT_DB = ROOT / 'data' / 'trading_system.db'
ARTIFACTS_DIR = ROOT / 'artifacts'

class WalkForwardSplit:
    """Walk-forward validation with embargo for time series data"""

    def __init__(self, n_splits: int = 5, embargo_bars: int = 3, min_train_size: int = 20000):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        """Generate walk-forward splits"""
        n_samples = len(X)

        # Calculate split points
        test_size = (n_samples - self.min_train_size) // self.n_splits

        for i in range(self.n_splits):
            # Training set: expanding window
            train_start = 0
            train_end = self.min_train_size + i * test_size - self.embargo_bars

            # Test set: fixed size window
            test_start = train_end + self.embargo_bars
            test_end = min(test_start + test_size, n_samples)

            if train_end > train_start and test_end > test_start:
                train_indices = list(range(train_start, train_end))
                test_indices = list(range(test_start, test_end))

                yield train_indices, test_indices

def load_training_data(db_path: str, symbol: str, limit: int = 200000):
    """Load and prepare training data for LSTM tail classifier"""
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
    ap.add_argument('--cv-folds', type=int, default=5, help='Walk-forward validation folds')
    ap.add_argument('--embargo-bars', type=int, default=3, help='Embargo bars for walk-forward CV')
    ap.add_argument('--epochs', type=int, default=15, help='Training epochs')
    ap.add_argument('--batch-size', type=int, default=64, help='Batch size')
    ap.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    args = ap.parse_args()

    logger.info(f"--- Starting LSTM Tail Training for {args.symbol} ---")

    # Load data
    df = load_training_data(args.db, args.symbol, args.limit)

    # Initialize pipelines
    feature_pipeline = BlackSwanFeaturePipeline()
    labeling_pipeline = BlackSwanLabeling(forecast_horizon=100)

    # Generate features
    logger.info("Generating LSTM features...")
    lstm_features = feature_pipeline.generate_lstm_features(df, args.symbol)
    logger.info(f"Generated {len(lstm_features)} feature rows.")

    if not feature_pipeline.validate_features(lstm_features, 'lstm'):
        logger.error("Feature validation failed")
        return

    logger.info("Generating tail event labels...")
    labels_dict = labeling_pipeline.generate_labels_for_symbol(df, args.symbol)
    logger.info(f"Generated labels for keys: {list(labels_dict.keys())}")

    if not labeling_pipeline.validate_labels(labels_dict):
        logger.error("Label validation failed")
        return

    combined_labels = pd.concat([
        labels_dict['lstm_long'],
        labels_dict['lstm_short']
    ]).sort_index()
    logger.info(f"Combined labels into shape: {combined_labels.shape}")

    common_idx = lstm_features.index.intersection(combined_labels.index)
    X_features = lstm_features.loc[common_idx]
    y_labels = combined_labels.loc[common_idx]['tail_class']
    logger.info(f"Aligned features and labels on {len(common_idx)} common indices.")

    logger.info("Preparing LSTM sequences...")
    X_sequences, y_sequences = feature_pipeline.prepare_lstm_sequences(X_features, y_labels)

    if len(X_sequences) == 0:
        logger.error("No sequences generated - insufficient data. Check sequence length vs. data length.")
        return

    logger.info(f"Generated {len(X_sequences)} sequences with shape {X_sequences.shape}")
    class_dist = pd.Series(y_sequences).value_counts().sort_index().to_dict()
    logger.info(f"Class distribution: {class_dist}")

    # Calculate class weights for imbalanced data
    class_weights = labeling_pipeline.calculate_class_weights(pd.Series(y_sequences))

    # Initialize model
    model = LSTMTailClassifier(
        sequence_length=feature_pipeline.sequence_length,
        n_features=X_sequences.shape[2],
        n_classes=4,  # 0: <5R, 1: 5-10R, 2: 10-20R, 3: >=20R
        hidden_units=64,
        n_layers=2,
        dropout_rate=0.2,
        learning_rate=args.learning_rate,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    )

    # Setup walk-forward validation
    cv_splitter = WalkForwardSplit(
        n_splits=args.cv_folds,
        embargo_bars=args.embargo_bars,
        min_train_size=max(10000, len(X_sequences) // 2)
    )

    # Train model with walk-forward validation
    training_results = model.train(
        X=X_sequences,
        y=y_sequences,
        cv_splitter=cv_splitter,
        class_weights=class_weights,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Save model
    model_dir = ARTIFACTS_DIR / 'models' / 'lstm_tail' / args.symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model.h5'

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
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = value.item()
            elif isinstance(value, dict):
                # Handle nested dictionaries (like classification reports)
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serializable_dict[k] = v.item()
                    else:
                        serializable_dict[k] = v
                serializable_results[key] = serializable_dict
            else:
                serializable_results[key] = value

        json.dump(serializable_results, f, indent=2)

    logger.info(f"Training results saved to {results_path}")

    # Save the training data for future analysis
    data_path = model_dir / 'training_data.npz'
    try:
        np.savez_compressed(data_path, X=X_sequences, y=y_sequences)
        logger.info(f"Training data saved to {data_path}")
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")

    # Print summary
    cv_accuracy = training_results.get('cv_accuracy_mean', 0)
    cv_precision = training_results.get('cv_precision_mean', 0)
    cv_recall = training_results.get('cv_recall_mean', 0)

    logger.info(f"Cross-validation Accuracy: {cv_accuracy:.4f}")
    logger.info(f"Cross-validation Precision (macro): {cv_precision:.4f}")
    logger.info(f"Cross-validation Recall (macro): {cv_recall:.4f}")

    # Class-specific performance
    final_classification_report = training_results.get('final_classification_report', {})
    if final_classification_report:
        logger.info("Final model performance by class:")
        for class_id in range(4):
            class_key = str(class_id)
            if class_key in final_classification_report:
                metrics = final_classification_report[class_key]
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                logger.info(f"  Class {class_id}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # Cleanup
    gc.collect()
    logger.info(f"--- LSTM Tail Training for {args.symbol} Completed ---")

if __name__ == '__main__':
    main()
