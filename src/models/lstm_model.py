"""
LSTM trend classifier for M5 Multi-Symbol Trend Bot
Implements sequence-based trend classification with purged K-fold validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.data.database import TradingDatabase

logger = logging.getLogger(__name__)

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        ce_loss = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        focal_loss = focal_weight * ce_loss
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

class PurgedTimeSeriesSplit:
    """Purged Time Series Split for financial data"""

    def __init__(self, n_splits=5, embargo_bars=3):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // self.n_splits
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = list(range(test_start, test_end))
            train_end = max(0, test_start - self.embargo_bars)
            train_indices = list(range(0, train_end))
            if test_end < n_samples:
                train_start_after = min(n_samples, test_end + self.embargo_bars)
                train_indices.extend(list(range(train_start_after, n_samples)))
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

class LSTMTrendClassifier:
    """LSTM model for trend classification"""

    def __init__(self, sequence_length=60, n_features=None, n_classes=3,
                 hidden_units=64, n_layers=2, dropout_rate=0.2,
                 learning_rate=0.001, use_focal_loss=True):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_focal_loss = use_focal_loss
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def build_model(self):
        if self.n_features is None:
            raise ValueError("n_features must be set before building model")
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        for i in range(self.n_layers):
            return_sequences = i < self.n_layers - 1
            model.add(LSTM(units=self.hidden_units, return_sequences=return_sequences,
                           dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
            if i < self.n_layers - 1:
                model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.n_classes, activation='softmax'))
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = FocalLoss() if self.use_focal_loss else 'categorical_crossentropy'
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'precision', 'recall'])
        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        return model

    def prepare_sequences(self, features: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(features) < self.sequence_length:
            logger.warning(f"Not enough data for sequences ({len(features)} < {self.sequence_length}). Returning empty arrays.")
            empty_X = np.empty((0, self.sequence_length, features.shape[-1]), dtype=features.dtype)
            empty_y = np.empty((0,), dtype=labels.dtype) if labels is not None else None
            return empty_X, empty_y
        X_sequences = np.array([features[i-self.sequence_length:i] for i in range(self.sequence_length, len(features))])
        y_sequences = np.array([labels[i] for i in range(self.sequence_length, len(labels))]) if labels is not None else None
        return X_sequences, y_sequences

    def preprocess_features(self, features: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        original_shape = features.shape
        features_reshaped = features.reshape(-1, features.shape[-1])
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features_reshaped)
        else:
            features_scaled = self.scaler.transform(features_reshaped)
        return features_scaled.reshape(original_shape)

    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        labels_mapped = labels + 1
        return to_categorical(labels_mapped, num_classes=self.n_classes)

    def decode_predictions(self, predictions: np.ndarray) -> np.ndarray:
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes - 1

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 100,
              batch_size: int = 128, sample_weights: Optional[np.ndarray] = None, use_purged_cv: bool = True) -> Dict:
        if self.n_features is None: self.n_features = X.shape[-1]
        if self.model is None: self.build_model()
        X_seq, y_seq = self.prepare_sequences(X, y)
        if X_seq.shape[0] == 0:
            logger.warning("Cannot train model: Not enough data to create sequences.")
            return {'history': {}, 'cv_scores': [], 'final_loss': None, 'final_accuracy': None}
        X_scaled = self.preprocess_features(X_seq, fit_scaler=True)
        y_encoded = self.encode_labels(y_seq)
        seq_weights = sample_weights[self.sequence_length:] if sample_weights is not None else None
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)]
        training_history = {}
        if use_purged_cv:
            cv_scores = []
            purged_cv = PurgedTimeSeriesSplit(n_splits=5, embargo_bars=3)
            for fold, (train_idx, val_idx) in enumerate(purged_cv.split(X_scaled)):
                logger.info(f"Training fold {fold + 1}/5")
                X_train_fold, y_train_fold = X_scaled[train_idx], y_encoded[train_idx]
                X_val_fold, y_val_fold = X_scaled[val_idx], y_encoded[val_idx]
                fold_weights = seq_weights[train_idx] if seq_weights is not None else None
                history = self.model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                                       epochs=epochs, batch_size=batch_size, sample_weight=fold_weights,
                                       callbacks=callbacks, verbose=0)
                val_loss, val_acc, _, _ = self.model.evaluate(X_val_fold, y_val_fold, verbose=0)
                cv_scores.append({'loss': val_loss, 'accuracy': val_acc})
                if fold == 4: training_history = history.history
            avg_loss = np.mean([score['loss'] for score in cv_scores])
            avg_acc = np.mean([score['accuracy'] for score in cv_scores])
            logger.info(f"Cross-validation results: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
        else:
            history = self.model.fit(X_scaled, y_encoded, validation_split=validation_split, epochs=epochs,
                                   batch_size=batch_size, sample_weight=seq_weights, callbacks=callbacks, verbose=1)
            training_history = history.history
        self.is_fitted = True
        return {'history': training_history, 'cv_scores': cv_scores if use_purged_cv else None,
                'final_loss': training_history.get('val_loss', [])[-1] if training_history.get('val_loss') else None,
                'final_accuracy': training_history.get('val_accuracy', [])[-1] if training_history.get('val_accuracy') else None}

    def predict(self, X: np.ndarray, return_probabilities: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self.is_fitted: raise ValueError("Model must be trained before making predictions")
        X_seq, _ = self.prepare_sequences(X)
        if X_seq.shape[0] == 0:
            empty_probs = np.empty((0, self.n_classes))
            empty_classes = np.empty((0,))
            return (empty_probs, empty_classes) if return_probabilities else empty_classes
        X_scaled = self.preprocess_features(X_seq, fit_scaler=False)
        predictions = self.model.predict(X_scaled, verbose=0)
        class_predictions = self.decode_predictions(predictions)
        return (predictions, class_predictions) if return_probabilities else class_predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        if not self.is_fitted: raise ValueError("Model must be trained before evaluation")
        probabilities, predictions = self.predict(X, return_probabilities=True)
        _, y_seq = self.prepare_sequences(X, y)
        if len(predictions) != len(y_seq):
             raise ValueError(f"Mismatch in prediction and label lengths after sequencing: {len(predictions)} vs {len(y_seq)}")
        accuracy = np.mean(predictions == y_seq)
        class_report = classification_report(y_seq, predictions, target_names=['Down', 'Neutral', 'Up'], output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_seq, predictions)
        return {'accuracy': accuracy, 'classification_report': class_report, 'confusion_matrix': conf_matrix.tolist(),
                'predictions': predictions.tolist(), 'probabilities': probabilities.tolist()}

    def save_model(self, filepath: str, include_scaler: bool = True):
        if not self.is_fitted: raise ValueError("Model must be trained before saving")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(filepath))
        if include_scaler:
            metadata = {'sequence_length': self.sequence_length, 'n_features': self.n_features, 'n_classes': self.n_classes,
                        'hidden_units': self.hidden_units, 'n_layers': self.n_layers, 'dropout_rate': self.dropout_rate,
                        'learning_rate': self.learning_rate, 'use_focal_loss': self.use_focal_loss}
            with open(filepath.parent / f"{filepath.stem}_scaler.pkl", 'wb') as f: pickle.dump(self.scaler, f)
            with open(filepath.parent / f"{filepath.stem}_metadata.json", 'w') as f: json.dump(metadata, f, indent=2)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str, load_scaler: bool = True):
        filepath = Path(filepath)
        self.model = tf.keras.models.load_model(str(filepath), custom_objects={'FocalLoss': FocalLoss})
        if load_scaler:
            scaler_path = filepath.parent / f"{filepath.stem}_scaler.pkl"
            metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f: self.scaler = pickle.load(f)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f: metadata = json.load(f)
                for key, value in metadata.items(): setattr(self, key, value)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray, method: str = 'permutation') -> Dict:
        if not self.is_fitted: raise ValueError("Model must be trained before calculating importance")
        if method == 'permutation':
            baseline_score = self.evaluate(X, y)['accuracy']
            X_seq, _ = self.prepare_sequences(X)
            X_scaled = self.preprocess_features(X_seq, fit_scaler=False)
            importance_scores = []
            for feature_idx in range(self.n_features):
                X_permuted = X_scaled.copy()
                np.random.shuffle(X_permuted[:, :, feature_idx])
                predictions = self.decode_predictions(self.model.predict(X_permuted, verbose=0))
                _, y_seq = self.prepare_sequences(X, y)
                permuted_score = np.mean(predictions == y_seq)
                importance_scores.append(baseline_score - permuted_score)
            return {'method': 'permutation', 'baseline_score': baseline_score, 'importance_scores': importance_scores,
                    'feature_ranking': np.argsort(importance_scores)[::-1].tolist()}
        else:
            raise ValueError(f"Unknown importance method: {method}")
