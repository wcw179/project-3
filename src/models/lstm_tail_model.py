"""
LSTM Tail Classifier for Black-Swan Hunter Trading Bot
Predicts probability of tail events with Focal Loss for class imbalance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling severe class imbalance in tail events"""

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Apply softmax to predictions
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        
        # Calculate cross entropy
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config

class LSTMTailClassifier:
    """LSTM classifier for tail event prediction"""

    def __init__(self, sequence_length=60, n_features=None, n_classes=4,
                 hidden_units=64, n_layers=2, dropout_rate=0.2,
                 learning_rate=0.001, use_focal_loss=True,
                 focal_alpha=0.25, focal_gamma=2.0):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.model = None
        self.is_fitted = False

    def build_model(self):
        """Build LSTM architecture optimized for tail event classification"""
        if self.n_features is None:
            raise ValueError("n_features must be set before building model")
        
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # LSTM layers
        for i in range(self.n_layers):
            return_sequences = i < self.n_layers - 1
            model.add(LSTM(
                units=self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
            if i < self.n_layers - 1:
                model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.n_classes, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        
        if self.use_focal_loss:
            loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = 'categorical_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        return model

    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    class_weights: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare data for training"""
        # Encode labels to categorical
        y_categorical = to_categorical(y, num_classes=self.n_classes)
        
        # Calculate sample weights from class weights
        sample_weights = None
        if class_weights is not None:
            sample_weights = np.array([class_weights[label] for label in y])
        
        return X, y_categorical, sample_weights

    def train(self, X: np.ndarray, y: np.ndarray, cv_splitter,
              class_weights: Optional[Dict] = None,
              epochs: int = 50, batch_size: int = 64) -> Dict:
        """Train LSTM with walk-forward cross-validation"""
        
        if self.n_features is None:
            self.n_features = X.shape[2]
        
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X_processed, y_categorical, sample_weights = self.prepare_data(X, y, class_weights)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Cross-validation
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        logger.info("Starting walk-forward cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_processed)):
            logger.info(f"Training fold {fold + 1}")
            
            X_train_fold = X_processed[train_idx]
            X_val_fold = X_processed[val_idx]
            y_train_fold = y_categorical[train_idx]
            y_val_fold = y_categorical[val_idx]
            
            fold_weights = sample_weights[train_idx] if sample_weights is not None else None
            
            # Reset model for each fold
            self.build_model()
            
            # Train fold
            history = self.model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                batch_size=batch_size,
                sample_weight=fold_weights,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate fold
            val_pred_proba = self.model.predict(X_val_fold, verbose=0)
            val_pred_classes = np.argmax(val_pred_proba, axis=1)
            val_true_classes = np.argmax(y_val_fold, axis=1)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true_classes, val_pred_classes, average='macro', zero_division=0
            )
            accuracy = np.mean(val_pred_classes == val_true_classes)
            
            cv_scores['accuracy'].append(accuracy)
            cv_scores['precision'].append(precision)
            cv_scores['recall'].append(recall)
            cv_scores['f1'].append(f1)
        
        # Train final model on all data
        logger.info("Training final model on all data...")
        
        # Use last 20% as validation
        split_idx = int(len(X_processed) * 0.8)
        X_train_final = X_processed[:split_idx]
        X_val_final = X_processed[split_idx:]
        y_train_final = y_categorical[:split_idx]
        y_val_final = y_categorical[split_idx:]
        
        final_weights = sample_weights[:split_idx] if sample_weights is not None else None
        
        # Reset and train final model
        self.build_model()
        
        final_history = self.model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=epochs,
            batch_size=batch_size,
            sample_weight=final_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        
        # Final evaluation
        final_pred_proba = self.model.predict(X_val_final, verbose=0)
        final_pred_classes = np.argmax(final_pred_proba, axis=1)
        final_true_classes = np.argmax(y_val_final, axis=1)
        
        final_accuracy = np.mean(final_pred_classes == final_true_classes)
        final_classification_report = classification_report(
            final_true_classes, final_pred_classes,
            target_names=[f'Class_{i}' for i in range(self.n_classes)],
            output_dict=True,
            zero_division=0
        )
        final_confusion_matrix = confusion_matrix(final_true_classes, final_pred_classes)
        
        # Compile results
        training_results = {
            'cv_accuracy_mean': np.mean(cv_scores['accuracy']),
            'cv_accuracy_std': np.std(cv_scores['accuracy']),
            'cv_precision_mean': np.mean(cv_scores['precision']),
            'cv_precision_std': np.std(cv_scores['precision']),
            'cv_recall_mean': np.mean(cv_scores['recall']),
            'cv_recall_std': np.std(cv_scores['recall']),
            'cv_f1_mean': np.mean(cv_scores['f1']),
            'cv_f1_std': np.std(cv_scores['f1']),
            'final_accuracy': final_accuracy,
            'final_classification_report': final_classification_report,
            'final_confusion_matrix': final_confusion_matrix.tolist(),
            'training_history': {
                'loss': final_history.history.get('loss', []),
                'val_loss': final_history.history.get('val_loss', []),
                'accuracy': final_history.history.get('accuracy', []),
                'val_accuracy': final_history.history.get('val_accuracy', [])
            }
        }
        
        logger.info(f"Training completed. CV Accuracy: {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}")
        logger.info(f"Final validation accuracy: {final_accuracy:.4f}")
        
        return training_results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(X, verbose=0)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        
        accuracy = np.mean(predictions == y)
        
        classification_rep = classification_report(
            y, predictions,
            target_names=[f'Class_{i}' for i in range(self.n_classes)],
            output_dict=True,
            zero_division=0
        )
        
        confusion_mat = confusion_matrix(y, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }

    def save_model(self, filepath: str, include_metadata: bool = True):
        """Save model and metadata"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(str(filepath))
        
        if include_metadata:
            # Save metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'n_classes': self.n_classes,
                'hidden_units': self.hidden_units,
                'n_layers': self.n_layers,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'use_focal_loss': self.use_focal_loss,
                'focal_alpha': self.focal_alpha,
                'focal_gamma': self.focal_gamma
            }
            
            metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"LSTM Tail model saved to {filepath}")

    def load_model(self, filepath: str, load_metadata: bool = True):
        """Load model and metadata"""
        filepath = Path(filepath)
        
        # Load Keras model with custom objects
        custom_objects = {'FocalLoss': FocalLoss}
        self.model = tf.keras.models.load_model(str(filepath), custom_objects=custom_objects)
        
        if load_metadata:
            metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                for key, value in metadata.items():
                    setattr(self, key, value)
        
        self.is_fitted = True
        logger.info(f"LSTM Tail model loaded from {filepath}")
