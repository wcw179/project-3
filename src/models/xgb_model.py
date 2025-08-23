"""
XGBoost meta-model for M5 Multi-Symbol Trend Bot
Implements execution filtering using LSTM probabilities and market features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import optuna
from xgboost.callback import EarlyStopping



logger = logging.getLogger(__name__)

class XGBMetaModel:
    """XGBoost meta-model for trade execution filtering"""
    
    def __init__(self, objective='binary:logistic', use_calibration=True, 
                 random_state=42):
        self.objective = objective
        self.use_calibration = use_calibration
        self.random_state = random_state
        
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Default hyperparameters
        self.params = {
            'objective': objective,
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': random_state,
            'n_jobs': -1
        }
    
    def prepare_features(self, features: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Prepare features for XGBoost"""
        # Handle missing values
        features_clean = features.fillna(0)
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = features_clean.columns.tolist()
        
        # Scale features (optional for XGBoost, but can help)
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features_clean)
        else:
            features_scaled = self.scaler.transform(features_clean)
        
        return features_scaled
    
    def calculate_sample_weights(self, y: np.ndarray, method: str = 'balanced') -> np.ndarray:
        """Calculate sample weights for imbalanced classes"""
        if method == 'balanced':
            from sklearn.utils.class_weight import compute_sample_weight
            return compute_sample_weight('balanced', y)
        elif method == 'custom':
            # Custom weighting: higher weight for positive class
            weights = np.ones(len(y))
            positive_ratio = np.mean(y)
            if positive_ratio > 0:
                weights[y == 1] = (1 - positive_ratio) / positive_ratio
            return weights
        else:
            return np.ones(len(y))
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                sample_weights: Optional[np.ndarray] = None,
                                n_trials: int = 100) -> Dict:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'objective': self.objective,
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                train_weights = sample_weights[train_idx] if sample_weights is not None else None
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, sample_weight=train_weights, verbose=False)
                
                val_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Update parameters with best trial
        best_params = study.best_params
        best_params.update({
            'objective': self.objective,
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_jobs': -1
        })
        
        self.params = best_params
        
        logger.info(f"Hyperparameter optimization completed. Best AUC: {study.best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def train(self, X: pd.DataFrame, y: np.ndarray,
              sample_weights: Optional[np.ndarray] = None,
              optimize_hyperparams: bool = True,
              validation_split: float = 0.2) -> Dict:
        """Train XGBoost meta-model"""
        
        # Prepare features
        X_processed = self.prepare_features(X, fit_scaler=True)
        
        # Calculate sample weights if not provided
        if sample_weights is None:
            sample_weights = self.calculate_sample_weights(y, method='balanced')
        
        # Optimize hyperparameters
        if optimize_hyperparams:
            optimization_results = self.optimize_hyperparameters(
                X_processed, y, sample_weights, n_trials=50
            )
        else:
            optimization_results = None
        
        # Split data for validation
        split_idx = int(len(X_processed) * (1 - validation_split))
        X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = sample_weights[:split_idx]
        
        # Train model
        self.model = xgb.XGBClassifier(**self.params)
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=eval_set,
            callbacks=[xgb.callback.EarlyStopping(rounds=20)],
            verbose=False
        )
        
        # Calibrate probabilities
        if self.use_calibration:
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic', cv=3
            )
            self.calibrated_model.fit(X_train, y_train, sample_weight=train_weights)
        
        self.is_fitted = True
        
        # Evaluate on validation set
        val_predictions = self.predict_proba(X.iloc[split_idx:])
        val_auc = roc_auc_score(y_val, val_predictions)
        val_brier = brier_score_loss(y_val, val_predictions)
        
        training_results = {
            'validation_auc': val_auc,
            'validation_brier_score': val_brier,
            'feature_importance': self.get_feature_importance(),
            'optimization_results': optimization_results,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        }
        
        logger.info(f"XGB training completed. Validation AUC: {val_auc:.4f}, Brier Score: {val_brier:.4f}")
        
        return training_results
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.prepare_features(X, fit_scaler=False)
        
        if self.use_calibration and self.calibrated_model is not None:
            probabilities = self.calibrated_model.predict_proba(X_processed)[:, 1]
        else:
            probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        return probabilities
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> Dict:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(y, probabilities)
        brier_score = brier_score_loss(y, probabilities)
        
        # Classification report
        class_report = classification_report(
            y, predictions,
            target_names=['Skip', 'Trade'],
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y, predictions)
        
        return {
            'auc_score': auc_score,
            'brier_score': brier_score,
            'accuracy': class_report['accuracy'],
            'precision': class_report['Trade']['precision'],
            'recall': class_report['Trade']['recall'],
            'f1_score': class_report['Trade']['f1-score'],
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_scores = self.model.feature_importances_
        
        if self.feature_names is not None:
            importance_dict = dict(zip(self.feature_names, importance_scores))
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_importance = [(f'feature_{i}', score) for i, score in enumerate(importance_scores)]
            sorted_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'importance_type': importance_type,
            'feature_importance': dict(sorted_importance),
            'top_10_features': sorted_importance[:10]
        }
    
    def optimize_threshold(self, X: pd.DataFrame, y: np.ndarray, 
                          metric: str = 'f1') -> Dict:
        """Optimize classification threshold"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before threshold optimization")
        
        probabilities = self.predict_proba(X)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            
            if metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y, predictions)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y, predictions, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y, predictions, zero_division=0)
            elif metric == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y, predictions)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        return {
            'best_threshold': best_threshold,
            'best_score': best_score,
            'metric': metric,
            'threshold_scores': dict(zip(thresholds, scores))
        }
    
    def save_model(self, filepath: str):
        """Save model and associated components"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main model
        self.model.save_model(str(filepath))
        
        # Save additional components
        components = {
            'scaler': self.scaler,
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'params': self.params,
            'use_calibration': self.use_calibration,
            'objective': self.objective
        }
        
        components_path = filepath.parent / f"{filepath.stem}_components.pkl"
        with open(components_path, 'wb') as f:
            pickle.dump(components, f)
        
        logger.info(f"XGB model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and associated components"""
        filepath = Path(filepath)
        
        # Load main model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(filepath))
        
        # Load additional components
        components_path = filepath.parent / f"{filepath.stem}_components.pkl"
        if components_path.exists():
            with open(components_path, 'rb') as f:
                components = pickle.load(f)
            
            self.scaler = components['scaler']
            self.calibrated_model = components['calibrated_model']
            self.feature_names = components['feature_names']
            self.params = components['params']
            self.use_calibration = components['use_calibration']
            self.objective = components['objective']
        
        self.is_fitted = True
        logger.info(f"XGB model loaded from {filepath}")
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict:
        """Explain a single prediction using SHAP-like approach"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before explaining predictions")
        
        try:
            import shap
            
            X_processed = self.prepare_features(X, fit_scaler=False)
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_processed[index:index+1])
            
            # Get feature contributions
            feature_contributions = dict(zip(self.feature_names, shap_values[0]))
            sorted_contributions = sorted(
                feature_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            return {
                'prediction': self.predict_proba(X.iloc[index:index+1])[0],
                'feature_contributions': dict(sorted_contributions),
                'top_positive_features': [(k, v) for k, v in sorted_contributions if v > 0][:5],
                'top_negative_features': [(k, v) for k, v in sorted_contributions if v < 0][:5]
            }
            
        except ImportError:
            logger.warning("SHAP not available. Using feature importance as proxy.")
            
            # Fallback: use feature importance
            importance = self.get_feature_importance()
            prediction = self.predict_proba(X.iloc[index:index+1])[0]
            
            return {
                'prediction': prediction,
                'feature_importance': importance['feature_importance'],
                'note': 'SHAP not available, showing feature importance instead'
            }
    
    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, 
                      cv_folds: int = 5, sample_weights: Optional[np.ndarray] = None) -> Dict:
        """Perform time series cross-validation"""
        if not self.is_fitted:
            # Train model first
            self.train(X, y, sample_weights, optimize_hyperparams=False)
        
        X_processed = self.prepare_features(X, fit_scaler=False)
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = {
            'auc': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_processed)):
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_weights = sample_weights[train_idx] if sample_weights is not None else None
            
            # Train fold model
            fold_model = xgb.XGBClassifier(**self.params)
            fold_model.fit(X_train, y_train, sample_weight=train_weights, verbose=False)
            
            # Evaluate
            val_proba = fold_model.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            cv_scores['auc'].append(roc_auc_score(y_val, val_proba))
            cv_scores['accuracy'].append(accuracy_score(y_val, val_pred))
            cv_scores['precision'].append(precision_score(y_val, val_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val, val_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val, val_pred, zero_division=0))
        
        # Calculate statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            cv_results[f'{metric}_scores'] = scores
        
        return cv_results
