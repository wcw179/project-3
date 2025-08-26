"""
XGBoost MFE Regressor for Black-Swan Hunter Trading Bot
Predicts Maximum Favorable Excursion in risk multiples (R)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import pickle
import json

import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import optuna

logger = logging.getLogger(__name__)

class XGBMFERegressor:
    """XGBoost regressor for Maximum Favorable Excursion prediction"""
    
    def __init__(self, objective='reg:squarederror', random_state=42):
        self.objective = objective
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Default hyperparameters optimized for MFE regression
        self.params = {
            'objective': objective,
            'eval_metric': 'rmse',
            'max_depth': 5,
            'learning_rate': 0.02,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def prepare_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Prepare features for XGBoost training/inference"""
        # Handle missing values
        X_clean = X.fillna(0)
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = X_clean.columns.tolist()
        
        # Optional scaling (can help with convergence)
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            X_scaled = self.scaler.transform(X_clean)
        
        return X_scaled
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                cv_splitter, n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna with custom CV"""
        
        def objective(trial):
            params = {
                'objective': self.objective,
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Custom cross-validation with purged splits
            cv_scores = []
            for train_idx, val_idx in cv_splitter.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=50,
                         verbose=False)
                
                val_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                cv_scores.append(rmse)
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='minimize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Update parameters with best trial
        best_params = study.best_params
        best_params.update({
            'objective': self.objective,
            'eval_metric': 'rmse',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        })
        
        self.params = best_params
        
        logger.info(f"Hyperparameter optimization completed. Best RMSE: {study.best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def train(self, X: pd.DataFrame, y: np.ndarray, cv_splitter,
              optimize_hyperparams: bool = True, n_trials: int = 50,
              validation_split: float = 0.2) -> Dict:
        """Train XGBoost MFE regressor with purged cross-validation"""
        
        # Prepare features
        X_processed = self.prepare_features(X, fit_scaler=True)
        
        # Optimize hyperparameters if requested
        optimization_results = None
        if optimize_hyperparams:
            logger.info("Optimizing hyperparameters...")
            optimization_results = self.optimize_hyperparameters(
                X_processed, y, cv_splitter, n_trials
            )
        
        # Cross-validation evaluation
        logger.info("Performing cross-validation...")
        cv_rmse_scores = []
        cv_r2_scores = []
        cv_mae_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_processed)):
            logger.info(f"Training fold {fold + 1}")
            
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train fold model
            fold_model = xgb.XGBRegressor(**self.params)
            fold_model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          early_stopping_rounds=50,
                          verbose=False)
            
            # Evaluate
            val_pred = fold_model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            r2 = r2_score(y_val, val_pred)
            mae = mean_absolute_error(y_val, val_pred)
            
            cv_rmse_scores.append(rmse)
            cv_r2_scores.append(r2)
            cv_mae_scores.append(mae)
        
        # Train final model on all data
        logger.info("Training final model...")
        split_idx = int(len(X_processed) * (1 - validation_split))
        X_train_final = X_processed[:split_idx]
        X_val_final = X_processed[split_idx:]
        y_train_final = y[:split_idx]
        y_val_final = y[split_idx:]
        
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train_final, y_train_final,
                      eval_set=[(X_val_final, y_val_final)],
                      early_stopping_rounds=50,
                      verbose=False)
        
        self.is_fitted = True
        
        # Final validation metrics
        val_pred_final = self.model.predict(X_val_final)
        final_rmse = np.sqrt(mean_squared_error(y_val_final, val_pred_final))
        final_r2 = r2_score(y_val_final, val_pred_final)
        final_mae = mean_absolute_error(y_val_final, val_pred_final)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        training_results = {
            'cv_rmse_mean': np.mean(cv_rmse_scores),
            'cv_rmse_std': np.std(cv_rmse_scores),
            'cv_r2_mean': np.mean(cv_r2_scores),
            'cv_r2_std': np.std(cv_r2_scores),
            'cv_mae_mean': np.mean(cv_mae_scores),
            'cv_mae_std': np.std(cv_mae_scores),
            'final_rmse': final_rmse,
            'final_r2': final_r2,
            'final_mae': final_mae,
            'feature_importance': feature_importance,
            'optimization_results': optimization_results,
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }
        
        logger.info(f"Training completed. CV RMSE: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
        logger.info(f"Final validation RMSE: {final_rmse:.4f}, R²: {final_r2:.4f}")
        
        return training_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict MFE values"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.prepare_features(X, fit_scaler=False)
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        
        # Additional regression metrics
        mape = np.mean(np.abs((y - predictions) / np.maximum(y, 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'mape': mape,
            'predictions': predictions.tolist()
        }
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_scores = self.model.feature_importances_
        
        if self.feature_names is not None:
            importance_dict = dict(zip(self.feature_names, importance_scores))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_importance = [(f'feature_{i}', score) for i, score in enumerate(importance_scores)]
            sorted_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'importance_type': importance_type,
            'feature_importance': dict(sorted_importance),
            'top_10_features': sorted_importance[:10]
        }
    
    def save_model(self, filepath: str):
        """Save model and associated components"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(filepath))
        
        # Save additional components
        components = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params,
            'objective': self.objective
        }
        
        components_path = filepath.parent / f"{filepath.stem}_components.pkl"
        with open(components_path, 'wb') as f:
            pickle.dump(components, f)
        
        logger.info(f"XGB MFE model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and associated components"""
        filepath = Path(filepath)
        
        # Load XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(filepath))
        
        # Load additional components
        components_path = filepath.parent / f"{filepath.stem}_components.pkl"
        if components_path.exists():
            with open(components_path, 'rb') as f:
                components = pickle.load(f)
            
            self.scaler = components['scaler']
            self.feature_names = components['feature_names']
            self.params = components['params']
            self.objective = components['objective']
        
        self.is_fitted = True
        logger.info(f"XGB MFE model loaded from {filepath}")
