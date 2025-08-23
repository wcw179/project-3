"""
Training pipeline for M5 Multi-Symbol Trend Bot
Handles GPU training on Colab with artifact persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import gc

from src.data.database import M5Database
from src.features.feature_pipeline import FeaturePipeline
from src.features.labeling import TripleBarrierLabeling
from src.models.lstm_model import LSTMTrendClassifier
from src.models.xgb_model import XGBMetaModel

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Complete training pipeline for LSTM and XGB models"""
    
    def __init__(self, db_path: str = "data/m5_trading.db",
                 artifacts_dir: str = "artifacts"):
        self.db = M5Database(db_path)
        self.feature_pipeline = FeaturePipeline()
        self.labeling = TripleBarrierLabeling()
        
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model-specific directories
        (self.artifacts_dir / "models" / "lstm").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "models" / "xgb").mkdir(parents=True, exist_ok=True)
        
    def prepare_training_data(self, symbol: str, start_date: datetime,
                            end_date: datetime, rr_preset: str = '1:2') -> Dict:
        """Prepare training data for a symbol by loading pre-generated bar data, features, and labels."""
        logger.info(f"Preparing training data for {symbol} from '{self.db.db_path}'.")

        # 1. Load all necessary data using the M5Database adapter
        ohlc = self.db.get_ohlcv_data(symbol, start_date, end_date)
        lstm_features = self.db.get_features_subset(symbol, 'lstm', start_date, end_date)
        xgb_features = self.db.get_features_subset(symbol, 'xgb', start_date, end_date)
        labels_meta = self.db.get_labels_meta(symbol, start_date, end_date)

        if ohlc.empty: raise ValueError(f"No bar data (OHLCV) found for {symbol}")
        if lstm_features.empty: raise ValueError(f"No LSTM features found for {symbol}")
        if xgb_features.empty: raise ValueError(f"No XGB features found for {symbol}")
        if labels_meta.empty: raise ValueError(f"No labels metadata found for {symbol}")

        # 2. Reconstruct the complete labels DataFrame from the 'meta' column
        # This is necessary because the runner now stores the full label info in the meta field.
        labels_data = []
        for timestamp, row in labels_meta.iterrows():
            meta = row.get('meta_parsed', {})
            # The meta field now contains the full dictionary from the labeling process
            if 'label' in meta and 'holding_period' in meta:
                meta['timestamp'] = timestamp
                labels_data.append(meta)

        if not labels_data:
            raise ValueError(f"No valid labels could be reconstructed from the database.")

        labels_df = pd.DataFrame(labels_data).set_index('timestamp')

        # 3. Post-process labels (calculate weights, remove overlaps)
        sample_weights = self.labeling.calculate_sample_weights(labels_df)
        labels_df = self.labeling.remove_overlapping_events(labels_df)

        # 4. Align all dataframes to a common set of timestamps
        common_timestamps = lstm_features.index.intersection(labels_df.index)

        if len(common_timestamps) < 100:
            raise ValueError(f"Insufficient aligned data for {symbol}: {len(common_timestamps)} samples")

        aligned_lstm_features = lstm_features.loc[common_timestamps]
        aligned_xgb_features = xgb_features.loc[common_timestamps]
        aligned_labels = labels_df.loc[common_timestamps]
        aligned_weights = sample_weights.loc[common_timestamps]
        aligned_ohlc = ohlc.loc[common_timestamps]

        logger.info(f"Prepared {len(common_timestamps)} training samples for {symbol}")

        return {
            'lstm_features': aligned_lstm_features,
            'xgb_features': aligned_xgb_features,
            'labels': aligned_labels,
            'sample_weights': aligned_weights,
            'ohlcv': aligned_ohlc,
            'timestamps': common_timestamps
        }
    
    def train_lstm_model(self, training_data: Dict, symbol: str, 
                        hyperparams: Optional[Dict] = None) -> Dict:
        """Train LSTM model for a symbol"""
        logger.info(f"Training LSTM model for {symbol}")
        
        # Default hyperparameters
        default_hyperparams = {
            'sequence_length': 60,
            'hidden_units': 128,
            'n_layers': 2,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 128,
            'epochs': 100,
            'use_focal_loss': True
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)
        
        # Prepare data
        lstm_features = training_data['lstm_features'].values
        labels = training_data['labels']['label'].values
        sample_weights = training_data['sample_weights'].values
        
        # Initialize model
        lstm_model = LSTMTrendClassifier(
            sequence_length=default_hyperparams['sequence_length'],
            n_features=lstm_features.shape[1],
            hidden_units=default_hyperparams['hidden_units'],
            n_layers=default_hyperparams['n_layers'],
            dropout_rate=default_hyperparams['dropout_rate'],
            learning_rate=default_hyperparams['learning_rate'],
            use_focal_loss=default_hyperparams['use_focal_loss']
        )
        
        # Train model
        training_results = lstm_model.train(
            X=lstm_features,
            y=labels,
            epochs=default_hyperparams['epochs'],
            batch_size=default_hyperparams['batch_size'],
            sample_weights=sample_weights,
            use_purged_cv=True
        )
        
        # Evaluate model
        evaluation_results = lstm_model.evaluate(lstm_features, labels)
        
        # Save model
        model_path = self.artifacts_dir / "models" / "lstm" / symbol / "model.h5"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        lstm_model.save_model(str(model_path))
        
        # Save training metadata
        metadata = {
            'symbol': symbol,
            'model_type': 'lstm',
            'hyperparameters': default_hyperparams,
            'training_results': {
                'final_loss': training_results.get('final_loss'),
                'final_accuracy': training_results.get('final_accuracy')
            },
            'evaluation_results': {
                'accuracy': evaluation_results['accuracy'],
                'classification_report': evaluation_results['classification_report']
            },
            'training_period': {
                'start': training_data['timestamps'].min().isoformat(),
                'end': training_data['timestamps'].max().isoformat()
            },
            'n_samples': len(training_data['timestamps']),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = model_path.parent / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in database
        self.db.save_model_artifact(
            model_type='lstm',
            symbol=symbol,
            model_version=datetime.now().strftime('%Y%m%d_%H%M%S'),
            artifact_path=str(model_path),
            metrics=evaluation_results,
            hyperparameters=default_hyperparams,
            training_period=(training_data['timestamps'].min(), training_data['timestamps'].max())
        )
        
        logger.info(f"LSTM model trained and saved for {symbol}")
        
        # Clean up memory
        del lstm_model
        gc.collect()
        
        return {
            'model_path': str(model_path),
            'metadata': metadata,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
    
    def generate_lstm_probabilities(self, symbol: str, training_data: Dict) -> pd.DataFrame:
        """Generate LSTM probabilities for XGB training"""
        logger.info(f"Generating LSTM probabilities for {symbol}")
        
        # Load trained LSTM model
        model_path = self.artifacts_dir / "models" / "lstm" / symbol / "model.h5"
        
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model not found for {symbol}")
        
        lstm_model = LSTMTrendClassifier()
        lstm_model.load_model(str(model_path))
        
        # Generate probabilities
        lstm_features = training_data['lstm_features'].values
        probabilities, predictions = lstm_model.predict(lstm_features, return_probabilities=True)
        
        # Create DataFrame with probabilities
        # Account for sequence length offset
        seq_len = lstm_model.sequence_length
        prob_timestamps = training_data['timestamps'][seq_len:]
        
        prob_df = pd.DataFrame({
            'p_down': probabilities[:, 0],
            'p_neutral': probabilities[:, 1], 
            'p_up': probabilities[:, 2],
            'prediction': predictions
        }, index=prob_timestamps)
        
        # Probabilities are used directly in the next step and not stored.
        
        logger.info(f"Generated {len(prob_df)} LSTM probabilities for {symbol}")
        
        return prob_df
    
    def train_xgb_model(self, training_data: Dict, lstm_probabilities: pd.DataFrame, 
                       symbol: str, hyperparams: Optional[Dict] = None) -> Dict:
        """Train XGB meta-model for a symbol"""
        logger.info(f"Training XGB model for {symbol}")
        
        # Prepare XGB features with LSTM probabilities
        xgb_features = training_data['xgb_features'].copy()
        
        # Align LSTM probabilities with XGB features
        common_timestamps = xgb_features.index.intersection(lstm_probabilities.index)
        
        if len(common_timestamps) < 50:
            raise ValueError(f"Insufficient aligned data for XGB training: {len(common_timestamps)}")
        
        aligned_xgb_features = xgb_features.loc[common_timestamps]
        aligned_lstm_probs = lstm_probabilities.loc[common_timestamps]
        aligned_labels = training_data['labels'].loc[common_timestamps]
        
        # Add LSTM probabilities to XGB features
        for col in ['p_up', 'p_down', 'p_neutral']:
            aligned_xgb_features[f'lstm_{col}'] = aligned_lstm_probs[col]
        
        # Generate meta-labels (binary: trade/skip)
        meta_labels = self.labeling.generate_meta_labels(
            aligned_labels, aligned_lstm_probs, threshold=0.55
        )
        
        if meta_labels.empty:
            raise ValueError(f"No meta-labels generated for {symbol}")
        
        # Align all data
        final_timestamps = aligned_xgb_features.index.intersection(meta_labels.index)
        final_features = aligned_xgb_features.loc[final_timestamps]
        final_labels = meta_labels.loc[final_timestamps]['meta_label'].values
        
        # Initialize XGB model
        xgb_model = XGBMetaModel(use_calibration=True)
        
        # Train model
        training_results = xgb_model.train(
            X=final_features,
            y=final_labels,
            optimize_hyperparams=True,
            validation_split=0.2
        )
        
        # Evaluate model
        evaluation_results = xgb_model.evaluate(final_features, final_labels)
        
        # Save model
        model_path = self.artifacts_dir / "models" / "xgb" / symbol / "model.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        xgb_model.save_model(str(model_path))
        
        # Save training metadata
        metadata = {
            'symbol': symbol,
            'model_type': 'xgb',
            'hyperparameters': xgb_model.params,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'training_period': {
                'start': final_timestamps.min().isoformat(),
                'end': final_timestamps.max().isoformat()
            },
            'n_samples': len(final_timestamps),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = model_path.parent / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in database
        self.db.save_model_artifact(
            model_type='xgb',
            symbol=symbol,
            model_version=datetime.now().strftime('%Y%m%d_%H%M%S'),
            artifact_path=str(model_path),
            metrics=evaluation_results,
            hyperparameters=xgb_model.params,
            training_period=(final_timestamps.min(), final_timestamps.max())
        )
        
        logger.info(f"XGB model trained and saved for {symbol}")
        
        return {
            'model_path': str(model_path),
            'metadata': metadata,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
    
    def train_symbol_models(self, symbol: str, start_date: datetime, 
                           end_date: datetime, rr_preset: str = '1:2') -> Dict:
        """Train both LSTM and XGB models for a symbol"""
        logger.info(f"Starting complete training pipeline for {symbol}")
        
        try:
            # Prepare training data
            training_data = self.prepare_training_data(symbol, start_date, end_date, rr_preset)
            
            # Train LSTM model
            lstm_results = self.train_lstm_model(training_data, symbol)
            
            # Generate LSTM probabilities
            lstm_probabilities = self.generate_lstm_probabilities(symbol, training_data)
            
            # Train XGB model
            xgb_results = self.train_xgb_model(training_data, lstm_probabilities, symbol)
            
            results = {
                'symbol': symbol,
                'status': 'success',
                'lstm_results': lstm_results,
                'xgb_results': xgb_results,
                'training_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'rr_preset': rr_preset
            }
            
            logger.info(f"Training pipeline completed successfully for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }
    
    def train_multi_symbol_models(self, symbols: List[str], start_date: datetime,
                                 end_date: datetime, rr_preset: str = '1:2') -> Dict[str, Dict]:
        """Train models for multiple symbols"""
        logger.info(f"Starting multi-symbol training for {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"Training models for {symbol} ({symbols.index(symbol) + 1}/{len(symbols)})")
            
            try:
                symbol_results = self.train_symbol_models(symbol, start_date, end_date, rr_preset)
                results[symbol] = symbol_results
                
                # Clean up memory between symbols
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to train models for {symbol}: {e}")
                results[symbol] = {
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Generate summary
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        logger.info(f"Multi-symbol training completed: {successful}/{len(symbols)} successful")
        
        return results
    
    def validate_trained_models(self, symbol: str) -> Dict:
        """Validate that trained models exist and are functional"""
        validation_results = {
            'symbol': symbol,
            'lstm_model': {'exists': False, 'functional': False},
            'xgb_model': {'exists': False, 'functional': False}
        }
        
        # Check LSTM model
        lstm_path = self.artifacts_dir / "models" / "lstm" / symbol / "model.h5"
        if lstm_path.exists():
            validation_results['lstm_model']['exists'] = True
            
            try:
                lstm_model = LSTMTrendClassifier()
                lstm_model.load_model(str(lstm_path))
                validation_results['lstm_model']['functional'] = True
                validation_results['lstm_model']['sequence_length'] = lstm_model.sequence_length
                validation_results['lstm_model']['n_features'] = lstm_model.n_features
            except Exception as e:
                validation_results['lstm_model']['error'] = str(e)
        
        # Check XGB model
        xgb_path = self.artifacts_dir / "models" / "xgb" / symbol / "model.json"
        if xgb_path.exists():
            validation_results['xgb_model']['exists'] = True
            
            try:
                xgb_model = XGBMetaModel()
                xgb_model.load_model(str(xgb_path))
                validation_results['xgb_model']['functional'] = True
                validation_results['xgb_model']['n_features'] = len(xgb_model.feature_names) if xgb_model.feature_names else 0
            except Exception as e:
                validation_results['xgb_model']['error'] = str(e)
        
        return validation_results
