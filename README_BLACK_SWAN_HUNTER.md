# Black-Swan Hunter Trading Bot

A production-ready dual-model trading system designed to detect and capitalize on extreme price movements (Black Swan events) in M5 forex markets.

## System Architecture

### Dual-Model Approach
- **XGB_MFE (Regressor)**: Predicts Maximum Favorable Excursion (MFE) in risk multiples (R)
- **LSTM_TAIL (Classifier)**: Predicts probability of tail events occurring within forecast horizon

### Key Specifications
- **Timeframe**: M5 OHLCV bars
- **Symbols**: Major forex pairs (EURUSD, GBPUSD, etc.)
- **LSTM sequence length**: 60 bars (5 hours of M5 data)
- **Forecast horizon**: 288 bars (~8.3 hours ahead)
- **Risk unit definition**: 1R = ATR(14) at entry bar

## Features

### XGB Features (~25 tabular features)
- **Trend**: EMA ratios and slopes (20/50/200)
- **Momentum**: RSI(14), MACD signal/histogram
- **Volatility**: ATR/Close ratio, Bollinger Band metrics
- **Volume**: VWAP distance, volume regime
- **Price Action**: HL range, return statistics
- **Patterns**: 20+ candlestick patterns
- **Context**: Hour of day, day of week, spread proxy

### LSTM Features (~10 sequential features)
- **Core**: Log returns, ATR-normalized OHLC
- **Indicators**: RSI(14), ATR normalized, HL range
- **Trend**: EMA deviations for periods 20, 50
- **Volume**: Normalized volume relative to 20-period average

## Labeling Strategy

### XGB Regression Labels
- `label_mfe = max(high[t+1:t+T] - close[t]) / ATR14[t]` for long setups
- Values clipped to [0, 50] R-multiples
- Forward-looking window of T=100 bars

### LSTM Classification Labels (Multi-class)
- **Class 0**: MFE < 5R (normal moves)
- **Class 1**: 5R ≤ MFE < 10R (moderate tail)
- **Class 2**: 10R ≤ MFE < 20R (strong tail)
- **Class 3**: MFE ≥ 20R (extreme tail)
- Handles class imbalance using Focal Loss (alpha=0.25, gamma=2.0)

## Installation & Setup

### Dependencies
```bash
pip install tensorflow xgboost optuna pandas numpy scikit-learn MetaTrader5
```

### Project Structure
```
project-3/
├── src/
│   ├── features/
│   │   ├── black_swan_pipeline.py      # Enhanced feature pipeline
│   │   └── black_swan_labeling.py      # MFE labeling & tail classification
│   ├── models/
│   │   ├── xgb_mfe_model.py           # XGBoost MFE regressor
│   │   ├── lstm_tail_model.py         # LSTM tail classifier
│   │   ├── train_xgb_mfe.py          # XGB training pipeline
│   │   └── train_lstm_tail.py        # LSTM training pipeline
│   ├── backtesting/
│   │   └── black_swan_backtest.py    # Comprehensive backtesting engine
│   └── trading/
│       └── live_bot.py               # Live trading bot with MT5 integration
└── artifacts/
    └── models/                       # Trained model storage
```

## Usage

### 1. Data Preparation
```bash
# Load M5 OHLCV data into database
python src/scripts/migrate_to_trading_db.py
```

### 2. Model Training

#### Train XGBoost MFE Regressor
```bash
python src/models/train_xgb_mfe.py --symbol EURUSD --optimize-hyperparams --n-trials 50
```

#### Train LSTM Tail Classifier
```bash
python src/models/train_lstm_tail.py --symbol EURUSD --epochs 50 --batch-size 64
```

### 3. Backtesting
```python
from src.backtesting.black_swan_backtest import BlackSwanBacktester, BacktestConfig

# Configure backtest
config = BacktestConfig(
    initial_capital=100000.0,
    base_risk_per_trade=0.005,  # 0.5% risk per trade
    min_mfe_prediction=5.0,
    min_tail_probability=0.3
)

# Run backtest
backtester = BlackSwanBacktester(config)
results = backtester.run_backtest(market_data, predictions)
```

### 4. Live Trading
```bash
python src/trading/live_bot.py --models-dir artifacts/models --symbols EURUSD,GBPUSD
```

## Trading Logic & Entry Rules

### Signal Generation
```python
# Compute predictions
mfe_pred = xgb_model.predict(current_features)
tail_probs = lstm_model.predict_proba(sequence_features)
p_tail = tail_probs[1] + tail_probs[2] + tail_probs[3]  # P(MFE >= 5R)

# Entry conditions (LONG example)
trend_filter = (ema20 > ema50 > ema200)
signal_strength = (mfe_pred >= 5.0) and (p_tail >= 0.3)
entry_signal = trend_filter and signal_strength
```

### Position Sizing Rules
- **Base risk**: 0.5% of account balance per trade
- **Size multipliers** based on prediction confidence:
  - High confidence (mfe_pred ≥ 15R, p_tail ≥ 0.4): 1.5x base size
  - Medium confidence (mfe_pred ≥ 10R, p_tail ≥ 0.3): 1.0x base size
  - Low confidence (mfe_pred ≥ 5R, p_tail ≥ 0.2): 0.5x base size
- **Maximum 3 concurrent positions**

### Risk Management Framework
- **Stop Loss**: 1.0 × ATR(14) below entry (defines 1R loss)
- **Trailing Stop**: ATR-based trailing after 3R profit
- **Take Profit**: Partial close 30% at 5R, remainder trails
- **Maximum Hold**: 500 bars (41.7 hours) forced exit
- **Daily Loss Limit**: 2% of account balance

## Model Training Specifications

### XGBoost Configuration
- **Objective**: `reg:squarederror` with custom evaluation metric
- **Validation**: Purged Time Series Split (5 folds, 3-bar embargo)
- **Hyperparameters**: max_depth=5, n_estimators=500, learning_rate=0.02
- **Early stopping**: 50 rounds on validation RMSE

### LSTM Configuration
- **Architecture**: 2 LSTM layers (64 units each), 0.2 dropout, BatchNorm, Dense(32), softmax output
- **Loss**: Focal Loss with class weights for severe imbalance
- **Optimizer**: Adam(lr=0.001) with ReduceLROnPlateau(factor=0.5, patience=5)
- **Validation**: Walk-forward with 3-bar embargo between train/test

## Performance Metrics

### Success Criteria
- **Sharpe ratio** > 1.5 on out-of-sample data
- **Maximum drawdown** < 10%
- **Tail event detection precision** > 60% for Class 2+ events
- **Average trade execution time** < 100ms
- **System uptime** > 99.5% during market hours

### Backtesting Metrics
- Performance: Sharpe ratio, CAGR, maximum drawdown
- Tail Detection: Precision/Recall for Class 2+ events, PR-AUC
- Model Quality: Out-of-sample R² for XGB, classification accuracy for LSTM

## Production Deployment

### MT5 Integration
- Real-time price feeds and order execution
- Position monitoring and risk management
- Automated trade logging and performance tracking

### Monitoring & Maintenance
- Log all predictions vs. realized outcomes for model drift detection
- Retrain models monthly with expanding window
- Alert system for prediction confidence degradation
- Performance dashboard with key metrics visualization

## Technical Requirements
- **Database**: SQLite for development, PostgreSQL for production
- **ML Libraries**: XGBoost 1.7+, TensorFlow 2.12+, scikit-learn 1.3+
- **Broker Integration**: MT5 Python API
- **Deployment**: Docker containers with automated model serving
- **Compute**: XGB training on CPU (4+ cores), LSTM inference optimized for low latency

## License
This project is for educational and research purposes. Use at your own risk in live trading environments.

## Disclaimer
Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. The Black Swan Hunter system is designed to detect extreme market movements but cannot guarantee profitable trades.
