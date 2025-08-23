"""
Backtesting engine for M5 Multi-Symbol Trend Bot
Implements single & multi-symbol backtesting with comprehensive metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path

from src.data.database import TradingDatabase
from src.features.hierarchical_extremes import HierarchicalExtremes
from src.models.lstm_model import LSTMTrendClassifier
from src.models.xgb_model import XGBMetaModel

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade record structure"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: Optional[float]
    size: float
    stop_loss: float
    profit_target: float
    exit_reason: str  # 'profit_target', 'stop_loss', 'trailing_stop', 'timeout'
    pnl: float
    return_pct: float
    holding_period: int  # in bars
    lstm_prob: float
    xgb_score: float
    rr_achieved: float

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 5
    min_rr_filter: float = 5.0  # Minimum RR for evaluation (Rule: >= 1:5)
    lstm_threshold: float = 0.55
    xgb_threshold: float = 0.6
    use_ema_confirmation: bool = True
    use_black_swan_filter: bool = True
    use_he_trailing_stop: bool = True
    max_holding_period: int = 100  # bars
    commission: float = 0.0001  # 0.01%
    slippage: float = 0.0001  # 0.01%

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, db_path: str = "data/trading_system.db"):
        self.db = TradingDatabase(db_path)
        self.he = HierarchicalExtremes(levels=3, atr_lookback=1440)
        
        # Models (loaded when needed)
        self.lstm_model = None
        self.xgb_model = None
        
        # State tracking
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_capital: float = 0.0
        
    def load_models(self, lstm_path: str, xgb_path: str):
        """Load trained models"""
        self.lstm_model = LSTMTrendClassifier()
        self.lstm_model.load_model(lstm_path)
        
        self.xgb_model = XGBMetaModel()
        self.xgb_model.load_model(xgb_path)
        
        logger.info("Models loaded successfully")
    
    def get_signals(self, ohlc: pd.DataFrame, features_lstm: pd.DataFrame, 
                   features_xgb: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        """Generate trading signals"""
        signals = pd.DataFrame(index=ohlc.index)
        signals['signal'] = 0
        signals['lstm_prob_up'] = 0.33
        signals['lstm_prob_down'] = 0.33
        signals['xgb_score'] = 0.5
        signals['ema_confirmation'] = True
        signals['black_swan_filter'] = True
        
        if self.lstm_model is None or self.xgb_model is None:
            logger.warning("Models not loaded, returning neutral signals")
            return signals
        
        # Get LSTM predictions
        try:
            lstm_features_array = features_lstm.values
            lstm_probs, lstm_classes = self.lstm_model.predict(lstm_features_array, return_probabilities=True)
            
            # Align predictions with timestamps (account for sequence length)
            seq_len = self.lstm_model.sequence_length
            if len(lstm_probs) > 0:
                pred_timestamps = ohlc.index[seq_len:]
                signals.loc[pred_timestamps, 'lstm_prob_up'] = lstm_probs[:, 2]  # Up class
                signals.loc[pred_timestamps, 'lstm_prob_down'] = lstm_probs[:, 0]  # Down class
        
        except Exception as e:
            logger.error(f"Error getting LSTM predictions: {e}")
        
        # Get XGB predictions
        try:
            # Add LSTM probabilities to XGB features
            xgb_features_with_lstm = features_xgb.copy()
            xgb_features_with_lstm['lstm_p_up'] = signals['lstm_prob_up']
            xgb_features_with_lstm['lstm_p_down'] = signals['lstm_prob_down']
            
            xgb_scores = self.xgb_model.predict_proba(xgb_features_with_lstm)
            signals['xgb_score'] = xgb_scores
        
        except Exception as e:
            logger.error(f"Error getting XGB predictions: {e}")
        
        # Apply filters
        if config.use_ema_confirmation:
            signals['ema_confirmation'] = self._check_ema_confirmation(ohlc)
        
        if config.use_black_swan_filter:
            signals['black_swan_filter'] = self._check_black_swan_filter(ohlc)
        
        # Generate final signals
        long_condition = (
            (signals['lstm_prob_up'] > config.lstm_threshold) &
            (signals['xgb_score'] > config.xgb_threshold) &
            signals['ema_confirmation'] &
            signals['black_swan_filter']
        )
        
        short_condition = (
            (signals['lstm_prob_down'] > config.lstm_threshold) &
            (signals['xgb_score'] > config.xgb_threshold) &
            signals['ema_confirmation'] &
            signals['black_swan_filter']
        )
        
        signals.loc[long_condition, 'signal'] = 1
        signals.loc[short_condition, 'signal'] = -1
        
        return signals
    
    def _check_ema_confirmation(self, ohlc: pd.DataFrame) -> pd.Series:
        """Check EMA trend confirmation"""
        # Simple EMA confirmation: price above EMA20 for long, below for short
        ema20 = ohlc['close'].ewm(span=20).mean()
        ema50 = ohlc['close'].ewm(span=50).mean()
        ema200 = ohlc['close'].ewm(span=200).mean()
        
        # Trend alignment: EMAs in proper order
        uptrend = (ema20 > ema50) & (ema50 > ema200) & (ohlc['close'] > ema20)
        downtrend = (ema20 < ema50) & (ema50 < ema200) & (ohlc['close'] < ema20)
        
        return uptrend | downtrend
    
    def _check_black_swan_filter(self, ohlc: pd.DataFrame) -> pd.Series:
        """Check black swan filter (avoid trading during extreme events)"""
        returns = ohlc['close'].pct_change()
        vol = returns.rolling(20).std()
        
        # Avoid trading when recent volatility is extremely high
        extreme_vol = vol > vol.rolling(100).quantile(0.95)
        
        # Avoid trading after large moves
        large_moves = np.abs(returns) > (vol * 3)
        recent_large_moves = large_moves.rolling(5).sum() > 0
        
        return ~(extreme_vol | recent_large_moves)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               risk_amount: float) -> float:
        """Calculate position size based on risk"""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        
        position_size = risk_amount / risk_per_share
        return position_size
    
    def update_trailing_stops(self, ohlc_row: pd.Series, hierarchical_levels: Dict):
        """Update trailing stops using HE levels"""
        if not self.open_trades:
            return
        
        current_price = ohlc_row['close']
        current_atr = ohlc_row.get('atr', current_price * 0.01)  # Fallback ATR
        
        for trade in self.open_trades:
            if trade.direction == 1:  # Long position
                # Calculate trailing stop using HE support
                he_stop = self.he.calculate_trailing_stop_level(
                    hierarchical_levels, current_price, 'long', current_atr, level=1
                )
                
                if he_stop and he_stop > trade.stop_loss:
                    trade.stop_loss = he_stop
            
            else:  # Short position
                # Calculate trailing stop using HE resistance
                he_stop = self.he.calculate_trailing_stop_level(
                    hierarchical_levels, current_price, 'short', current_atr, level=1
                )
                
                if he_stop and he_stop < trade.stop_loss:
                    trade.stop_loss = he_stop
    
    def check_exits(self, ohlc_row: pd.Series, bar_index: int) -> List[Trade]:
        """Check for trade exits"""
        exits = []
        current_price = ohlc_row['close']
        current_time = ohlc_row.name
        
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            exit_trade = False
            exit_reason = ""
            exit_price = current_price
            
            # Check profit target
            if trade.direction == 1 and current_price >= trade.profit_target:
                exit_trade = True
                exit_reason = "profit_target"
                exit_price = trade.profit_target
            elif trade.direction == -1 and current_price <= trade.profit_target:
                exit_trade = True
                exit_reason = "profit_target"
                exit_price = trade.profit_target
            
            # Check stop loss
            elif trade.direction == 1 and current_price <= trade.stop_loss:
                exit_trade = True
                exit_reason = "stop_loss"
                exit_price = trade.stop_loss
            elif trade.direction == -1 and current_price >= trade.stop_loss:
                exit_trade = True
                exit_reason = "stop_loss"
                exit_price = trade.stop_loss
            
            # Check timeout
            elif trade.holding_period >= 100:  # Max holding period
                exit_trade = True
                exit_reason = "timeout"
                exit_price = current_price
            
            if exit_trade:
                # Calculate PnL
                if trade.direction == 1:
                    pnl = (exit_price - trade.entry_price) * trade.size
                else:
                    pnl = (trade.entry_price - exit_price) * trade.size
                
                # Apply commission and slippage
                total_commission = (trade.entry_price + exit_price) * trade.size * 0.0001
                total_slippage = (trade.entry_price + exit_price) * trade.size * 0.0001
                pnl -= (total_commission + total_slippage)
                
                # Update trade
                trade.exit_time = current_time
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl = pnl
                trade.return_pct = pnl / (trade.entry_price * trade.size)
                
                # Calculate achieved RR
                if trade.direction == 1:
                    risk = trade.entry_price - trade.stop_loss
                    reward = exit_price - trade.entry_price
                else:
                    risk = trade.stop_loss - trade.entry_price
                    reward = trade.entry_price - exit_price
                
                trade.rr_achieved = reward / risk if risk > 0 else 0
                
                # Move to closed trades
                self.open_trades.remove(trade)
                self.closed_trades.append(trade)
                exits.append(trade)
            else:
                # Update holding period
                trade.holding_period += 1
        
        return exits
    
    def enter_trade(self, signal_row: pd.Series, ohlc_row: pd.Series, 
                   config: BacktestConfig) -> Optional[Trade]:
        """Enter a new trade"""
        if len(self.open_trades) >= config.max_positions:
            return None
        
        signal = signal_row['signal']
        if signal == 0:
            return None
        
        current_time = signal_row.name
        current_price = ohlc_row['close']
        current_atr = ohlc_row.get('atr', current_price * 0.01)
        
        # Calculate stop loss and profit target
        if signal == 1:  # Long
            stop_loss = current_price - (current_atr * 1.5)
            profit_target = current_price + (current_atr * 3.0)  # 1:2 RR minimum
        else:  # Short
            stop_loss = current_price + (current_atr * 1.5)
            profit_target = current_price - (current_atr * 3.0)
        
        # Calculate position size
        risk_amount = self.current_capital * config.risk_per_trade
        position_size = self.calculate_position_size(current_price, stop_loss, risk_amount)
        
        if position_size <= 0:
            return None
        
        # Create trade
        trade = Trade(
            entry_time=current_time,
            exit_time=None,
            symbol=ohlc_row.get('symbol', 'UNKNOWN'),
            direction=signal,
            entry_price=current_price,
            exit_price=None,
            size=position_size,
            stop_loss=stop_loss,
            profit_target=profit_target,
            exit_reason="",
            pnl=0.0,
            return_pct=0.0,
            holding_period=0,
            lstm_prob=signal_row.get('lstm_prob_up' if signal == 1 else 'lstm_prob_down', 0.5),
            xgb_score=signal_row.get('xgb_score', 0.5),
            rr_achieved=0.0
        )
        
        self.open_trades.append(trade)
        return trade
    
    def run_backtest(self, symbol: str, config: BacktestConfig) -> Dict:
        """Run backtest for single symbol"""
        logger.info(f"Starting backtest for {symbol}")
        
        # Initialize
        self.current_capital = config.initial_capital
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = []
        
        # Get data
        ohlc = self.db.get_ohlcv_data(symbol, config.start_date, config.end_date)
        if ohlc.empty:
            logger.error(f"No OHLCV data for {symbol}")
            return {}
        
        # Load features
        # Load features from database
        features_lstm_raw = self.db.get_features(symbol, 'lstm', config.start_date, config.end_date)
        features_xgb_raw = self.db.get_features(symbol, 'xgb', config.start_date, config.end_date)

        # Expand features from JSON
        if not features_lstm_raw.empty:
            lstm_records = features_lstm_raw['features_parsed'].tolist()
            features_lstm = pd.DataFrame.from_records(lstm_records, index=features_lstm_raw.index)
        else:
            features_lstm = pd.DataFrame()

        if not features_xgb_raw.empty:
            xgb_records = features_xgb_raw['features_parsed'].tolist()
            features_xgb = pd.DataFrame.from_records(xgb_records, index=features_xgb_raw.index)
        else:
            features_xgb = pd.DataFrame()
        
        if features_lstm.empty or features_xgb.empty:
            logger.error(f"No features available for {symbol}")
            return {}
        
        # Calculate ATR
        from src.features.technical_indicators import TechnicalIndicators
        indicators = TechnicalIndicators()
        atr = indicators.calculate_atr(ohlc['high'], ohlc['low'], ohlc['close'])
        ohlc['atr'] = atr
        
        # Generate signals
        signals = self.get_signals(ohlc, features_lstm, features_xgb, config)
        
        # Calculate HE levels
        hierarchical_levels = self.he.identify_hierarchical_levels(ohlc)
        
        # Run simulation
        for i, (timestamp, ohlc_row) in enumerate(ohlc.iterrows()):
            if timestamp not in signals.index:
                continue
            
            signal_row = signals.loc[timestamp]
            
            # Update trailing stops
            if config.use_he_trailing_stop:
                self.update_trailing_stops(ohlc_row, hierarchical_levels)
            
            # Check exits
            exits = self.check_exits(ohlc_row, i)
            
            # Enter new trades
            new_trade = self.enter_trade(signal_row, ohlc_row, config)
            
            # Update equity curve
            total_pnl = sum(trade.pnl for trade in self.closed_trades)
            unrealized_pnl = 0
            
            for trade in self.open_trades:
                if trade.direction == 1:
                    unrealized_pnl += (ohlc_row['close'] - trade.entry_price) * trade.size
                else:
                    unrealized_pnl += (trade.entry_price - ohlc_row['close']) * trade.size
            
            current_equity = config.initial_capital + total_pnl + unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'realized_pnl': total_pnl,
                'unrealized_pnl': unrealized_pnl,
                'open_trades': len(self.open_trades),
                'closed_trades': len(self.closed_trades)
            })
        
        # Close remaining open trades
        if self.open_trades:
            final_price = ohlc.iloc[-1]['close']
            final_time = ohlc.index[-1]
            
            for trade in self.open_trades[:]:
                if trade.direction == 1:
                    pnl = (final_price - trade.entry_price) * trade.size
                else:
                    pnl = (trade.entry_price - final_price) * trade.size
                
                trade.exit_time = final_time
                trade.exit_price = final_price
                trade.exit_reason = "backtest_end"
                trade.pnl = pnl
                trade.return_pct = pnl / (trade.entry_price * trade.size)
                
                self.closed_trades.append(trade)
            
            self.open_trades = []
        
        # Calculate metrics
        metrics = self.calculate_metrics(config)
        
        logger.info(f"Backtest completed for {symbol}: {len(self.closed_trades)} trades")
        
        return {
            'symbol': symbol,
            'config': config,
            'trades': self.closed_trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics
        }

    def calculate_metrics(self, config: BacktestConfig) -> Dict:
        """Calculate comprehensive backtest metrics"""
        if not self.closed_trades:
            return {'error': 'No trades to analyze'}

        trades_df = pd.DataFrame([
            {
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'direction': trade.direction,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'holding_period': trade.holding_period,
                'rr_achieved': trade.rr_achieved,
                'exit_reason': trade.exit_reason
            }
            for trade in self.closed_trades
        ])

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)

        # Filter trades by minimum RR for evaluation
        qualified_trades = trades_df[trades_df['rr_achieved'] >= config.min_rr_filter]

        # Basic metrics
        total_trades = len(trades_df)
        qualified_trade_count = len(qualified_trades)
        win_trades = qualified_trades[qualified_trades['pnl'] > 0]
        loss_trades = qualified_trades[qualified_trades['pnl'] <= 0]

        win_rate = len(win_trades) / qualified_trade_count if qualified_trade_count > 0 else 0

        # PnL metrics
        total_pnl = qualified_trades['pnl'].sum()
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0

        # Risk metrics
        returns = equity_df['equity'].pct_change().dropna()

        # Sharpe Ratio (annualized)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() * 252 * 24 * 12) / (returns.std() * np.sqrt(252 * 24 * 12))  # 5-min bars
        else:
            sharpe_ratio = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() * 252 * 24 * 12) / (downside_returns.std() * np.sqrt(252 * 24 * 12))
        else:
            sortino_ratio = 0

        # Maximum Drawdown
        equity_values = equity_df['equity']
        rolling_max = equity_values.expanding().max()
        drawdown = (equity_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # CAGR
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days > 0:
            cagr = (equity_values.iloc[-1] / config.initial_capital) ** (365.25 / days) - 1
        else:
            cagr = 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Profit Factor
        gross_profit = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
        gross_loss = abs(loss_trades['pnl'].sum()) if len(loss_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_trades': total_trades,
            'qualified_trades': qualified_trade_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'final_equity': equity_values.iloc[-1],
            'return_pct': (equity_values.iloc[-1] / config.initial_capital - 1) * 100,
            'avg_holding_period': qualified_trades['holding_period'].mean(),
            'avg_rr_achieved': qualified_trades['rr_achieved'].mean(),
            'exit_reasons': qualified_trades['exit_reason'].value_counts().to_dict()
        }
