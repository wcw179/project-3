"""
Comprehensive Backtesting Engine for Black-Swan Hunter Trading Bot
Implements walk-forward analysis with ATR-based risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: Optional[float]
    position_size: float
    atr_at_entry: float
    mfe_prediction: float
    tail_probability: float
    
    # Realized metrics
    pnl_points: Optional[float] = None
    pnl_r_multiple: Optional[float] = None
    pnl_percentage: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    bars_held: Optional[int] = None
    exit_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.exit_price is not None and self.pnl_points is None:
            self.calculate_realized_metrics()
    
    def calculate_realized_metrics(self):
        """Calculate realized trade metrics"""
        if self.exit_price is None:
            return
        
        if self.direction == 'long':
            self.pnl_points = self.exit_price - self.entry_price
        else:  # short
            self.pnl_points = self.entry_price - self.exit_price
        
        self.pnl_r_multiple = self.pnl_points / self.atr_at_entry
        self.pnl_percentage = (self.pnl_points / self.entry_price) * 100

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    base_risk_per_trade: float = 0.005  # 0.5% of capital
    max_concurrent_positions: int = 3
    max_daily_loss_pct: float = 0.02  # 2% daily loss limit
    
    # Entry criteria
    min_mfe_prediction: float = 5.0
    min_tail_probability: float = 0.3
    
    # Position sizing multipliers
    high_confidence_mfe: float = 15.0
    high_confidence_prob: float = 0.4
    high_confidence_multiplier: float = 1.5
    
    medium_confidence_mfe: float = 10.0
    medium_confidence_prob: float = 0.3
    medium_confidence_multiplier: float = 1.0
    
    low_confidence_multiplier: float = 0.5
    
    # Risk management
    stop_loss_atr_multiple: float = 1.0
    trailing_stop_trigger_r: float = 3.0
    partial_take_profit_r: float = 5.0
    partial_take_profit_pct: float = 0.3
    max_hold_bars: int = 500  # 41.7 hours for M5
    
    # Trend filter
    use_trend_filter: bool = True

class BlackSwanBacktester:
    """Comprehensive backtesting engine for Black Swan Hunter"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.config.initial_capital
        self.trades: List[Trade] = []
        self.open_positions: List[Trade] = []
        self.daily_pnl = {}
        self.equity_curve = []
        self.current_bar_index = 0
        
    def calculate_position_size(self, mfe_pred: float, tail_prob: float, atr: float, price: float) -> float:
        """Calculate position size based on prediction confidence"""
        # Determine confidence level
        if mfe_pred >= self.config.high_confidence_mfe and tail_prob >= self.config.high_confidence_prob:
            multiplier = self.config.high_confidence_multiplier
        elif mfe_pred >= self.config.medium_confidence_mfe and tail_prob >= self.config.medium_confidence_prob:
            multiplier = self.config.medium_confidence_multiplier
        else:
            multiplier = self.config.low_confidence_multiplier
        
        # Base risk amount
        risk_amount = self.capital * self.config.base_risk_per_trade * multiplier
        
        # Position size based on ATR stop loss
        stop_loss_distance = atr * self.config.stop_loss_atr_multiple
        position_size = risk_amount / stop_loss_distance
        
        return position_size
    
    def check_trend_filter(self, bar: pd.Series, direction: str) -> bool:
        """Check trend filter conditions"""
        if not self.config.use_trend_filter:
            return True
        
        # Simple trend filter using EMAs (assuming they're in the data)
        if 'ema20' in bar.index and 'ema50' in bar.index and 'ema200' in bar.index:
            if direction == 'long':
                return bar['ema20'] > bar['ema50'] > bar['ema200']
            else:  # short
                return bar['ema20'] < bar['ema50'] < bar['ema200']
        
        return True  # Default to True if EMA data not available
    
    def check_entry_signal(self, bar: pd.Series, predictions: Dict) -> Optional[str]:
        """Check for entry signals"""
        mfe_pred = predictions.get('mfe_prediction', 0)
        tail_probs = predictions.get('tail_probabilities', [1, 0, 0, 0])
        
        # Calculate tail probability (classes 1, 2, 3)
        tail_prob = sum(tail_probs[1:])
        
        # Check minimum criteria
        if mfe_pred < self.config.min_mfe_prediction or tail_prob < self.config.min_tail_probability:
            return None
        
        # Check position limits
        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return None
        
        # Check daily loss limit
        today = bar.name.date() if hasattr(bar.name, 'date') else None
        if today and today in self.daily_pnl:
            if self.daily_pnl[today] <= -self.capital * self.config.max_daily_loss_pct:
                return None
        
        # Determine direction (simplified - could be more sophisticated)
        # For now, assume long direction if trend filter passes
        direction = 'long'  # Could be enhanced with directional prediction
        
        if self.check_trend_filter(bar, direction):
            return direction
        
        return None
    
    def open_position(self, bar: pd.Series, direction: str, predictions: Dict):
        """Open a new position"""
        mfe_pred = predictions.get('mfe_prediction', 0)
        tail_probs = predictions.get('tail_probabilities', [1, 0, 0, 0])
        tail_prob = sum(tail_probs[1:])
        
        atr = bar.get('atr14', bar['close'] * 0.01)  # Fallback ATR
        price = bar['close']
        
        # Calculate position size
        position_size = self.calculate_position_size(mfe_pred, tail_prob, atr, price)
        
        # Calculate stop loss
        if direction == 'long':
            stop_loss = price - (atr * self.config.stop_loss_atr_multiple)
        else:  # short
            stop_loss = price + (atr * self.config.stop_loss_atr_multiple)
        
        # Create trade
        trade = Trade(
            entry_time=bar.name,
            exit_time=None,
            symbol=bar.get('symbol', 'UNKNOWN'),
            direction=direction,
            entry_price=price,
            exit_price=None,
            stop_loss=stop_loss,
            take_profit=None,
            position_size=position_size,
            atr_at_entry=atr,
            mfe_prediction=mfe_pred,
            tail_probability=tail_prob
        )
        
        self.open_positions.append(trade)
        logger.debug(f"Opened {direction} position at {price:.5f}, stop: {stop_loss:.5f}, size: {position_size:.2f}")
    
    def update_position_metrics(self, trade: Trade, bar: pd.Series):
        """Update MFE/MAE for open position"""
        current_price = bar['close']
        
        if trade.direction == 'long':
            unrealized_pnl = current_price - trade.entry_price
            if trade.max_favorable_excursion is None or unrealized_pnl > trade.max_favorable_excursion:
                trade.max_favorable_excursion = unrealized_pnl
            if trade.max_adverse_excursion is None or unrealized_pnl < trade.max_adverse_excursion:
                trade.max_adverse_excursion = unrealized_pnl
        else:  # short
            unrealized_pnl = trade.entry_price - current_price
            if trade.max_favorable_excursion is None or unrealized_pnl > trade.max_favorable_excursion:
                trade.max_favorable_excursion = unrealized_pnl
            if trade.max_adverse_excursion is None or unrealized_pnl < trade.max_adverse_excursion:
                trade.max_adverse_excursion = unrealized_pnl
    
    def check_exit_conditions(self, trade: Trade, bar: pd.Series) -> Tuple[bool, str]:
        """Check if position should be closed"""
        current_price = bar['close']
        bars_held = self.current_bar_index - getattr(trade, '_entry_bar_index', 0)
        
        # Stop loss
        if trade.direction == 'long' and current_price <= trade.stop_loss:
            return True, 'stop_loss'
        elif trade.direction == 'short' and current_price >= trade.stop_loss:
            return True, 'stop_loss'
        
        # Maximum hold time
        if bars_held >= self.config.max_hold_bars:
            return True, 'max_hold_time'
        
        # Partial take profit
        if trade.max_favorable_excursion is not None:
            mfe_r = trade.max_favorable_excursion / trade.atr_at_entry
            if mfe_r >= self.config.partial_take_profit_r:
                return True, 'take_profit'
        
        # Trailing stop (simplified)
        if trade.max_favorable_excursion is not None:
            mfe_r = trade.max_favorable_excursion / trade.atr_at_entry
            if mfe_r >= self.config.trailing_stop_trigger_r:
                # Implement trailing stop logic
                trailing_distance = trade.atr_at_entry * 2.0  # 2 ATR trailing
                if trade.direction == 'long':
                    trailing_stop = current_price - trailing_distance
                    if current_price <= trailing_stop:
                        return True, 'trailing_stop'
                else:  # short
                    trailing_stop = current_price + trailing_distance
                    if current_price >= trailing_stop:
                        return True, 'trailing_stop'
        
        return False, ''
    
    def close_position(self, trade: Trade, bar: pd.Series, exit_reason: str):
        """Close an open position"""
        trade.exit_time = bar.name
        trade.exit_price = bar['close']
        trade.exit_reason = exit_reason
        trade.bars_held = self.current_bar_index - getattr(trade, '_entry_bar_index', 0)
        
        # Calculate final metrics
        trade.calculate_realized_metrics()
        
        # Update daily P&L
        today = bar.name.date() if hasattr(bar.name, 'date') else None
        if today:
            if today not in self.daily_pnl:
                self.daily_pnl[today] = 0
            self.daily_pnl[today] += trade.pnl_points * trade.position_size
        
        # Update capital
        pnl_dollars = trade.pnl_points * trade.position_size
        self.capital += pnl_dollars
        
        # Move to closed trades
        self.trades.append(trade)
        self.open_positions.remove(trade)
        
        logger.debug(f"Closed {trade.direction} position: {trade.pnl_r_multiple:.2f}R, reason: {exit_reason}")
    
    def run_backtest(self, data: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Run complete backtest"""
        logger.info(f"Starting backtest on {len(data)} bars")
        
        self.reset()
        
        # Align data and predictions
        common_index = data.index.intersection(predictions.index)
        data_aligned = data.loc[common_index]
        predictions_aligned = predictions.loc[common_index]
        
        for i, (timestamp, bar) in enumerate(data_aligned.iterrows()):
            self.current_bar_index = i
            
            # Get predictions for this bar
            if timestamp in predictions_aligned.index:
                pred_row = predictions_aligned.loc[timestamp]
                predictions_dict = {
                    'mfe_prediction': pred_row.get('mfe_prediction', 0),
                    'tail_probabilities': [
                        pred_row.get('tail_prob_0', 0.7),
                        pred_row.get('tail_prob_1', 0.2),
                        pred_row.get('tail_prob_2', 0.08),
                        pred_row.get('tail_prob_3', 0.02)
                    ]
                }
            else:
                continue
            
            # Update open positions
            for trade in self.open_positions.copy():
                self.update_position_metrics(trade, bar)
                
                should_exit, exit_reason = self.check_exit_conditions(trade, bar)
                if should_exit:
                    self.close_position(trade, bar, exit_reason)
            
            # Check for new entry signals
            entry_direction = self.check_entry_signal(bar, predictions_dict)
            if entry_direction:
                self.open_position(bar, entry_direction, predictions_dict)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.capital,
                'open_positions': len(self.open_positions)
            })
        
        # Close any remaining open positions
        if self.open_positions and len(data_aligned) > 0:
            final_bar = data_aligned.iloc[-1]
            for trade in self.open_positions.copy():
                self.close_position(trade, final_bar, 'end_of_data')
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics()
        
        logger.info(f"Backtest completed: {len(self.trades)} trades, Final equity: ${self.capital:,.2f}")
        
        return {
            'trades': [asdict(trade) for trade in self.trades],
            'equity_curve': self.equity_curve,
            'performance_metrics': performance_metrics,
            'config': asdict(self.config)
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl_r_multiple > 0]
        losing_trades = [t for t in self.trades if t.pnl_r_multiple <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital
        
        returns = [t.pnl_r_multiple for t in self.trades]
        avg_return_r = np.mean(returns) if returns else 0
        
        # Risk metrics
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288)  # Annualized for M5
        else:
            sharpe_ratio = 0
        
        # Drawdown calculation
        equity_values = [eq['equity'] for eq in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # Tail event detection metrics
        tail_trades = [t for t in self.trades if t.pnl_r_multiple >= 5.0]  # 5R+ moves
        extreme_tail_trades = [t for t in self.trades if t.pnl_r_multiple >= 20.0]  # 20R+ moves
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return_r': avg_return_r,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'tail_events_captured': len(tail_trades),
            'extreme_tail_events': len(extreme_tail_trades),
            'avg_winning_trade_r': np.mean([t.pnl_r_multiple for t in winning_trades]) if winning_trades else 0,
            'avg_losing_trade_r': np.mean([t.pnl_r_multiple for t in losing_trades]) if losing_trades else 0,
            'largest_win_r': max([t.pnl_r_multiple for t in self.trades]) if self.trades else 0,
            'largest_loss_r': min([t.pnl_r_multiple for t in self.trades]) if self.trades else 0
        }
