"""
Live Trading Bot for Black-Swan Hunter System
Real-time inference and order execution with MT5 integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import threading
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.features.black_swan_pipeline import BlackSwanFeaturePipeline
from src.models.xgb_mfe_model import XGBMFERegressor
from src.models.lstm_tail_model import LSTMTailClassifier
from src.backtesting.black_swan_backtest import BacktestConfig

logger = logging.getLogger(__name__)

@dataclass
class LivePosition:
    """Live trading position"""
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: Optional[float]
    entry_time: datetime
    atr_at_entry: float
    mfe_prediction: float
    tail_probability: float

class MT5Interface:
    """MetaTrader 5 interface for live trading"""
    
    def __init__(self):
        self.connected = False
        self.mt5 = None
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Connected to MT5 account: {account_info.login}")
            self.connected = True
            return True
            
        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.mt5 and self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask prices"""
        if not self.connected:
            return None
        
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def get_bars(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """Get historical bars"""
        if not self.connected:
            return None
        
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def place_order(self, symbol: str, order_type: int, volume: float, 
                   price: float, sl: float, tp: Optional[float] = None) -> Optional[int]:
        """Place market order"""
        if not self.connected:
            return None
        
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "BlackSwanHunter",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        
        result = self.mt5.order_send(request)
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode}")
            return None
        
        return result.order
    
    def close_position(self, ticket: int) -> bool:
        """Close position by ticket"""
        if not self.connected:
            return False
        
        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        
        position = positions[0]
        
        if position.type == self.mt5.POSITION_TYPE_BUY:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "BlackSwanHunter_Close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        
        result = self.mt5.order_send(request)
        return result.retcode == self.mt5.TRADE_RETCODE_DONE
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self.connected:
            return []
        
        positions = self.mt5.positions_get()
        if positions is None:
            return []
        
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': pos.type,
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'time': datetime.fromtimestamp(pos.time),
                'profit': pos.profit
            }
            for pos in positions
        ]

class BlackSwanLiveBot:
    """Live trading bot for Black Swan Hunter system"""
    
    def __init__(self, config: BacktestConfig = None, symbols: List[str] = None):
        self.config = config or BacktestConfig()
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        # Initialize components
        self.feature_pipeline = BlackSwanFeaturePipeline()
        self.mt5 = MT5Interface()
        self.models = {}
        self.positions: Dict[int, LivePosition] = {}
        
        # Control flags
        self.running = False
        self.last_signal_check = {}
        
        # Performance tracking
        self.daily_pnl = 0
        self.trade_count = 0
        
    def load_models(self, models_dir: Path):
        """Load trained XGB and LSTM models"""
        logger.info("Loading trained models...")
        
        for symbol in self.symbols:
            symbol_models = {}
            
            # Load XGB MFE model
            xgb_path = models_dir / 'xgb_mfe' / symbol / 'model.json'
            if xgb_path.exists():
                xgb_model = XGBMFERegressor()
                xgb_model.load_model(str(xgb_path))
                symbol_models['xgb'] = xgb_model
                logger.info(f"Loaded XGB model for {symbol}")
            
            # Load LSTM Tail model
            lstm_path = models_dir / 'lstm_tail' / symbol / 'model.h5'
            if lstm_path.exists():
                lstm_model = LSTMTailClassifier()
                lstm_model.load_model(str(lstm_path))
                symbol_models['lstm'] = lstm_model
                logger.info(f"Loaded LSTM model for {symbol}")
            
            if symbol_models:
                self.models[symbol] = symbol_models
        
        logger.info(f"Loaded models for {len(self.models)} symbols")
    
    def connect_to_broker(self) -> bool:
        """Connect to MT5 broker"""
        return self.mt5.connect()
    
    def get_market_data(self, symbol: str, bars_needed: int = 300) -> Optional[pd.DataFrame]:
        """Get current market data for analysis"""
        # Get M5 bars (timeframe = 5)
        df = self.mt5.get_bars(symbol, 5, bars_needed)
        
        if df is None or len(df) < bars_needed:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        return df
    
    def generate_predictions(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate MFE and tail probability predictions"""
        if symbol not in self.models:
            return None
        
        symbol_models = self.models[symbol]
        
        try:
            # Generate features
            xgb_features = self.feature_pipeline.generate_xgb_features(df, symbol)
            lstm_features = self.feature_pipeline.generate_lstm_features(df, symbol)
            
            # Get latest features (most recent bar)
            latest_xgb = xgb_features.iloc[[-1]]
            
            # XGB MFE prediction
            mfe_prediction = 0
            if 'xgb' in symbol_models:
                mfe_pred = symbol_models['xgb'].predict(latest_xgb)
                mfe_prediction = mfe_pred[0] if len(mfe_pred) > 0 else 0
            
            # LSTM tail probability prediction
            tail_probabilities = [0.7, 0.2, 0.08, 0.02]  # Default
            if 'lstm' in symbol_models:
                # Prepare sequences
                X_seq, _ = self.feature_pipeline.prepare_lstm_sequences(lstm_features)
                if len(X_seq) > 0:
                    tail_probs = symbol_models['lstm'].predict_proba(X_seq[-1:])
                    tail_probabilities = tail_probs[0].tolist()
            
            return {
                'mfe_prediction': mfe_prediction,
                'tail_probabilities': tail_probabilities,
                'tail_probability_sum': sum(tail_probabilities[1:])  # Classes 1,2,3
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None
    
    def check_entry_signal(self, symbol: str, df: pd.DataFrame, predictions: Dict) -> Optional[str]:
        """Check for entry signals"""
        latest_bar = df.iloc[-1]
        
        mfe_pred = predictions['mfe_prediction']
        tail_prob = predictions['tail_probability_sum']
        
        # Check minimum criteria
        if mfe_pred < self.config.min_mfe_prediction:
            return None
        if tail_prob < self.config.min_tail_probability:
            return None
        
        # Check position limits
        current_positions = len([p for p in self.positions.values() if p.symbol == symbol])
        if current_positions >= 1:  # Max 1 position per symbol
            return None
        
        total_positions = len(self.positions)
        if total_positions >= self.config.max_concurrent_positions:
            return None
        
        # Check daily loss limit
        if self.daily_pnl <= -self.config.initial_capital * self.config.max_daily_loss_pct:
            return None
        
        # Simple trend filter (could be enhanced)
        # For now, just check if we have a strong signal
        if mfe_pred >= 10.0 and tail_prob >= 0.4:
            return 'long'  # Could add short logic here
        
        return None
    
    def calculate_position_size(self, symbol: str, mfe_pred: float, tail_prob: float, atr: float) -> float:
        """Calculate position size for live trading"""
        # Get current account balance
        if self.mt5.connected:
            account_info = self.mt5.mt5.account_info()
            current_capital = account_info.balance if account_info else self.config.initial_capital
        else:
            current_capital = self.config.initial_capital
        
        # Determine confidence multiplier
        if mfe_pred >= self.config.high_confidence_mfe and tail_prob >= self.config.high_confidence_prob:
            multiplier = self.config.high_confidence_multiplier
        elif mfe_pred >= self.config.medium_confidence_mfe and tail_prob >= self.config.medium_confidence_prob:
            multiplier = self.config.medium_confidence_multiplier
        else:
            multiplier = self.config.low_confidence_multiplier
        
        # Calculate risk amount
        risk_amount = current_capital * self.config.base_risk_per_trade * multiplier
        
        # Position size based on ATR stop
        stop_distance = atr * self.config.stop_loss_atr_multiple
        
        # Get current price to calculate lot size
        price_info = self.mt5.get_current_price(symbol)
        if not price_info:
            return 0
        
        current_price = price_info['ask']  # For long positions
        
        # Calculate lot size (simplified - should consider contract size)
        lot_size = risk_amount / (stop_distance * 100000)  # Assuming standard lot
        
        # Round to broker's minimum lot size
        min_lot = 0.01
        lot_size = max(min_lot, round(lot_size / min_lot) * min_lot)
        
        return lot_size
    
    def open_position(self, symbol: str, direction: str, df: pd.DataFrame, predictions: Dict) -> bool:
        """Open a new live position"""
        latest_bar = df.iloc[-1]
        atr = latest_bar.get('atr14', latest_bar['close'] * 0.01)
        
        # Get current price
        price_info = self.mt5.get_current_price(symbol)
        if not price_info:
            return False
        
        if direction == 'long':
            entry_price = price_info['ask']
            stop_loss = entry_price - (atr * self.config.stop_loss_atr_multiple)
            order_type = self.mt5.mt5.ORDER_TYPE_BUY
        else:
            entry_price = price_info['bid']
            stop_loss = entry_price + (atr * self.config.stop_loss_atr_multiple)
            order_type = self.mt5.mt5.ORDER_TYPE_SELL
        
        # Calculate position size
        mfe_pred = predictions['mfe_prediction']
        tail_prob = predictions['tail_probability_sum']
        lot_size = self.calculate_position_size(symbol, mfe_pred, tail_prob, atr)
        
        if lot_size <= 0:
            return False
        
        # Place order
        ticket = self.mt5.place_order(
            symbol=symbol,
            order_type=order_type,
            volume=lot_size,
            price=entry_price,
            sl=stop_loss
        )
        
        if ticket:
            # Record position
            position = LivePosition(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=lot_size,
                stop_loss=stop_loss,
                take_profit=None,
                entry_time=datetime.now(),
                atr_at_entry=atr,
                mfe_prediction=mfe_pred,
                tail_probability=tail_prob
            )
            
            self.positions[ticket] = position
            self.trade_count += 1
            
            logger.info(f"Opened {direction} position: {symbol} @ {entry_price:.5f}, "
                       f"SL: {stop_loss:.5f}, Size: {lot_size}, MFE: {mfe_pred:.2f}R")
            return True
        
        return False
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        for ticket, position in list(self.positions.items()):
            # Get current market data
            df = self.get_market_data(position.symbol, 50)
            if df is None:
                continue
            
            current_price = df.iloc[-1]['close']
            
            # Check exit conditions
            should_close, reason = self.check_exit_conditions(position, current_price)
            
            if should_close:
                if self.mt5.close_position(ticket):
                    logger.info(f"Closed position {ticket}: {reason}")
                    del self.positions[ticket]
                else:
                    logger.error(f"Failed to close position {ticket}")
    
    def check_exit_conditions(self, position: LivePosition, current_price: float) -> Tuple[bool, str]:
        """Check if position should be closed"""
        # Time-based exit
        time_held = datetime.now() - position.entry_time
        if time_held.total_seconds() > self.config.max_hold_bars * 300:  # 5 minutes per bar
            return True, 'max_hold_time'
        
        # Calculate current P&L in R-multiples
        if position.direction == 'long':
            pnl_points = current_price - position.entry_price
        else:
            pnl_points = position.entry_price - current_price
        
        pnl_r = pnl_points / position.atr_at_entry
        
        # Take profit at 5R
        if pnl_r >= self.config.partial_take_profit_r:
            return True, 'take_profit'
        
        # Additional exit logic could be added here
        
        return False, ''
    
    def run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting live trading loop...")
        self.running = True
        
        while self.running:
            try:
                # Monitor existing positions
                self.monitor_positions()
                
                # Check for new signals
                for symbol in self.symbols:
                    # Rate limiting - check each symbol every 5 minutes
                    now = datetime.now()
                    last_check = self.last_signal_check.get(symbol, datetime.min)
                    
                    if (now - last_check).total_seconds() < 300:  # 5 minutes
                        continue
                    
                    self.last_signal_check[symbol] = now
                    
                    # Get market data
                    df = self.get_market_data(symbol)
                    if df is None:
                        continue
                    
                    # Generate predictions
                    predictions = self.generate_predictions(symbol, df)
                    if predictions is None:
                        continue
                    
                    # Check for entry signal
                    direction = self.check_entry_signal(symbol, df, predictions)
                    if direction:
                        self.open_position(symbol, direction, df, predictions)
                
                # Sleep for 30 seconds before next iteration
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait longer on error
        
        self.running = False
        logger.info("Trading loop stopped")
    
    def start(self, models_dir: Path):
        """Start the live trading bot"""
        logger.info("Starting Black Swan Hunter Live Bot...")
        
        # Load models
        self.load_models(models_dir)
        
        if not self.models:
            logger.error("No models loaded, cannot start trading")
            return False
        
        # Connect to broker
        if not self.connect_to_broker():
            logger.error("Failed to connect to broker")
            return False
        
        # Start trading loop in separate thread
        trading_thread = threading.Thread(target=self.run_trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        logger.info("Live bot started successfully")
        return True
    
    def stop(self):
        """Stop the live trading bot"""
        logger.info("Stopping live bot...")
        self.running = False
        
        # Close all open positions
        for ticket in list(self.positions.keys()):
            if self.mt5.close_position(ticket):
                logger.info(f"Closed position {ticket} on shutdown")
        
        # Disconnect from broker
        self.mt5.disconnect()
        
        logger.info("Live bot stopped")

def main():
    """Main entry point for live trading"""
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', type=str, required=True, help='Directory containing trained models')
    ap.add_argument('--symbols', type=str, default='EURUSD,GBPUSD,USDJPY,AUDUSD', help='Comma-separated symbols')
    ap.add_argument('--config', type=str, help='JSON config file path')
    args = ap.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Load config
    config = BacktestConfig()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Initialize and start bot
    bot = BlackSwanLiveBot(config=config, symbols=symbols)
    
    try:
        if bot.start(Path(args.models_dir)):
            # Keep main thread alive
            while bot.running:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop()

if __name__ == '__main__':
    main()
