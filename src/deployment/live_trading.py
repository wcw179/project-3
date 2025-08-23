"""
Live trading system for M5 Multi-Symbol Trend Bot
Real-time execution with MT5/broker API integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import time
import threading
from dataclasses import dataclass
import json
from pathlib import Path
import MetaTrader5 as mt5

from src.data.database import TradingDatabase
from src.data.ingestion import DataIngestion
from src.features.feature_pipeline import FeaturePipeline
from src.features.hierarchical_extremes import HierarchicalExtremes
from src.models.lstm_model import LSTMTrendClassifier
from src.models.xgb_model import XGBMetaModel

logger = logging.getLogger(__name__)

@dataclass
class LivePosition:
    """Live position tracking"""
    ticket: int
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss: float
    profit_target: float
    current_price: float
    unrealized_pnl: float
    lstm_prob: float
    xgb_score: float
    trailing_stop: Optional[float] = None

@dataclass
class TradingConfig:
    """Live trading configuration"""
    symbols: List[str]
    risk_per_trade: float = 0.02
    max_positions_per_symbol: int = 1
    max_total_positions: int = 5
    lstm_threshold: float = 0.55
    xgb_threshold: float = 0.6
    min_rr_ratio: float = 1.5
    use_ema_confirmation: bool = True
    use_black_swan_filter: bool = True
    use_he_trailing_stop: bool = True
    update_interval: int = 300  # seconds (5 minutes)
    max_slippage: float = 0.0005
    max_spread: float = 0.001

class LiveTradingEngine:
    """Main live trading engine"""
    
    def __init__(self, config: TradingConfig, db_path: str = "data/trading_system.db",
                 models_dir: str = "artifacts/models"):
        self.config = config
        self.db = TradingDatabase(db_path)
        self.data_ingestion = DataIngestion(db_path)
        self.feature_pipeline = FeaturePipeline(db_path)
        self.he = HierarchicalExtremes(levels=3, atr_lookback=1440)
        
        self.models_dir = Path(models_dir)
        self.models = {}  # symbol -> {'lstm': model, 'xgb': model}
        
        # Trading state
        self.positions: Dict[str, List[LivePosition]] = {}
        self.is_running = False
        self.last_update = {}
        
        # Performance tracking
        self.trade_log = []
        self.equity_history = []
        
    def initialize_mt5(self, login: int, password: str, server: str) -> bool:
        """Initialize MT5 connection"""
        if not self.data_ingestion.initialize_mt5(login, password, server):
            logger.error("Failed to initialize MT5")
            return False
        
        logger.info("MT5 initialized for live trading")
        return True
    
    def load_models(self) -> bool:
        """Load trained models for all symbols"""
        logger.info("Loading trained models...")
        
        for symbol in self.config.symbols:
            try:
                # Load LSTM model
                lstm_path = self.models_dir / "lstm" / symbol / "model.h5"
                if lstm_path.exists():
                    lstm_model = LSTMTrendClassifier()
                    lstm_model.load_model(str(lstm_path))
                    
                    # Load XGB model
                    xgb_path = self.models_dir / "xgb" / symbol / "model.json"
                    if xgb_path.exists():
                        xgb_model = XGBMetaModel()
                        xgb_model.load_model(str(xgb_path))
                        
                        self.models[symbol] = {
                            'lstm': lstm_model,
                            'xgb': xgb_model
                        }
                        
                        logger.info(f"Models loaded for {symbol}")
                    else:
                        logger.warning(f"XGB model not found for {symbol}")
                else:
                    logger.warning(f"LSTM model not found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load models for {symbol}: {e}")
        
        loaded_symbols = len(self.models)
        logger.info(f"Loaded models for {loaded_symbols}/{len(self.config.symbols)} symbols")
        
        return loaded_symbols > 0
    
    def get_current_data(self, symbol: str, bars: int = 300) -> pd.DataFrame:
        """Get current market data for a symbol"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=bars * 5)
            
            # Update data from MT5
            self.data_ingestion.update_symbol_data(symbol, lookback_days=2)
            
            # Get data from database
            ohlc = self.db.get_ohlcv_data(symbol, start_time, end_time)
            
            if ohlc.empty:
                logger.warning(f"No current data available for {symbol}")
                return pd.DataFrame()
            
            return ohlc.tail(bars)
            
        except Exception as e:
            logger.error(f"Error getting current data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, symbol: str, ohlc: pd.DataFrame) -> Dict:
        """Generate trading signals for a symbol"""
        if symbol not in self.models:
            return {'signal': 0, 'lstm_prob': 0.33, 'xgb_score': 0.5}
        
        try:
            # Generate features
            lstm_features = self.feature_pipeline.generate_lstm_features(ohlc, symbol)
            
            # Get LSTM predictions
            lstm_model = self.models[symbol]['lstm']
            lstm_features_array = lstm_features.values
            
            if len(lstm_features_array) < lstm_model.sequence_length:
                return {'signal': 0, 'lstm_prob': 0.33, 'xgb_score': 0.5}
            
            lstm_probs, lstm_classes = lstm_model.predict(lstm_features_array, return_probabilities=True)
            
            # Get latest LSTM probabilities
            latest_lstm_probs = {
                'p_up': lstm_probs[-1, 2],
                'p_down': lstm_probs[-1, 0],
                'p_neutral': lstm_probs[-1, 1]
            }
            
            # Generate XGB features with LSTM probabilities
            xgb_features = self.feature_pipeline.generate_xgb_features(ohlc, symbol)
            
            # Add LSTM probabilities to XGB features
            for key, value in latest_lstm_probs.items():
                xgb_features.loc[xgb_features.index[-1], f'lstm_{key}'] = value
            
            # Get XGB prediction
            xgb_model = self.models[symbol]['xgb']
            xgb_score = xgb_model.predict_proba(xgb_features.tail(1))[0]
            
            # Apply filters
            ema_confirmation = self._check_ema_confirmation(ohlc)
            black_swan_filter = self._check_black_swan_filter(ohlc)
            
            # Generate signal
            signal = 0
            max_lstm_prob = max(latest_lstm_probs['p_up'], latest_lstm_probs['p_down'])
            
            if (max_lstm_prob > self.config.lstm_threshold and 
                xgb_score > self.config.xgb_threshold and
                ema_confirmation and black_swan_filter):
                
                if latest_lstm_probs['p_up'] > latest_lstm_probs['p_down']:
                    signal = 1  # Long
                else:
                    signal = -1  # Short
            
            return {
                'signal': signal,
                'lstm_prob': max_lstm_prob,
                'xgb_score': xgb_score,
                'lstm_probs': latest_lstm_probs,
                'ema_confirmation': ema_confirmation,
                'black_swan_filter': black_swan_filter
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return {'signal': 0, 'lstm_prob': 0.33, 'xgb_score': 0.5}
    
    def _check_ema_confirmation(self, ohlc: pd.DataFrame) -> bool:
        """Check EMA trend confirmation"""
        if len(ohlc) < 200:
            return False
        
        ema20 = ohlc['close'].ewm(span=20).mean()
        ema50 = ohlc['close'].ewm(span=50).mean()
        ema200 = ohlc['close'].ewm(span=200).mean()
        
        current_price = ohlc['close'].iloc[-1]
        current_ema20 = ema20.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_ema200 = ema200.iloc[-1]
        
        # Check for clear trend
        uptrend = (current_ema20 > current_ema50 > current_ema200 and 
                  current_price > current_ema20)
        downtrend = (current_ema20 < current_ema50 < current_ema200 and 
                    current_price < current_ema20)
        
        return uptrend or downtrend
    
    def _check_black_swan_filter(self, ohlc: pd.DataFrame) -> bool:
        """Check black swan filter"""
        if len(ohlc) < 100:
            return True
        
        returns = ohlc['close'].pct_change()
        vol = returns.rolling(20).std()
        
        # Check recent volatility
        current_vol = vol.iloc[-1]
        vol_percentile = vol.rolling(100).rank(pct=True).iloc[-1]
        
        # Check recent large moves
        recent_returns = returns.tail(5)
        large_moves = (np.abs(recent_returns) > vol.iloc[-1] * 2).any()
        
        # Filter out extreme conditions
        return vol_percentile < 0.9 and not large_moves
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account information
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
            
            account_balance = account_info.balance
            risk_amount = account_balance * self.config.risk_per_trade
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit == 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            # Get symbol info for lot size constraints
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.0
            
            # Adjust for minimum lot size
            min_lot = symbol_info.volume_min
            lot_step = symbol_info.volume_step
            
            # Round to valid lot size
            position_size = max(min_lot, round(position_size / lot_step) * lot_step)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def place_order(self, symbol: str, signal: Dict, current_price: float) -> Optional[int]:
        """Place market order"""
        try:
            direction = signal['signal']
            if direction == 0:
                return None
            
            # Calculate stop loss and profit target
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            point = symbol_info.point
            spread = (symbol_info.ask - symbol_info.bid) / point
            
            # Check spread
            if spread * point > self.config.max_spread:
                logger.warning(f"Spread too wide for {symbol}: {spread * point}")
                return None
            
            # Calculate levels
            atr_estimate = current_price * 0.01  # Rough ATR estimate
            
            if direction == 1:  # Long
                entry_price = symbol_info.ask
                stop_loss = entry_price - (atr_estimate * 1.5)
                profit_target = entry_price + (atr_estimate * 3.0)
                order_type = mt5.ORDER_TYPE_BUY
            else:  # Short
                entry_price = symbol_info.bid
                stop_loss = entry_price + (atr_estimate * 1.5)
                profit_target = entry_price - (atr_estimate * 3.0)
                order_type = mt5.ORDER_TYPE_SELL
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, entry_price, stop_loss)
            if position_size == 0:
                return None
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": order_type,
                "price": entry_price,
                "sl": stop_loss,
                "tp": profit_target,
                "deviation": int(self.config.max_slippage / point),
                "magic": 12345,
                "comment": f"M5Bot_{direction}_{datetime.now().strftime('%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"Order send failed for {symbol}: {mt5.last_error()}")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed for {symbol}: {result.retcode}")
                return None
            
            # Create position tracking
            position = LivePosition(
                ticket=result.order,
                symbol=symbol,
                direction=direction,
                entry_time=datetime.now(),
                entry_price=result.price,
                size=position_size,
                stop_loss=stop_loss,
                profit_target=profit_target,
                current_price=result.price,
                unrealized_pnl=0.0,
                lstm_prob=signal['lstm_prob'],
                xgb_score=signal['xgb_score']
            )
            
            if symbol not in self.positions:
                self.positions[symbol] = []
            
            self.positions[symbol].append(position)
            
            logger.info(f"Order placed for {symbol}: {direction} {position_size} @ {result.price}")
            
            return result.order
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def update_positions(self):
        """Update all open positions"""
        try:
            # Get current positions from MT5
            mt5_positions = mt5.positions_get()
            if mt5_positions is None:
                mt5_positions = []
            
            # Update tracked positions
            for symbol in list(self.positions.keys()):
                for position in self.positions[symbol][:]:
                    # Find corresponding MT5 position
                    mt5_pos = None
                    for pos in mt5_positions:
                        if pos.ticket == position.ticket:
                            mt5_pos = pos
                            break
                    
                    if mt5_pos is None:
                        # Position closed
                        self.positions[symbol].remove(position)
                        logger.info(f"Position closed: {position.symbol} {position.ticket}")
                    else:
                        # Update position
                        position.current_price = mt5_pos.price_current
                        position.unrealized_pnl = mt5_pos.profit
                        
                        # Update trailing stop if enabled
                        if self.config.use_he_trailing_stop:
                            self.update_trailing_stop(position)
                
                # Clean up empty symbol entries
                if not self.positions[symbol]:
                    del self.positions[symbol]
                    
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def update_trailing_stop(self, position: LivePosition):
        """Update trailing stop using HE levels"""
        try:
            # Get recent data
            ohlc = self.get_current_data(position.symbol, 100)
            if ohlc.empty:
                return
            
            # Calculate HE levels
            hierarchical_levels = self.he.identify_hierarchical_levels(ohlc)
            current_atr = ohlc['close'].pct_change().rolling(14).std().iloc[-1] * position.current_price
            
            # Calculate new trailing stop
            new_stop = self.he.calculate_trailing_stop_level(
                hierarchical_levels, position.current_price, 
                'long' if position.direction == 1 else 'short',
                current_atr, level=1
            )
            
            if new_stop is None:
                return
            
            # Check if we should update the stop
            should_update = False
            if position.direction == 1 and new_stop > position.stop_loss:
                should_update = True
            elif position.direction == -1 and new_stop < position.stop_loss:
                should_update = True
            
            if should_update:
                # Modify position
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": position.symbol,
                    "position": position.ticket,
                    "sl": new_stop,
                    "tp": position.profit_target
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    position.stop_loss = new_stop
                    position.trailing_stop = new_stop
                    logger.info(f"Trailing stop updated for {position.symbol}: {new_stop}")
                    
        except Exception as e:
            logger.error(f"Error updating trailing stop for {position.symbol}: {e}")
    
    def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update positions
                self.update_positions()
                
                # Process each symbol
                for symbol in self.config.symbols:
                    if symbol not in self.models:
                        continue
                    
                    # Check position limits
                    current_positions = len(self.positions.get(symbol, []))
                    total_positions = sum(len(positions) for positions in self.positions.values())
                    
                    if (current_positions >= self.config.max_positions_per_symbol or
                        total_positions >= self.config.max_total_positions):
                        continue
                    
                    # Get current data
                    ohlc = self.get_current_data(symbol)
                    if ohlc.empty:
                        continue
                    
                    # Generate signals
                    signal = self.generate_signals(symbol, ohlc)
                    
                    # Place order if signal is valid
                    if signal['signal'] != 0:
                        current_price = ohlc['close'].iloc[-1]
                        self.place_order(symbol, signal, current_price)
                
                # Update trading state in database
                self.update_trading_state()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.update_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def update_trading_state(self):
        """Update trading state in database"""
        try:
            for symbol in self.config.symbols:
                positions_data = []
                if symbol in self.positions:
                    for pos in self.positions[symbol]:
                        positions_data.append({
                            'ticket': pos.ticket,
                            'direction': pos.direction,
                            'entry_time': pos.entry_time.isoformat(),
                            'entry_price': pos.entry_price,
                            'size': pos.size,
                            'current_price': pos.current_price,
                            'unrealized_pnl': pos.unrealized_pnl
                        })
                
                # Update database
                with self.db.get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO live_trading_state 
                        (symbol, position_state, last_processed_bar, updated_at)
                        VALUES (?, ?, ?, ?)
                    """, (
                        symbol,
                        json.dumps(positions_data),
                        datetime.now(),
                        datetime.now()
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating trading state: {e}")
    
    def start_trading(self) -> bool:
        """Start live trading"""
        logger.info("Starting live trading system...")
        
        # Load models
        if not self.load_models():
            logger.error("Failed to load models")
            return False
        
        # Initialize positions tracking
        for symbol in self.config.symbols:
            self.positions[symbol] = []
        
        # Start trading loop in separate thread
        self.is_running = True
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()
        
        logger.info("Live trading system started")
        return True
    
    def stop_trading(self):
        """Stop live trading"""
        logger.info("Stopping live trading system...")
        
        self.is_running = False
        
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join(timeout=30)
        
        # Close MT5 connection
        self.data_ingestion.shutdown_mt5()
        
        logger.info("Live trading system stopped")
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        total_positions = sum(len(positions) for positions in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for positions in self.positions.values() for pos in positions)
        
        return {
            'is_running': self.is_running,
            'total_positions': total_positions,
            'positions_by_symbol': {symbol: len(positions) for symbol, positions in self.positions.items()},
            'total_unrealized_pnl': total_pnl,
            'loaded_models': list(self.models.keys()),
            'last_update': datetime.now().isoformat()
        }
