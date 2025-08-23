"""
Main orchestrator for M5 Multi-Symbol Trend Bot
Provides unified interface for training, backtesting, and live trading
"""

import argparse
import sys
from pathlib import Path
# Add project root to sys.path to allow absolute imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import List, Dict, Optional


from src.features.feature_pipeline import FeaturePipeline
from src.deployment.training_pipeline import TrainingPipeline
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.backtesting.reporting import BacktestReporter


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class M5TradingBot:
    """Main M5 Multi-Symbol Trend Bot orchestrator"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

        # Initialize components
        self.data_ingestion = None  # Lazily initialized
        self.feature_pipeline = FeaturePipeline()
        self.training_pipeline = TrainingPipeline()
        self.backtest_engine = BacktestEngine()
        self.reporter = BacktestReporter()

    def load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
                "data": {
                    "start_date": "2023-01-01",
                    "end_date": "2024-01-01"
                },
                "training": {
                    "rr_presets": ["1:2", "1:3", "1:4"],
                    "lstm": {
                        "sequence_length": 60,
                        "hidden_units": 128,
                        "n_layers": 2,
                        "dropout_rate": 0.2,
                        "learning_rate": 0.001,
                        "batch_size": 128,
                        "epochs": 100
                    },
                    "xgb": {
                        "optimize_hyperparams": True,
                        "use_calibration": True
                    }
                },
                "backtesting": {
                    "initial_capital": 10000.0,
                    "risk_per_trade": 0.05,
                    "min_rr_filter": 1.5,
                    "lstm_threshold": 0.55,
                    "xgb_threshold": 0.6
                },
                "live_trading": {
                    "risk_per_trade": 0.05,
                    "max_positions_per_symbol": 1,
                    "max_total_positions": 5,
                    "update_interval": 300
                },
                "mt5": {
                    "login": None,
                    "password": None,
                    "server": None
                }
            }

            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

            return default_config

    def ingest_data(self, symbols: Optional[List[str]] = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Ingest historical data for symbols"""
        logger.info("Starting data ingestion...")

        if self.data_ingestion is None:
            from data.ingestion import DataIngestion
            self.data_ingestion = DataIngestion()

        symbols = symbols or self.config["symbols"]
        start_date = start_date or self.config["data"]["start_date"]
        end_date = end_date or self.config["data"]["end_date"]

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Initialize MT5 if credentials provided
        mt5_config = self.config.get("mt5", {})
        if all(mt5_config.get(key) for key in ["login", "password", "server"]):
            if not self.data_ingestion.initialize_mt5(
                mt5_config["login"], mt5_config["password"], mt5_config["server"]
            ):
                logger.error("Failed to initialize MT5")
                return False

        # Ingest data for all symbols
        results = self.data_ingestion.ingest_multi_symbol_data(symbols, start_dt, end_dt)

        successful = sum(results.values())
        logger.info(f"Data ingestion completed: {successful}/{len(symbols)} symbols successful")

        return successful > 0



    def train_models(self, symbols: Optional[List[str]] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None,
                    rr_preset: str = "1:2") -> Dict:
        """Train LSTM and XGB models for symbols"""
        logger.info("Starting model training...")

        symbols = symbols or self.config["symbols"]
        start_date = start_date or self.config["data"]["start_date"]
        end_date = end_date or self.config["data"]["end_date"]

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        results = self.training_pipeline.train_multi_symbol_models(
            symbols, start_dt, end_dt, rr_preset
        )

        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        logger.info(f"Model training completed: {successful}/{len(symbols)} symbols successful")

        return results

    def run_backtest(self, symbol: str, start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """Run backtest for a symbol"""
        logger.info(f"Starting backtest for {symbol}")

        start_date = start_date or self.config["data"]["start_date"]
        end_date = end_date or self.config["data"]["end_date"]

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Create backtest config
        bt_config = self.config["backtesting"]
        config = BacktestConfig(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=bt_config["initial_capital"],
            risk_per_trade=bt_config["risk_per_trade"],
            min_rr_filter=bt_config["min_rr_filter"],
            lstm_threshold=bt_config["lstm_threshold"],
            xgb_threshold=bt_config["xgb_threshold"]
        )

        # Load models
        lstm_path = f"artifacts/models/lstm/{symbol}/model.h5"
        xgb_path = f"artifacts/models/xgb/{symbol}/model.json"

        self.backtest_engine.load_models(lstm_path, xgb_path)

        # Run backtest
        results = self.backtest_engine.run_backtest(symbol, config)

        if results:
            # Generate reports
            reports = self.reporter.generate_complete_report(results)
            results['reports'] = reports

            logger.info(f"Backtest completed for {symbol}")

        return results

    def run_multi_symbol_backtest(self, symbols: Optional[List[str]] = None) -> Dict:
        """Run backtests for multiple symbols"""
        symbols = symbols or self.config["symbols"]
        results = {}

        for symbol in symbols:
            try:
                symbol_results = self.run_backtest(symbol)
                results[symbol] = symbol_results
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")
                results[symbol] = {"error": str(e)}

        return results

    def start_live_trading(self) -> bool:
        """Start live trading"""
        logger.info("Starting live trading...")
        from src.deployment.live_trading import LiveTradingEngine, TradingConfig

        # Create trading config
        lt_config = self.config["live_trading"]
        trading_config = TradingConfig(
            symbols=self.config["symbols"],
            risk_per_trade=lt_config["risk_per_trade"],
            max_positions_per_symbol=lt_config["max_positions_per_symbol"],
            max_total_positions=lt_config["max_total_positions"],
            update_interval=lt_config["update_interval"]
        )

        # Initialize live trading engine
        live_engine = LiveTradingEngine(trading_config)

        # Initialize MT5
        mt5_config = self.config["mt5"]
        if not live_engine.initialize_mt5(
            mt5_config["login"], mt5_config["password"], mt5_config["server"]
        ):
            logger.error("Failed to initialize MT5 for live trading")
            return False

        # Start trading
        return live_engine.start_trading()

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "data_summary": {},
            "model_status": {},
            "backtest_results": {},
            "live_trading_status": {}
        }

        # Data summary
        try:
            if self.data_ingestion is None:
                from data.ingestion import DataIngestion
                self.data_ingestion = DataIngestion()
            status["data_summary"] = self.data_ingestion.get_data_summary(self.config["symbols"])
        except Exception as e:
            status["data_summary"] = {"error": str(e)}

        # Model validation
        for symbol in self.config["symbols"]:
            try:
                status["model_status"][symbol] = self.training_pipeline.validate_trained_models(symbol)
            except Exception as e:
                status["model_status"][symbol] = {"error": str(e)}

        return status

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="M5 Multi-Symbol Trend Bot")
    parser.add_argument("command", choices=[
        "ingest", "train", "backtest", "live", "status"
    ], help="Command to execute")
    parser.add_argument("--symbols", nargs="+", help="Symbols to process")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config.json", help="Config file path")

    args = parser.parse_args()

    # Initialize bot
    bot = M5TradingBot(args.config)

    try:
        if args.command == "ingest":
            success = bot.ingest_data(args.symbols, args.start_date, args.end_date)
            print(f"Data ingestion {'successful' if success else 'failed'}")



        elif args.command == "train":
            results = bot.train_models(args.symbols, args.start_date, args.end_date)
            successful = sum(1 for r in results.values() if r.get('status') == 'success')
            print(f"Model training: {successful}/{len(results)} symbols successful")

        elif args.command == "backtest":
            if args.symbols and len(args.symbols) == 1:
                results = bot.run_backtest(args.symbols[0], args.start_date, args.end_date)
                if results and 'metrics' in results:
                    metrics = results['metrics']
                    print(f"Backtest Results for {args.symbols[0]}:")
                    print(f"  Total Return: {metrics.get('return_pct', 0):.2f}%")
                    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                    print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            else:
                results = bot.run_multi_symbol_backtest(args.symbols)
                print(f"Multi-symbol backtest completed for {len(results)} symbols")

        elif args.command == "live":
            success = bot.start_live_trading()
            if success:
                print("Live trading started successfully")
                input("Press Enter to stop trading...")
            else:
                print("Failed to start live trading")

        elif args.command == "status":
            status = bot.get_system_status()
            print(json.dumps(status, indent=2, default=str))

    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
