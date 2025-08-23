"""
Database management for M5 Multi-Symbol Trend Bot
Handles SQLite operations for OHLCV data, features, labels, and model artifacts
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Main database interface for the trading system"""

    def __init__(self, db_path: str = "data/trading_system.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            # OHLCV data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    tick_volume INTEGER,
                    spread INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)

            # Features table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    feature_set TEXT NOT NULL,  -- 'lstm' or 'xgb'
                    features TEXT NOT NULL,     -- JSON serialized features
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, feature_set)
                )
            """)

            # Labels table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    label_type TEXT NOT NULL,   -- 'y_base' or 'y_meta'
                    rr_preset TEXT NOT NULL,    -- '1:2', '1:3', '1:4'
                    label_value INTEGER NOT NULL,
                    barrier_meta TEXT,          -- JSON with barrier info
                    sample_weight REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, label_type, rr_preset)
                )
            """)

            # Model artifacts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,   -- 'lstm' or 'xgb'
                    symbol TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    metrics TEXT,               -- JSON serialized metrics
                    hyperparameters TEXT,       -- JSON serialized hyperparams
                    training_period_start DATETIME,
                    training_period_end DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Backtest results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_config TEXT NOT NULL,  -- JSON config
                    metrics TEXT NOT NULL,          -- JSON metrics
                    trades TEXT,                    -- JSON trades data
                    equity_curve TEXT,              -- JSON equity curve
                    report_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Live trading state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_trading_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    position_state TEXT,        -- JSON position info
                    last_signal_time DATETIME,
                    last_processed_bar DATETIME,
                    model_version TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv_data(symbol, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp ON features(symbol, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_symbol_timestamp ON labels(symbol, timestamp)")

            conn.commit()
            logger.info("Database initialized successfully")

    def insert_ohlcv_batch(self, symbol: str, df: pd.DataFrame) -> int:
        """Insert OHLCV data in batch"""
        with self.get_connection() as conn:
            # Prepare data
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            df_copy = df_copy.reset_index()  # Ensure timestamp is a column

            # Insert data
            inserted = 0
            for _, row in df_copy.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO ohlcv_data
                        (symbol, timestamp, open, high, low, close, volume, tick_volume, spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'], row.name if hasattr(row, 'name') else row['timestamp'],
                        row['open'], row['high'], row['low'], row['close'], row['volume'],
                        row.get('tick_volume'), row.get('spread')
                    ))
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert row for {symbol}: {e}")

            conn.commit()
            return inserted

    def get_ohlcv_data(self, symbol: str, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve OHLCV data for a symbol"""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            if not df.empty:
                df.set_index('timestamp', inplace=True)

        return df

    def insert_features(self, symbol: str, timestamp: datetime, feature_set: str, features: Dict) -> bool:
        """Insert feature vector"""
        with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO features
                    (symbol, timestamp, feature_set, features)
                    VALUES (?, ?, ?, ?)
                """, (symbol, timestamp, feature_set, json.dumps(features)))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert features: {e}")
                return False

    def get_features(self, symbol: str, feature_set: str, start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve features for a symbol"""
        query = "SELECT * FROM features WHERE symbol = ? AND feature_set = ?"
        params = [symbol, feature_set]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            if not df.empty:
                # Parse JSON features
                df['features_parsed'] = df['features'].apply(json.loads)
                df.set_index('timestamp', inplace=True)

        return df

    def insert_labels(self, symbol: str, timestamp: datetime, label_type: str,
                      rr_preset: str, label_value: int, barrier_meta: Dict = None,
                      sample_weight: float = None) -> bool:
        """Insert label data"""
        with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO labels
                    (symbol, timestamp, label_type, rr_preset, label_value, barrier_meta, sample_weight)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, timestamp, label_type, rr_preset, label_value,
                      json.dumps(barrier_meta) if barrier_meta else None, sample_weight))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert labels: {e}")
                return False

    def get_labels(self, symbol: str, label_type: str, rr_preset: str,
                   start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve labels for a symbol"""
        query = "SELECT * FROM labels WHERE symbol = ? AND label_type = ? AND rr_preset = ?"
        params = [symbol, label_type, rr_preset]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            if not df.empty:
                df.set_index('timestamp', inplace=True)

        return df

    def save_model_artifact(self, model_type: str, symbol: str, model_version: str,
                           artifact_path: str, metrics: Dict, hyperparameters: Dict,
                           training_period: Tuple[datetime, datetime]) -> bool:
        """Save model artifact metadata"""
        with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO model_artifacts
                    (model_type, symbol, model_version, artifact_path, metrics,
                     hyperparameters, training_period_start, training_period_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (model_type, symbol, model_version, artifact_path,
                      json.dumps(metrics), json.dumps(hyperparameters),
                      training_period[0], training_period[1]))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save model artifact: {e}")
                return False

    def get_latest_model_artifact(self, model_type: str, symbol: str) -> Optional[Dict]:
        """Get latest model artifact for symbol"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM model_artifacts
                WHERE model_type = ? AND symbol = ?
                ORDER BY created_at DESC LIMIT 1
            """, (model_type, symbol))

            row = cursor.fetchone()
            if row:
                return dict(row)
        return None



class M5Database:
    """Database interface for legacy m5_trading.db schema
    Schema:
      - bars(symbol TEXT, time DATETIME, open, high, low, close, volume)
      - features(symbol TEXT, time DATETIME, feat TEXT JSON)
      - labels(symbol TEXT, time DATETIME, y INTEGER, meta TEXT JSON)
      - ticks, models, extremes, trades (optional for this interface)
    """

    def __init__(self, db_path: str = "data/m5_trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Do not auto-migrate existing DBs; only ensure tables if new
        self._ensure_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self.get_connection() as conn:
            # Create minimal required tables if they don't exist
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT NOT NULL,
                    time DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(symbol, time)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_time ON bars(symbol, time)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS features (
                  symbol TEXT NOT NULL,
                  time   DATETIME NOT NULL,
                  feat   TEXT NOT NULL,
                  PRIMARY KEY(symbol, time)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                  symbol TEXT NOT NULL,
                  time   DATETIME NOT NULL,
                  y      INTEGER NOT NULL,
                  meta   TEXT,
                  PRIMARY KEY(symbol, time)
                )
                """
            )
            conn.commit()

    # ---------------------------- Bars ----------------------------
    def insert_bars_batch(self, symbol: str, df: pd.DataFrame) -> int:
        """Insert OHLCV rows into bars. Expects df indexed by timestamp with columns open,high,low,close,volume"""
        if df.empty:
            return 0
        df_copy = df.copy()
        df_copy = df_copy.reset_index()
        # Ensure column name is 'time'
        if 'index' in df_copy.columns:
            df_copy.rename(columns={'index': 'time'}, inplace=True)
        elif 'timestamp' in df_copy.columns:
            df_copy.rename(columns={'timestamp': 'time'}, inplace=True)
        elif 'time' not in df_copy.columns:
            df_copy.rename(columns={df_copy.columns[0]: 'time'}, inplace=True)
        inserted = 0
        with self.get_connection() as conn:
            cur = conn.cursor()
            for _, row in df_copy.iterrows():
                try:
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO bars (symbol, time, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            row['time'],
                            float(row['open']),
                            float(row['high']),
                            float(row['low']),
                            float(row['close']),
                            float(row.get('volume', 0.0)),
                        ),
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert bar for {symbol}: {e}")
            conn.commit()
        return inserted

    def get_ohlcv_data(self, symbol: str, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve OHLCV data from bars table and return with index 'timestamp'"""
        query = "SELECT * FROM bars WHERE symbol = ?"
        params: List[Any] = [symbol]
        if start_date:
            query += " AND time >= ?"; params.append(start_date)
        if end_date:
            query += " AND time <= ?"; params.append(end_date)
        query += " ORDER BY time"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['time'])
        if df.empty:
            return df
        df.rename(columns={'time': 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    # ---------------------------- Features ----------------------------
    def upsert_feature_subset(self, symbol: str, timestamp: datetime, feature_set: str, features: Dict) -> bool:
        """Upsert a subset ('lstm' or 'xgb') into the combined features.feat JSON"""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT feat FROM features WHERE symbol = ? AND time = ?", (symbol, timestamp))
                row = cur.fetchone()
                base = {'lstm': {}, 'xgb': {}}
                if row and row['feat']:
                    try:
                        base = json.loads(row['feat'])
                    except Exception:
                        base = {'lstm': {}, 'xgb': {}}
                base[feature_set] = features
                cur.execute(
                    "INSERT OR REPLACE INTO features (symbol, time, feat) VALUES (?, ?, ?)",
                    (symbol, timestamp, json.dumps(base))
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to upsert features for {symbol} @ {timestamp}: {e}")
            return False

    def get_features_subset(self, symbol: str, feature_set: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Return a DataFrame of the requested subset from combined features JSON"""
        query = "SELECT symbol, time, feat FROM features WHERE symbol = ?"
        params: List[Any] = [symbol]
        if start_date:
            query += " AND time >= ?"; params.append(start_date)
        if end_date:
            query += " AND time <= ?"; params.append(end_date)

        if limit:
            query = f"SELECT * FROM ({query} ORDER BY time DESC LIMIT {limit}) ORDER BY time ASC"
        else:
            query += " ORDER BY time"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['time'])
        if df.empty:
            return pd.DataFrame()
        # Parse JSON and expand
        records = []
        for _, row in df.iterrows():
            try:
                feat = json.loads(row['feat'])
                subset = feat.get(feature_set, {})
            except Exception:
                subset = {}
            rec = {'timestamp': row['time'], **subset}
            records.append(rec)
        out = pd.DataFrame.from_records(records)
        if out.empty:
            return out
        out.set_index('timestamp', inplace=True)
        return out

    def _ensure_schema(self):
        """Ensure the legacy database schema exists."""
        with self.get_connection() as conn:
            # Note: This is a simplified schema for compatibility.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    type TEXT,
                    params TEXT,
                    metrics TEXT,
                    created_at DATETIME NOT NULL
                )
            """)

    # ---------------------------- Labels ----------------------------
    def upsert_labels_meta(self, symbol: str, timestamp: datetime, y: int, meta: Dict) -> bool:
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO labels (symbol, time, y, meta) VALUES (?, ?, ?, ?)",
                    (symbol, timestamp, y, json.dumps(meta) if meta is not None else None)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to upsert labels for {symbol} @ {timestamp}: {e}")
            return False

    def get_labels_meta(self, symbol: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        query = "SELECT symbol, time, y, meta FROM labels WHERE symbol = ?"
        params: List[Any] = [symbol]
        if start_date:
            query += " AND time >= ?"; params.append(start_date)
        if end_date:
            query += " AND time <= ?"; params.append(end_date)

        if limit:
            query = f"SELECT * FROM ({query} ORDER BY time DESC LIMIT {limit}) ORDER BY time ASC"
        else:
            query += " ORDER BY time"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['time'])
        if df.empty:
            return pd.DataFrame()
        # Parse meta JSON
        df.rename(columns={'time': 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)
        try:
            df['meta_parsed'] = df['meta'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
        except Exception:
            df['meta_parsed'] = [{} for _ in range(len(df))]
        return df

    # ---------------------------- Models & Backtests (for compatibility) ----------------------------
    def save_model_artifact(self, model_type: str, symbol: str, model_version: str,
                           artifact_path: str, metrics: Dict, hyperparameters: Dict,
                           training_period: Tuple[datetime, datetime]) -> bool:
        """Saves model artifact metadata to the `models` table for compatibility."""
        try:
            model_id = f"{model_type}_{symbol}_{model_version}"
            with self.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO models (model_id, type, params, metrics, created_at) VALUES (?, ?, ?, ?, ?)",
                    (model_id, model_type, json.dumps(hyperparameters), json.dumps(metrics), datetime.now())
                )
                conn.commit()
                logger.info(f"Saved model artifact for {model_id} to legacy `models` table.")
                return True
        except Exception as e:
            logger.error(f"Failed to save model artifact {model_id}: {e}")
            return False

    def insert_backtest_result(self, backtest_id: str, symbol: str, strategy_config: str,
                               metrics: str, trades: str, equity_curve: str, report_path: str) -> bool:
        """Saves backtest results to the `trades` table for compatibility."""
        # The legacy schema doesn't have a dedicated backtest table.
        # We can log the main result as a special trade record if needed, but for now, we just log it.
        logger.info(f"Backtest result for {backtest_id} (not saved to legacy DB). Metrics: {metrics}")
        return True

