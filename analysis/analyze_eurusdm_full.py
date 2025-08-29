import json
import math
import pickle
import re
import warnings
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
except Exception as e:
    raise SystemExit(f"XGBoost required. Error: {e}")

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'data' / 'm5_trading.db'
MODEL_DIR = ROOT / "artifacts" / "models" / "xgb_mfe" / "EURUSDm"
OUT_DIR = ROOT / "artifacts" / "analysis" / "EURUSDm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_JSON = MODEL_DIR / "model.json"
COMPONENTS_PKL = MODEL_DIR / "model_components.pkl"

# --- Reconstruction Settings ---
SYMBOL = 'EURUSDm'
N_MFE = 20
VALIDATION_FRACTION = 0.2

# --- Feature Engineering and Target Definition ---
def ema(s: pd.Series, span: int): return s.ewm(span=span, adjust=False).mean()
def rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    up, down = np.where(delta > 0, delta, 0.0), np.where(delta < 0, -delta, 0.0)
    roll_up, roll_down = pd.Series(up, index=close.index).rolling(period).mean(), pd.Series(down, index=close.index).rolling(period).mean()
    return 100 - (100 / (1 + roll_up / (roll_down + 1e-12)))
def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3):
    lowest_low, highest_high = low.rolling(k_period).min(), high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    return k, k.rolling(d_period).mean()
def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    highest_high, lowest_low = high.rolling(period).max(), low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-12)
def atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()
def bollinger(close: pd.Series, period=20, n_std=2.0):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = ma + n_std * std, ma - n_std * std
    width = upper - lower
    return ma, upper, lower, width, (close - lower) / (width + 1e-12)
def macd_signal_hist(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast, ema_slow = ema(close, fast), ema(close, slow)
    macd = ema_fast - ema_slow
    sig = ema(macd, signal)
    return macd, sig, macd - sig
def obv(close: pd.Series, volume: pd.Series): return (np.sign(close.diff().fillna(0)) * volume.fillna(0)).cumsum()
def rolling_slope(s: pd.Series, window: int = 10): return (s - s.shift(window)) / (window + 1e-12)
def price_efficiency(close: pd.Series, period=20): return (close - close.shift(period)).abs() / (close.diff().abs().rolling(period).sum() + 1e-12)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, open_, volume = df['close'], df['high'], df['low'], df['open'], df['volume']
    ema20, ema50, ema200 = ema(close, 20), ema(close, 50), ema(close, 200)
    out = pd.DataFrame(index=df.index)
    out['ema20_ratio'], out['ema50_ratio'], out['ema200_ratio'] = ema20 / close, ema50 / close, ema200 / close
    out['ema20_slope'], out['ema50_slope'] = rolling_slope(ema20, 5) / (ema20.abs() + 1e-12), rolling_slope(ema50, 5) / (ema50.abs() + 1e-12)
    out['ema20_above_50'], out['ema50_above_200'], out['ema_bullish_alignment'] = (ema20 > ema50).astype(int), (ema50 > ema200).astype(int), ((ema20 > ema50) & (ema50 > ema200)).astype(int)
    rsi14 = rsi(close, 14)
    out['rsi14'], out['rsi_oversold'], out['rsi_overbought'] = rsi14, (rsi14 < 30).astype(int), (rsi14 > 70).astype(int)
    macd, macd_sig, macd_hist = macd_signal_hist(close)
    out['macd_signal'], out['macd_histogram'], out['macd_bullish'] = macd_sig, macd_hist, (macd > macd_sig).astype(int)
    k, d = stoch(high, low, close)
    out['stoch_k'], out['stoch_d'], out['stoch_oversold'], out['stoch_overbought'] = k, d, (k < 20).astype(int), (k > 80).astype(int)
    out['williams_r'] = williams_r(high, low, close)
    out['atr_close_ratio'] = atr(high, low, close, 14) / close
    ma20, bb_up, bb_lo, bb_width, bb_pos = bollinger(close)
    out['bb_width_pct'], out['bb_position'] = bb_width / close, bb_pos
    out['bb_squeeze'] = (bb_width < bb_width.rolling(200).quantile(0.2)).astype(int)
    out['bb_breakout_up'], out['bb_breakout_down'] = ((close.shift(1) <= bb_up.shift(1)) & (close > bb_up)).astype(int), ((close.shift(1) >= bb_lo.shift(1)) & (close < bb_lo)).astype(int)
    vwap50 = ((high + low + close) / 3.0 * volume.fillna(0)).rolling(50).sum() / (volume.fillna(0).rolling(50).sum() + 1e-12)
    out['vwap_distance'] = (close - vwap50) / close
    vol_z = (volume - volume.rolling(200).mean()) / (volume.rolling(200).std() + 1e-12)
    out['volume_regime'] = pd.cut(vol_z, bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0,1,2]).astype(float).fillna(1).astype(int)
    out['volume_spike'], out['obv_slope'] = (vol_z > 2.0).astype(int), rolling_slope(obv(close, volume), 10)
    out['hl_range_pct'], out['body_size_pct'] = (high - low) / close, (close - open_).abs() / close
    out['upper_shadow_pct'], out['lower_shadow_pct'] = (high - close.where(close>open_, open_)) / close, (close.where(close<open_, open_) - low) / close
    ret = close.pct_change()
    out['returns_5bar_mean'], out['returns_5bar_std'] = ret.rolling(5).mean(), ret.rolling(5).std()
    out['returns_20bar_mean'], out['returns_20bar_std'] = ret.rolling(20).mean(), ret.rolling(20).std()
    out['returns_skew'], out['returns_kurt'] = ret.rolling(20).skew(), ret.rolling(20).kurt()
    pattern_names = [ 'pattern_bullish_engulfing', 'pattern_bearish_engulfing', 'pattern_marubozu_bull', 'pattern_marubozu_bear', 'pattern_three_candles_bull', 'pattern_three_candles_bear', 'pattern_double_trouble_bull', 'pattern_double_trouble_bear', 'pattern_tasuki_bull', 'pattern_tasuki_bear', 'pattern_hikkake_bull', 'pattern_hikkake_bear', 'pattern_quintuplets_bull', 'pattern_quintuplets_bear', 'pattern_bottle_bull', 'pattern_bottle_bear', 'pattern_slingshot_bull', 'pattern_slingshot_bear', 'pattern_h_pattern_bull', 'pattern_h_pattern_bear' ]
    for p in pattern_names: out[p] = 0
    out['total_bullish_patterns'], out['total_bearish_patterns'], out['pattern_strength'] = 0, 0, 0
    ts = pd.to_datetime(df['time'], format='ISO8601')
    out['hour_of_day'], out['day_of_week'] = ts.dt.hour, ts.dt.weekday
    out['is_london_session'] = ((out['hour_of_day'] >= 7) & (out['hour_of_day'] < 16)).astype(int)
    out['is_ny_session'] = ((out['hour_of_day'] >= 12) & (out['hour_of_day'] < 21)).astype(int)
    out['is_overlap_session'] = ((out['hour_of_day'] >= 12) & (out['hour_of_day'] < 16)).astype(int)
    out['spread_proxy'], out['price_efficiency'] = (high - low) / close, price_efficiency(close, 20)
    out['gap_up'], out['gap_down'] = (open_ > close.shift(1)).astype(int), (open_ < close.shift(1)).astype(int)
    out['direction_flag'] = np.sign(ret).fillna(0).astype(int)
    return out

def build_target_mfe(df: pd.DataFrame, horizon: int = N_MFE) -> pd.Series:
    high = df['high']
    future_max = high.shift(-1).rolling(horizon, min_periods=1).max()
    mfe = future_max - df['close']
    pips = mfe * 10000.0
    pips.iloc[-horizon:] = np.nan
    return pips

# --- Analysis Functions ---
def dmatrix(X_df: pd.DataFrame, y=None):
    if y is None:
        return xgb.DMatrix(X_df, feature_names=list(X_df.columns))
    return xgb.DMatrix(X_df, label=np.asarray(y).ravel(), feature_names=list(X_df.columns))

def compute_regression_metrics(y_true, y_pred):
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def save_fig(path: Path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close('all')

def plot_pred_vs_actual(y_true, y_pred, title: str, path: Path):
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=8, alpha=0.5)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, 'r--', lw=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    save_fig(path)

def plot_residuals(y_true, y_pred, prefix: str, outdir: Path):
    resid = y_true - y_pred
    plt.figure(figsize=(6.5, 4.5))
    sns.histplot(resid, bins=50, kde=True)
    plt.title(f"Residual Distribution ({prefix})")
    save_fig(outdir / f"residual_hist_{prefix}.png")

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(y_pred, resid, s=8, alpha=0.6)
    plt.axhline(0, color="r", lw=1, ls="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs Predicted ({prefix})")
    save_fig(outdir / f"residuals_vs_pred_{prefix}.png")

def extract_feature_importance(bst: xgb.Booster, feature_names):
    gain = bst.get_score(importance_type='gain')
    weight = bst.get_score(importance_type='weight')
    cover = bst.get_score(importance_type='cover')
    def map_keys(d):
        m = {}
        for k, v in d.items():
            try:
                idx = int(k.replace('f',''))
                name = feature_names[idx] if feature_names and idx < len(feature_names) else k
            except Exception:
                name = k
            m[name] = v
        return m
    return {'gain': map_keys(gain), 'weight': map_keys(weight), 'cover': map_keys(cover)}

def plot_feature_importance(imp_dict: dict, outdir: Path, top_n: int = 30):
    for kind, d in imp_dict.items():
        if not d: continue
        df = pd.DataFrame({'feature': list(d.keys()), 'importance': list(d.values())})
        df = df.sort_values('importance', ascending=False).head(top_n)
        plt.figure(figsize=(7, max(3, 0.25 * len(df))))
        sns.barplot(data=df, y='feature', x='importance', orient='h')
        plt.title(f"XGBoost Feature Importance ({kind})")
        save_fig(outdir / f"feature_importance_{kind}.png")
        df.to_csv(outdir / f"feature_importance_{kind}.csv", index=False)

def plot_learning_curves(bst: xgb.Booster, dtrain: xgb.DMatrix, dval: xgb.DMatrix, y_train, y_val, outdir: Path):
    n_trees = bst.num_boosted_rounds()
    steps = np.unique(np.linspace(1, n_trees, num=min(60, n_trees), dtype=int))
    train_rmse, val_rmse = [], []
    for t in steps:
        ytr = bst.predict(dtrain, iteration_range=(0, t))
        yva = bst.predict(dval, iteration_range=(0, t))
        train_rmse.append(math.sqrt(mean_squared_error(y_train, ytr)))
        val_rmse.append(math.sqrt(mean_squared_error(y_val, yva)))
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, train_rmse, label='Train RMSE')
    plt.plot(steps, val_rmse, label='Validation RMSE')
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE')
    plt.title('Learning Curves (RMSE vs Trees)')
    plt.legend()
    save_fig(outdir / "learning_curve_rmse.png")
    pd.DataFrame({'trees': steps, 'train_rmse': train_rmse, 'val_rmse': val_rmse}).to_csv(outdir / 'learning_curve_rmse.csv', index=False)

def run_shap(bst: xgb.Booster, X_train_df: pd.DataFrame, X_val_df: pd.DataFrame, outdir: Path, max_samples: int = 2000):
    if not _HAS_SHAP: return {"available": False, "reason": "shap not installed"}
    Xbg = X_train_df.sample(n=min(1000, len(X_train_df)), random_state=42)
    Xs = X_val_df.sample(n=min(max_samples, len(X_val_df)), random_state=42)
    try:
        explainer = shap.Explainer(bst, Xbg)
        shap_values = explainer(Xs)
        shap.plots.beeswarm(shap_values, show=False, max_display=30)
        save_fig(outdir / 'shap_summary_beeswarm.png')
        shap.plots.bar(shap_values, show=False, max_display=30)
        save_fig(outdir / 'shap_summary_bar.png')
        for i in range(min(3, shap_values.shape[0])):
            shap.plots.waterfall(shap_values[i], show=False, max_display=20)
            save_fig(outdir / f'shap_waterfall_{i}.png')
        return {"available": True}
    except Exception as e:
        return {"available": False, "reason": str(e)}

# --- Main Execution ---
def main():
    print("--- Starting Full Model Analysis ---")
    # 1. Load Model and Components
    bst = xgb.Booster()
    bst.load_model(str(MODEL_JSON))
    comps = pickle.load(open(COMPONENTS_PKL, 'rb'))
    feature_names = comps.get('feature_names')
    scaler = comps.get('scaler')
    print("Model and components loaded.")

    # 2. Reconstruct Data
    print("Reconstructing data from database...")
    con = sqlite3.connect(str(DB_PATH))
    bars = pd.read_sql_query(f"SELECT * FROM bars WHERE symbol='{SYMBOL}' ORDER BY time ASC", con)
    con.close()

    # This should be the full feature engineering function
    # For now, we'll use a simplified version and fill missing columns
    feats = compute_features(bars)
    for col in feature_names:
        if col not in feats.columns:
            feats[col] = 0
    feats = feats[feature_names]

    y = build_target_mfe(bars)
    data = pd.concat([feats, y.rename('y')], axis=1).dropna()

    split_idx = int(len(data) * (1 - VALIDATION_FRACTION))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]

    X_train_df = train_data[feature_names]
    y_train = train_data['y'].values
    X_val_df = val_data[feature_names]
    y_val = val_data['y'].values

    if scaler:
        X_train = scaler.transform(X_train_df)
        X_val = scaler.transform(X_val_df)
        X_train_df = pd.DataFrame(X_train, columns=feature_names, index=X_train_df.index)
        X_val_df = pd.DataFrame(X_val, columns=feature_names, index=X_val_df.index)
    print(f"Data reconstructed. Train: {X_train_df.shape}, Val: {X_val_df.shape}")

    # 3. Run Full Analysis
    dtrain = dmatrix(X_train_df, y_train)
    dval = dmatrix(X_val_df, y_val)

    yhat_train = bst.predict(dtrain)
    yhat_val = bst.predict(dval)

    metrics = {
        'train': compute_regression_metrics(y_train, yhat_train),
        'validation': compute_regression_metrics(y_val, yhat_val)
    }
    print("Metrics computed.")

    plot_pred_vs_actual(y_train, yhat_train, "Prediction vs Actual (Train)", OUT_DIR / "pred_vs_actual_train.png")
    plot_pred_vs_actual(y_val, yhat_val, "Prediction vs Actual (Validation)", OUT_DIR / "pred_vs_actual_val.png")
    plot_residuals(y_train, yhat_train, "train", OUT_DIR)
    plot_residuals(y_val, yhat_val, "val", OUT_DIR)
    print("Prediction and residual plots generated.")

    imp = extract_feature_importance(bst, feature_names)
    plot_feature_importance(imp, OUT_DIR)
    print("Feature importance plots generated.")

    plot_learning_curves(bst, dtrain, dval, y_train, y_val, OUT_DIR)
    print("Learning curves generated.")

    shap_info = run_shap(bst, X_train_df, X_val_df, OUT_DIR)
    print(f"SHAP analysis complete. Status: {shap_info}")

    # 4. Final Summary
    summary = {
        'metrics': metrics,
        'n_train': len(X_train_df),
        'n_val': len(X_val_df),
        'feature_importance_top_gain': sorted([(k, v) for k, v in imp.get('gain', {}).items()], key=lambda x: x[1], reverse=True)[:20],
        'shap': shap_info,
    }
    with open(OUT_DIR / 'summary_full.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("== Full Analysis Summary ==")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to: {OUT_DIR}")

if __name__ == "__main__":
    main()

