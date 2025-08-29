import os
import json
import math
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
except Exception as e:
    raise SystemExit(f"XGBoost is required to run this analysis. Error importing xgboost: {e}")

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models" / "xgb_mfe" / "EURUSDm"
OUT_DIR = ROOT / "artifacts" / "analysis" / "EURUSDm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_JSON = MODEL_DIR / "model.json"
COMPONENTS_PKL = MODEL_DIR / "model_components.pkl"
TRAINING_RESULTS_JSON = MODEL_DIR / "training_results.json"


def load_components(path: Path):
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # Expect dict-like
    if isinstance(obj, dict):
        return obj
    # Fall back: try joblib-like structures
    try:
        return dict(obj)
    except Exception:
        return {}


def to_dataframe(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        return X
    if X is None:
        return None
    if feature_names is None:
        if isinstance(X, np.ndarray):
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        else:
            X = np.asarray(X)
            feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names)


def dmatrix(X_df: pd.DataFrame, y=None):
    if y is None:
        return xgb.DMatrix(X_df, feature_names=list(X_df.columns))
    return xgb.DMatrix(X_df, label=np.asarray(y).ravel(), feature_names=list(X_df.columns))


def compute_regression_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_fig(path: Path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pred_vs_actual(y_true, y_pred, title: str, path: Path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(6.5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, 'r--', lw=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    save_fig(path)


def plot_residuals(y_true, y_pred, prefix: str, outdir: Path):
    resid = y_true - y_pred
    sns.set(style="whitegrid")
    # Residual histogram
    plt.figure(figsize=(6.5, 4.5))
    sns.histplot(resid, bins=50, kde=True, color="tab:blue")
    plt.title(f"Residual Distribution ({prefix})")
    plt.xlabel("Residual (y - y_hat)")
    plt.ylabel("Count")
    save_fig(outdir / f"residual_hist_{prefix}.png")

    # Residuals vs fitted
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(y_pred, resid, s=8, alpha=0.6)
    plt.axhline(0, color="r", lw=1, ls="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs Predicted ({prefix})")
    save_fig(outdir / f"residuals_vs_pred_{prefix}.png")


def extract_feature_importance(bst: xgb.Booster, feature_names):
    # Use gain importance by default
    gain = bst.get_score(importance_type='gain')
    weight = bst.get_score(importance_type='weight')
    cover = bst.get_score(importance_type='cover')

    def map_keys(d):
        m = {}
        for k, v in d.items():
            # keys like 'f12' -> 12
            try:
                idx = int(k.replace('f',''))
                name = feature_names[idx] if feature_names and idx < len(feature_names) else k
            except Exception:
                name = k
            m[name] = v
        return m

    return {
        'gain': map_keys(gain),
        'weight': map_keys(weight),
        'cover': map_keys(cover)
    }


def plot_feature_importance(imp_dict: dict, outdir: Path, top_n: int = 30):
    for kind, d in imp_dict.items():
        if not d:
            continue
        df = pd.DataFrame({'feature': list(d.keys()), 'importance': list(d.values())})
        df = df.sort_values('importance', ascending=False).head(top_n)
        plt.figure(figsize=(7, max(3, 0.25 * len(df))))
        sns.barplot(data=df, y='feature', x='importance', orient='h', color='tab:blue')
        plt.title(f"XGBoost Feature Importance ({kind})")
        plt.xlabel(kind)
        plt.ylabel("")
        save_fig(outdir / f"feature_importance_{kind}.png")
        df.to_csv(outdir / f"feature_importance_{kind}.csv", index=False)


def plot_learning_curves(bst: xgb.Booster, dtrain: xgb.DMatrix, dval: xgb.DMatrix, y_train, y_val, outdir: Path):
    # Determine number of trees
    try:
        n_trees = bst.num_boosted_rounds()
    except Exception:
        # Fallback: use attribute or parse from JSON path
        n_trees = 0
        try:
            attrs = bst.attributes()
            if 'best_iteration' in attrs:
                n_trees = int(attrs['best_iteration']) + 1
        except Exception:
            pass
        if not n_trees:
            # Conservative default
            n_trees = 200
    # Sample points to limit computation
    steps = np.unique(np.linspace(1, n_trees, num=min(60, n_trees), dtype=int))
    train_rmse, val_rmse = [], []
    for t in steps:
        try:
            ytr = bst.predict(dtrain, iteration_range=(0, t))
            yva = bst.predict(dval, iteration_range=(0, t))
        except Exception:
            ytr = bst.predict(dtrain, ntree_limit=t)
            yva = bst.predict(dval, ntree_limit=t)
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

    # Save raw LC data
    lc_df = pd.DataFrame({'trees': steps, 'train_rmse': train_rmse, 'val_rmse': val_rmse})
    lc_df.to_csv(outdir / 'learning_curve_rmse.csv', index=False)


def run_shap(bst: xgb.Booster, X_train_df: pd.DataFrame, X_val_df: pd.DataFrame, outdir: Path, max_samples: int = 2000):
    if not _HAS_SHAP:
        return {"available": False, "reason": "shap not installed"}
    # Sample for speed
    Xbg = X_train_df.sample(n=min(1000, len(X_train_df)), random_state=42)
    Xs = X_val_df.sample(n=min(max_samples, len(X_val_df)), random_state=42)
    try:
        explainer = shap.Explainer(bst, Xbg)
        shap_values = explainer(Xs)
        # Summary plot (beeswarm)
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False, max_display=30)
        save_fig(outdir / 'shap_summary_beeswarm.png')
        # Bar summary
        plt.figure()
        shap.plots.bar(shap_values, show=False, max_display=30)
        save_fig(outdir / 'shap_summary_bar.png')
        # Waterfall for a few top errors: we'll just use first 3 samples here
        for i in range(min(3, shap_values.shape[0])):
            try:
                plt.figure()
                shap.plots.waterfall(shap_values[i], show=False, max_display=20)
                save_fig(outdir / f'shap_waterfall_{i}.png')
            except Exception:
                pass
        # Dependence plots for top features by mean |shap|
        vals = np.abs(shap_values.values).mean(0)
        order = np.argsort(vals)[::-1][:5]
        for idx in order:
            feat = shap_values.feature_names[idx]
            plt.figure()
            shap.plots.scatter(shap_values[:, idx], color=shap_values, show=False)
            plt.title(f"SHAP dependence: {feat}")
            save_fig(outdir / f'shap_dependence_{feat}.png')
        return {"available": True}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def parse_training_results(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(encoding='utf-8', errors='ignore')
    try:
        return json.loads(txt)
    except Exception:
        # Try to regex extract key metrics from possibly truncated JSON
        metrics = {}
        patterns = {
            'cv_rmse_mean': r'"cv_rmse_mean"\s*:\s*([0-9eE+\-.]+)',
            'cv_rmse_std': r'"cv_rmse_std"\s*:\s*([0-9eE+\-.]+)',
            'cv_r2_mean': r'"cv_r2_mean"\s*:\s*([0-9eE+\-.]+)',
            'cv_r2_std': r'"cv_r2_std"\s*:\s*([0-9eE+\-.]+)',
            'cv_mae_mean': r'"cv_mae_mean"\s*:\s*([0-9eE+\-.]+)',
            'cv_mae_std': r'"cv_mae_std"\s*:\s*([0-9eE+\-.]+)',
            'final_rmse': r'"final_rmse"\s*:\s*([0-9eE+\-.]+)',
            'final_r2': r'"final_r2"\s*:\s*([0-9eE+\-.]+)',
            'final_mae': r'"final_mae"\s*:\s*([0-9eE+\-.]+)'
        }
        for k, pat in patterns.items():
            m = re.search(pat, txt)
            if m:
                try:
                    metrics[k] = float(m.group(1))
                except Exception:
                    pass
        return metrics if metrics else None


def main():
    print("--- Starting Model Analysis ---")
    # Load model
    if not MODEL_JSON.exists():
        raise FileNotFoundError(f"Missing model.json at {MODEL_JSON}")
    bst = xgb.Booster()
    bst.load_model(str(MODEL_JSON))
    print("Model loaded successfully.")

    # Load components and extract data
    comps = load_components(COMPONENTS_PKL)

    # Load data: first try reconstructed splits, then fall back to components
    X_train, y_train, X_val, y_val = None, None, None, None
    split_path = ROOT / 'artifacts' / 'data' / 'splits' / 'EURUSDm' / 'splits.pkl'
    if split_path.exists():
        print(f"Loading reconstructed splits from {split_path}")
        with open(split_path, 'rb') as f:
            splits = pickle.load(f)
        X_train, y_train, X_val, y_val = splits.get('X_train'), splits.get('y_train'), splits.get('X_val'), splits.get('y_val')

    if X_train is None:
        print("Splits not found or empty, falling back to model_components.pkl")
        X_train = comps.get('X_train') if isinstance(comps, dict) else None
        y_train = comps.get('y_train') if isinstance(comps, dict) else None
        X_val = comps.get('X_val') if isinstance(comps, dict) else None
        y_val = comps.get('y_val') if isinstance(comps, dict) else None

    feature_names = (comps.get('feature_names') if isinstance(comps, dict) else None) or comps.get('features') if isinstance(comps, dict) else None

    # Convert to DataFrame if data available
    X_train_df = to_dataframe(X_train, feature_names) if X_train is not None else None
    X_val_df = to_dataframe(X_val, feature_names) if X_val is not None else None
    if feature_names is None and X_train_df is not None:
        feature_names = list(X_train_df.columns)

    metrics = {}
    created_plots = []

    print("Generating feature importance...")
    if feature_names is None:
        score = bst.get_score(importance_type='gain')
        if score:
            max_idx = max(int(k.replace('f','')) for k in score.keys() if k.startswith('f'))
            feature_names = [f'f{i}' for i in range(max_idx+1)]
        else:
            feature_names = []
    imp = extract_feature_importance(bst, feature_names)
    plot_feature_importance(imp, OUT_DIR, top_n=30)
    created_plots += [f for f in os.listdir(OUT_DIR) if f.startswith('feature_importance_') and f.endswith('.png')]
    print("Feature importance plots generated.")

    training_results = parse_training_results(TRAINING_RESULTS_JSON)

    shap_info = {"available": False, "reason": "no data"}
    n_train = n_val = 0
    if X_train_df is not None and X_val_df is not None and y_train is not None and y_val is not None:
        print(f"Data loaded. Train shape: {X_train_df.shape}, Val shape: {X_val_df.shape}")
        dtrain = dmatrix(X_train_df, y_train)
        dval = dmatrix(X_val_df, y_val)

        print("Making predictions...")
        yhat_train = bst.predict(dtrain)
        yhat_val = bst.predict(dval)

        print("Computing metrics...")
        metrics = {
            'train': compute_regression_metrics(y_train, yhat_train),
            'validation': compute_regression_metrics(y_val, yhat_val)
        }
        n_train = int(len(X_train_df)); n_val = int(len(X_val_df))

        print("Generating prediction and residual plots...")
        plot_pred_vs_actual(y_train, yhat_train, "Prediction vs Actual (Train)", OUT_DIR / "pred_vs_actual_train.png")
        plot_pred_vs_actual(y_val, yhat_val, "Prediction vs Actual (Validation)", OUT_DIR / "pred_vs_actual_val.png")
        plot_residuals(np.asarray(y_train).ravel(), yhat_train, "train", OUT_DIR)
        plot_residuals(np.asarray(y_val).ravel(), yhat_val, "val", OUT_DIR)
        created_plots += [
            'pred_vs_actual_train.png','pred_vs_actual_val.png',
            'residual_hist_train.png','residuals_vs_pred_train.png',
            'residual_hist_val.png','residuals_vs_pred_val.png'
        ]
        print("Plots generated.")

        print("Generating learning curves...")
        plot_learning_curves(bst, dtrain, dval, np.asarray(y_train).ravel(), np.asarray(y_val).ravel(), OUT_DIR)
        created_plots.append('learning_curve_rmse.png')
        print("Learning curves generated.")

        print("Running SHAP analysis...")
        shap_info = run_shap(bst, X_train_df, X_val_df, OUT_DIR)
        if shap_info.get('available'):
            created_plots += [
                'shap_summary_beeswarm.png','shap_summary_bar.png',
                'shap_waterfall_0.png','shap_waterfall_1.png','shap_waterfall_2.png'
            ]
        print("SHAP analysis complete.")
    else:
        print("Training/validation data not available. Skipping full evaluation.")
        if training_results:
            metrics['validation_like'] = training_results

    print("Finalizing summary...")
    booster_attrs = bst.attributes()
    num_trees = None
    try:
        num_trees = bst.num_boosted_rounds()
    except Exception:
        pass
    if not num_trees:
        try:
            js = json.loads(MODEL_JSON.read_text())
            num_trees = int(js['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'])
        except Exception:
            num_trees = None

    summary = {
        'metrics': metrics,
        'n_train': int(n_train),
        'n_val': int(n_val),
        'n_features': int(len(feature_names) if feature_names else 0),
        'feature_names_available': bool(feature_names),
        'num_trees': num_trees,
        'booster_attrs': booster_attrs,
        'feature_importance_top_gain': sorted([(k, v) for k, v in imp.get('gain', {}).items()], key=lambda x: x[1], reverse=True)[:20],
        'training_results_present': training_results is not None,
        'shap': shap_info,
        'created_plots': sorted(list(set(created_plots))),
        'output_dir': str(OUT_DIR)
    }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("== Model Evaluation Summary ==")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

