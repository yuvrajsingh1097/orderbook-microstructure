"""
ML Trade Direction Predictor
==============================
Predicts short-term price direction using microstructure features.

Features (engineered from L2 orderbook + trades):
    OBI, weighted OBI, queue imbalance, depth ratio
    VPIN, spread (bps), trade flow imbalance
    Rolling means and stds (5, 10, 20 window)
    Lagged returns (1, 5, 10 steps)

Models:
    1. Random Forest (main)
    2. Logistic Regression (baseline)
    3. Gradient Boosting (optional)

Target:
    Binary: 1 if mid_price[t+horizon] > mid_price[t] else 0

Pipeline:
    Feature engineering → train/test split → fit → evaluate
    → feature importance → walk-forward backtest
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Evaluation results for a trained model."""
    model_name:   str
    accuracy:     float
    precision:    float
    recall:       float
    f1:           float
    roc_auc:      float
    cv_mean:      float
    cv_std:       float
    n_train:      int
    n_test:       int
    feature_importance: dict = field(default_factory=dict)
    confusion:    np.ndarray = None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(
    obi_df:     pd.DataFrame,
    vpin_arr:   np.ndarray = None,
    tfi_arr:    np.ndarray = None,
    windows:    list = [5, 10, 20],
    lag_steps:  list = [1, 5, 10],
) -> pd.DataFrame:
    """
    Build full feature matrix from OBI signals.

    Parameters
    ----------
    obi_df   : DataFrame from OBIEngine.to_dataframe()
    vpin_arr : optional VPIN series aligned to obi_df
    tfi_arr  : optional TFI series aligned to obi_df
    windows  : rolling window sizes
    lag_steps: lagged return steps

    Returns feature DataFrame (no NaN).
    """
    df = obi_df.copy()

    # Base microstructure features
    base_cols = ["obi", "weighted_obi", "queue_imbalance",
                 "depth_ratio", "spread_bps"]

    # Rolling stats
    for col in base_cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_ma{w}"]   = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_std{w}"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
            df[f"{col}_zscore{w}"] = (
                (df[col] - df[f"{col}_ma{w}"]) /
                (df[f"{col}_std{w}"] + 1e-8)
            ).clip(-5, 5)

    # Price returns and lags
    df["return_1"]  = df["mid_price"].pct_change(1).fillna(0)
    df["return_5"]  = df["mid_price"].pct_change(5).fillna(0)
    df["return_10"] = df["mid_price"].pct_change(10).fillna(0)

    for lag in lag_steps:
        df[f"obi_lag{lag}"]          = df["obi"].shift(lag).fillna(0)
        df[f"weighted_obi_lag{lag}"] = df["weighted_obi"].shift(lag).fillna(0)
        df[f"return_lag{lag}"]       = df["return_1"].shift(lag).fillna(0)

    # Spread momentum
    df["spread_ma5"]    = df["spread_bps"].rolling(5, min_periods=1).mean()
    df["spread_change"] = df["spread_bps"].diff().fillna(0)

    # VPIN feature
    if vpin_arr is not None and len(vpin_arr) == len(df):
        df["vpin"] = vpin_arr
        df["vpin_ma5"] = pd.Series(vpin_arr).rolling(5, min_periods=1).mean().values
    else:
        df["vpin"]     = 0.5
        df["vpin_ma5"] = 0.5

    # TFI feature
    if tfi_arr is not None and len(tfi_arr) == len(df):
        df["tfi"] = tfi_arr if isinstance(tfi_arr, np.ndarray) else tfi_arr.values
    else:
        df["tfi"] = 0.0

    # Interaction features
    df["obi_x_vpin"]   = df["obi"] * df["vpin"]
    df["obi_x_spread"] = df["obi"] * df["spread_bps"]
    df["depth_x_obi"]  = df["depth_ratio"] * df["obi"]

    return df.dropna().reset_index(drop=True)


def build_labels(
    df:      pd.DataFrame,
    horizon: int = 5,
) -> pd.Series:
    """
    Binary label: 1 if price rises over next `horizon` steps, else 0.
    """
    future_price = df["mid_price"].shift(-horizon)
    label = (future_price > df["mid_price"]).astype(int)
    return label


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature column names (exclude non-feature cols)."""
    exclude = {"timestamp", "mid_price", "label", "regime"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

class DirectionModel:
    """
    Trains and evaluates ML models for trade direction prediction.

    Usage:
        model = DirectionModel(model_type='rf')
        model.fit(features_df, labels)
        result = model.evaluate(X_test, y_test)
        importance = model.feature_importance()
    """

    MODELS = {
        "rf":  lambda: RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "lr":  lambda: LogisticRegression(
            C=1.0, max_iter=500, random_state=42,
        ),
        "gb":  lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42,
        ),
    }

    def __init__(self, model_type: str = "rf"):
        if model_type not in self.MODELS:
            raise ValueError(f"model_type must be one of {list(self.MODELS)}")
        self.model_type  = model_type
        self.model       = self.MODELS[model_type]()
        self.scaler      = StandardScaler()
        self.feature_cols = None
        self.is_fitted   = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list = None,
    ) -> "DirectionModel":
        """Fit model on training data."""
        self.feature_cols = feature_cols or get_feature_cols(X)
        Xf = X[self.feature_cols].values.astype(np.float32)
        Xs = self.scaler.fit_transform(Xf)
        self.model.fit(Xs, y.values)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary direction."""
        Xf = X[self.feature_cols].values.astype(np.float32)
        Xs = self.scaler.transform(Xf)
        return self.model.predict(Xs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of up-move."""
        Xf = X[self.feature_cols].values.astype(np.float32)
        Xs = self.scaler.transform(Xf)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(Xs)[:, 1]
        return self.model.decision_function(Xs)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelResult:
        """Evaluate model on test set."""
        preds  = self.predict(X)
        probas = self.predict_proba(X)

        # CV on full dataset
        Xf = X[self.feature_cols].values.astype(np.float32)
        Xs = self.scaler.transform(Xf)
        cv = cross_val_score(self.model, Xs, y.values, cv=5, scoring="accuracy")

        return ModelResult(
            model_name=self.model_type,
            accuracy=round(float(accuracy_score(y, preds)), 4),
            precision=round(float(precision_score(y, preds, zero_division=0)), 4),
            recall=round(float(recall_score(y, preds, zero_division=0)), 4),
            f1=round(float(f1_score(y, preds, zero_division=0)), 4),
            roc_auc=round(float(roc_auc_score(y, probas)), 4),
            cv_mean=round(float(cv.mean()), 4),
            cv_std=round(float(cv.std()), 4),
            n_train=0,
            n_test=len(y),
            feature_importance=self.feature_importance(),
            confusion=confusion_matrix(y, preds),
        )

    def feature_importance(self) -> dict:
        """Return top feature importances."""
        if not self.is_fitted or self.feature_cols is None:
            return {}
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_[0])
        else:
            return {}
        idx     = np.argsort(imp)[::-1]
        result  = {self.feature_cols[i]: round(float(imp[i]), 6) for i in idx[:20]}
        # Normalise
        total   = sum(result.values())
        if total > 0:
            result = {k: round(v/total, 6) for k, v in result.items()}
        return result


# ---------------------------------------------------------------------------
# Walk-forward backtest of ML signal
# ---------------------------------------------------------------------------

def ml_walk_forward(
    features_df: pd.DataFrame,
    labels:      pd.Series,
    n_folds:     int = 5,
    model_type:  str = "rf",
    horizon:     int = 5,
) -> dict:
    """
    Walk-forward evaluation: train on past, test on future.

    Returns aggregated metrics across all folds.
    """
    n         = len(features_df)
    fold_size = n // (n_folds + 1)
    results   = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_end  = min(train_end + fold_size, n)

        if test_end <= train_end + 10:
            continue

        X_train = features_df.iloc[:train_end]
        y_train = labels.iloc[:train_end]
        X_test  = features_df.iloc[train_end:test_end]
        y_test  = labels.iloc[train_end:test_end]

        # Skip if labels are all one class
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        m = DirectionModel(model_type)
        m.fit(X_train, y_train)
        r = m.evaluate(X_test, y_test)
        r.n_train = len(X_train)
        results.append(r)

    if not results:
        return {}

    return {
        "n_folds":      len(results),
        "mean_accuracy":round(float(np.mean([r.accuracy for r in results])), 4),
        "mean_f1":      round(float(np.mean([r.f1       for r in results])), 4),
        "mean_roc_auc": round(float(np.mean([r.roc_auc  for r in results])), 4),
        "std_accuracy": round(float(np.std( [r.accuracy for r in results])), 4),
        "results":      results,
    }


# ---------------------------------------------------------------------------
# Signal-to-PnL converter
# ---------------------------------------------------------------------------

def signal_backtest(
    probas:      np.ndarray,
    prices:      np.ndarray,
    threshold:   float = 0.55,
    hold_steps:  int   = 5,
    cost_bps:    float = 2.0,
) -> dict:
    """
    Convert model probability predictions to a simple PnL backtest.

    Parameters
    ----------
    probas      : model predicted probabilities of up-move
    prices      : mid-price series
    threshold   : min probability to generate signal
    hold_steps  : holding period
    cost_bps    : transaction cost in bps per trade

    Returns dict with PnL metrics.
    """
    pnls = []
    cost = cost_bps / 10_000

    for i in range(len(probas) - hold_steps):
        p = probas[i]
        if p > threshold:          # buy signal
            ret = (prices[i+hold_steps] - prices[i]) / prices[i] - cost
            pnls.append(ret)
        elif p < (1 - threshold):  # sell signal
            ret = (prices[i] - prices[i+hold_steps]) / prices[i] - cost
            pnls.append(ret)

    if not pnls:
        return {"n_trades":0,"win_rate":0.0,"avg_pnl":0.0,"sharpe":0.0,"total_pnl":0.0}

    pnls = np.array(pnls)
    sharpe = float(pnls.mean() / (pnls.std() + 1e-8)) * np.sqrt(252 * 390)

    return {
        "n_trades":  len(pnls),
        "win_rate":  round(float((pnls > 0).mean()), 4),
        "avg_pnl":   round(float(pnls.mean()), 6),
        "sharpe":    round(float(sharpe), 4),
        "total_pnl": round(float(pnls.sum()), 6),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feed.orderbook_ws import OrderbookFeedSimulator
    from microstructure.obi import OBIEngine, compute_tfi
    from microstructure.vpin import VPINCalculator

    print("=" * 55)
    print("ML Trade Direction Model Demo")
    print("=" * 55)

    feed = OrderbookFeedSimulator(n_events=5000, seed=42)
    feed.generate()

    engine    = OBIEngine()
    obi_df    = engine.to_dataframe(engine.compute_signals(feed.snapshots))
    snaps_df  = feed.to_dataframe()
    trades_df = feed.trades_dataframe()
    tfi       = compute_tfi(trades_df, snaps_df)

    calc      = VPINCalculator(bucket_size=400.0, window=20)
    result    = calc.compute(trades_df, snaps_df)
    valid     = ~np.isnan(result.vpin)
    vpin_al   = np.interp(obi_df["timestamp"].values,
                          result.timestamps[valid], result.vpin[valid]) \
                if valid.any() else np.full(len(obi_df), 0.3)

    feat_df = build_feature_matrix(obi_df, vpin_al, tfi)
    labels  = build_labels(feat_df, horizon=5)

    # Align
    min_len  = min(len(feat_df), len(labels))
    feat_df  = feat_df.iloc[:min_len].reset_index(drop=True)
    labels   = labels.iloc[:min_len].reset_index(drop=True)
    mask     = labels.notna()
    feat_df  = feat_df[mask].reset_index(drop=True)
    labels   = labels[mask].reset_index(drop=True)

    split = int(len(feat_df) * 0.8)
    X_train, X_test = feat_df.iloc[:split], feat_df.iloc[split:]
    y_train, y_test = labels.iloc[:split], labels.iloc[split:]

    print(f"\n  Features: {len(get_feature_cols(feat_df))}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Label balance: {labels.mean():.3f} (0.5 = balanced)")

    for model_type in ["lr", "rf"]:
        m = DirectionModel(model_type)
        m.fit(X_train, y_train)
        r = m.evaluate(X_test, y_test)
        r.n_train = len(X_train)
        print(f"\n  [{model_type.upper()}] acc={r.accuracy:.4f}  f1={r.f1:.4f}  "
              f"auc={r.roc_auc:.4f}  cv={r.cv_mean:.4f}±{r.cv_std:.4f}")
        if r.feature_importance:
            top3 = list(r.feature_importance.items())[:3]
            print(f"    Top features: {top3}")

    # Walk-forward
    print("\n  Walk-forward evaluation (RF, 4 folds):")
    wf = ml_walk_forward(feat_df, labels, n_folds=4, model_type="rf")
    print(f"    mean_accuracy={wf.get('mean_accuracy',0):.4f}  "
          f"mean_auc={wf.get('mean_roc_auc',0):.4f}")
