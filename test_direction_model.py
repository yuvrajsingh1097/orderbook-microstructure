"""
Unit Tests — ML Trade Direction Model
Run with: python -m pytest tests/test_direction_model.py -v
"""
import pytest
import numpy as np
import pandas as pd
from ml.direction_model import (
    DirectionModel, ModelResult,
    build_feature_matrix, build_labels, get_feature_cols,
    ml_walk_forward, signal_backtest,
)


@pytest.fixture(scope="module")
def obi_df():
    import sys; sys.path.insert(0, ".")
    from feed.orderbook_ws import OrderbookFeedSimulator
    from microstructure.obi import OBIEngine
    f = OrderbookFeedSimulator(n_events=1000, seed=42); f.generate()
    eng = OBIEngine()
    return eng.to_dataframe(eng.compute_signals(f.snapshots))


@pytest.fixture(scope="module")
def feat_df(obi_df):
    return build_feature_matrix(obi_df)


@pytest.fixture(scope="module")
def labels(feat_df):
    lb = build_labels(feat_df, horizon=5)
    return lb.iloc[:len(feat_df)].fillna(0).astype(int)


@pytest.fixture(scope="module")
def split_data(feat_df, labels):
    min_len  = min(len(feat_df), len(labels))
    X = feat_df.iloc[:min_len].reset_index(drop=True)
    y = labels.iloc[:min_len].reset_index(drop=True)
    mask = y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    split = int(len(X) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


@pytest.fixture(scope="module")
def fitted_rf(split_data):
    X_tr, _, y_tr, _ = split_data
    m = DirectionModel("rf")
    m.fit(X_tr, y_tr)
    return m


# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------
class TestBuildFeatures:
    def test_returns_dataframe(self, feat_df):
        assert isinstance(feat_df, pd.DataFrame)

    def test_has_features(self, feat_df):
        assert len(get_feature_cols(feat_df)) > 10

    def test_no_nan(self, feat_df):
        fcols = get_feature_cols(feat_df)
        assert feat_df[fcols].isna().sum().sum() == 0

    def test_has_rolling_cols(self, feat_df):
        assert any("ma5" in c for c in feat_df.columns)
        assert any("std10" in c for c in feat_df.columns)

    def test_has_interaction_cols(self, feat_df):
        assert "obi_x_vpin" in feat_df.columns

    def test_has_lag_cols(self, feat_df):
        assert any("lag" in c for c in feat_df.columns)

    def test_with_vpin(self, obi_df):
        vpin = np.full(len(obi_df), 0.3)
        df   = build_feature_matrix(obi_df, vpin_arr=vpin)
        assert "vpin" in df.columns

    def test_with_tfi(self, obi_df):
        tfi = pd.Series(np.zeros(len(obi_df)))
        df  = build_feature_matrix(obi_df, tfi_arr=tfi)
        assert "tfi" in df.columns


# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------
class TestBuildLabels:
    def test_returns_series(self, feat_df):
        lb = build_labels(feat_df, horizon=5)
        assert isinstance(lb, pd.Series)

    def test_binary(self, feat_df):
        lb = build_labels(feat_df, horizon=5).dropna()
        assert set(lb.unique()).issubset({0, 1})

    def test_roughly_balanced(self, feat_df):
        lb = build_labels(feat_df, horizon=5).dropna()
        mean = lb.mean()
        assert 0.3 <= mean <= 0.7


# ---------------------------------------------------------------------------
# 3. DirectionModel
# ---------------------------------------------------------------------------
class TestDirectionModel:
    def test_init_rf(self):
        m = DirectionModel("rf")
        assert m.model_type == "rf"

    def test_init_lr(self):
        m = DirectionModel("lr")
        assert m.model_type == "lr"

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            DirectionModel("invalid_model")

    def test_fit_returns_self(self, split_data):
        X_tr, _, y_tr, _ = split_data
        m = DirectionModel("lr")
        assert m.fit(X_tr, y_tr) is m

    def test_is_fitted_after_fit(self, fitted_rf):
        assert fitted_rf.is_fitted

    def test_predict_shape(self, fitted_rf, split_data):
        _, X_te, _, _ = split_data
        preds = fitted_rf.predict(X_te)
        assert len(preds) == len(X_te)

    def test_predict_binary(self, fitted_rf, split_data):
        _, X_te, _, _ = split_data
        preds = fitted_rf.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self, fitted_rf, split_data):
        _, X_te, _, _ = split_data
        probas = fitted_rf.predict_proba(X_te)
        assert ((probas >= 0) & (probas <= 1)).all()

    def test_evaluate_returns_result(self, fitted_rf, split_data):
        _, X_te, _, y_te = split_data
        r = fitted_rf.evaluate(X_te, y_te)
        assert isinstance(r, ModelResult)

    def test_accuracy_in_range(self, fitted_rf, split_data):
        _, X_te, _, y_te = split_data
        r = fitted_rf.evaluate(X_te, y_te)
        assert 0.0 <= r.accuracy <= 1.0

    def test_roc_auc_in_range(self, fitted_rf, split_data):
        _, X_te, _, y_te = split_data
        r = fitted_rf.evaluate(X_te, y_te)
        assert 0.0 <= r.roc_auc <= 1.0

    def test_feature_importance_sum(self, fitted_rf):
        imp = fitted_rf.feature_importance()
        assert isinstance(imp, dict)
        assert len(imp) > 0
        total = sum(imp.values())
        assert abs(total - 1.0) < 0.01

    def test_confusion_matrix_shape(self, fitted_rf, split_data):
        _, X_te, _, y_te = split_data
        r = fitted_rf.evaluate(X_te, y_te)
        assert r.confusion.shape == (2, 2)


# ---------------------------------------------------------------------------
# 4. Walk-forward
# ---------------------------------------------------------------------------
class TestWalkForward:
    def test_returns_dict(self, feat_df, labels):
        min_len = min(len(feat_df), len(labels))
        X = feat_df.iloc[:min_len].reset_index(drop=True)
        y = labels.iloc[:min_len].reset_index(drop=True)
        res = ml_walk_forward(X, y, n_folds=3, model_type="lr")
        assert isinstance(res, dict)

    def test_mean_accuracy_in_range(self, feat_df, labels):
        min_len = min(len(feat_df), len(labels))
        X = feat_df.iloc[:min_len].reset_index(drop=True)
        y = labels.iloc[:min_len].reset_index(drop=True)
        res = ml_walk_forward(X, y, n_folds=3, model_type="lr")
        if res:
            assert 0.0 <= res["mean_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Signal backtest
# ---------------------------------------------------------------------------
class TestSignalBacktest:
    def test_returns_dict(self, fitted_rf, split_data):
        _, X_te, _, _ = split_data
        probas = fitted_rf.predict_proba(X_te)
        prices = X_te["mid_price"].values
        bt = signal_backtest(probas, prices)
        assert isinstance(bt, dict)

    def test_win_rate_in_range(self, fitted_rf, split_data):
        _, X_te, _, _ = split_data
        probas = fitted_rf.predict_proba(X_te)
        prices = X_te["mid_price"].values
        bt = signal_backtest(probas, prices)
        assert 0.0 <= bt.get("win_rate", 0) <= 1.0

    def test_no_trades_high_threshold(self, fitted_rf, split_data):
        _, X_te, _, _ = split_data
        probas = fitted_rf.predict_proba(X_te)
        prices = X_te["mid_price"].values
        bt = signal_backtest(probas, prices, threshold=0.99)
        assert bt["n_trades"] == 0
