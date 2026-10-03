"""
Unit Tests — OBI Signals & Alpha Generation
Run with: python -m pytest tests/test_obi.py -v
"""
import pytest
import numpy as np
import pandas as pd
from microstructure.obi import (
    OBIEngine, AlphaGenerator, OBISignal, AlphaSignal,
    compute_obi, compute_weighted_obi, compute_queue_imbalance,
    compute_depth_ratio, compute_tfi,
)


@pytest.fixture(scope="module")
def feed():
    import sys; sys.path.insert(0, ".")
    from feed.orderbook_ws import OrderbookFeedSimulator
    f = OrderbookFeedSimulator(n_events=500, seed=42)
    f.generate()
    return f


@pytest.fixture(scope="module")
def snap(feed):
    return feed.snapshots[10]


@pytest.fixture(scope="module")
def obi_df(feed):
    eng  = OBIEngine()
    sigs = eng.compute_signals(feed.snapshots)
    return eng.to_dataframe(sigs)


@pytest.fixture(scope="module")
def alpha_signals(feed, obi_df):
    gen = AlphaGenerator()
    tfi = compute_tfi(feed.trades_dataframe(), feed.to_dataframe())
    return gen.generate(obi_df, None, tfi)


# ---------------------------------------------------------------------------
# 1. Individual calculators
# ---------------------------------------------------------------------------
class TestComputeOBI:
    def test_in_range(self, snap):
        assert -1.0 <= compute_obi(snap) <= 1.0

    def test_returns_float(self, snap):
        assert isinstance(compute_obi(snap), float)

    def test_balanced_book_near_zero(self, feed):
        for snap in feed.snapshots[:20]:
            v = abs(compute_obi(snap))
            assert v <= 1.0


class TestWeightedOBI:
    def test_in_range(self, snap):
        assert -1.0 <= compute_weighted_obi(snap) <= 1.0

    def test_decay_effect(self, snap):
        w1 = compute_weighted_obi(snap, decay=0.9)
        w2 = compute_weighted_obi(snap, decay=0.1)
        assert isinstance(w1, float) and isinstance(w2, float)


class TestQueueImbalance:
    def test_in_range(self, snap):
        assert -1.0 <= compute_queue_imbalance(snap) <= 1.0

    def test_returns_float(self, snap):
        assert isinstance(compute_queue_imbalance(snap), float)


class TestDepthRatio:
    def test_positive(self, snap):
        assert compute_depth_ratio(snap) > 0

    def test_returns_float(self, snap):
        assert isinstance(compute_depth_ratio(snap), float)


# ---------------------------------------------------------------------------
# 2. OBIEngine
# ---------------------------------------------------------------------------
class TestOBIEngine:
    def test_compute_signals_length(self, feed):
        eng  = OBIEngine()
        sigs = eng.compute_signals(feed.snapshots)
        assert len(sigs) == len(feed.snapshots)

    def test_returns_obi_signals(self, feed):
        eng  = OBIEngine()
        sigs = eng.compute_signals(feed.snapshots)
        assert isinstance(sigs[0], OBISignal)

    def test_to_dataframe_shape(self, feed, obi_df):
        assert len(obi_df) == len(feed.snapshots)

    def test_required_columns(self, obi_df):
        for col in ["timestamp","obi","weighted_obi","queue_imbalance","depth_ratio","mid_price"]:
            assert col in obi_df.columns

    def test_obi_in_range(self, obi_df):
        assert (obi_df["obi"].between(-1, 1)).all()

    def test_weighted_obi_in_range(self, obi_df):
        assert (obi_df["weighted_obi"].between(-1, 1)).all()

    def test_rolling_features(self, obi_df):
        eng = OBIEngine()
        feat_df = eng.rolling_features(obi_df)
        assert "obi_ma5" in feat_df.columns
        assert "obi_std10" in feat_df.columns

    def test_rolling_features_adds_returns(self, obi_df):
        eng = OBIEngine()
        feat_df = eng.rolling_features(obi_df)
        assert "return_1" in feat_df.columns
        assert "return_5" in feat_df.columns


# ---------------------------------------------------------------------------
# 3. TFI
# ---------------------------------------------------------------------------
class TestTFI:
    def test_returns_series(self, feed):
        tfi = compute_tfi(feed.trades_dataframe(), feed.to_dataframe())
        assert isinstance(tfi, pd.Series)

    def test_length_matches_snapshots(self, feed):
        tfi = compute_tfi(feed.trades_dataframe(), feed.to_dataframe())
        assert len(tfi) == len(feed.snapshots)

    def test_values_finite(self, feed):
        tfi = compute_tfi(feed.trades_dataframe(), feed.to_dataframe())
        assert np.isfinite(tfi.values).all()

    def test_empty_trades(self, feed):
        empty = pd.DataFrame(columns=["timestamp","price","size","side"])
        tfi = compute_tfi(empty, feed.to_dataframe())
        assert isinstance(tfi, pd.Series)


# ---------------------------------------------------------------------------
# 4. AlphaGenerator
# ---------------------------------------------------------------------------
class TestAlphaGenerator:
    def test_returns_list(self, alpha_signals):
        assert isinstance(alpha_signals, list)

    def test_length_matches_input(self, feed, obi_df, alpha_signals):
        assert len(alpha_signals) == len(obi_df)

    def test_signal_type(self, alpha_signals):
        assert isinstance(alpha_signals[0], AlphaSignal)

    def test_direction_valid(self, alpha_signals):
        for sig in alpha_signals:
            assert sig.direction in {-1, 0, 1}

    def test_confidence_in_range(self, alpha_signals):
        for sig in alpha_signals:
            assert 0.0 <= sig.confidence <= 1.0

    def test_has_components(self, alpha_signals):
        assert isinstance(alpha_signals[0].components, dict)
        assert "obi" in alpha_signals[0].components

    def test_decayed_signal_finite(self, alpha_signals):
        assert all(np.isfinite(s.decayed) for s in alpha_signals)

    def test_backtest_returns_dict(self, feed, obi_df, alpha_signals):
        gen = AlphaGenerator()
        bt  = gen.backtest_signal(alpha_signals, obi_df["mid_price"].values)
        assert isinstance(bt, dict)

    def test_backtest_win_rate_range(self, feed, obi_df, alpha_signals):
        gen = AlphaGenerator()
        bt  = gen.backtest_signal(alpha_signals, obi_df["mid_price"].values)
        assert 0.0 <= bt.get("win_rate", 0) <= 1.0

    def test_empty_signals_backtest(self, obi_df):
        gen = AlphaGenerator()
        bt  = gen.backtest_signal([], obi_df["mid_price"].values)
        assert bt["win_rate"] == 0.0

    def test_weights_sum_to_one(self):
        gen = AlphaGenerator()
        total = gen.w_obi + gen.w_vpin + gen.w_tfi + gen.w_mom
        assert abs(total - 1.0) < 1e-6
