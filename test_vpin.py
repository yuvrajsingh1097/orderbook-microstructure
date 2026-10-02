"""
Unit Tests — VPIN Engine
Run with: python -m pytest tests/test_vpin.py -v
"""
import pytest
import numpy as np
import pandas as pd
from microstructure.vpin import (
    VPINCalculator, VPINResult, VolumeBucket,
    classify_tick_rule, classify_lee_ready, classify_bulk_volume,
    build_volume_buckets,
)


@pytest.fixture(scope="module")
def feed_data():
    import sys; sys.path.insert(0, ".")
    from feed.orderbook_ws import OrderbookFeedSimulator
    f = OrderbookFeedSimulator(n_events=2000, seed=42, start_price=100.0)
    f.generate()
    return f.trades_dataframe(), f.to_dataframe()


@pytest.fixture(scope="module")
def vpin_bvc(feed_data):
    trades_df, snaps_df = feed_data
    calc = VPINCalculator(bucket_size=300.0, window=15, method="bvc")
    return calc.compute(trades_df, snaps_df)


# ---------------------------------------------------------------------------
# 1. Classifiers
# ---------------------------------------------------------------------------
class TestTickRule:
    def test_returns_array(self):
        p = np.array([100.0, 100.1, 100.0, 100.2])
        r = classify_tick_rule(p)
        assert isinstance(r, np.ndarray) and len(r) == 4

    def test_uptick_is_buy(self):
        p = np.array([100.0, 100.1])
        assert classify_tick_rule(p)[1] == 1.0

    def test_downtick_is_sell(self):
        p = np.array([100.1, 100.0])
        assert classify_tick_rule(p)[1] == 0.0

    def test_in_range(self):
        p = np.random.uniform(99, 101, 100)
        r = classify_tick_rule(p)
        assert ((r >= 0) & (r <= 1)).all()


class TestLeeReady:
    def test_above_mid_is_buy(self):
        p   = np.array([100.1])
        mid = np.array([100.0])
        assert classify_lee_ready(p, mid)[0] == 1.0

    def test_below_mid_is_sell(self):
        p   = np.array([99.9])
        mid = np.array([100.0])
        assert classify_lee_ready(p, mid)[0] == 0.0

    def test_in_range(self):
        p   = np.random.uniform(99, 101, 50)
        mid = np.ones(50) * 100.0
        r   = classify_lee_ready(p, mid)
        assert ((r >= 0) & (r <= 1)).all()


class TestBVC:
    def test_returns_array(self):
        p = np.random.uniform(99, 101, 100)
        v = np.ones(100) * 100
        r = classify_bulk_volume(p, v)
        assert isinstance(r, np.ndarray) and len(r) == 100

    def test_values_in_range(self):
        p = np.random.uniform(99, 101, 200)
        v = np.ones(200) * 100
        r = classify_bulk_volume(p, v)
        assert ((r >= 0) & (r <= 1)).all()

    def test_rising_prices_high_buy(self):
        p = np.linspace(100, 110, 100)
        v = np.ones(100)
        r = classify_bulk_volume(p, v)
        assert r[50:].mean() > 0.5


# ---------------------------------------------------------------------------
# 2. Volume bucket builder
# ---------------------------------------------------------------------------
class TestBuildBuckets:
    def test_returns_list(self):
        p  = np.ones(100) * 100.0
        v  = np.ones(100) * 10.0
        bf = np.ones(100) * 0.5
        ts = np.arange(100, dtype=float)
        buckets = build_volume_buckets(p, v, bf, ts, bucket_size=50.0)
        assert isinstance(buckets, list)

    def test_bucket_volume_approx(self):
        p  = np.ones(200) * 100.0
        v  = np.ones(200) * 10.0
        bf = np.ones(200) * 0.5
        ts = np.arange(200, dtype=float)
        buckets = build_volume_buckets(p, v, bf, ts, bucket_size=50.0)
        for b in buckets[:-1]:
            assert abs(b.volume - 50.0) < 10.1

    def test_buy_sell_sum_to_volume(self):
        p  = np.random.uniform(99, 101, 100)
        v  = np.ones(100) * 20.0
        bf = np.random.uniform(0, 1, 100)
        ts = np.arange(100, dtype=float)
        buckets = build_volume_buckets(p, v, bf, ts, bucket_size=100.0)
        for b in buckets:
            assert abs(b.buy_volume + b.sell_volume - b.volume) < 1e-4

    def test_oi_in_range(self):
        p  = np.random.uniform(99, 101, 200)
        v  = np.ones(200) * 10.0
        bf = np.random.uniform(0, 1, 200)
        ts = np.arange(200, dtype=float)
        buckets = build_volume_buckets(p, v, bf, ts, bucket_size=50.0)
        for b in buckets:
            assert 0.0 <= b.order_imbalance <= 1.0


# ---------------------------------------------------------------------------
# 3. VPINCalculator
# ---------------------------------------------------------------------------
class TestVPINCalculator:
    def test_returns_result(self, feed_data):
        trades_df, snaps_df = feed_data
        calc = VPINCalculator(bucket_size=500.0, window=10, method="bvc")
        r    = calc.compute(trades_df, snaps_df)
        assert isinstance(r, VPINResult)

    def test_vpin_in_range(self, vpin_bvc):
        valid = vpin_bvc.vpin[~np.isnan(vpin_bvc.vpin)]
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_n_buckets_positive(self, vpin_bvc):
        assert vpin_bvc.n_buckets > 0

    def test_timestamps_increasing(self, vpin_bvc):
        ts = vpin_bvc.timestamps
        assert (np.diff(ts) >= 0).all()

    def test_all_methods_run(self, feed_data):
        trades_df, snaps_df = feed_data
        for method in ["tick", "lee_ready", "bvc"]:
            calc = VPINCalculator(bucket_size=500.0, window=10, method=method)
            r    = calc.compute(trades_df, snaps_df)
            assert r.n_buckets > 0

    def test_summary_stats_keys(self, feed_data):
        trades_df, snaps_df = feed_data
        calc = VPINCalculator(bucket_size=500.0, window=10)
        r    = calc.compute(trades_df, snaps_df)
        s    = calc.summary_stats(r)
        for k in ["n_buckets","vpin_mean","vpin_max","vpin_p95"]:
            assert k in s

    def test_alert_threshold_in_range(self, vpin_bvc):
        calc   = VPINCalculator()
        thresh = calc.alert_threshold(vpin_bvc.vpin)
        assert 0.0 <= thresh <= 1.0

    def test_empty_trades(self):
        calc = VPINCalculator()
        empty_df = pd.DataFrame(columns=["timestamp","price","size","side"])
        r = calc.compute(empty_df)
        assert r.n_buckets == 0

    def test_window_effect(self, feed_data):
        trades_df, snaps_df = feed_data
        c1 = VPINCalculator(bucket_size=500.0, window=5)
        c2 = VPINCalculator(bucket_size=500.0, window=20)
        r1 = c1.compute(trades_df, snaps_df)
        r2 = c2.compute(trades_df, snaps_df)
        v1 = r1.vpin[~np.isnan(r1.vpin)]
        v2 = r2.vpin[~np.isnan(r2.vpin)]
        # Larger window → smoother (lower std)
        assert v2.std() <= v1.std() + 0.05
