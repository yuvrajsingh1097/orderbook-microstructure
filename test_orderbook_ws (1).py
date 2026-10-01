"""
Unit Tests — Orderbook Feed Simulator
Run with: python -m pytest tests/test_orderbook_ws.py -v
"""
import pytest
import numpy as np
import pandas as pd
from feed.orderbook_ws import (
    PriceLevel, OrderbookSnapshot, TradeEvent,
    OrderbookFeedSimulator, REGIMES,
)


@pytest.fixture(scope="module")
def feed():
    f = OrderbookFeedSimulator(n_events=500, seed=42, n_levels=5)
    f.generate()
    return f


@pytest.fixture(scope="module")
def snap(feed):
    return feed.snapshots[0]


# ---------------------------------------------------------------------------
# 1. PriceLevel & OrderbookSnapshot
# ---------------------------------------------------------------------------
class TestPriceLevel:
    def test_fields(self):
        pl = PriceLevel(price=100.0, size=500.0)
        assert pl.price == 100.0 and pl.size == 500.0


class TestOrderbookSnapshot:
    def test_mid_price(self, snap):
        expected = (snap.best_bid + snap.best_ask) / 2
        assert abs(snap.mid_price - expected) < 1e-6

    def test_spread_positive(self, snap):
        assert snap.spread > 0

    def test_best_bid_below_ask(self, snap):
        assert snap.best_bid < snap.best_ask

    def test_imbalance_in_range(self, snap):
        assert -1.0 <= snap.imbalance <= 1.0

    def test_bid_depth_positive(self, snap):
        assert snap.bid_depth > 0

    def test_ask_depth_positive(self, snap):
        assert snap.ask_depth > 0

    def test_total_depth(self, snap):
        assert abs(snap.total_depth - snap.bid_depth - snap.ask_depth) < 1e-6

    def test_bids_sorted_descending(self, snap):
        prices = [l.price for l in snap.bids]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self, snap):
        prices = [l.price for l in snap.asks]
        assert prices == sorted(prices)


# ---------------------------------------------------------------------------
# 2. TradeEvent
# ---------------------------------------------------------------------------
class TestTradeEvent:
    def test_buy_flag(self):
        t = TradeEvent(timestamp=1000, price=100.0, size=50.0, side="buy")
        assert t.is_buy is True

    def test_sell_flag(self):
        t = TradeEvent(timestamp=1000, price=100.0, size=50.0, side="sell")
        assert t.is_buy is False

    def test_positive_size(self, feed):
        for t in feed.trades[:20]:
            assert t.size > 0


# ---------------------------------------------------------------------------
# 3. Feed generation
# ---------------------------------------------------------------------------
class TestFeedGeneration:
    def test_correct_n_snapshots(self, feed):
        assert len(feed.snapshots) == 500

    def test_trades_generated(self, feed):
        assert len(feed.trades) > 0

    def test_timestamps_increasing(self, feed):
        ts = [s.timestamp for s in feed.snapshots]
        assert all(ts[i] < ts[i+1] for i in range(len(ts)-1))

    def test_prices_positive(self, feed):
        assert all(s.mid_price > 0 for s in feed.snapshots)

    def test_spreads_positive(self, feed):
        assert all(s.spread > 0 for s in feed.snapshots)

    def test_reproducible(self):
        f1 = OrderbookFeedSimulator(n_events=100, seed=0).generate()
        f2 = OrderbookFeedSimulator(n_events=100, seed=0).generate()
        p1 = [s.mid_price for s in f1.snapshots]
        p2 = [s.mid_price for s in f2.snapshots]
        assert np.allclose(p1, p2)

    def test_different_seeds(self):
        f1 = OrderbookFeedSimulator(n_events=100, seed=0).generate()
        f2 = OrderbookFeedSimulator(n_events=100, seed=99).generate()
        p1 = [s.mid_price for s in f1.snapshots]
        p2 = [s.mid_price for s in f2.snapshots]
        assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------------
# 4. DataFrame output
# ---------------------------------------------------------------------------
class TestDataFrames:
    def test_to_dataframe_shape(self, feed):
        df = feed.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500

    def test_required_columns(self, feed):
        df = feed.to_dataframe()
        for col in ["timestamp","mid_price","spread","imbalance","bid_depth","regime"]:
            assert col in df.columns

    def test_spread_bps_positive(self, feed):
        df = feed.to_dataframe()
        assert (df["spread_bps"] > 0).all()

    def test_trades_dataframe(self, feed):
        td = feed.trades_dataframe()
        assert isinstance(td, pd.DataFrame)
        assert "price" in td.columns and "side" in td.columns

    def test_regime_column(self, feed):
        df = feed.to_dataframe()
        assert df["regime"].isin(list(REGIMES.keys())).all()


# ---------------------------------------------------------------------------
# 5. Stats
# ---------------------------------------------------------------------------
class TestStats:
    def test_returns_dict(self, feed):
        assert isinstance(feed.stats(), dict)

    def test_n_snapshots_key(self, feed):
        assert feed.stats()["n_snapshots"] == 500

    def test_mean_spread_positive(self, feed):
        assert feed.stats()["mean_spread_bps"] > 0

    def test_regimes_present(self, feed):
        assert "regimes" in feed.stats()


# ---------------------------------------------------------------------------
# 6. Regimes
# ---------------------------------------------------------------------------
class TestRegimes:
    def test_stressed_higher_spread(self):
        assert REGIMES["stressed"]["spread_bps"] > REGIMES["normal"]["spread_bps"]

    def test_illiquid_lower_depth(self):
        assert REGIMES["illiquid"]["depth_mean"] < REGIMES["normal"]["depth_mean"]

    def test_regime_schedule(self):
        f = OrderbookFeedSimulator(
            n_events=200, seed=0,
            regime_schedule=[(0,"normal"),(100,"stressed")]
        ).generate()
        regimes = [getattr(s,"_regime","normal") for s in f.snapshots]
        assert "normal"   in regimes
        assert "stressed" in regimes
