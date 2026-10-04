"""
Unit Tests — TWAP/VWAP Execution Engine
Run with: python -m pytest tests/test_twap.py -v
"""
import pytest
import numpy as np
import pandas as pd
from execution.twap import (
    Order, SliceExecution, ExecutionResult,
    AlmgrenChrissImpact, TWAPExecutor, VWAPExecutor, ISExecutor,
    intraday_volume_profile, execution_summary, compare_algos,
)


@pytest.fixture(scope="module")
def price_data():
    rng = np.random.default_rng(42)
    n   = 20
    prices = 185.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.001))
    spread = rng.uniform(1.5, 4.0, n)
    ts     = np.linspace(0, 1200_000, n)
    return prices, spread, ts


@pytest.fixture(scope="module")
def buy_order(price_data):
    prices, _, _ = price_data
    return Order("ORD001", "buy",  10_000, float(prices[0]))


@pytest.fixture(scope="module")
def sell_order(price_data):
    prices, _, _ = price_data
    return Order("ORD002", "sell", 10_000, float(prices[0]))


@pytest.fixture(scope="module")
def twap_result(buy_order, price_data):
    prices, spread, ts = price_data
    return TWAPExecutor(n_slices=20).execute(buy_order, prices, spread, ts)


@pytest.fixture(scope="module")
def vwap_result(buy_order, price_data):
    prices, spread, ts = price_data
    return VWAPExecutor(n_slices=20).execute(buy_order, prices, spread, ts)


@pytest.fixture(scope="module")
def is_result(buy_order, price_data):
    prices, spread, ts = price_data
    return ISExecutor(n_slices=20).execute(buy_order, prices, spread, ts)


# ---------------------------------------------------------------------------
# 1. AlmgrenChrissImpact
# ---------------------------------------------------------------------------
class TestAlmgrenChriss:
    def test_temp_impact_positive(self):
        m = AlmgrenChrissImpact()
        assert m.temporary_impact_bps(1000) > 0

    def test_perm_impact_positive(self):
        m = AlmgrenChrissImpact()
        assert m.permanent_impact_bps(10_000) > 0

    def test_larger_qty_higher_impact(self):
        m  = AlmgrenChrissImpact()
        i1 = m.temporary_impact_bps(1_000)
        i2 = m.temporary_impact_bps(100_000)
        assert i2 > i1

    def test_total_impact_sum(self):
        m = AlmgrenChrissImpact()
        total = m.total_impact_bps(1_000, 10_000)
        assert total == m.temporary_impact_bps(1_000) + m.permanent_impact_bps(10_000)


# ---------------------------------------------------------------------------
# 2. Volume profile
# ---------------------------------------------------------------------------
class TestVolumeProfile:
    def test_sums_to_one(self):
        vp = intraday_volume_profile(20)
        assert abs(vp.sum() - 1.0) < 1e-6

    def test_correct_length(self):
        vp = intraday_volume_profile(15)
        assert len(vp) == 15

    def test_all_positive(self):
        vp = intraday_volume_profile(20)
        assert (vp > 0).all()


# ---------------------------------------------------------------------------
# 3. TWAP
# ---------------------------------------------------------------------------
class TestTWAP:
    def test_returns_result(self, twap_result):
        assert isinstance(twap_result, ExecutionResult)

    def test_algo_label(self, twap_result):
        assert twap_result.algo == "TWAP"

    def test_n_slices(self, twap_result):
        assert twap_result.n_slices == 20
        assert len(twap_result.slices) == 20

    def test_exec_price_positive(self, twap_result):
        assert twap_result.avg_exec_price > 0

    def test_participation_rate_one(self, twap_result):
        assert abs(twap_result.participation_rate - 1.0) < 1e-4

    def test_total_cost_positive(self, twap_result):
        assert twap_result.total_cost_bps > 0

    def test_buy_higher_than_market(self, twap_result):
        # Buying pays spread + impact, so exec > market
        mkt_avg = np.mean([s.market_price for s in twap_result.slices])
        assert twap_result.avg_exec_price > mkt_avg

    def test_sell_lower_than_market(self, sell_order, price_data):
        prices, spread, ts = price_data
        res = TWAPExecutor(n_slices=20).execute(sell_order, prices, spread, ts)
        mkt_avg = np.mean([s.market_price for s in res.slices])
        assert res.avg_exec_price < mkt_avg


# ---------------------------------------------------------------------------
# 4. VWAP
# ---------------------------------------------------------------------------
class TestVWAP:
    def test_returns_result(self, vwap_result):
        assert isinstance(vwap_result, ExecutionResult)

    def test_algo_label(self, vwap_result):
        assert vwap_result.algo == "VWAP"

    def test_slice_qty_varies(self, vwap_result):
        qtys = [s.qty for s in vwap_result.slices]
        assert max(qtys) > min(qtys)   # VWAP varies qty per slice

    def test_total_qty_conserved(self, buy_order, vwap_result):
        total = sum(s.qty for s in vwap_result.slices)
        assert abs(total - buy_order.total_qty) < 1e-4

    def test_exec_price_positive(self, vwap_result):
        assert vwap_result.avg_exec_price > 0


# ---------------------------------------------------------------------------
# 5. IS executor
# ---------------------------------------------------------------------------
class TestIS:
    def test_returns_result(self, is_result):
        assert isinstance(is_result, ExecutionResult)

    def test_algo_label(self, is_result):
        assert is_result.algo == "IS"

    def test_front_loaded(self, is_result):
        # IS should have more volume early (slice 0 > slice -1)
        assert is_result.slices[0].qty > is_result.slices[-1].qty

    def test_total_qty_conserved(self, buy_order, is_result):
        total = sum(s.qty for s in is_result.slices)
        assert abs(total - buy_order.total_qty) < 1e-4


# ---------------------------------------------------------------------------
# 6. Compare algos
# ---------------------------------------------------------------------------
class TestCompareAlgos:
    def test_returns_df_and_results(self, buy_order, price_data):
        prices, spread, ts = price_data
        df, results = compare_algos(buy_order, prices, spread, ts)
        assert isinstance(df, pd.DataFrame)
        assert len(results) == 3

    def test_all_algos_present(self, buy_order, price_data):
        prices, spread, ts = price_data
        df, _ = compare_algos(buy_order, prices, spread, ts)
        assert "TWAP" in df.index
        assert "VWAP" in df.index
        assert "IS"   in df.index

    def test_summary_keys(self, twap_result):
        s = execution_summary(twap_result)
        for k in ["algo","is_bps","total_cost_bps","participation_rate"]:
            assert k in s
