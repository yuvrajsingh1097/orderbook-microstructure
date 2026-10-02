"""
VPIN — Volume-Synchronized Probability of Informed Trading
===========================================================
Easley, López de Prado & O'Hara (2012) implementation.

Key concepts:
    Volume bucket : a fixed quantum of traded volume V*
    Buy volume     : fraction of V* attributed to buyer-initiated trades
    Sell volume    : V* minus buy volume
    VPIN           : avg |buy_vol - sell_vol| / V* over trailing n buckets

Pipeline:
    Trade ticks → volume buckets → classify buy/sell →
    compute |ΔV| per bucket → rolling mean → VPIN ∈ [0, 1]

Buy/sell classification methods:
    1. Tick rule   : up-tick → buy, down-tick → sell
    2. Lee-Ready   : compare trade price to mid-quote
    3. Bulk volume  : use normal CDF of price change (BVC method)

Higher VPIN → more informed trading → higher adverse selection risk.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VolumeBucket:
    """One VPIN volume bucket."""
    bucket_id:   int
    start_idx:   int
    end_idx:     int
    volume:      float     # total volume (≈ V*)
    buy_volume:  float
    sell_volume: float
    vwap:        float     # volume-weighted avg price
    start_time:  float
    end_time:    float

    @property
    def order_imbalance(self) -> float:
        return abs(self.buy_volume - self.sell_volume) / (self.volume + 1e-8)


@dataclass
class VPINResult:
    """VPIN time series result."""
    timestamps:   np.ndarray   # one per bucket
    vpin:         np.ndarray   # VPIN ∈ [0,1]
    buckets:      list         # list of VolumeBucket
    bucket_size:  float
    n_buckets:    int
    window:       int


# ---------------------------------------------------------------------------
# Buy/sell classifiers
# ---------------------------------------------------------------------------

def classify_tick_rule(prices: np.ndarray) -> np.ndarray:
    """
    Tick rule: buy if price increased vs previous trade, sell otherwise.
    First trade classified as buy.

    Returns array of floats ∈ {0, 1} (1 = buy).
    """
    n     = len(prices)
    buys  = np.zeros(n, dtype=np.float32)
    buys[0] = 0.5   # ambiguous first trade → split 50/50

    for i in range(1, n):
        if prices[i] > prices[i-1]:
            buys[i] = 1.0
        elif prices[i] < prices[i-1]:
            buys[i] = 0.0
        else:
            buys[i] = buys[i-1]   # last tick rule for ties
    return buys


def classify_lee_ready(
    prices:    np.ndarray,
    mid_prices: np.ndarray,
) -> np.ndarray:
    """
    Lee-Ready rule: compare trade price to prevailing mid-quote.
        price > mid → buy
        price < mid → sell
        price == mid → tick rule fallback
    """
    n    = len(prices)
    buys = np.zeros(n, dtype=np.float32)

    for i in range(n):
        mid = mid_prices[i] if i < len(mid_prices) else 0
        if prices[i] > mid:
            buys[i] = 1.0
        elif prices[i] < mid:
            buys[i] = 0.0
        else:
            # Tie-break with tick rule
            if i > 0:
                buys[i] = buys[i-1]
            else:
                buys[i] = 0.5
    return buys


def classify_bulk_volume(
    prices:  np.ndarray,
    volumes: np.ndarray,
    window:  int = 1,
) -> np.ndarray:
    """
    Bulk Volume Classification (BVC) — Easley, López de Prado & O'Hara 2012.

    Z_t = ΔP_t / σ(ΔP, window)
    buy_fraction = Φ(Z_t)   [normal CDF]

    Returns array of buy fractions ∈ [0, 1].
    """
    n       = len(prices)
    dp      = np.diff(prices, prepend=prices[0])
    sigma   = pd.Series(dp).rolling(window, min_periods=1).std().fillna(1e-8).values
    sigma   = np.where(sigma < 1e-8, 1e-8, sigma)
    Z       = dp / sigma
    buy_frac = norm.cdf(Z).astype(np.float32)
    return buy_frac


# ---------------------------------------------------------------------------
# Volume bucket builder
# ---------------------------------------------------------------------------

def build_volume_buckets(
    prices:    np.ndarray,
    volumes:   np.ndarray,
    buy_fracs: np.ndarray,
    timestamps: np.ndarray,
    bucket_size: float,
) -> list:
    """
    Aggregate trades into fixed-volume buckets of size `bucket_size`.

    Each bucket accumulates trades until total volume ≥ bucket_size.
    Fractional trades are split across bucket boundaries.

    Returns list of VolumeBucket objects.
    """
    buckets      = []
    bucket_id    = 0
    accum_vol    = 0.0
    accum_buy    = 0.0
    accum_sell   = 0.0
    vwap_num     = 0.0
    start_idx    = 0
    start_time   = timestamps[0] if len(timestamps) > 0 else 0.0

    n = len(prices)
    i = 0

    while i < n:
        remaining = bucket_size - accum_vol
        trade_vol = volumes[i]
        buy_frac  = buy_fracs[i]

        if trade_vol <= remaining:
            # Whole trade fits in current bucket
            accum_vol  += trade_vol
            accum_buy  += trade_vol * buy_frac
            accum_sell += trade_vol * (1 - buy_frac)
            vwap_num   += prices[i] * trade_vol
            i += 1
        else:
            # Split trade: fill current bucket then start new one
            frac      = remaining / trade_vol
            accum_vol  += remaining
            accum_buy  += remaining * buy_frac
            accum_sell += remaining * (1 - buy_frac)
            vwap_num   += prices[i] * remaining

            # Close bucket
            vwap = vwap_num / accum_vol if accum_vol > 0 else prices[i]
            buckets.append(VolumeBucket(
                bucket_id=bucket_id,
                start_idx=start_idx,
                end_idx=i,
                volume=accum_vol,
                buy_volume=accum_buy,
                sell_volume=accum_sell,
                vwap=round(vwap, 6),
                start_time=start_time,
                end_time=timestamps[i] if i < len(timestamps) else 0.0,
            ))

            # Reset for new bucket
            bucket_id  += 1
            accum_vol   = trade_vol * (1 - frac)
            accum_buy   = accum_vol * buy_frac
            accum_sell  = accum_vol * (1 - buy_frac)
            vwap_num    = prices[i] * accum_vol
            start_idx   = i
            start_time  = timestamps[i] if i < len(timestamps) else 0.0
            i += 1

    # Close any remaining partial bucket
    if accum_vol > 0:
        vwap = vwap_num / accum_vol if accum_vol > 0 else prices[-1]
        buckets.append(VolumeBucket(
            bucket_id=bucket_id,
            start_idx=start_idx,
            end_idx=n-1,
            volume=accum_vol,
            buy_volume=accum_buy,
            sell_volume=accum_sell,
            vwap=round(vwap, 6),
            start_time=start_time,
            end_time=timestamps[-1] if len(timestamps) > 0 else 0.0,
        ))

    return buckets


# ---------------------------------------------------------------------------
# VPIN calculator
# ---------------------------------------------------------------------------

class VPINCalculator:
    """
    Computes VPIN from a stream of trade events.

    Parameters
    ----------
    bucket_size    : volume per bucket V* (default: 1/50 of daily volume)
    window         : rolling window of buckets for VPIN (default: 50)
    method         : 'tick' | 'lee_ready' | 'bvc'
    """

    def __init__(
        self,
        bucket_size: float = 1000.0,
        window:      int   = 50,
        method:      str   = "bvc",
    ):
        self.bucket_size = bucket_size
        self.window      = window
        self.method      = method

    def compute(
        self,
        trades_df:    pd.DataFrame,
        snapshots_df: pd.DataFrame = None,
    ) -> VPINResult:
        """
        Compute VPIN from trade tick data.

        Parameters
        ----------
        trades_df    : DataFrame with columns [timestamp, price, size, side]
        snapshots_df : optional DataFrame with [timestamp, mid_price] for Lee-Ready

        Returns VPINResult with full time series.
        """
        if len(trades_df) < 10:
            empty = np.array([])
            return VPINResult(empty, empty, [], self.bucket_size, 0, self.window)

        prices     = trades_df["price"].values.astype(np.float64)
        volumes    = trades_df["size"].values.astype(np.float64)
        timestamps = trades_df["timestamp"].values.astype(np.float64)

        # Buy/sell classification
        if self.method == "tick":
            buy_fracs = classify_tick_rule(prices)

        elif self.method == "lee_ready":
            if snapshots_df is not None and len(snapshots_df) > 0:
                mid_prices = np.interp(
                    timestamps,
                    snapshots_df["timestamp"].values,
                    snapshots_df["mid_price"].values,
                )
            else:
                mid_prices = prices
            buy_fracs = classify_lee_ready(prices, mid_prices)

        else:  # bvc (default)
            buy_fracs = classify_bulk_volume(prices, volumes)

        # Build volume buckets
        buckets = build_volume_buckets(
            prices, volumes, buy_fracs, timestamps, self.bucket_size
        )

        if len(buckets) < 2:
            empty = np.array([])
            return VPINResult(empty, empty, buckets, self.bucket_size, 0, self.window)

        # Compute VPIN as rolling mean of |buy_vol - sell_vol| / bucket_size
        oi_series = np.array([b.order_imbalance for b in buckets])
        ts_series = np.array([b.end_time for b in buckets])

        vpin = np.full(len(oi_series), np.nan)
        for i in range(self.window - 1, len(oi_series)):
            vpin[i] = np.mean(oi_series[i - self.window + 1 : i + 1])

        return VPINResult(
            timestamps=ts_series,
            vpin=vpin,
            buckets=buckets,
            bucket_size=self.bucket_size,
            n_buckets=len(buckets),
            window=self.window,
        )

    def alert_threshold(
        self,
        vpin_series: np.ndarray,
        percentile:  float = 95.0,
    ) -> float:
        """
        Compute VPIN alert threshold at given percentile.
        When VPIN > threshold → high informed trading risk.
        """
        valid = vpin_series[~np.isnan(vpin_series)]
        if len(valid) == 0:
            return 0.5
        return float(np.percentile(valid, percentile))

    def summary_stats(self, result: VPINResult) -> dict:
        """Summary statistics for a VPIN result."""
        valid = result.vpin[~np.isnan(result.vpin)]
        if len(valid) == 0:
            return {}
        oi = np.array([b.order_imbalance for b in result.buckets])
        return {
            "n_buckets":       result.n_buckets,
            "vpin_mean":       round(float(valid.mean()), 4),
            "vpin_std":        round(float(valid.std()), 4),
            "vpin_max":        round(float(valid.max()), 4),
            "vpin_min":        round(float(valid.min()), 4),
            "vpin_p95":        round(float(np.percentile(valid, 95)), 4),
            "mean_oi":         round(float(oi.mean()), 4),
            "pct_high_vpin":   round(float((valid > 0.5).mean() * 100), 2),
            "bucket_size":     result.bucket_size,
            "window":          result.window,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feed.orderbook_ws import OrderbookFeedSimulator

    print("=" * 55)
    print("VPIN Engine Demo")
    print("=" * 55)

    feed = OrderbookFeedSimulator(n_events=5_000, seed=42, start_price=185.0)
    feed.generate()

    trades_df = feed.trades_dataframe()
    snaps_df  = feed.to_dataframe()

    print(f"\n  Trades: {len(trades_df):,}  |  Snapshots: {len(snaps_df):,}")

    for method in ["tick", "lee_ready", "bvc"]:
        calc   = VPINCalculator(bucket_size=500.0, window=20, method=method)
        result = calc.compute(trades_df, snaps_df)
        stats  = calc.summary_stats(result)
        thresh = calc.alert_threshold(result.vpin)
        print(f"\n  [{method}] n_buckets={stats.get('n_buckets',0)}  "
              f"vpin_mean={stats.get('vpin_mean',0):.4f}  "
              f"vpin_max={stats.get('vpin_max',0):.4f}  "
              f"alert_threshold(p95)={thresh:.4f}")
