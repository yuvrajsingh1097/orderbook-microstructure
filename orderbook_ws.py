"""
Orderbook Feed Simulator — L2 Limit Order Book
================================================
Simulates a Level-2 orderbook with realistic market microstructure:

    - Bid/ask price levels with size at each level
    - Market orders, limit orders, cancellations
    - Tick-by-tick trade events
    - Synthetic feed with configurable regimes

Data structures:
    OrderbookSnapshot : one point-in-time L2 snapshot
    TradeEvent        : individual trade (price, size, side)
    OrderbookFeed     : generator of snapshots + trades

Realistic properties simulated:
    - Bid-ask spread mean-reverts around a base spread
    - Depth follows log-normal distribution
    - Trade arrival: Poisson process
    - Price follows geometric Brownian motion with microstructure noise
    - Regime shifts: normal / stressed / illiquid
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterator
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PriceLevel:
    """A single price level in the orderbook."""
    price: float
    size:  float   # total quantity available


@dataclass
class OrderbookSnapshot:
    """
    A single L2 orderbook snapshot.

    bids: list of PriceLevel, sorted descending (best bid first)
    asks: list of PriceLevel, sorted ascending  (best ask first)
    """
    timestamp:  float          # unix ms
    bids:       list           # [PriceLevel, ...]
    asks:       list           # [PriceLevel, ...]
    mid_price:  float = 0.0
    spread:     float = 0.0
    imbalance:  float = 0.0    # (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def __post_init__(self):
        if self.bids and self.asks:
            self.mid_price = (self.bids[0].price + self.asks[0].price) / 2
            self.spread    = self.asks[0].price - self.bids[0].price
            bid_vol = sum(l.size for l in self.bids)
            ask_vol = sum(l.size for l in self.asks)
            denom   = bid_vol + ask_vol
            self.imbalance = (bid_vol - ask_vol) / denom if denom > 0 else 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def bid_depth(self) -> float:
        return sum(l.size for l in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(l.size for l in self.asks)

    @property
    def total_depth(self) -> float:
        return self.bid_depth + self.ask_depth


@dataclass
class TradeEvent:
    """A single trade execution."""
    timestamp: float
    price:     float
    size:      float
    side:      str       # 'buy' | 'sell'  (aggressor side)
    is_buy:    bool = True

    def __post_init__(self):
        self.is_buy = (self.side == 'buy')


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

REGIMES = {
    "normal": dict(
        vol=0.0005,        # per-tick vol
        spread_bps=2.0,    # base spread in bps
        depth_mean=500.0,  # mean size per level
        trade_rate=5.0,    # trades per second
        spread_noise=0.3,
    ),
    "stressed": dict(
        vol=0.002,
        spread_bps=8.0,
        depth_mean=150.0,
        trade_rate=15.0,
        spread_noise=0.8,
    ),
    "illiquid": dict(
        vol=0.0008,
        spread_bps=15.0,
        depth_mean=50.0,
        trade_rate=1.0,
        spread_noise=1.2,
    ),
}


# ---------------------------------------------------------------------------
# Orderbook Feed Simulator
# ---------------------------------------------------------------------------

class OrderbookFeedSimulator:
    """
    Generates a realistic synthetic L2 orderbook feed.

    Parameters
    ----------
    ticker      : asset symbol
    start_price : initial mid price
    n_levels    : number of price levels each side
    tick_size   : minimum price increment
    n_events    : total events to generate
    seed        : random seed
    regime_schedule : list of (start_event, regime_name) pairs
    """

    def __init__(
        self,
        ticker:          str   = "AAPL",
        start_price:     float = 100.0,
        n_levels:        int   = 10,
        tick_size:       float = 0.01,
        n_events:        int   = 10_000,
        seed:            int   = 42,
        regime_schedule: list  = None,
    ):
        self.ticker      = ticker
        self.price       = start_price
        self.n_levels    = n_levels
        self.tick_size   = tick_size
        self.n_events    = n_events
        self.rng         = np.random.default_rng(seed)
        self.regime_schedule = regime_schedule or [(0, "normal"), (4000, "stressed"), (7000, "normal")]

        self._snapshots: list = []
        self._trades:    list = []
        self._generated  = False

    def _get_regime(self, event_idx: int) -> dict:
        """Return regime parameters for a given event index."""
        regime_name = "normal"
        for start, name in self.regime_schedule:
            if event_idx >= start:
                regime_name = name
        return REGIMES[regime_name], regime_name

    def _make_snapshot(
        self,
        timestamp: float,
        mid: float,
        spread_bps: float,
        depth_mean: float,
        spread_noise: float,
    ) -> OrderbookSnapshot:
        """Build a single L2 snapshot around a given mid price."""
        half_spread = mid * spread_bps / 10_000 / 2
        half_spread *= (1 + self.rng.exponential(spread_noise))
        half_spread  = max(half_spread, self.tick_size)

        best_bid = round(mid - half_spread, 4)
        best_ask = round(mid + half_spread, 4)

        bids, asks = [], []
        for i in range(self.n_levels):
            bid_price  = round(best_bid - i * self.tick_size, 4)
            ask_price  = round(best_ask + i * self.tick_size, 4)
            bid_size   = max(1.0, self.rng.lognormal(np.log(depth_mean), 0.6))
            ask_size   = max(1.0, self.rng.lognormal(np.log(depth_mean), 0.6))
            bids.append(PriceLevel(bid_price, round(bid_size, 1)))
            asks.append(PriceLevel(ask_price, round(ask_size, 1)))

        return OrderbookSnapshot(timestamp=timestamp, bids=bids, asks=asks)

    def generate(self) -> "OrderbookFeedSimulator":
        """Generate full feed. Returns self for chaining."""
        snapshots, trades = [], []
        t = 0.0   # timestamp in seconds

        for i in range(self.n_events):
            params, regime = self._get_regime(i)

            # Price evolution: GBM with microstructure noise
            dt   = 1.0 / params["trade_rate"]
            dW   = self.rng.standard_normal() * params["vol"]
            self.price = max(0.01, self.price * np.exp(dW))
            t   += dt + self.rng.exponential(dt * 0.2)

            snap = self._make_snapshot(
                timestamp=round(t * 1000, 1),   # ms
                mid=self.price,
                spread_bps=params["spread_bps"],
                depth_mean=params["depth_mean"],
                spread_noise=params["spread_noise"],
            )
            snap._regime = regime
            snapshots.append(snap)

            # Generate trade events (Poisson arrival)
            n_trades = self.rng.poisson(0.4)
            for _ in range(n_trades):
                side  = "buy" if self.rng.random() > 0.5 else "sell"
                price = snap.best_ask if side == "buy" else snap.best_bid
                size  = max(1.0, self.rng.lognormal(np.log(params["depth_mean"] * 0.1), 0.5))
                trades.append(TradeEvent(
                    timestamp=round(t * 1000 + self.rng.uniform(0, dt * 1000), 1),
                    price=round(price, 4),
                    size=round(size, 1),
                    side=side,
                ))

        self._snapshots = snapshots
        self._trades    = trades
        self._generated = True
        return self

    @property
    def snapshots(self) -> list:
        if not self._generated:
            self.generate()
        return self._snapshots

    @property
    def trades(self) -> list:
        if not self._generated:
            self.generate()
        return self._trades

    def to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to a flat DataFrame for analysis."""
        rows = []
        for snap in self.snapshots:
            rows.append({
                "timestamp":   snap.timestamp,
                "mid_price":   snap.mid_price,
                "best_bid":    snap.best_bid,
                "best_ask":    snap.best_ask,
                "spread":      round(snap.spread, 6),
                "spread_bps":  round(snap.spread / snap.mid_price * 10_000, 4) if snap.mid_price > 0 else 0,
                "imbalance":   round(snap.imbalance, 6),
                "bid_depth":   round(snap.bid_depth, 1),
                "ask_depth":   round(snap.ask_depth, 1),
                "total_depth": round(snap.total_depth, 1),
                "regime":      getattr(snap, "_regime", "normal"),
            })
        return pd.DataFrame(rows)

    def trades_dataframe(self) -> pd.DataFrame:
        """Convert trade events to DataFrame."""
        rows = [{"timestamp": t.timestamp, "price": t.price,
                 "size": t.size, "side": t.side, "is_buy": t.is_buy}
                for t in self.trades]
        return pd.DataFrame(rows)

    def stats(self) -> dict:
        """Summary statistics of the generated feed."""
        df = self.to_dataframe()
        return {
            "n_snapshots":    len(self._snapshots),
            "n_trades":       len(self._trades),
            "price_start":    round(df["mid_price"].iloc[0], 4),
            "price_end":      round(df["mid_price"].iloc[-1], 4),
            "price_range":    round(df["mid_price"].max() - df["mid_price"].min(), 4),
            "mean_spread_bps":round(df["spread_bps"].mean(), 4),
            "mean_imbalance": round(df["imbalance"].mean(), 4),
            "mean_bid_depth": round(df["bid_depth"].mean(), 1),
            "regimes":        df["regime"].value_counts().to_dict(),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("Orderbook Feed Simulator Demo")
    print("=" * 55)

    feed = OrderbookFeedSimulator(
        ticker="AAPL", start_price=185.0,
        n_levels=5, n_events=5_000, seed=42,
    )
    feed.generate()

    df = feed.to_dataframe()
    td = feed.trades_dataframe()

    print(f"\n  Feed stats:")
    for k, v in feed.stats().items():
        print(f"    {k:<20}: {v}")

    print(f"\n  Sample snapshot:")
    snap = feed.snapshots[0]
    print(f"    mid={snap.mid_price:.4f}  spread={snap.spread:.4f}  imbalance={snap.imbalance:.4f}")
    print(f"    best bid={snap.best_bid:.4f}  best ask={snap.best_ask:.4f}")
    print(f"    bid depth={snap.bid_depth:.0f}  ask depth={snap.ask_depth:.0f}")

    print(f"\n  Sample trades (first 3):")
    for t in feed.trades[:3]:
        print(f"    {t.side:4s}  price={t.price:.4f}  size={t.size:.1f}")
