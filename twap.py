"""
TWAP / VWAP Execution Engine
==============================
Simulates institutional order execution with realistic market impact.

Algorithms:
    1. TWAP  — Time-Weighted Average Price
               Split order equally across N time slices
    2. VWAP  — Volume-Weighted Average Price
               Schedule slices proportional to historical volume profile
    3. IS    — Implementation Shortfall
               Minimise deviation from arrival price (aggressive early)
    4. POV   — Participation of Volume
               Execute at fixed % of market volume each period

Market impact model (Almgren-Chriss):
    Temporary impact : η * (v / ADV)^0.6  [in bps]
    Permanent impact : γ * (V / ADV)       [price shift]

Slippage model:
    Bid-ask spread cost: spread_bps / 2
    Market impact cost : Almgren-Chriss
    Timing risk        : volatility × sqrt(T)

Metrics reported:
    - Arrival price vs execution price (IS bps)
    - Slippage vs TWAP benchmark
    - Slippage vs VWAP benchmark
    - Market impact cost
    - Participation rate
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """A parent order to be executed."""
    order_id:    str
    side:        str        # 'buy' | 'sell'
    total_qty:   float      # total shares/contracts to execute
    arrival_px:  float      # price at order arrival
    urgency:     str = "normal"   # low | normal | high


@dataclass
class SliceExecution:
    """A single child order (slice) execution."""
    slice_id:      int
    timestamp:     float
    qty:           float
    exec_price:    float     # actual execution price
    arrival_price: float     # price at parent order arrival
    market_price:  float     # mid-price at execution time
    spread_cost:   float     # bps paid crossing spread
    impact_cost:   float     # bps of market impact
    total_cost_bps:float     # spread + impact


@dataclass
class ExecutionResult:
    """Full execution summary for a parent order."""
    order:            Order
    slices:           list        # list of SliceExecution
    algo:             str
    avg_exec_price:   float
    arrival_price:    float
    twap_benchmark:   float
    vwap_benchmark:   float
    is_bps:           float       # Implementation Shortfall in bps
    slippage_vs_twap: float       # bps vs TWAP
    slippage_vs_vwap: float       # bps vs VWAP
    total_cost_bps:   float
    participation_rate: float
    n_slices:         int
    duration_s:       float


# ---------------------------------------------------------------------------
# Market impact model
# ---------------------------------------------------------------------------

class AlmgrenChrissImpact:
    """
    Almgren-Chriss (2001) market impact model.

    Temporary impact:  η * (v/ADV)^0.6   (per-slice, in bps)
    Permanent impact:  γ * (V/ADV)        (cumulative, in bps)

    Parameters
    ----------
    eta   : temporary impact coefficient (default 0.1)
    gamma : permanent impact coefficient (default 0.05)
    adv   : average daily volume (shares)
    """

    def __init__(
        self,
        eta:   float = 0.1,
        gamma: float = 0.05,
        adv:   float = 1_000_000.0,
    ):
        self.eta   = eta
        self.gamma = gamma
        self.adv   = adv

    def temporary_impact_bps(self, slice_qty: float, duration_s: float = 60.0) -> float:
        """Temporary impact for a single slice execution."""
        rate = slice_qty / (self.adv * duration_s / 23_400)   # annualised rate
        impact = self.eta * (rate ** 0.6) * 10_000
        return round(float(impact), 4)

    def permanent_impact_bps(self, total_qty: float) -> float:
        """Permanent (lasting) impact of the entire order."""
        impact = self.gamma * (total_qty / self.adv) * 10_000
        return round(float(impact), 4)

    def total_impact_bps(self, slice_qty: float, total_qty: float, duration_s: float = 60.0) -> float:
        return self.temporary_impact_bps(slice_qty, duration_s) + self.permanent_impact_bps(total_qty)


# ---------------------------------------------------------------------------
# Volume profile for VWAP scheduling
# ---------------------------------------------------------------------------

def intraday_volume_profile(n_periods: int = 20, seed: int = 0) -> np.ndarray:
    """
    Simulate intraday U-shaped volume profile.
    Higher volume at open and close, lower midday.
    Returns normalised weights summing to 1.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 1, n_periods)
    # U-shape: higher at open and close
    profile = 0.5 * np.exp(-8 * (t - 0)**2) + 0.5 * np.exp(-8 * (t - 1)**2) + 0.1
    # Add noise
    profile += rng.uniform(0, 0.05, n_periods)
    return profile / profile.sum()


# ---------------------------------------------------------------------------
# TWAP executor
# ---------------------------------------------------------------------------

class TWAPExecutor:
    """
    Time-Weighted Average Price execution.
    Splits order into equal-sized slices over a fixed duration.

    Parameters
    ----------
    n_slices    : number of child orders
    duration_s  : total execution duration in seconds
    impact_model: AlmgrenChrissImpact instance
    """

    def __init__(
        self,
        n_slices:     int   = 20,
        duration_s:   float = 1200.0,   # 20 minutes
        impact_model: AlmgrenChrissImpact = None,
    ):
        self.n_slices    = n_slices
        self.duration_s  = duration_s
        self.impact_model = impact_model or AlmgrenChrissImpact()

    def execute(
        self,
        order:       Order,
        price_path:  np.ndarray,   # mid-price at each slice time
        spread_bps:  np.ndarray,   # spread in bps at each slice
        timestamps:  np.ndarray,   # timestamp for each slice
    ) -> ExecutionResult:
        """
        Simulate TWAP execution.

        Parameters
        ----------
        order      : Order to execute
        price_path : mid-price observed at each slice (length = n_slices)
        spread_bps : bid-ask spread in bps at each slice
        timestamps : timestamp (ms) at each slice

        Returns ExecutionResult.
        """
        n       = min(self.n_slices, len(price_path))
        qty_per = order.total_qty / n
        sign    = 1 if order.side == "buy" else -1

        slices = []
        total_exec_qty   = 0.0
        total_exec_notional = 0.0
        slice_dur = self.duration_s / n

        for i in range(n):
            mkt_px   = float(price_path[i])
            spr_bps  = float(spread_bps[i]) if i < len(spread_bps) else 2.0

            # Spread cost (half spread paid on crossing)
            spread_cost = spr_bps / 2.0

            # Market impact
            impact = self.impact_model.temporary_impact_bps(qty_per, slice_dur)

            # Execution price: buy pays spread + impact premium
            exec_px = mkt_px * (1 + sign * (spread_cost + impact) / 10_000)

            total_cost = spread_cost + impact

            slices.append(SliceExecution(
                slice_id=i,
                timestamp=float(timestamps[i]) if i < len(timestamps) else float(i),
                qty=qty_per,
                exec_price=round(exec_px, 6),
                arrival_price=order.arrival_px,
                market_price=mkt_px,
                spread_cost=round(spread_cost, 4),
                impact_cost=round(impact, 4),
                total_cost_bps=round(total_cost, 4),
            ))
            total_exec_qty     += qty_per
            total_exec_notional += qty_per * exec_px

        avg_exec = total_exec_notional / max(total_exec_qty, 1e-8)
        twap_bm  = float(np.mean(price_path[:n]))
        vwap_bm  = twap_bm   # same for TWAP algo

        # IS = (avg_exec - arrival_px) / arrival_px * sign * 10000
        is_bps = sign * (avg_exec - order.arrival_px) / order.arrival_px * 10_000

        return ExecutionResult(
            order=order,
            slices=slices,
            algo="TWAP",
            avg_exec_price=round(avg_exec, 6),
            arrival_price=order.arrival_px,
            twap_benchmark=round(twap_bm, 6),
            vwap_benchmark=round(vwap_bm, 6),
            is_bps=round(is_bps, 4),
            slippage_vs_twap=round(sign * (avg_exec - twap_bm) / twap_bm * 10_000, 4),
            slippage_vs_vwap=round(sign * (avg_exec - vwap_bm) / vwap_bm * 10_000, 4),
            total_cost_bps=round(sum(s.total_cost_bps for s in slices) / n, 4),
            participation_rate=round(total_exec_qty / max(order.total_qty, 1e-8), 4),
            n_slices=n,
            duration_s=self.duration_s,
        )


# ---------------------------------------------------------------------------
# VWAP executor
# ---------------------------------------------------------------------------

class VWAPExecutor:
    """
    Volume-Weighted Average Price execution.
    Schedules slices proportional to expected volume profile.
    """

    def __init__(
        self,
        n_slices:     int   = 20,
        duration_s:   float = 1200.0,
        impact_model: AlmgrenChrissImpact = None,
        seed:         int   = 42,
    ):
        self.n_slices    = n_slices
        self.duration_s  = duration_s
        self.impact_model = impact_model or AlmgrenChrissImpact()
        self.vol_profile  = intraday_volume_profile(n_slices, seed)

    def execute(
        self,
        order:      Order,
        price_path: np.ndarray,
        spread_bps: np.ndarray,
        timestamps: np.ndarray,
        mkt_volumes: Optional[np.ndarray] = None,
    ) -> ExecutionResult:
        """Simulate VWAP execution with volume-scheduled slices."""
        n    = min(self.n_slices, len(price_path))
        sign = 1 if order.side == "buy" else -1

        # Schedule quantities by volume profile
        qty_schedule = order.total_qty * self.vol_profile[:n]
        qty_schedule = qty_schedule / qty_schedule.sum() * order.total_qty

        slices = []
        total_exec_qty = total_exec_notional = 0.0
        slice_dur = self.duration_s / n

        for i in range(n):
            mkt_px   = float(price_path[i])
            spr_bps  = float(spread_bps[i]) if i < len(spread_bps) else 2.0
            qty_i    = float(qty_schedule[i])

            spread_cost = spr_bps / 2.0
            impact      = self.impact_model.temporary_impact_bps(qty_i, slice_dur)
            exec_px     = mkt_px * (1 + sign * (spread_cost + impact) / 10_000)
            total_cost  = spread_cost + impact

            slices.append(SliceExecution(
                slice_id=i,
                timestamp=float(timestamps[i]) if i < len(timestamps) else float(i),
                qty=qty_i,
                exec_price=round(exec_px, 6),
                arrival_price=order.arrival_px,
                market_price=mkt_px,
                spread_cost=round(spread_cost, 4),
                impact_cost=round(impact, 4),
                total_cost_bps=round(total_cost, 4),
            ))
            total_exec_qty     += qty_i
            total_exec_notional += qty_i * exec_px

        avg_exec = total_exec_notional / max(total_exec_qty, 1e-8)

        # VWAP benchmark: volume-weighted average of market prices
        vwap_bm = float(np.average(price_path[:n], weights=self.vol_profile[:n]))
        twap_bm = float(np.mean(price_path[:n]))

        is_bps = sign * (avg_exec - order.arrival_px) / order.arrival_px * 10_000

        return ExecutionResult(
            order=order,
            slices=slices,
            algo="VWAP",
            avg_exec_price=round(avg_exec, 6),
            arrival_price=order.arrival_px,
            twap_benchmark=round(twap_bm, 6),
            vwap_benchmark=round(vwap_bm, 6),
            is_bps=round(is_bps, 4),
            slippage_vs_twap=round(sign * (avg_exec - twap_bm) / twap_bm * 10_000, 4),
            slippage_vs_vwap=round(sign * (avg_exec - vwap_bm) / vwap_bm * 10_000, 4),
            total_cost_bps=round(sum(s.total_cost_bps for s in slices) / n, 4),
            participation_rate=round(total_exec_qty / max(order.total_qty, 1e-8), 4),
            n_slices=n,
            duration_s=self.duration_s,
        )


# ---------------------------------------------------------------------------
# Implementation Shortfall executor
# ---------------------------------------------------------------------------

class ISExecutor:
    """
    Implementation Shortfall executor.
    Trades more aggressively early to reduce timing risk.
    Uses exponential decay schedule: more volume early.
    """

    def __init__(
        self,
        n_slices:     int   = 20,
        duration_s:   float = 1200.0,
        urgency:      float = 0.5,    # 0=patient, 1=aggressive
        impact_model: AlmgrenChrissImpact = None,
    ):
        self.n_slices    = n_slices
        self.duration_s  = duration_s
        self.urgency     = urgency
        self.impact_model = impact_model or AlmgrenChrissImpact()

    def _is_schedule(self, n: int) -> np.ndarray:
        """Front-loaded schedule: more volume in early slices."""
        decay = 1.0 + self.urgency * 2
        weights = np.exp(-decay * np.linspace(0, 1, n))
        return weights / weights.sum()

    def execute(
        self,
        order:      Order,
        price_path: np.ndarray,
        spread_bps: np.ndarray,
        timestamps: np.ndarray,
    ) -> ExecutionResult:
        n    = min(self.n_slices, len(price_path))
        sign = 1 if order.side == "buy" else -1
        qty_schedule = order.total_qty * self._is_schedule(n)
        slice_dur    = self.duration_s / n

        slices = []
        total_exec_qty = total_exec_notional = 0.0

        for i in range(n):
            mkt_px   = float(price_path[i])
            spr_bps  = float(spread_bps[i]) if i < len(spread_bps) else 2.0
            qty_i    = float(qty_schedule[i])

            spread_cost = spr_bps / 2.0
            impact      = self.impact_model.temporary_impact_bps(qty_i, slice_dur)
            exec_px     = mkt_px * (1 + sign * (spread_cost + impact) / 10_000)
            total_cost  = spread_cost + impact

            slices.append(SliceExecution(
                slice_id=i,
                timestamp=float(timestamps[i]) if i < len(timestamps) else float(i),
                qty=qty_i,
                exec_price=round(exec_px, 6),
                arrival_price=order.arrival_px,
                market_price=mkt_px,
                spread_cost=round(spread_cost, 4),
                impact_cost=round(impact, 4),
                total_cost_bps=round(total_cost, 4),
            ))
            total_exec_qty     += qty_i
            total_exec_notional += qty_i * exec_px

        avg_exec = total_exec_notional / max(total_exec_qty, 1e-8)
        twap_bm  = float(np.mean(price_path[:n]))
        vwap_bm  = twap_bm
        is_bps   = sign * (avg_exec - order.arrival_px) / order.arrival_px * 10_000

        return ExecutionResult(
            order=order, slices=slices, algo="IS",
            avg_exec_price=round(avg_exec, 6),
            arrival_price=order.arrival_px,
            twap_benchmark=round(twap_bm, 6),
            vwap_benchmark=round(vwap_bm, 6),
            is_bps=round(is_bps, 4),
            slippage_vs_twap=round(sign*(avg_exec-twap_bm)/twap_bm*10_000, 4),
            slippage_vs_vwap=round(sign*(avg_exec-vwap_bm)/vwap_bm*10_000, 4),
            total_cost_bps=round(sum(s.total_cost_bps for s in slices)/n, 4),
            participation_rate=round(total_exec_qty/max(order.total_qty,1e-8), 4),
            n_slices=n, duration_s=self.duration_s,
        )


# ---------------------------------------------------------------------------
# Execution analyser
# ---------------------------------------------------------------------------

def execution_summary(result: ExecutionResult) -> dict:
    """Return flat summary dict for a single execution result."""
    costs = [s.total_cost_bps for s in result.slices]
    return {
        "algo":              result.algo,
        "n_slices":          result.n_slices,
        "avg_exec_price":    result.avg_exec_price,
        "arrival_price":     result.arrival_price,
        "is_bps":            result.is_bps,
        "slippage_vs_twap":  result.slippage_vs_twap,
        "slippage_vs_vwap":  result.slippage_vs_vwap,
        "total_cost_bps":    result.total_cost_bps,
        "participation_rate":result.participation_rate,
        "max_slice_cost_bps":round(float(max(costs)), 4) if costs else 0,
        "min_slice_cost_bps":round(float(min(costs)), 4) if costs else 0,
    }


def compare_algos(
    order:      Order,
    price_path: np.ndarray,
    spread_bps: np.ndarray,
    timestamps: np.ndarray,
    n_slices:   int = 20,
) -> pd.DataFrame:
    """Run TWAP, VWAP, IS on same order and return comparison DataFrame."""
    impact = AlmgrenChrissImpact()
    results = [
        TWAPExecutor(n_slices, impact_model=impact).execute(order, price_path, spread_bps, timestamps),
        VWAPExecutor(n_slices, impact_model=impact).execute(order, price_path, spread_bps, timestamps),
        ISExecutor(n_slices, urgency=0.5, impact_model=impact).execute(order, price_path, spread_bps, timestamps),
    ]
    rows = [execution_summary(r) for r in results]
    return pd.DataFrame(rows).set_index("algo"), results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feed.orderbook_ws import OrderbookFeedSimulator

    print("=" * 55)
    print("TWAP/VWAP Execution Engine Demo")
    print("=" * 55)

    feed = OrderbookFeedSimulator(n_events=2000, seed=42, start_price=185.0)
    feed.generate()
    df   = feed.to_dataframe()

    n_slices   = 20
    step       = max(1, len(df) // n_slices)
    price_path = df["mid_price"].values[::step][:n_slices]
    spread_bps = df["spread_bps"].values[::step][:n_slices]
    timestamps = df["timestamp"].values[::step][:n_slices]

    order = Order(
        order_id="ORD001", side="buy",
        total_qty=10_000, arrival_px=float(price_path[0]),
    )

    print(f"\n  Order: {order.side.upper()} {order.total_qty:,} shares @ arrival={order.arrival_px:.4f}")

    cmp_df, results = compare_algos(order, price_path, spread_bps, timestamps, n_slices)
    print(f"\n  Algo Comparison:")
    print(cmp_df[["is_bps","slippage_vs_twap","total_cost_bps","participation_rate"]].to_string())

    for res in results:
        print(f"\n  [{res.algo}] avg_exec={res.avg_exec_price:.4f}  "
              f"IS={res.is_bps:.2f}bps  cost={res.total_cost_bps:.2f}bps")
