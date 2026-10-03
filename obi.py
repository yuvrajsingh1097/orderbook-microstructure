"""
Order Book Imbalance (OBI) — Signals & Alpha Generation
=========================================================
Computes microstructure signals from L2 orderbook depth.

Signals:
    1. OBI (Order Book Imbalance)
       (bid_vol - ask_vol) / (bid_vol + ask_vol)  ∈ [-1, +1]
       Positive → buy pressure → price likely to rise

    2. Weighted OBI (depth-weighted, closer levels count more)
       Σ w_i * (bid_i - ask_i) / Σ w_i * (bid_i + ask_i)
       w_i = 1 / (1 + i)  [exponential decay by level]

    3. Queue Imbalance (top-of-book pressure)
       (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)

    4. Trade Flow Imbalance (TFI)
       Rolling (buy_trades - sell_trades) / total_trades

    5. Depth Ratio
       bid_depth / ask_depth  — simple ratio form

Alpha signal construction:
    - Combine OBI + VPIN + momentum for direction signal
    - Signal decay modelling (alpha decays over time)
    - Threshold crossing strategy
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
class OBISignal:
    """All imbalance signals for a single snapshot."""
    timestamp:        float
    obi:              float    # raw OBI ∈ [-1,+1]
    weighted_obi:     float    # depth-weighted OBI
    queue_imbalance:  float    # top-of-book only
    depth_ratio:      float    # bid_depth / ask_depth
    mid_price:        float
    spread_bps:       float


@dataclass
class AlphaSignal:
    """Combined directional alpha signal."""
    timestamp:   float
    raw_signal:  float    # composite ∈ [-1,+1]
    decayed:     float    # after signal decay
    direction:   int      # +1 buy / -1 sell / 0 flat
    confidence:  float    # ∈ [0,1]
    components:  dict     # breakdown by signal source


# ---------------------------------------------------------------------------
# OBI calculators
# ---------------------------------------------------------------------------

def compute_obi(snapshot) -> float:
    """
    Raw Order Book Imbalance from a snapshot.
    (bid_vol - ask_vol) / (bid_vol + ask_vol)
    """
    bid_vol = snapshot.bid_depth
    ask_vol = snapshot.ask_depth
    denom   = bid_vol + ask_vol
    if denom < 1e-8:
        return 0.0
    return float((bid_vol - ask_vol) / denom)


def compute_weighted_obi(snapshot, decay: float = 0.7) -> float:
    """
    Weighted OBI: deeper levels weighted less by exponential decay.
    w_i = decay^i  (level 0 = best bid/ask, highest weight)
    """
    bid_wt = sum(decay**i * l.size for i, l in enumerate(snapshot.bids))
    ask_wt = sum(decay**i * l.size for i, l in enumerate(snapshot.asks))
    denom  = bid_wt + ask_wt
    if denom < 1e-8:
        return 0.0
    return float((bid_wt - ask_wt) / denom)


def compute_queue_imbalance(snapshot) -> float:
    """Top-of-book queue imbalance (best bid vs best ask size only)."""
    if not snapshot.bids or not snapshot.asks:
        return 0.0
    bb_sz = snapshot.bids[0].size
    ba_sz = snapshot.asks[0].size
    denom = bb_sz + ba_sz
    if denom < 1e-8:
        return 0.0
    return float((bb_sz - ba_sz) / denom)


def compute_depth_ratio(snapshot) -> float:
    """bid_depth / ask_depth ratio (>1 means more buy support)."""
    if snapshot.ask_depth < 1e-8:
        return 1.0
    return float(snapshot.bid_depth / snapshot.ask_depth)


# ---------------------------------------------------------------------------
# Signal series builder
# ---------------------------------------------------------------------------

class OBIEngine:
    """
    Computes the full suite of OBI signals from a sequence of snapshots.
    Also builds rolling/lagged features for ML.
    """

    def __init__(self, decay: float = 0.7):
        self.decay = decay

    def compute_signals(self, snapshots: list) -> list:
        """
        Compute OBISignal for each snapshot in the list.
        Returns list of OBISignal objects.
        """
        signals = []
        for snap in snapshots:
            obi    = compute_obi(snap)
            w_obi  = compute_weighted_obi(snap, self.decay)
            qi     = compute_queue_imbalance(snap)
            dr     = compute_depth_ratio(snap)
            spr    = snap.spread / snap.mid_price * 10_000 if snap.mid_price > 0 else 0

            signals.append(OBISignal(
                timestamp=snap.timestamp,
                obi=round(obi, 6),
                weighted_obi=round(w_obi, 6),
                queue_imbalance=round(qi, 6),
                depth_ratio=round(dr, 6),
                mid_price=round(snap.mid_price, 6),
                spread_bps=round(spr, 4),
            ))
        return signals

    def to_dataframe(self, signals: list) -> pd.DataFrame:
        """Convert list of OBISignal to DataFrame."""
        return pd.DataFrame([{
            "timestamp":       s.timestamp,
            "obi":             s.obi,
            "weighted_obi":    s.weighted_obi,
            "queue_imbalance": s.queue_imbalance,
            "depth_ratio":     s.depth_ratio,
            "mid_price":       s.mid_price,
            "spread_bps":      s.spread_bps,
        } for s in signals])

    def rolling_features(
        self,
        df: pd.DataFrame,
        windows: list = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        Add rolling mean and std features for OBI signals.
        Useful for ML feature engineering.
        """
        base_cols = ["obi","weighted_obi","queue_imbalance","depth_ratio"]
        result = df.copy()

        for col in base_cols:
            if col not in result.columns:
                continue
            for w in windows:
                result[f"{col}_ma{w}"]  = result[col].rolling(w, min_periods=1).mean()
                result[f"{col}_std{w}"] = result[col].rolling(w, min_periods=1).std().fillna(0)

        # Mid-price returns
        result["return_1"] = result["mid_price"].pct_change(1).fillna(0)
        result["return_5"] = result["mid_price"].pct_change(5).fillna(0)

        # Spread change
        result["spread_change"] = result["spread_bps"].diff().fillna(0)

        return result.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trade Flow Imbalance
# ---------------------------------------------------------------------------

def compute_tfi(
    trades_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Trade Flow Imbalance (TFI): rolling (buy - sell) / total trades.

    Aligned to snapshot timestamps via nearest-time matching.
    Returns a Series aligned to snapshot index.
    """
    if len(trades_df) == 0:
        return pd.Series(np.zeros(len(snapshots_df)))

    trades = trades_df.copy()
    trades["buy_flag"]  = (trades["side"] == "buy").astype(float)
    trades["sell_flag"] = (trades["side"] == "sell").astype(float)

    # Rolling TFI on trade stream
    trades["tfi_raw"] = (
        trades["buy_flag"].rolling(window, min_periods=1).sum() -
        trades["sell_flag"].rolling(window, min_periods=1).sum()
    ) / window

    # Align to snapshot timestamps (nearest trade)
    snap_ts   = snapshots_df["timestamp"].values
    trade_ts  = trades["timestamp"].values
    tfi_vals  = trades["tfi_raw"].values

    tfi_aligned = np.interp(snap_ts, trade_ts, tfi_vals)
    return pd.Series(tfi_aligned, index=snapshots_df.index)


# ---------------------------------------------------------------------------
# Alpha signal generator
# ---------------------------------------------------------------------------

class AlphaGenerator:
    """
    Combines OBI, VPIN, TFI and momentum into a directional alpha signal.

    Signal = w_obi*OBI + w_vpin*(0.5-VPIN)*2 + w_tfi*TFI + w_mom*MOM
    (VPIN is inverted: high VPIN = uncertain direction → reduces signal)

    Parameters
    ----------
    w_obi   : weight for OBI signal
    w_vpin  : weight for VPIN toxicity adjustment
    w_tfi   : weight for Trade Flow Imbalance
    w_mom   : weight for price momentum
    decay   : exponential signal decay per step
    threshold : abs(signal) must exceed this to generate direction
    """

    def __init__(
        self,
        w_obi:     float = 0.40,
        w_vpin:    float = 0.20,
        w_tfi:     float = 0.25,
        w_mom:     float = 0.15,
        decay:     float = 0.95,
        threshold: float = 0.10,
    ):
        self.w_obi     = w_obi
        self.w_vpin    = w_vpin
        self.w_tfi     = w_tfi
        self.w_mom     = w_mom
        self.decay     = decay
        self.threshold = threshold

    def generate(
        self,
        obi_df:   pd.DataFrame,
        vpin_ts:  Optional[np.ndarray] = None,
        tfi_ts:   Optional[pd.Series]  = None,
        mom_window: int = 10,
    ) -> list:
        """
        Generate alpha signals from feature DataFrame.

        Parameters
        ----------
        obi_df   : DataFrame from OBIEngine.to_dataframe()
        vpin_ts  : optional VPIN time series (aligned to obi_df)
        tfi_ts   : optional TFI series (aligned to obi_df)

        Returns list of AlphaSignal objects.
        """
        n = len(obi_df)
        if n == 0:
            return []

        # Momentum: rolling return
        prices = obi_df["mid_price"].values
        mom    = pd.Series(prices).pct_change(mom_window).fillna(0).values

        # VPIN component (if available, else zeros)
        if vpin_ts is not None and len(vpin_ts) == n:
            vpin_comp = (0.5 - np.clip(vpin_ts, 0, 1)) * 2   # flip: high VPIN → reduce
        else:
            vpin_comp = np.zeros(n)

        # TFI component
        if tfi_ts is not None and len(tfi_ts) == n:
            tfi_comp = tfi_ts.values
        else:
            tfi_comp = np.zeros(n)

        signals   = []
        prev_decayed = 0.0

        for i in range(n):
            obi_val = obi_df["weighted_obi"].iloc[i]
            raw = (
                self.w_obi  * obi_val   +
                self.w_vpin * float(vpin_comp[i]) +
                self.w_tfi  * float(tfi_comp[i])  +
                self.w_mom  * float(mom[i]) * 100   # scale momentum
            )
            raw = float(np.clip(raw, -1, 1))

            # Signal decay
            decayed = self.decay * prev_decayed + (1 - self.decay) * raw
            prev_decayed = decayed

            # Direction
            if abs(decayed) > self.threshold:
                direction = int(np.sign(decayed))
            else:
                direction = 0

            confidence = min(abs(decayed) / max(self.threshold, 1e-8), 1.0)

            signals.append(AlphaSignal(
                timestamp=float(obi_df["timestamp"].iloc[i]),
                raw_signal=round(raw, 6),
                decayed=round(float(decayed), 6),
                direction=direction,
                confidence=round(float(confidence), 4),
                components={
                    "obi":      round(self.w_obi  * obi_val, 6),
                    "vpin_adj": round(self.w_vpin * float(vpin_comp[i]), 6),
                    "tfi":      round(self.w_tfi  * float(tfi_comp[i]), 6),
                    "momentum": round(self.w_mom  * float(mom[i]) * 100, 6),
                },
            ))

        return signals

    def backtest_signal(
        self,
        signals: list,
        prices:  np.ndarray,
        hold_periods: int = 5,
    ) -> dict:
        """
        Simple PnL backtest of the alpha signal.
        Enter on direction signal, hold for hold_periods, then exit.

        Returns dict with win_rate, avg_pnl, sharpe.
        """
        if len(signals) == 0 or len(prices) < 2:
            return {"win_rate": 0.0, "avg_pnl": 0.0, "sharpe": 0.0}

        pnls = []
        n    = min(len(signals), len(prices))

        for i in range(n - hold_periods):
            sig = signals[i]
            if sig.direction == 0:
                continue
            entry  = prices[i]
            exit_  = prices[min(i + hold_periods, n - 1)]
            ret    = (exit_ - entry) / (entry + 1e-8)
            pnl    = sig.direction * ret
            pnls.append(pnl)

        if not pnls:
            return {"win_rate": 0.0, "avg_pnl": 0.0, "sharpe": 0.0}

        pnls   = np.array(pnls)
        sharpe = float(pnls.mean() / (pnls.std() + 1e-8)) * np.sqrt(252)

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
    from microstructure.vpin import VPINCalculator

    print("=" * 55)
    print("OBI Signals & Alpha Generation Demo")
    print("=" * 55)

    feed = OrderbookFeedSimulator(n_events=3000, seed=42)
    feed.generate()

    # OBI signals
    engine  = OBIEngine(decay=0.7)
    signals = engine.compute_signals(feed.snapshots)
    obi_df  = engine.to_dataframe(signals)
    feat_df = engine.rolling_features(obi_df)

    print(f"\n  OBI signals: {len(signals):,}")
    print(f"  Feature cols: {feat_df.shape[1]}")
    print(f"  Mean OBI:         {obi_df['obi'].mean():.4f}")
    print(f"  Mean Weighted OBI:{obi_df['weighted_obi'].mean():.4f}")
    print(f"  Mean Queue Imbal: {obi_df['queue_imbalance'].mean():.4f}")

    # TFI
    snaps_df = feed.to_dataframe()
    trades_df = feed.trades_dataframe()
    tfi = compute_tfi(trades_df, snaps_df, window=15)
    print(f"\n  TFI mean: {tfi.mean():.4f}  std: {tfi.std():.4f}")

    # VPIN for alpha
    calc   = VPINCalculator(bucket_size=400.0, window=20)
    result = calc.compute(trades_df, snaps_df)
    vpin_aligned = np.interp(
        obi_df["timestamp"].values,
        result.timestamps[~np.isnan(result.vpin)],
        result.vpin[~np.isnan(result.vpin)],
    ) if (~np.isnan(result.vpin)).any() else np.full(len(obi_df), 0.3)

    # Alpha
    gen      = AlphaGenerator()
    alphas   = gen.generate(obi_df, vpin_aligned, tfi)
    bt       = gen.backtest_signal(alphas, obi_df["mid_price"].values)

    directions = [a.direction for a in alphas]
    print(f"\n  Alpha signals: {len(alphas):,}")
    print(f"  Buy  signals : {directions.count(1)}")
    print(f"  Sell signals : {directions.count(-1)}")
    print(f"  Flat signals : {directions.count(0)}")
    print(f"\n  Backtest (hold=5 steps):")
    for k, v in bt.items():
        print(f"    {k:<12}: {v}")
