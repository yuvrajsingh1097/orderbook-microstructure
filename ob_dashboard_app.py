"""
Streamlit Dashboard — Orderbook Microstructure & VPIN Engine
=============================================================
Real-time L2 orderbook analysis dashboard.

Pages:
    🏠 Overview      — live feed stats, regime detection
    📊 VPIN          — toxicity score, buy/sell volume buckets
    🔮 OBI & Alpha   — imbalance signals, directional alpha
    ⚙️  Execution     — TWAP/VWAP/IS comparison
    🤖 ML Model      — direction prediction, feature importance

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feed.orderbook_ws import OrderbookFeedSimulator, REGIMES
from microstructure.vpin import VPINCalculator
from microstructure.obi import OBIEngine, AlphaGenerator, compute_tfi
from execution.twap import Order, compare_algos, AlmgrenChrissImpact
from ml.direction_model import (
    build_feature_matrix, build_labels, DirectionModel,
    get_feature_cols, signal_backtest,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Orderbook Microstructure",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.metric-card { background:#161b22; border:1px solid #30363d;
               border-radius:8px; padding:12px; text-align:center; }
.positive { color:#3fb950; font-weight:700; }
.negative { color:#ff7b72; font-weight:700; }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Orderbook Microstructure")
st.sidebar.markdown("---")

n_events   = st.sidebar.slider("Feed events",    1000, 10000, 4000, 500)
start_price= st.sidebar.number_input("Start price ($)", value=185.0, step=1.0)
seed       = st.sidebar.number_input("Seed", value=42, min_value=0)
vpin_bucket= st.sidebar.slider("VPIN bucket size", 100, 2000, 400, 100)
vpin_window= st.sidebar.slider("VPIN window", 5, 50, 20)

st.sidebar.markdown("---")
page = st.sidebar.radio("Page", [
    "🏠 Overview",
    "📊 VPIN",
    "🔮 OBI & Alpha",
    "⚙️ Execution",
    "🤖 ML Model",
])

# ---------------------------------------------------------------------------
# Cached system builder
# ---------------------------------------------------------------------------
@st.cache_resource
def build_system(n_events, start_price, seed, vpin_bucket, vpin_window):
    feed = OrderbookFeedSimulator(
        n_events=n_events, start_price=start_price, seed=seed,
        regime_schedule=[(0,"normal"),(n_events//3,"stressed"),
                         (2*n_events//3,"normal")],
    )
    feed.generate()
    snaps_df  = feed.to_dataframe()
    trades_df = feed.trades_dataframe()

    engine  = OBIEngine()
    obi_sigs= engine.compute_signals(feed.snapshots)
    obi_df  = engine.to_dataframe(obi_sigs)
    tfi     = compute_tfi(trades_df, snaps_df, window=20)

    calc    = VPINCalculator(bucket_size=float(vpin_bucket), window=vpin_window)
    vpin_r  = calc.compute(trades_df, snaps_df)
    valid   = ~np.isnan(vpin_r.vpin)
    vpin_al = np.interp(obi_df["timestamp"].values,
                        vpin_r.timestamps[valid], vpin_r.vpin[valid]
                        ) if valid.any() else np.full(len(obi_df), 0.3)

    gen    = AlphaGenerator()
    alphas = gen.generate(obi_df, vpin_al, tfi)

    feat_df = build_feature_matrix(obi_df, vpin_al, tfi)
    labels  = build_labels(feat_df, horizon=5)
    ml      = min(len(feat_df), len(labels))
    feat_df = feat_df.iloc[:ml].reset_index(drop=True)
    labels  = labels.iloc[:ml].reset_index(drop=True)
    mask    = labels.notna()
    feat_df = feat_df[mask].reset_index(drop=True)
    labels  = labels[mask].reset_index(drop=True)

    sp = int(len(feat_df) * 0.8)
    model = DirectionModel("rf")
    if sp > 20 and labels.iloc[:sp].nunique() > 1:
        model.fit(feat_df.iloc[:sp], labels.iloc[:sp])

    return dict(
        feed=feed, snaps_df=snaps_df, trades_df=trades_df,
        obi_df=obi_df, tfi=tfi,
        vpin_result=vpin_r, vpin_al=vpin_al,
        vpin_stats=calc.summary_stats(vpin_r),
        vpin_thresh=calc.alert_threshold(vpin_r.vpin),
        alphas=alphas, gen=gen,
        feat_df=feat_df, labels=labels, model=model,
        split=sp,
    )

with st.spinner("Building microstructure system..."):
    sys = build_system(n_events, start_price, int(seed), vpin_bucket, vpin_window)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("📈 Orderbook Microstructure & VPIN Engine")
    stats = sys["feed"].stats()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Snapshots",        f"{stats['n_snapshots']:,}")
    c2.metric("Trades",           f"{stats['n_trades']:,}")
    c3.metric("Mean Spread (bps)",f"{stats['mean_spread_bps']:.2f}")
    c4.metric("VPIN Mean",        f"{sys['vpin_stats'].get('vpin_mean',0):.4f}")
    c5.metric("Price Range ($)",  f"{stats['price_range']:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mid Price by Regime")
        df = sys["snaps_df"]
        chart_df = pd.DataFrame({"Mid Price": df["mid_price"].values})
        st.line_chart(chart_df)

    with col2:
        st.subheader("Spread (bps) Over Time")
        st.line_chart(pd.DataFrame({"Spread bps": sys["snaps_df"]["spread_bps"].values}))

    st.markdown("---")
    st.subheader("Regime Distribution")
    reg_counts = sys["snaps_df"]["regime"].value_counts()
    st.bar_chart(reg_counts)

# ---------------------------------------------------------------------------
# Page: VPIN
# ---------------------------------------------------------------------------
elif page == "📊 VPIN":
    st.title("📊 VPIN — Volume-Synchronized Probability of Informed Trading")

    vs = sys["vpin_stats"]
    thresh = sys["vpin_thresh"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("VPIN Mean",   f"{vs.get('vpin_mean',0):.4f}")
    c2.metric("VPIN Max",    f"{vs.get('vpin_max',0):.4f}")
    c3.metric("Alert (p95)", f"{vs.get('vpin_p95',0):.4f}")
    c4.metric("High VPIN%",  f"{vs.get('pct_high_vpin',0):.1f}%")

    st.markdown("---")
    vpin_ts = sys["vpin_result"].vpin
    valid   = ~np.isnan(vpin_ts)
    if valid.any():
        vpin_df = pd.DataFrame({"VPIN": vpin_ts[valid]})
        vpin_df["Alert threshold"] = thresh
        st.subheader("VPIN Time Series")
        st.line_chart(vpin_df)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Buy vs Sell Volume per Bucket")
        buckets = sys["vpin_result"].buckets[:80]
        if buckets:
            bdf = pd.DataFrame({
                "Buy Volume":  [b.buy_volume  for b in buckets],
                "Sell Volume": [b.sell_volume for b in buckets],
            })
            st.bar_chart(bdf)

    with col2:
        st.subheader("VPIN Distribution")
        if valid.any():
            hist_df = pd.DataFrame({"VPIN": vpin_ts[valid]})
            st.dataframe(hist_df.describe().round(4), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: OBI & Alpha
# ---------------------------------------------------------------------------
elif page == "🔮 OBI & Alpha":
    st.title("🔮 OBI Signals & Alpha Generation")

    obi = sys["obi_df"]
    alphas = sys["alphas"]

    dirs = [a.direction for a in alphas]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean OBI",      f"{obi['obi'].mean():.4f}")
    c2.metric("Mean W-OBI",    f"{obi['weighted_obi'].mean():.4f}")
    c3.metric("Buy signals",   str(dirs.count(1)))
    c4.metric("Sell signals",  str(dirs.count(-1)))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("OBI & Weighted OBI")
        st.line_chart(obi[["obi","weighted_obi"]].rename(
            columns={"obi":"OBI","weighted_obi":"Weighted OBI"}))

    with col2:
        st.subheader("Decayed Alpha Signal")
        dec = [a.decayed for a in alphas]
        sig_df = pd.DataFrame({"Alpha": dec})
        st.line_chart(sig_df)

    st.markdown("---")
    st.subheader("Alpha Signal Backtest")
    prices = obi["mid_price"].values
    hold   = st.slider("Hold periods", 1, 20, 5)
    pnls   = []
    for i in range(len(alphas)-hold):
        if alphas[i].direction == 0: continue
        ret = (prices[i+hold]-prices[i])/prices[i]
        pnls.append(alphas[i].direction*ret)
    if pnls:
        cum = np.cumsum(pnls)
        c1,c2,c3 = st.columns(3)
        c1.metric("Trades",   str(len(pnls)))
        c2.metric("Win Rate", f"{(np.array(pnls)>0).mean():.2%}")
        c3.metric("Sharpe",   f"{float(np.mean(pnls)/(np.std(pnls)+1e-8))*np.sqrt(252):.3f}")
        st.line_chart(pd.DataFrame({"Cumulative PnL": cum}))

# ---------------------------------------------------------------------------
# Page: Execution
# ---------------------------------------------------------------------------
elif page == "⚙️ Execution":
    st.title("⚙️ TWAP / VWAP / IS Execution Engine")

    df = sys["snaps_df"]
    n_slices = st.slider("Number of slices", 5, 50, 20)
    total_qty= st.number_input("Order quantity (shares)", value=10_000, step=1_000)
    side     = st.radio("Side", ["buy","sell"], horizontal=True)

    step = max(1, len(df)//n_slices)
    pp   = df["mid_price"].values[::step][:n_slices]
    sp   = df["spread_bps"].values[::step][:n_slices]
    ts   = df["timestamp"].values[::step][:n_slices]
    order = Order("ORD001", side, float(total_qty), float(pp[0]))

    with st.spinner("Running execution simulation..."):
        cmp_df, results = compare_algos(order, pp, sp, ts, n_slices)

    st.subheader("Execution Quality Comparison")
    disp_cols = ["is_bps","slippage_vs_twap","slippage_vs_vwap","total_cost_bps","participation_rate"]
    st.dataframe(
        cmp_df[disp_cols].style.background_gradient(subset=["is_bps","total_cost_bps"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Execution Price vs Market")
        ep_df = pd.DataFrame({"Market": pp})
        for res in results:
            ep_df[res.algo] = [s.exec_price for s in res.slices]
        st.line_chart(ep_df)

    with col2:
        st.subheader("Quantity Schedule")
        qs_df = pd.DataFrame({res.algo: [s.qty for s in res.slices] for res in results})
        st.bar_chart(qs_df)

# ---------------------------------------------------------------------------
# Page: ML Model
# ---------------------------------------------------------------------------
elif page == "🤖 ML Model":
    st.title("🤖 ML Trade Direction Predictor")

    feat_df = sys["feat_df"]
    labels  = sys["labels"]
    model   = sys["model"]
    sp      = sys["split"]

    c1,c2,c3 = st.columns(3)
    c1.metric("Train samples", str(sp))
    c2.metric("Test samples",  str(len(feat_df)-sp))
    c3.metric("Features",      str(len(get_feature_cols(feat_df))))

    if model.is_fitted:
        X_te = feat_df.iloc[sp:]; y_te = labels.iloc[sp:]
        r    = model.evaluate(X_te, y_te)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy", f"{r.accuracy:.4f}")
        c2.metric("F1",       f"{r.f1:.4f}")
        c3.metric("ROC-AUC",  f"{r.roc_auc:.4f}")
        c4.metric("CV Mean",  f"{r.cv_mean:.4f}±{r.cv_std:.4f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Importance (top 15)")
            imp    = r.feature_importance
            top_df = pd.DataFrame(list(imp.items())[:15],
                                  columns=["Feature","Importance"])
            st.bar_chart(top_df.set_index("Feature"))

        with col2:
            st.subheader("Signal Backtest")
            threshold = st.slider("Signal threshold", 0.50, 0.70, 0.55, 0.01)
            hold      = st.slider("Hold steps", 1, 20, 5)
            probas    = model.predict_proba(X_te)
            prices    = X_te["mid_price"].values
            bt = signal_backtest(probas, prices, threshold=threshold, hold_steps=hold)
            bc1,bc2,bc3 = st.columns(3)
            bc1.metric("Trades",   str(bt["n_trades"]))
            bc2.metric("Win Rate", f"{bt['win_rate']:.2%}")
            bc3.metric("Sharpe",   f"{bt['sharpe']:.3f}")

            pnls, cost = [], 2.0/10_000
            for i in range(len(probas)-hold):
                p = probas[i]
                if   p > threshold: pnls.append((prices[i+hold]-prices[i])/prices[i]-cost)
                elif p < 1-threshold: pnls.append((prices[i]-prices[i+hold])/prices[i]-cost)
            if pnls:
                st.line_chart(pd.DataFrame({"Cumulative PnL": np.cumsum(pnls)}))
    else:
        st.warning("Not enough data to train model. Increase feed events.")
