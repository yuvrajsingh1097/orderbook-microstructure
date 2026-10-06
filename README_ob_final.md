# Orderbook Microstructure & VPIN Engine

Real-time L2 orderbook analysis pipeline with VPIN toxicity detection, OBI alpha signals, TWAP/VWAP/IS execution engine, and ML trade direction prediction.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-143%20passing-brightgreen)](#testing)

---

## Architecture

```
L2 Orderbook Feed (WebSocket / Synthetic)
    ↓
Snapshots + Trades (bid/ask levels, depth, imbalance)
    ↓
┌──────────────┬────────────────┬──────────────┬─────────────────┐
│ VPIN Engine  │ OBI Signals    │ Execution    │ ML Predictor    │
│ BVC classify │ Weighted OBI   │ TWAP/VWAP/IS │ Random Forest   │
│ Vol buckets  │ Queue imbal.   │ Almgren-     │ 70 microstruc.  │
│ Toxicity     │ Alpha decay    │ Chriss impact│ features        │
└──────────────┴────────────────┴──────────────┴─────────────────┘
    ↓
Streamlit Dashboard
```

---

## Modules

| File | Description |
|------|-------------|
| `feed/orderbook_ws.py` | L2 feed simulator, bid/ask depth, trade events, regime shifts |
| `microstructure/vpin.py` | VPIN (Easley et al. 2012): BVC/tick/Lee-Ready, volume buckets, toxicity alert |
| `microstructure/obi.py` | OBI, weighted OBI, queue imbalance, TFI, alpha generation, backtest |
| `execution/twap.py` | TWAP, VWAP, IS executors with Almgren-Chriss market impact |
| `ml/direction_model.py` | RF/LR direction model, 70 features, walk-forward backtest |
| `dashboard/app.py` | Streamlit dashboard — 5 pages |

---

## Results

| Module | Key Result |
|--------|-----------|
| VPIN (BVC) | Mean VPIN ≈ 0.27, P95 alert threshold ≈ 0.45 |
| OBI Alpha | Win rate ≈ 51%, Sharpe ≈ 0.04 (microstructure noise) |
| TWAP IS | ≈ 211 bps above arrival (spread + impact) |
| RF Direction | Accuracy ≈ 0.50, AUC ≈ 0.51 (near-random, realistic) |

---

## Output Samples

![Dashboard](outputs/dashboard_preview.png)
![Direction Model](outputs/direction_model.png)
![Execution Engine](outputs/execution_engine.png)
![OBI Signals](outputs/obi_signals.png)
![VPIN Engine](outputs/vpin_engine.png)
![Orderbook Feed](outputs/orderbook_feed.png)

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/orderbook-microstructure
cd orderbook-microstructure
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## License
MIT
