# Orderbook Microstructure & VPIN Engine

Real-time L2 orderbook analysis pipeline with VPIN (Volume-synchronized Probability of Informed Trading) calculation, OBI (Order Book Imbalance) signals, TWAP/VWAP execution engine, and ML-based trade direction prediction.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-32%20passing-brightgreen)](#testing)

---m

## What this does

| Module | Description |
|--------|-------------|
| `feed/orderbook_ws.py` | L2 orderbook feed simulator with bid/ask levels, trade events, regime shifts |
| `microstructure/vpin.py` | VPIN calculation: volume buckets, buy/sell classification, toxicity score |
| `microstructure/obi.py` | Order Book Imbalance: weighted depth, pressure signals, alpha generation |
| `execution/twap.py` | TWAP/VWAP execution engine with slippage and market impact modelling |
| `ml/direction_model.py` | ML trade direction predictor (Random Forest on microstructure features) |
| `dashboard/app.py` | Streamlit real-time dashboard |

---

## Architecture

```
L2 Feed (WebSocket / Synthetic)
    ↓
Orderbook Snapshot (bid/ask levels, depth, imbalance)
    ↓
┌────────────────┬──────────────────┬────────────────┐
│ VPIN Engine    │ OBI Signal       │ Execution      │
│ Volume buckets │ Depth imbalance  │ TWAP / VWAP    │
│ Trade classify │ Pressure alpha   │ Market impact  │
│ Toxicity score │ Signal decay     │ Slippage model │
└────────────────┴──────────────────┴────────────────┘
    ↓
ML Direction Model → trade signals
```

---

## Output Samples

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
