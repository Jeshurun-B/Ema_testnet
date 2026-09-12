# Ema_testnet
This project is to stress-test the AI powered simple ema strategy on the actual real world trading environment

# EMA-Testnet-Production-v1

[![Architecture: Production](https://img.shields.io/badge/Architecture-Dual--Engine%20Production-blue.svg)](https://github.com/Jeshurun-B/ema_testnet)
[![Exchange: Binance Futures](https://img.shields.io/badge/Exchange-Binance%20Futures%20Testnet-yellow.svg)](https://testnet.binancefuture.com)
[![Models: CatBoost & PyTorch](https://img.shields.io/badge/Models-CatBoost%20%7C%20PyTorch%20GRU-orange.svg)]()
[![Database: Supabase](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-green.svg)](https://supabase.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

Institutional-grade, autonomous algorithmic trading pipeline designed for cryptocurrency perpetual futures (`BTCUSDT`, `DOGEUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`) on 15-minute candle intervals. 

Driven by a **Dual-Engine Model Architecture** combining 24 CatBoost continuous regressors and 24 PyTorch Funnel GRU classifiers, the system operates on a **1.0x unleveraged, volatility-budgeted framework** with native exchange resting brackets and real-time friction accounting.

---

## Key Performance Benchmarks (1,000-Candle Untouched Holdout)

Evaluated across the untouched 1,000-candle simulation holdout (August 1 to September 6, 2026) under full institutional friction (**$0.08\%$ taker fees $+ 0.10\%$ adverse slippage**):

| Metric | Benchmark Milestone | Production System (`SC_24` Pure Static) | Production System (`SC_14` Trailing) |
| :--- | :---: | :---: | :---: |
| **Trade Win Rate** | 33.50% *(Zero Filters)* | **36.20% – 50.68%** | **52.39%** |
| **Profit Factor** | 0.83 *(Negative Edge)* | **1.72 – 2.41** | **2.15** |
| **Annualized Sharpe Ratio** | -1.13 | **+3.57 – +6.37** | **+5.29** |
| **Maximum Portfolio Drawdown** | 3.13% | **0.29% – 0.30%** | **0.41%** |
| **Net Realized PnL ($10k Capital)** | -$157.07 | **+$136.77 to +$242.20** | **+$297.65** |
| **Execution Latency** | N/A | **< 5.50 Seconds** | **< 5.50 Seconds** |

---

## Architectural Division of Labor
