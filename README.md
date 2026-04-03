# Stock Signal Testing

A signal-based intraday backtesting framework with evolutionary parameter optimization.

The system backtests a multi-indicator voting strategy, then uses **EvoTorch** (SNES algorithm)
to optimize indicator weights, stop-loss, take-profit, and the signal threshold — continuously
adapting to changing market conditions via walk-forward re-optimization.

---

## Project Structure

```
stock-signal-testing/
├── signal_testing/          # Core package
│   ├── data.py              # fetch_data() — single-batch yahooquery API calls
│   ├── indicators.py        # 14 indicator functions + INDICATORS registry
│   ├── backtest.py          # backtest() — pure simulation engine
│   ├── optimizer.py         # BacktestProblem (EvoTorch) + run_optimization()
│   ├── walk_forward.py      # walk_forward_optimize() — rolling window driver
│   └── results.py           # save/load params, plot_walk_forward()
├── optimize.py              # CLI: single-window optimization
├── run_walk_forward.py      # CLI: walk-forward optimization
├── signal-testing.py        # Legacy monolithic file (kept for reference)
└── requirements.txt
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install PyTorch for your hardware first
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA

pip install -r requirements.txt
```

---

## Quick Start

### Run a standard backtest

```python
from signal_testing import backtest

results = backtest(
    ticker="AAPL",
    n=2,
    time_frame="60d",
    interval="5m",
    take_profit=0.02,
    stop_loss=0.01,
)
print(results["sharpe_ratio"], results["total_trades"])
```

### Run a backtest with custom weights

```python
from signal_testing import backtest, INDICATORS

# Double the weight of RSI (index 1), disable PE ratio (index 8)
weights = [w for _, w in INDICATORS]
weights[1] = 2.0

results = backtest("AAPL", n=2, time_frame="60d", interval="5m",
                   take_profit=0.02, stop_loss=0.01, weights=weights)
```

---

## Evolutionary Optimization

Optimizes all parameters simultaneously using **SNES** (Separable Natural Evolution Strategy)
from EvoTorch. The solution vector encodes:

| Parameter | Bounds | Description |
|-----------|--------|-------------|
| `weights[0..13]` | [0, 5] | Per-indicator weight; 0 disables the indicator |
| `stop_loss` | [0.001, 0.10] | Stop-loss threshold (0.1% – 10%) |
| `take_profit` | [0.001, 0.20] | Take-profit threshold (0.1% – 20%) |
| `n` | [0.5, 8.0] | Weighted signal sum threshold for entry |

Fitness = Sharpe ratio. Solutions with fewer than `min_trades` trades receive a penalty of -10.

### CLI

```bash
# Optimize AAPL over 60 days of 5m bars
python optimize.py AAPL

# More generations, larger population
python optimize.py AAPL --generations 300 --popsize 100

# Save best params to a custom path
python optimize.py AAPL --save results/aapl_best.json
```

### Programmatic

```python
from signal_testing.data import fetch_data
from signal_testing.optimizer import run_optimization
from signal_testing.results import save_params, load_params

data = fetch_data("AAPL", "60d", "5m")

result = run_optimization(
    ticker="AAPL",
    time_frame="60d",
    interval="5m",
    preloaded_data=data,
    n_generations=200,
    popsize=50,
)

save_params(result, "results/aapl_best.json")
print(result["weights"], result["stop_loss"], result["take_profit"], result["n"])
```

---

## Walk-Forward Optimization

Walk-forward re-optimization is the mechanism for adapting to a chaotic market.
The dataset is split into rolling train/test windows. Parameters are optimized on
the training window and evaluated out-of-sample on the test window, then the window
slides forward and the process repeats.

```
Timeline:  |-----TRAIN-----|--TEST--|
               slide       |-----TRAIN-----|--TEST--|
                                slide      |-----TRAIN-----|--TEST--|
```

This prevents over-fitting to any single market regime and measures how well
optimized parameters generalize forward in time.

### CLI

```bash
# Default: 4-week train window, 1-week test window, sliding weekly
python run_walk_forward.py AAPL

# Shorter windows for higher-frequency adaptation
python run_walk_forward.py AAPL --train-bars 780 --test-bars 195 --interval 5m

# Plot train vs. test Sharpe across all windows after completion
python run_walk_forward.py AAPL --plot
```

### Results

Each window's best parameters are saved as `results/window_NNN.json`.
A summary CSV is saved to `results/walk_forward_TICKER.csv`.

To load the latest optimized parameters for live use:

```python
from signal_testing.results import load_latest_params

params = load_latest_params("results/")
results = backtest(
    ticker="AAPL",
    n=params["n"],
    time_frame="1w",
    interval="5m",
    take_profit=params["take_profit"],
    stop_loss=params["stop_loss"],
    weights=params["weights"],
)
```

---

## Indicators

| # | Indicator | Category | Default Weight |
|---|-----------|----------|---------------|
| 0 | SMA Crossover | Technical | 1 |
| 1 | RSI | Technical | 1 |
| 2 | MACD | Technical | 1 |
| 3 | Bollinger Bands | Technical | 1 |
| 4 | VWAP | Technical | 1 |
| 5 | On-Balance Volume | Volume | 1 |
| 6 | Volume Surge | Volume | 1 |
| 7 | VIX Contrarian | Sentiment | 1 |
| 8 | P/E Ratio | Fundamental | **0** (disabled) |
| 9 | Debt-to-Equity | Fundamental | 1 |
| 10 | Short Interest | Fundamental | 1 |
| 11 | Advance/Decline Line | Breadth | 1 |
| 12 | McClellan Oscillator | Breadth | 1 |
| 13 | Relative Strength vs. SPX | Breadth | 1 |

Each indicator returns `1` (bullish), `-1` (bearish), or `0` (neutral).
The weighted sum of all signals is compared against threshold `n` to determine trade entry.

### Adding a new indicator

1. Define `indicator_my_signal(signals, i, **ctx) -> int` in `signal_testing/indicators.py`
2. Add `(indicator_my_signal, 1)` to the `INDICATORS` list
3. The optimizer will automatically include it in the weight vector on the next run

---

## Algorithm: Why SNES?

SNES (Separable Natural Evolution Strategy) is well-suited for this problem because:

- **Noisy fitness**: Sharpe ratio from a short backtest window has high variance. SNES's
  natural gradient update is robust to noise, unlike gradient-based methods.
- **Non-differentiable landscape**: trade entry/exit creates discontinuities in the fitness
  function. Evolution strategies don't require gradients.
- **~17 parameters**: SNES's O(n) diagonal covariance update is efficient at this scale.
- **No hyperparameter tuning**: SNES self-adapts its search distribution width.

EvoTorch also provides CMA-ES and PGPE if you want to experiment. Swap the algorithm
in `signal_testing/optimizer.py` by replacing `SNES(...)` with `CMAES(...)`.
