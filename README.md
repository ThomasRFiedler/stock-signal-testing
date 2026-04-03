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
│   ├── cross_validate.py    # cross_validate() — cross-ticker overfitting check
│   └── results.py           # save/load params, plot_walk_forward()
├── optimize.py              # CLI: single-window optimization
├── run_walk_forward.py      # CLI: walk-forward optimization
├── run_cross_validate.py    # CLI: cross-ticker overfitting check
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

## Cross-Ticker Overfitting Check

After optimizing, it is important to verify that the parameters discovered are
genuinely predictive rather than curve-fitted to one stock's quirks. The cross-ticker
validation tool optimizes on a **train ticker** and immediately evaluates those same
parameters on a set of **test tickers** that were never seen during optimization.

A robust strategy should deliver similar (if degraded) Sharpe ratios on unseen
instruments. A large gap (train Sharpe >> average test Sharpe) is a flag for
over-fitting.

### CLI

```bash
# Optimize on AAPL, test on the defaults (MSFT, NVDA, SPY, QQQ)
python run_cross_validate.py AAPL

# Custom test universe, longer window
python run_cross_validate.py AAPL --test MSFT TSLA SPY IWM --time-frame 1y

# Skip optimization — reuse already-saved params
python run_cross_validate.py AAPL --test MSFT NVDA SPY --load-params results/window_000.json

# Save optimized params and the results table
python run_cross_validate.py AAPL --save results/aapl_cv.json --csv results/cv_results.csv
```

### Programmatic

```python
from signal_testing import cross_validate

df = cross_validate(
    train_ticker="AAPL",
    test_tickers=["MSFT", "NVDA", "SPY", "QQQ"],
    time_frame="60d",
    interval="5m",
    n_generations=200,
    popsize=50,
)
# df has columns: ticker, role, sharpe_ratio, pnl, total_trades,
#                 profit_factor, max_drawdown, stop_loss, take_profit, n
```

### Interpreting results

| Sharpe gap (train − avg test) | Interpretation |
|-------------------------------|---------------|
| < 0.75 | Good generalization — parameters are likely robust |
| 0.75 – 1.5 | Moderate degradation — review high-weight indicators |
| > 1.5 | Likely over-fit — reduce complexity or add more instruments to optimization |

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

---

## Deployment Roadmap (IBKR Paper → Live)

### Phase 1 — Signal validation (current)
- [x] Multi-indicator backtest engine with vectorized pre-computation
- [x] Evolutionary parameter optimizer (SNES via EvoTorch)
- [x] Walk-forward optimization — rolling re-optimization to adapt to market regimes
- [x] Cross-ticker overfitting check — tests parameter generalization on unseen instruments
- [ ] **Expand cross-ticker validation** — run optimization across 5+ diverse instruments
      (large-cap, mid-cap, ETF, sector) and only proceed if avg test Sharpe > 0.5
- [ ] **Stress-test walk-forward** — run across multiple tickers and compare degradation
      patterns to identify which indicators are adding signal vs. noise

### Phase 2 — Live data & execution layer
- [ ] **Add `signal_testing/live_data.py`** — replace yahooquery historical pulls with a
      real-time bar feed.  IBKR provides 5-second snapshots via their API; aggregate to
      the target interval (5m, 15m) and emit a bar event when a candle closes.
      Candidate library: `ib_insync` (async, Pythonic wrapper around the TWS API).
- [ ] **Add `signal_testing/execution.py`** — order management module:
      - `submit_order(ticker, side, qty, order_type)` — wraps `ib_insync` bracket orders
      - `cancel_all_open_orders(ticker)` — safety function
      - Idempotent position tracking: read current IBKR positions before placing to avoid
        doubling up if the process restarts mid-trade.
- [ ] **Port indicators to streaming** — indicators currently work on a full history
      DataFrame.  For live use, each new bar appended to a rolling window (e.g., last 200
      bars) is enough; re-run `compute_signals_matrix` on the window tail.
- [ ] **EOD forced-exit via scheduler** — replace the backtest's next-bar-date check with
      a real-time clock check (e.g., `apscheduler`) that fires a market-on-close order at
      15:55 ET regardless of signal state.

### Phase 3 — Risk management
- [ ] **Position sizing** — replace fixed `position_size` with a volatility-scaled Kelly
      fraction: `position_size = account_equity * kelly_fraction * (1 / realized_vol)`.
- [ ] **Portfolio-level exposure cap** — reject new entries if total open notional >
      `max_portfolio_exposure` (e.g., 20% of account equity) to prevent correlated losses.
- [ ] **Daily loss limit** — halt all new entries for the session if intraday P/L falls
      below a configurable drawdown floor (e.g., -2% of equity).
- [ ] **Slippage and commission model** — add realistic cost estimates to the backtester
      (IBKR tiered: ~$0.005/share, ~$1 minimum) so Sharpe targets are net-of-costs.

### Phase 4 — Infrastructure & monitoring
- [ ] **Structured logging** — replace `print()` calls with Python `logging` to a rotating
      file + stdout; include bar timestamp, signal_sum, position state, and order IDs.
- [ ] **State persistence** — write current position, entry price, and trade_high/low to a
      JSON state file after every bar so the process can resume cleanly after a crash or
      restart.
- [ ] **Health checks** — watchdog thread that alerts (email/Telegram) if:
      - No bar received for > 2× the expected interval
      - IBKR connection drops (TWS disconnect)
      - Daily loss limit triggered
- [ ] **Parameter auto-refresh** — schedule weekly walk-forward re-optimization overnight
      (after market close Friday) and hot-swap the params file so Monday morning uses
      freshly optimized values.

### Phase 5 — Paper trading
- [ ] **Deploy to IBKR paper account** — connect TWS in paper mode, run the live engine
      for at least 4 weeks (≥ one full walk-forward test window).
- [ ] **Track slippage** — compare backtest entry/exit prices to actual fill prices;
      adjust the backtest slippage model if the gap is > 0.05%.
- [ ] **Compare live vs. backtest Sharpe** — expect ~30% degradation from market impact
      and latency; investigate if worse.
- [ ] **Validate re-optimization cadence** — confirm that freshly optimized params improve
      or hold forward performance across at least 4 consecutive walk-forward windows.

### Phase 6 — Live trading switchover
- [ ] **Compliance check** — confirm account type (margin/cash), PDT rule status (> $25k
      equity required for unlimited day trades), and that the instruments traded are
      approved for the account.
- [ ] **Switch TWS connection string from paper to live** — single config change; all
      other code paths are identical.
- [ ] **Start with reduced position sizing** — begin at 25% of target size for the first
      two weeks; scale up after confirming live fills match paper performance.
- [ ] **Circuit breaker** — hard-coded emergency stop: if account equity drops > 5% in a
      single day, the engine exits all positions and halts until manually re-enabled.
