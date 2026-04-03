"""
Walk-forward optimizer for continuous market adaptation.

Splits a historical dataset into rolling train/test windows and re-runs
SNES optimization on each training window, evaluating out-of-sample on
the following test window.

This is the primary mechanism for adapting to a chaotic market: optimal
parameters discovered in a recent training window are applied forward
and then re-optimized as new data arrives.

      Timeline:  |-----TRAIN-----|--TEST--|
                      slide      |-----TRAIN-----|--TEST--|
                                     slide       |-----TRAIN-----|--TEST--|
"""

import os
import pandas as pd

from .data import fetch_data
from .backtest import backtest
from .optimizer import run_optimization
from .results import save_params


def _slice_data(full_data: dict, start_idx: int, end_idx: int) -> dict:
    """
    Return a copy of full_data sliced to bar indices [start_idx:end_idx].

    All daily DataFrames (daily_price, vix, spx_daily) are filtered by
    the date range covered by the sliced intraday price bars to prevent
    look-ahead bias.
    """
    price_slice = full_data["price"].iloc[start_idx:end_idx]
    if price_slice.empty:
        raise ValueError(f"Empty price slice for indices [{start_idx}:{end_idx}].")

    date_min = price_slice.index.normalize().min()
    date_max = price_slice.index.normalize().max()

    def _filter_daily(df):
        if df is None or df.empty:
            return df
        idx = pd.to_datetime(df.index).normalize()
        return df[(idx >= date_min.tz_localize(None)) & (idx <= date_max.tz_localize(None))]

    spx_price_slice = full_data["spx_price"]
    if not spx_price_slice.empty:
        spx_price_slice = spx_price_slice[
            (spx_price_slice.index >= price_slice.index.min()) &
            (spx_price_slice.index <= price_slice.index.max())
        ]

    return {
        "price":        price_slice,
        "daily_price":  _filter_daily(full_data["daily_price"]),
        "vix":          _filter_daily(full_data["vix"]),
        "spx_price":    spx_price_slice,
        "spx_daily":    _filter_daily(full_data["spx_daily"]),
        "fundamentals": full_data["fundamentals"],
    }


def walk_forward_optimize(
    ticker: str,
    interval: str,
    train_bars: int = 1560,
    test_bars: int = 390,
    step_bars: int = 390,
    n_generations: int = 150,
    popsize: int = 50,
    position_size: float = 100.0,
    starting_equity: float = 10_000.0,
    min_trades: int = 5,
    save_dir: str = "results",
) -> pd.DataFrame:
    """
    Run walk-forward optimization across the full available history.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    interval : str
        Bar interval ("1m", "5m", "15m", "30m", "1h").
    train_bars : int
        Number of bars in each training window.
        Default 1560 ≈ 4 weeks of 5m bars (4 × 5 days × 78 bars/day).
    test_bars : int
        Number of bars in each out-of-sample test window.
        Default 390 ≈ 1 week of 5m bars.
    step_bars : int
        How far to advance the window each iteration.
        Default equals test_bars (non-overlapping test windows).
    n_generations : int
        SNES generations per optimization window.
    popsize : int
        SNES population size.
    position_size : float
    starting_equity : float
    min_trades : int
        Minimum trades for a valid fitness evaluation.
    save_dir : str
        Directory for persisting per-window best parameters (JSON).

    Returns
    -------
    pd.DataFrame
        One row per walk-forward window with columns:
        window, train_start, train_end, test_start, test_end,
        weights, stop_loss, take_profit, n,
        train_sharpe, test_sharpe, test_pnl, test_trades.
        Also saved to {save_dir}/walk_forward_{ticker}.csv.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nFetching 1-year history for {ticker} at {interval} interval...")
    full_data  = fetch_data(ticker, time_frame="1y", interval=interval)
    total_bars = len(full_data["price"])
    print(f"Total bars available: {total_bars}")

    if total_bars < train_bars + test_bars:
        raise ValueError(
            f"Not enough bars ({total_bars}) for train_bars={train_bars} + "
            f"test_bars={test_bars}. Reduce window sizes or use a longer interval."
        )

    records     = []
    window_idx  = 0
    train_start = 0

    while train_start + train_bars + test_bars <= total_bars:
        train_end = train_start + train_bars
        test_end  = train_end   + test_bars

        price_df   = full_data["price"]
        train_data = _slice_data(full_data, train_start, train_end)
        test_data  = _slice_data(full_data, train_end,   test_end)

        print(f"\n{'=' * 60}")
        print(f"  Walk-Forward Window {window_idx}")
        print(f"  Train: {price_df.index[train_start].date()} → "
              f"{price_df.index[train_end - 1].date()} ({train_bars} bars)")
        print(f"  Test:  {price_df.index[train_end].date()} → "
              f"{price_df.index[test_end - 1].date()} ({test_bars} bars)")
        print(f"{'=' * 60}")

        # Optimize on training slice
        opt = run_optimization(
            ticker=ticker,
            time_frame="1y",       # time_frame is unused when preloaded_data is given
            interval=interval,
            preloaded_data=train_data,
            n_generations=n_generations,
            popsize=popsize,
            position_size=position_size,
            starting_equity=starting_equity,
            min_trades=min_trades,
        )

        # Evaluate best parameters on out-of-sample test slice
        test_result = backtest(
            ticker=ticker,
            n=opt["n"],
            time_frame="1y",
            interval=interval,
            take_profit=opt["take_profit"],
            stop_loss=opt["stop_loss"],
            weights=opt["weights"],
            position_size=position_size,
            starting_equity=starting_equity,
            preloaded_data=test_data,
            verbose=False,
        )

        record = {
            "window":       window_idx,
            "train_start":  price_df.index[train_start].date(),
            "train_end":    price_df.index[train_end - 1].date(),
            "test_start":   price_df.index[train_end].date(),
            "test_end":     price_df.index[test_end - 1].date(),
            "weights":      opt["weights"],
            "stop_loss":    opt["stop_loss"],
            "take_profit":  opt["take_profit"],
            "n":            opt["n"],
            "train_sharpe": opt["sharpe"],
            "test_sharpe":  test_result["sharpe_ratio"],
            "test_pnl":     test_result["pnl"],
            "test_trades":  test_result["total_trades"],
        }
        records.append(record)

        save_params(opt, path=os.path.join(save_dir, f"window_{window_idx:03d}.json"))
        print(f"  Train Sharpe: {opt['sharpe']:.4f}  |  "
              f"Test Sharpe: {test_result['sharpe_ratio']:.4f}  |  "
              f"Test P/L: ${test_result['pnl']:,.2f}  |  "
              f"Test Trades: {test_result['total_trades']}")

        train_start += step_bars
        window_idx  += 1

    wf_df = pd.DataFrame(records)
    out_path = os.path.join(save_dir, f"walk_forward_{ticker}.csv")
    wf_df.to_csv(out_path, index=False)
    print(f"\nWalk-forward results saved to {out_path}")
    return wf_df
