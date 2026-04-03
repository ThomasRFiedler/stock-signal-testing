"""
Cross-ticker overfitting check.

Optimizes strategy parameters on a single "train ticker" then evaluates
those same parameters on a list of out-of-sample "test tickers" that
were never seen during optimization.

If the strategy is genuine, performance should degrade only modestly on
unseen tickers.  A large gap (train Sharpe >> test Sharpe across all
test tickers) is a strong signal of over-fitting.

Usage
-----
    from signal_testing.cross_validate import cross_validate

    result = cross_validate(
        train_ticker="AAPL",
        test_tickers=["MSFT", "NVDA", "SPY", "QQQ"],
        time_frame="60d",
        interval="5m",
    )
"""

import pandas as pd

from .data import fetch_data
from .backtest import backtest
from .optimizer import run_optimization
from .results import save_params


def cross_validate(
    train_ticker: str,
    test_tickers: list,
    time_frame: str = "60d",
    interval: str = "5m",
    extra_train_tickers: list = None,
    n_generations: int = 200,
    popsize: int = 50,
    position_size: float = 100.0,
    starting_equity: float = 10_000.0,
    min_trades: int = 15,
    reg_lambda: float = 0.01,
    patience: int = 20,
    min_delta: float = 0.05,
    preloaded_params: dict = None,
    save_train_params: str = None,
) -> pd.DataFrame:
    """
    Optimize on *train_ticker* (+ optional extra tickers), evaluate on
    each ticker in *test_tickers*.

    Parameters
    ----------
    train_ticker : str
        Primary ticker used for optimization (the "in-sample" instrument).
    test_tickers : list[str]
        Tickers to evaluate the optimized parameters on (out-of-sample).
        Must not overlap with the train ticker(s).
    time_frame : str
        History window: "1w", "60d", or "1y".  Same window used for all
        tickers so they cover the same calendar period.
    interval : str
        Bar interval: "1m", "5m", "15m", "30m", "1h".
    extra_train_tickers : list[str], optional
        Additional tickers included in the optimization.  Fitness becomes
        the average Sharpe across train_ticker + extra_train_tickers.
        Passing diverse instruments (e.g. ["MSFT", "SPY"]) is the most
        effective anti-overfitting measure.
    n_generations : int
        Maximum SNES generations.  Early stopping may halt sooner.
    popsize : int
        SNES population size.
    position_size : float
    starting_equity : float
    min_trades : int
        Minimum trades per training ticker required for non-penalty fitness.
    reg_lambda : float
        L2 regularization on indicator weights (default 0.01).
    patience : int
        Early stopping patience in generations (default 20).
    min_delta : float
        Minimum fitness improvement to reset early-stopping counter.
    preloaded_params : dict, optional
        If provided, skip optimization and use these params directly.
        Expects keys: weights, stop_loss, take_profit, n.
    save_train_params : str, optional
        If provided, save the optimized params to this path.

    Returns
    -------
    pd.DataFrame
        One row per ticker (train ticker(s) first, then test tickers) with
        columns: ticker, role, sharpe_ratio, pnl, total_trades,
                 profit_factor, max_drawdown, stop_loss, take_profit, n.
    """
    extra_train = list(extra_train_tickers) if extra_train_tickers else []
    all_train_tickers = [train_ticker] + extra_train

    # ------------------------------------------------------------------
    # Step 1 – Optimize on train ticker(s) (or use supplied params)
    # ------------------------------------------------------------------
    if preloaded_params is not None:
        opt = preloaded_params
        opt.setdefault("sharpe", float("nan"))
        print(f"\nUsing pre-loaded parameters (skipping optimization).")
        print(f"  stop_loss={opt['stop_loss']:.4f}  "
              f"take_profit={opt['take_profit']:.4f}  n={opt['n']:.4f}")
    else:
        print(f"\nOptimizing on: {all_train_tickers} "
              f"({time_frame} @ {interval}, "
              f"{n_generations} generations × {popsize} population)...")
        opt = run_optimization(
            train_tickers=all_train_tickers,
            time_frame=time_frame,
            interval=interval,
            n_generations=n_generations,
            popsize=popsize,
            position_size=position_size,
            starting_equity=starting_equity,
            min_trades=min_trades,
            reg_lambda=reg_lambda,
            patience=patience,
            min_delta=min_delta,
        )
        gens = opt.get("generations_run", n_generations)
        early = " (early stop)" if opt.get("stopped_early") else ""
        print(f"  Optimization complete: {gens} generations{early}, "
              f"best fitness={opt['sharpe']:.4f}")

        if save_train_params:
            save_params(opt, save_train_params)
            print(f"  Train params saved to {save_train_params}")

    weights     = opt["weights"]
    stop_loss   = opt["stop_loss"]
    take_profit = opt["take_profit"]
    n           = opt["n"]

    # ------------------------------------------------------------------
    # Step 2 – Evaluate on every ticker (train + test)
    # ------------------------------------------------------------------
    train_set   = set(all_train_tickers)
    all_tickers = all_train_tickers + [t for t in test_tickers if t not in train_set]
    records     = []

    for ticker in all_tickers:
        role = "TRAIN" if ticker in train_set else "TEST"
        print(f"\n  [{role}] Fetching and backtesting {ticker}...")

        try:
            data = fetch_data(ticker, time_frame, interval)
            result = backtest(
                ticker=ticker,
                n=n,
                time_frame=time_frame,
                interval=interval,
                take_profit=take_profit,
                stop_loss=stop_loss,
                weights=weights,
                position_size=position_size,
                starting_equity=starting_equity,
                preloaded_data=data,
                verbose=False,
            )
            records.append({
                "ticker":        ticker,
                "role":          role,
                "sharpe_ratio":  result["sharpe_ratio"],
                "pnl":           result["pnl"],
                "total_trades":  result["total_trades"],
                "profit_factor": result["profit_factor"],
                "max_drawdown":  result["max_drawdown"],
                "stop_loss":     round(stop_loss, 5),
                "take_profit":   round(take_profit, 5),
                "n":             round(n, 4),
            })
        except Exception as e:
            print(f"    WARNING: {ticker} failed — {e}")
            records.append({
                "ticker":        ticker,
                "role":          role,
                "sharpe_ratio":  float("nan"),
                "pnl":           float("nan"),
                "total_trades":  0,
                "profit_factor": float("nan"),
                "max_drawdown":  float("nan"),
                "stop_loss":     round(stop_loss, 5),
                "take_profit":   round(take_profit, 5),
                "n":             round(n, 4),
            })

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Step 3 – Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"  CROSS-TICKER VALIDATION SUMMARY")
    print(f"  Optimized on: {all_train_tickers}  |  "
          f"Time frame: {time_frame}  |  Interval: {interval}")
    print(f"  Parameters:  stop_loss={stop_loss:.4f}  "
          f"take_profit={take_profit:.4f}  n={n:.4f}")
    print("=" * 75)
    print(f"  {'Ticker':<8} {'Role':<6} {'Sharpe':>8} {'P/L ($)':>10} "
          f"{'Trades':>7} {'PF':>6} {'MaxDD ($)':>10}")
    print("  " + "-" * 63)

    train_sharpe = df.loc[df["role"] == "TRAIN", "sharpe_ratio"].mean()
    for _, row in df.iterrows():
        flag = ""
        if row["role"] == "TEST" and not pd.isna(row["sharpe_ratio"]):
            degradation = train_sharpe - row["sharpe_ratio"]
            if degradation > 1.5:
                flag = "  << large drop"
            elif degradation > 0.75:
                flag = "  < moderate drop"
        pf_str = f"{row['profit_factor']:.2f}" if not pd.isna(row["profit_factor"]) else "  N/A"
        print(f"  {row['ticker']:<8} {row['role']:<6} "
              f"{row['sharpe_ratio']:>8.4f} {row['pnl']:>10.2f} "
              f"{row['total_trades']:>7} {pf_str:>6} "
              f"{row['max_drawdown']:>10.2f}{flag}")

    test_rows = df[df["role"] == "TEST"]
    valid_test = test_rows.dropna(subset=["sharpe_ratio"])
    if len(valid_test) > 0:
        avg_test_sharpe = valid_test["sharpe_ratio"].mean()
        sharpe_gap = train_sharpe - avg_test_sharpe
        print("  " + "-" * 63)
        print(f"  {'Train Sharpe':<20} {train_sharpe:>8.4f}")
        print(f"  {'Avg Test Sharpe':<20} {avg_test_sharpe:>8.4f}")
        print(f"  {'Gap (overfit proxy)':<20} {sharpe_gap:>8.4f}", end="")
        if sharpe_gap > 1.5:
            print("  *** LIKELY OVERFIT ***")
        elif sharpe_gap > 0.75:
            print("  ** some degradation, review indicators **")
        else:
            print("  (good generalization)")

    print("=" * 75)

    return df
