import numpy as np
import pandas as pd

from .data import fetch_data
from .indicators import INDICATORS


def backtest(
    ticker: str,
    n: float,
    time_frame: str,
    interval: str,
    take_profit: float,
    stop_loss: float,
    weights: list = None,
    position_size: float = 100.0,
    starting_equity: float = 10_000.0,
    preloaded_data: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Run a backtest on *ticker* using a multi-indicator voting strategy.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    n : float
        Weighted signal sum threshold to open a position.
        Long when sum(weight * signal) >= n, short when <= -n.
    time_frame : str
        Historical window: "1w", "60d", or "1y".
    interval : str
        Bar interval: "1m", "5m", "15m", "30m", or "1h".
    take_profit : float
        Fractional take-profit threshold (e.g. 0.02 = 2%).
    stop_loss : float
        Fractional stop-loss threshold (e.g. 0.01 = 1%).
    weights : list[float], optional
        Per-indicator weights, one per entry in INDICATORS.
        Defaults to the weights stored in the INDICATORS registry.
    position_size : float
        Dollar amount per trade (default 100).
    starting_equity : float
        Starting account equity in dollars (default 10,000).
    preloaded_data : dict, optional
        Output of fetch_data(). If provided, skips the API call.
        Pass this when calling backtest() in a tight loop (optimizer).
    verbose : bool
        If False, suppresses all console output and chart generation.
        Set to False during optimization to avoid I/O overhead.

    Returns
    -------
    dict with keys:
        pnl, sharpe_ratio, max_drawdown, total_trades, profit_factor,
        trades_df, equity_series, price_df
    """
    # ------------------------------------------------------------------
    # Step 1 – Data
    # ------------------------------------------------------------------
    data = preloaded_data if preloaded_data is not None else fetch_data(ticker, time_frame, interval)

    price        = data["price"]
    daily_price  = data["daily_price"]
    vix          = data["vix"]
    spx_price    = data["spx_price"]
    spx_daily    = data["spx_daily"]
    fundamentals = data["fundamentals"]

    if price.empty:
        raise RuntimeError(f"No price data returned for {ticker}.")

    # ------------------------------------------------------------------
    # Step 2 – Signals dataframe + context
    # ------------------------------------------------------------------
    signals = price.copy()

    effective_weights = weights if weights is not None else [w for _, w in INDICATORS]

    ctx = {
        "daily_price":  daily_price,
        "vix":          vix,
        "spx_price":    spx_price,
        "spx_daily":    spx_daily,
        "fundamentals": fundamentals,
    }

    # ------------------------------------------------------------------
    # Steps 3-5 – Bar loop: compute signals, manage positions
    # ------------------------------------------------------------------
    trades         = []
    equity_curve   = []
    cumulative_pnl = 0.0

    in_position      = False
    position_type    = None
    entry_price      = None
    entry_time       = None
    entry_signal_sum = None
    trade_high       = None
    trade_low        = None

    num_bars = len(signals)

    for i in range(num_bars):
        bar_time  = signals.index[i]
        bar_close = signals["close"].iloc[i]
        bar_high  = signals["high"].iloc[i]
        bar_low   = signals["low"].iloc[i]

        # Step 3 – weighted indicator votes
        ind_values = [
            ew * ind_func(signals, i, **ctx)
            for (ind_func, _), ew in zip(INDICATORS, effective_weights)
        ]
        signal_sum = sum(ind_values)

        is_last_bar_of_day = (
            i == num_bars - 1
            or signals.index[i + 1].date() != bar_time.date()
        )

        # Step 5 – exit logic
        if in_position:
            trade_high = max(trade_high, bar_high)
            trade_low  = min(trade_low, bar_low)

            exit_price  = None
            exit_reason = None

            if position_type == "long":
                pct = (bar_close - entry_price) / entry_price
                if pct >= take_profit:
                    exit_price, exit_reason = bar_close, "take_profit"
                elif pct <= -stop_loss:
                    exit_price, exit_reason = bar_close, "stop_loss"
            elif position_type == "short":
                pct = (entry_price - bar_close) / entry_price
                if pct >= take_profit:
                    exit_price, exit_reason = bar_close, "take_profit"
                elif pct <= -stop_loss:
                    exit_price, exit_reason = bar_close, "stop_loss"

            if exit_price is None and is_last_bar_of_day:
                exit_price, exit_reason = bar_close, "eod_exit"

            if exit_price is not None:
                if position_type == "long":
                    net_pnl       = (exit_price - entry_price) / entry_price * position_size
                    favorable_exc = (trade_high - entry_price) / entry_price
                    adverse_exc   = (entry_price - trade_low) / entry_price
                else:
                    net_pnl       = (entry_price - exit_price) / entry_price * position_size
                    favorable_exc = (entry_price - trade_low) / entry_price
                    adverse_exc   = (trade_high - entry_price) / entry_price

                cumulative_pnl += net_pnl

                trades.append({
                    "trade_num":           len(trades) + 1,
                    "type":                position_type,
                    "entry_date":          entry_time.strftime("%Y-%m-%d"),
                    "entry_time":          entry_time.strftime("%H:%M"),
                    "exit_date":           bar_time.strftime("%Y-%m-%d"),
                    "exit_time":           bar_time.strftime("%H:%M"),
                    "signal":              entry_signal_sum,
                    "entry_price":         round(entry_price, 4),
                    "exit_price":          round(exit_price, 4),
                    "exit_reason":         exit_reason,
                    "position_size":       position_size,
                    "net_pnl":             round(net_pnl, 4),
                    "favorable_excursion": round(favorable_exc, 6),
                    "adverse_excursion":   round(adverse_exc, 6),
                    "cumulative_pnl":      round(cumulative_pnl, 4),
                })

                in_position      = False
                position_type    = None
                entry_price      = None
                entry_time       = None
                entry_signal_sum = None
                trade_high       = None
                trade_low        = None

        # Step 4 – entry logic
        if not in_position and not is_last_bar_of_day:
            if signal_sum >= n:
                in_position      = True
                position_type    = "long"
                entry_price      = bar_close
                entry_time       = bar_time
                entry_signal_sum = signal_sum
                trade_high       = bar_high
                trade_low        = bar_low
            elif signal_sum <= -n:
                in_position      = True
                position_type    = "short"
                entry_price      = bar_close
                entry_time       = bar_time
                entry_signal_sum = signal_sum
                trade_high       = bar_high
                trade_low        = bar_low

        equity_curve.append((bar_time, starting_equity + cumulative_pnl))

    # ------------------------------------------------------------------
    # Step 6 – Performance metrics
    # ------------------------------------------------------------------
    trades_df    = pd.DataFrame(trades)
    total_trades = len(trades_df)
    total_pnl    = cumulative_pnl

    eq_series = pd.Series(
        [e[1] for e in equity_curve],
        index=[e[0] for e in equity_curve],
    )

    if total_trades > 0:
        winning      = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"]
        losing       = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"]
        gross_profit = winning.sum() if len(winning) > 0 else 0.0
        gross_loss   = abs(losing.sum()) if len(losing) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss
                         if gross_loss != 0 else float("inf"))
        returns      = trades_df["net_pnl"] / position_size
        sharpe_ratio = ((returns.mean() / returns.std()) * np.sqrt(252)
                        if returns.std() != 0 else 0.0)
        running_max  = eq_series.cummax()
        max_drawdown = (eq_series - running_max).min()
    else:
        profit_factor = 0.0
        sharpe_ratio  = 0.0
        max_drawdown  = 0.0

    # ------------------------------------------------------------------
    # Step 7 – Console output (verbose only)
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print(f"  BACKTEST RESULTS  –  {ticker}")
        print("=" * 60)
        print(f"  Time frame  : {time_frame}   Interval: {interval}")
        print(f"  Indicators  : {len(INDICATORS)}   Required (n): {n}")
        print(f"  Take Profit : {take_profit:.2%}   Stop Loss: {stop_loss:.2%}")
        print("-" * 60)
        print(f"  Starting Equity  : ${starting_equity:,.2f}")
        print(f"  Ending Equity    : ${starting_equity + total_pnl:,.2f}")
        print(f"  Total P/L        : ${total_pnl:,.2f}")
        print(f"  Sharpe Ratio     : {sharpe_ratio:.4f}")
        print(f"  Max Drawdown     : ${max_drawdown:,.2f}")
        print(f"  Total Trades     : {total_trades}")
        print(f"  Profit Factor    : {profit_factor:.4f}")
        print("=" * 60)

        if total_trades > 0:
            print("\n  TRADE LOG")
            print("-" * 120)
            display_cols = [
                "trade_num", "type", "entry_date", "entry_time",
                "signal", "entry_price", "exit_price", "exit_reason",
                "position_size", "net_pnl",
                "favorable_excursion", "adverse_excursion", "cumulative_pnl",
            ]
            print(trades_df[display_cols].to_string(index=False))

        _plot_results(ticker, eq_series, signals, trades_df, starting_equity)

    return {
        "pnl":           round(total_pnl, 4),
        "sharpe_ratio":  round(sharpe_ratio, 4),
        "max_drawdown":  round(max_drawdown, 4),
        "total_trades":  total_trades,
        "profit_factor": round(profit_factor, 4),
        "trades_df":     trades_df,
        "equity_series": eq_series,
        "price_df":      signals,
    }


def _plot_results(ticker, eq_series, signals, trades_df, starting_equity):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.6], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eq_series.index, eq_series.values, color="#2196F3", linewidth=1.2)
    ax1.fill_between(eq_series.index, eq_series.values, alpha=0.15, color="#2196F3")
    ax1.set_title(f"Equity Curve – {ticker}", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.axhline(starting_equity, color="grey", linewidth=0.5, linestyle="--")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(signals.index, signals["close"], color="#455A64", linewidth=0.8, label="Close")
    ax2.set_title(f"Price Chart with Trades – {ticker}", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Price ($)")
    ax2.grid(True, alpha=0.3)

    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            entry_dt = pd.Timestamp(f"{trade['entry_date']} {trade['entry_time']}")
            exit_dt  = pd.Timestamp(f"{trade['exit_date']} {trade['exit_time']}")
            if trade["type"] == "long":
                ax2.annotate("▲", xy=(entry_dt, trade["entry_price"]),
                             fontsize=10, color="green", ha="center", va="bottom")
                ax2.annotate("×", xy=(exit_dt, trade["exit_price"]),
                             fontsize=12, color="darkgreen", ha="center", va="center",
                             fontweight="bold")
            else:
                ax2.annotate("▼", xy=(entry_dt, trade["entry_price"]),
                             fontsize=10, color="red", ha="center", va="top")
                ax2.annotate("×", xy=(exit_dt, trade["exit_price"]),
                             fontsize=12, color="darkred", ha="center", va="center",
                             fontweight="bold")

    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    ax3 = fig.add_subplot(gs[2])
    if len(trades_df) > 0:
        colors = ["green" if pnl > 0 else "red" for pnl in trades_df["net_pnl"]]
        ax3.bar(trades_df["trade_num"], trades_df["net_pnl"],
                color=colors, edgecolor="white")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("Net P/L ($)")
        ax3.set_title("Per-Trade P/L", fontsize=13, fontweight="bold")
        ax3.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No trades executed", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(f"backtest_{ticker}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Charts saved as backtest_{ticker}.png")
