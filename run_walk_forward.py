"""
CLI: run walk-forward optimization across the full available history.

Usage
-----
    python run_walk_forward.py AAPL
    python run_walk_forward.py AAPL --interval 5m --train-bars 1560 --test-bars 390
    python run_walk_forward.py AAPL --plot
"""

import argparse

from signal_testing.walk_forward import walk_forward_optimize
from signal_testing.results import plot_walk_forward


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evolutionary optimization.")
    parser.add_argument("ticker",                         help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--interval",     default="5m",   help="Bar interval (default: 5m)")
    parser.add_argument("--train-bars",   type=int, default=1560,
                        help="Bars per training window (default: 1560 ≈ 4 weeks at 5m)")
    parser.add_argument("--test-bars",    type=int, default=390,
                        help="Bars per test window (default: 390 ≈ 1 week at 5m)")
    parser.add_argument("--step-bars",    type=int, default=None,
                        help="Slide step size (default: same as test-bars)")
    parser.add_argument("--generations",  type=int, default=150)
    parser.add_argument("--popsize",      type=int, default=50)
    parser.add_argument("--min-trades",   type=int, default=5)
    parser.add_argument("--save-dir",     default="results")
    parser.add_argument("--plot",         action="store_true",
                        help="Plot walk-forward results after completion")
    args = parser.parse_args()

    step = args.step_bars if args.step_bars is not None else args.test_bars

    wf_df = walk_forward_optimize(
        ticker=args.ticker,
        interval=args.interval,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=step,
        n_generations=args.generations,
        popsize=args.popsize,
        min_trades=args.min_trades,
        save_dir=args.save_dir,
    )

    print("\nWalk-forward summary:")
    print(wf_df[["window", "train_start", "test_end",
                 "train_sharpe", "test_sharpe", "test_pnl", "test_trades"]].to_string(index=False))

    if args.plot:
        chart_path = f"{args.save_dir}/walk_forward_{args.ticker.upper()}.png"
        plot_walk_forward(wf_df, args.ticker.upper(), save_path=chart_path)


if __name__ == "__main__":
    main()
