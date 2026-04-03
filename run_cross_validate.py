"""
CLI: Cross-ticker overfitting check.

Optimizes parameters on a train ticker, then backtests those same
parameters on a set of test tickers.  A large Sharpe gap between the
train ticker and the test tickers indicates over-fitting.

Examples
--------
# Optimize on AAPL, test on MSFT / NVDA / SPY / QQQ (defaults)
python run_cross_validate.py AAPL

# Custom test tickers
python run_cross_validate.py AAPL --test MSFT NVDA TSLA SPY

# Use a different time-frame / interval
python run_cross_validate.py AAPL --test SPY QQQ --time-frame 1y --interval 15m

# Skip optimization — load existing params and just test generalization
python run_cross_validate.py AAPL --test SPY QQQ --load-params results/window_000.json

# Save optimized train params
python run_cross_validate.py AAPL --save results/aapl_cv.json

# Save the cross-validation results table to CSV
python run_cross_validate.py AAPL --csv results/cv_results.csv
"""

import argparse
import json

from signal_testing.cross_validate import cross_validate


DEFAULT_TEST_TICKERS = ["MSFT", "NVDA", "SPY", "QQQ"]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-ticker overfitting check for the signal-testing strategy."
    )
    parser.add_argument("train_ticker", type=str,
                        help="Ticker to optimize on (in-sample).")
    parser.add_argument("--test", nargs="+", default=DEFAULT_TEST_TICKERS,
                        metavar="TICKER",
                        help=f"Out-of-sample test tickers "
                             f"(default: {DEFAULT_TEST_TICKERS}).")
    parser.add_argument("--time-frame", default="60d",
                        choices=["1w", "60d", "1y"],
                        help="Historical window (default: 60d).")
    parser.add_argument("--interval", default="5m",
                        choices=["1m", "5m", "15m", "30m", "1h"],
                        help="Bar interval (default: 5m).")
    parser.add_argument("--generations", type=int, default=200,
                        help="SNES generations for optimization (default: 200).")
    parser.add_argument("--popsize", type=int, default=50,
                        help="SNES population size (default: 50).")
    parser.add_argument("--position-size", type=float, default=100.0,
                        help="Dollar amount per trade (default: 100).")
    parser.add_argument("--starting-equity", type=float, default=10_000.0,
                        help="Starting account equity (default: 10000).")
    parser.add_argument("--min-trades", type=int, default=5,
                        help="Minimum trades for valid fitness (default: 5).")
    parser.add_argument("--load-params", type=str, default=None,
                        metavar="PATH",
                        help="JSON file of pre-optimized params. If provided, "
                             "skip optimization and go straight to cross-ticker eval.")
    parser.add_argument("--save", type=str, default=None,
                        metavar="PATH",
                        help="Save train-ticker optimized params to this JSON path.")
    parser.add_argument("--csv", type=str, default=None,
                        metavar="PATH",
                        help="Save the results DataFrame to a CSV file.")
    args = parser.parse_args()

    preloaded = None
    if args.load_params:
        with open(args.load_params) as f:
            preloaded = json.load(f)
        print(f"Loaded params from {args.load_params}")

    df = cross_validate(
        train_ticker=args.train_ticker,
        test_tickers=args.test,
        time_frame=args.time_frame,
        interval=args.interval,
        n_generations=args.generations,
        popsize=args.popsize,
        position_size=args.position_size,
        starting_equity=args.starting_equity,
        min_trades=args.min_trades,
        preloaded_params=preloaded,
        save_train_params=args.save,
    )

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()
