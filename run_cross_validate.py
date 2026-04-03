"""
CLI: Cross-ticker overfitting check.

Optimizes parameters on a train ticker (+ optional extra train tickers),
then backtests those same parameters on a set of test tickers.  A large
Sharpe gap between the train set and the test set indicates over-fitting.

Anti-overfitting flags
----------------------
--extra-train   Add more tickers to the training objective (avg Sharpe)
--min-trades    Raise to prevent degenerate few-trade solutions
--reg-lambda    L2 weight regularization coefficient
--patience      Early stopping patience (generations without improvement)

Examples
--------
# Single-ticker optimize on AAPL, test on defaults
python run_cross_validate.py AAPL

# Multi-ticker training (most effective anti-overfit)
python run_cross_validate.py AAPL --extra-train MSFT SPY --test NVDA QQQ TSLA

# Longer window
python run_cross_validate.py AAPL --extra-train MSFT SPY --time-frame 1y --interval 15m

# Skip optimization — load existing params and just test generalization
python run_cross_validate.py AAPL --test SPY QQQ --load-params results/window_000.json

# Save results
python run_cross_validate.py AAPL --extra-train MSFT SPY --save results/multi_cv.json --csv results/cv_results.csv
"""

import argparse
import json

from signal_testing.cross_validate import cross_validate


DEFAULT_TEST_TICKERS = ["NVDA", "SPY", "QQQ", "TSLA"]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-ticker overfitting check for the signal-testing strategy."
    )
    parser.add_argument("train_ticker", type=str,
                        help="Primary ticker to optimize on (in-sample).")
    parser.add_argument("--extra-train", nargs="+", default=None,
                        metavar="TICKER",
                        help="Additional tickers included in the training objective. "
                             "Fitness = avg Sharpe across all train tickers. "
                             "Most effective anti-overfitting measure.")
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
                        help="Max SNES generations (default: 200; early stopping may halt sooner).")
    parser.add_argument("--popsize", type=int, default=50,
                        help="SNES population size (default: 50).")
    parser.add_argument("--position-size", type=float, default=100.0,
                        help="Dollar amount per trade (default: 100).")
    parser.add_argument("--starting-equity", type=float, default=10_000.0,
                        help="Starting account equity (default: 10000).")
    parser.add_argument("--min-trades", type=int, default=15,
                        help="Minimum trades per ticker for valid fitness (default: 15).")
    parser.add_argument("--reg-lambda", type=float, default=0.01,
                        help="L2 regularization on indicator weights (default: 0.01).")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience in generations (default: 20).")
    parser.add_argument("--min-delta", type=float, default=0.05,
                        help="Min fitness improvement to reset patience counter (default: 0.05).")
    parser.add_argument("--load-params", type=str, default=None,
                        metavar="PATH",
                        help="JSON file of pre-optimized params. Skips optimization.")
    parser.add_argument("--save", type=str, default=None,
                        metavar="PATH",
                        help="Save optimized train params to this JSON path.")
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
        extra_train_tickers=args.extra_train,
        time_frame=args.time_frame,
        interval=args.interval,
        n_generations=args.generations,
        popsize=args.popsize,
        position_size=args.position_size,
        starting_equity=args.starting_equity,
        min_trades=args.min_trades,
        reg_lambda=args.reg_lambda,
        patience=args.patience,
        min_delta=args.min_delta,
        preloaded_params=preloaded,
        save_train_params=args.save,
    )

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()
