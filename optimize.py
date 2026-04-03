"""
CLI: run evolutionary parameter optimization for a single ticker/window.

Usage
-----
    python optimize.py AAPL
    python optimize.py AAPL --time-frame 60d --interval 5m --generations 200
    python optimize.py AAPL --save results/aapl_best.json
"""

import argparse
import json

from signal_testing.data import fetch_data
from signal_testing.optimizer import run_optimization
from signal_testing.results import save_params


def main():
    parser = argparse.ArgumentParser(description="Optimize backtest parameters with SNES.")
    parser.add_argument("ticker",                      help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--time-frame",   default="60d",   help="1w | 60d | 1y  (default: 60d)")
    parser.add_argument("--interval",     default="5m",    help="1m | 5m | 15m | 30m | 1h  (default: 5m)")
    parser.add_argument("--generations",  type=int, default=200, help="SNES generations (default: 200)")
    parser.add_argument("--popsize",      type=int, default=50,  help="Population size (default: 50)")
    parser.add_argument("--min-trades",   type=int, default=5,   help="Min trades for valid fitness (default: 5)")
    parser.add_argument("--position-size", type=float, default=100.0)
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument("--save",         default=None,
                        help="Path to save best params JSON (default: results/<ticker>_best.json)")
    args = parser.parse_args()

    save_path = args.save or f"results/{args.ticker.upper()}_best.json"

    print(f"\nFetching data for {args.ticker.upper()} ({args.time_frame}, {args.interval})...")
    data = fetch_data(args.ticker, args.time_frame, args.interval)
    print(f"Loaded {len(data['price'])} bars.\n")

    result = run_optimization(
        ticker=args.ticker,
        time_frame=args.time_frame,
        interval=args.interval,
        preloaded_data=data,
        n_generations=args.generations,
        popsize=args.popsize,
        position_size=args.position_size,
        starting_equity=args.starting_equity,
        min_trades=args.min_trades,
    )

    print("\n" + "=" * 60)
    print(f"  OPTIMIZATION COMPLETE  –  {args.ticker.upper()}")
    print("=" * 60)
    print(f"  Best Sharpe   : {result['sharpe']:.4f}")
    print(f"  stop_loss     : {result['stop_loss']:.4f}")
    print(f"  take_profit   : {result['take_profit']:.4f}")
    print(f"  n threshold   : {result['n']:.4f}")
    print(f"  Weights       : {[round(w, 3) for w in result['weights']]}")
    print("=" * 60)

    save_params(result, save_path)
    print(f"\nBest parameters saved to {save_path}")


if __name__ == "__main__":
    main()
