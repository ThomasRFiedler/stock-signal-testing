"""
Evolutionary parameter optimizer using EvoTorch (SNES algorithm).

Optimizes indicator weights, stop_loss, take_profit, and n simultaneously
by maximizing a regularized, multi-ticker average Sharpe ratio.

Solution vector layout (len(INDICATORS) + 3 dimensions):
    [w_0, ..., w_13, stop_loss, take_profit, n]

Bounds:
    weights     : [0.0, 5.0]
    stop_loss   : [0.001, 0.10]
    take_profit : [0.001, 0.20]
    n           : [0.5,  8.0]

Anti-overfitting mechanisms
----------------------------
1. Multi-ticker training — fitness is the *average* Sharpe across all
   train tickers.  Parameters must generalize across instruments.
2. min_trades threshold — any solution producing fewer than min_trades
   on *any* single train ticker is penalized, preventing the optimizer
   from finding degenerate few-trade cherry-picks.
3. L2 weight regularization — a small penalty on the sum of squared
   indicator weights keeps weights from being pushed to their maximum
   bound just to inflate in-sample Sharpe.
4. Early stopping — optimization halts automatically when the best
   fitness has not improved by more than min_delta over the last
   `patience` generations, avoiding unnecessary computation and
   overfitting to noise in the later generations.
"""

import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger

from .backtest import backtest
from .data import fetch_data
from .indicators import INDICATORS

N_INDICATORS = len(INDICATORS)

# Parameter bounds
# TP/SL capped at realistic intraday ranges (3% / 6%).
# The previous 10%/20% upper limits allowed degenerate solutions that
# cherry-picked a handful of trades with extreme stop widths.
_LOWER = [0.0] * N_INDICATORS + [0.001, 0.001, 0.5]
_UPPER = [5.0] * N_INDICATORS + [0.03,  0.06,  8.0]

PARAM_LOWER = torch.tensor(_LOWER, dtype=torch.float32)
PARAM_UPPER = torch.tensor(_UPPER, dtype=torch.float32)


class BacktestProblem(Problem):
    """
    EvoTorch Problem that wraps the backtester as a fitness function.

    Fitness = mean Sharpe ratio across all train tickers, minus an L2
    regularization penalty on indicator weights.

    Solutions where *any* ticker produces fewer than min_trades receive
    a penalty fitness of -10 for that ticker (averaged in).
    """

    def __init__(
        self,
        train_data: dict,
        interval: str,
        position_size: float = 100.0,
        starting_equity: float = 10_000.0,
        min_trades: int = 15,
        reg_lambda: float = 0.01,
    ):
        """
        Parameters
        ----------
        train_data : dict[str, dict]
            Mapping of {ticker: preloaded_data} for every training ticker.
            Each value is the output of fetch_data().
        interval : str
            Bar interval — passed through to backtest().
        position_size : float
        starting_equity : float
        min_trades : int
            Minimum number of trades required on *each* train ticker before
            fitness is evaluated normally.  Default raised to 15 (was 5) to
            ensure Sharpe estimates are statistically meaningful.
        reg_lambda : float
            L2 regularization coefficient on indicator weights.
            Fitness -= reg_lambda * sum(w_i^2).
            Keeps weights from being driven to their 5.0 upper bound purely
            to inflate in-sample Sharpe.  Default: 0.01.
        """
        super().__init__(
            objective_sense="max",
            solution_length=N_INDICATORS + 3,
            initial_bounds=(0.5, 1.5),
            dtype=torch.float32,
            num_actors=1,
        )
        self._train_data      = train_data          # {ticker: data_dict}
        self._interval        = interval
        self._position_size   = position_size
        self._starting_equity = starting_equity
        self._min_trades      = min_trades
        self._reg_lambda      = reg_lambda

    def _evaluate(self, solution):
        params      = solution.values.clamp(PARAM_LOWER, PARAM_UPPER).cpu().tolist()
        weights     = params[:N_INDICATORS]
        stop_loss   = params[N_INDICATORS]
        take_profit = params[N_INDICATORS + 1]
        n           = params[N_INDICATORS + 2]

        sharpes = []
        for ticker, data in self._train_data.items():
            try:
                result = backtest(
                    ticker=ticker,
                    n=n,
                    time_frame="1y",        # unused when preloaded_data is given
                    interval=self._interval,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    weights=weights,
                    position_size=self._position_size,
                    starting_equity=self._starting_equity,
                    preloaded_data=data,
                    verbose=False,
                )
                sharpe = float(result["sharpe_ratio"])
                if result["total_trades"] < self._min_trades:
                    sharpe = -10.0
                if not np.isfinite(sharpe):
                    sharpe = -10.0
            except Exception:
                sharpe = -10.0
            sharpes.append(sharpe)

        mean_sharpe = float(np.mean(sharpes))

        # L2 regularization on indicator weights — penalize large weights
        weight_penalty = self._reg_lambda * float(np.sum(np.array(weights) ** 2))
        fitness = mean_sharpe - weight_penalty

        solution.set_evals(fitness)


def run_optimization(
    train_tickers: list,
    interval: str,
    preloaded_data_map: dict = None,
    time_frame: str = "60d",
    n_generations: int = 200,
    popsize: int = 50,
    stdev_init: float = 0.5,
    position_size: float = 100.0,
    starting_equity: float = 10_000.0,
    min_trades: int = 15,
    reg_lambda: float = 0.01,
    patience: int = 20,
    min_delta: float = 0.05,
    # Legacy single-ticker shim — kept for backward compatibility
    ticker: str = None,
    preloaded_data: dict = None,
) -> dict:
    """
    Run SNES optimization and return the best parameters found.

    Parameters
    ----------
    train_tickers : list[str]
        Tickers to train on simultaneously.  Fitness = avg Sharpe across all.
        Use multiple tickers (e.g. ["AAPL", "MSFT", "SPY"]) to improve
        generalization and reduce overfitting.
    interval : str
        Bar interval for all train tickers.
    preloaded_data_map : dict[str, dict], optional
        Pre-fetched data for each ticker: {ticker: fetch_data() output}.
        If None, data is fetched automatically for each ticker.
    time_frame : str
        Historical window used when fetching data ("1w", "60d", "1y").
    n_generations : int
        Maximum SNES generations (default 200).  May stop earlier via
        early stopping if the plateau criterion is met.
    popsize : int
        Population size per generation (default 50).
    stdev_init : float
        Initial standard deviation for SNES search distribution.
    position_size : float
    starting_equity : float
    min_trades : int
        Minimum trades per ticker required for non-penalty fitness (default 15).
    reg_lambda : float
        L2 weight regularization coefficient (default 0.01).
    patience : int
        Early stopping: halt if best fitness has not improved by more than
        min_delta for this many consecutive generations (default 20).
    min_delta : float
        Minimum improvement in best fitness to reset the patience counter
        (default 0.05).
    ticker : str
        Deprecated single-ticker parameter.  Use train_tickers instead.
    preloaded_data : dict
        Deprecated.  Use preloaded_data_map instead.

    Returns
    -------
    dict with keys:
        weights, stop_loss, take_profit, n, sharpe, progress_df,
        train_tickers, stopped_early, generations_run
    """
    # ------------------------------------------------------------------
    # Backward-compatibility shim for single-ticker callers
    # ------------------------------------------------------------------
    if ticker is not None and not train_tickers:
        train_tickers = [ticker]
    if preloaded_data is not None and preloaded_data_map is None:
        preloaded_data_map = {train_tickers[0]: preloaded_data}

    if not train_tickers:
        raise ValueError("Provide at least one ticker in train_tickers.")

    # ------------------------------------------------------------------
    # Fetch data for any ticker not already supplied
    # ------------------------------------------------------------------
    if preloaded_data_map is None:
        preloaded_data_map = {}

    train_data = {}
    for t in train_tickers:
        if t in preloaded_data_map:
            train_data[t] = preloaded_data_map[t]
        else:
            print(f"  Fetching data for train ticker {t} ({time_frame} @ {interval})...")
            train_data[t] = fetch_data(t, time_frame, interval)

    # ------------------------------------------------------------------
    # Build problem and searcher
    # ------------------------------------------------------------------
    problem = BacktestProblem(
        train_data=train_data,
        interval=interval,
        position_size=position_size,
        starting_equity=starting_equity,
        min_trades=min_trades,
        reg_lambda=reg_lambda,
    )

    searcher      = SNES(problem, stdev_init=stdev_init, popsize=popsize)
    pandas_logger = PandasLogger(searcher)
    StdOutLogger(searcher)

    # ------------------------------------------------------------------
    # Generation loop with early stopping
    # ------------------------------------------------------------------
    best_so_far   = -float("inf")
    no_improve    = 0
    stopped_early = False

    for gen in range(1, n_generations + 1):
        searcher.step()
        current_best = float(searcher.status["best"].evals)

        if current_best - best_so_far > min_delta:
            best_so_far = current_best
            no_improve  = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  [Early stopping] No improvement > {min_delta} for "
                  f"{patience} generations.  Stopping at generation {gen}.")
            stopped_early = True
            break

    # ------------------------------------------------------------------
    # Extract best params (clamp to valid ranges)
    # ------------------------------------------------------------------
    best_params = (
        searcher.status["best"].values
        .clamp(PARAM_LOWER, PARAM_UPPER)
        .cpu()
        .tolist()
    )
    best_sharpe = float(searcher.status["best"].evals)

    return {
        "weights":          [float(w) for w in best_params[:N_INDICATORS]],
        "stop_loss":        float(best_params[N_INDICATORS]),
        "take_profit":      float(best_params[N_INDICATORS + 1]),
        "n":                float(best_params[N_INDICATORS + 2]),
        "sharpe":           best_sharpe,
        "progress_df":      pandas_logger.to_dataframe(),
        "train_tickers":    train_tickers,
        "stopped_early":    stopped_early,
        "generations_run":  gen,
    }
