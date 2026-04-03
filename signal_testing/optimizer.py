"""
Evolutionary parameter optimizer using EvoTorch (SNES algorithm).

Optimizes indicator weights, stop_loss, take_profit, and n simultaneously
by maximizing Sharpe ratio over a backtested period.

Solution vector layout (len(INDICATORS) + 3 dimensions):
    [w_0, ..., w_13, stop_loss, take_profit, n]

Bounds:
    weights     : [0.0, 5.0]
    stop_loss   : [0.001, 0.10]
    take_profit : [0.001, 0.20]
    n           : [0.5,  8.0]
"""

import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger

from .backtest import backtest
from .indicators import INDICATORS

N_INDICATORS = len(INDICATORS)

# Parameter bounds
_LOWER = [0.0] * N_INDICATORS + [0.001, 0.001, 0.5]
_UPPER = [5.0] * N_INDICATORS + [0.10,  0.20,  8.0]

PARAM_LOWER = torch.tensor(_LOWER, dtype=torch.float32)
PARAM_UPPER = torch.tensor(_UPPER, dtype=torch.float32)


class BacktestProblem(Problem):
    """
    EvoTorch Problem that wraps the backtester as a fitness function.

    Each candidate solution encodes all tunable parameters as a single
    float32 vector. Fitness is the Sharpe ratio. Solutions with fewer
    than *min_trades* are penalized to discourage degenerate behaviour
    (e.g. all weights near zero → no trades → technically 0 Sharpe).
    """

    def __init__(
        self,
        ticker: str,
        time_frame: str,
        interval: str,
        preloaded_data: dict,
        position_size: float = 100.0,
        starting_equity: float = 10_000.0,
        min_trades: int = 5,
    ):
        """
        Parameters
        ----------
        ticker : str
        time_frame : str
        interval : str
        preloaded_data : dict
            Result of fetch_data(). Passed through to every backtest()
            call to avoid redundant API requests during optimization.
        position_size : float
        starting_equity : float
        min_trades : int
            Minimum number of trades required before fitness is
            evaluated normally. Solutions below this threshold receive
            a penalty fitness of -10.
        """
        super().__init__(
            objective_sense="max",
            solution_length=N_INDICATORS + 3,
            # Start population in the active region to avoid weight collapse.
            initial_bounds=(0.5, 1.5),
            bounds=(PARAM_LOWER, PARAM_UPPER),
            dtype=torch.float32,
            num_actors=1,   # keep single-process; yahooquery is not fork-safe
        )
        self._ticker          = ticker
        self._time_frame      = time_frame
        self._interval        = interval
        self._data            = preloaded_data
        self._position_size   = position_size
        self._starting_equity = starting_equity
        self._min_trades      = min_trades

    def _evaluate(self, solutions):
        fitnesses = torch.zeros(len(solutions), dtype=torch.float32)

        for i in range(len(solutions)):
            params      = solutions.values[i].cpu().tolist()
            weights     = params[:N_INDICATORS]
            stop_loss   = params[N_INDICATORS]
            take_profit = params[N_INDICATORS + 1]
            n           = params[N_INDICATORS + 2]

            try:
                result = backtest(
                    ticker=self._ticker,
                    n=n,
                    time_frame=self._time_frame,
                    interval=self._interval,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    weights=weights,
                    position_size=self._position_size,
                    starting_equity=self._starting_equity,
                    preloaded_data=self._data,
                    verbose=False,
                )
                sharpe = float(result["sharpe_ratio"])
                if result["total_trades"] < self._min_trades:
                    sharpe = -10.0
                if not np.isfinite(sharpe):
                    sharpe = -10.0
            except Exception:
                sharpe = -10.0

            fitnesses[i] = sharpe

        solutions.set_evals(fitnesses)


def run_optimization(
    ticker: str,
    time_frame: str,
    interval: str,
    preloaded_data: dict,
    n_generations: int = 200,
    popsize: int = 50,
    stdev_init: float = 0.5,
    position_size: float = 100.0,
    starting_equity: float = 10_000.0,
    min_trades: int = 5,
) -> dict:
    """
    Run SNES optimization and return the best parameters found.

    Parameters
    ----------
    ticker, time_frame, interval : str
    preloaded_data : dict
        Output of fetch_data() — fetched once outside the optimizer.
    n_generations : int
        Number of SNES generations to run (default 200).
    popsize : int
        Population size per generation (default 50).
    stdev_init : float
        Initial standard deviation for SNES search distribution.
    position_size, starting_equity : float
    min_trades : int

    Returns
    -------
    dict with keys:
        weights, stop_loss, take_profit, n, sharpe, progress_df
    """
    problem = BacktestProblem(
        ticker=ticker,
        time_frame=time_frame,
        interval=interval,
        preloaded_data=preloaded_data,
        position_size=position_size,
        starting_equity=starting_equity,
        min_trades=min_trades,
    )

    searcher = SNES(problem, stdev_init=stdev_init, popsize=popsize)
    pandas_logger = PandasLogger(searcher)
    StdOutLogger(searcher)

    searcher.run(n_generations)

    best_params = searcher.status["best"].values.cpu().tolist()
    best_sharpe = float(searcher.status["best"].evals)

    return {
        "weights":     [float(w) for w in best_params[:N_INDICATORS]],
        "stop_loss":   float(best_params[N_INDICATORS]),
        "take_profit": float(best_params[N_INDICATORS + 1]),
        "n":           float(best_params[N_INDICATORS + 2]),
        "sharpe":      best_sharpe,
        "progress_df": pandas_logger.to_dataframe(),
    }
