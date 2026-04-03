"""
Persistence and visualization for optimization results.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .indicators import INDICATORS


def save_params(opt_result: dict, path: str) -> None:
    """
    Save optimizer output to a JSON file.

    Converts all numpy/torch scalars to native Python types for
    JSON serializability.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {
        "weights":     [float(w) for w in opt_result["weights"]],
        "stop_loss":   float(opt_result["stop_loss"]),
        "take_profit": float(opt_result["take_profit"]),
        "n":           float(opt_result["n"]),
        "sharpe":      float(opt_result["sharpe"]),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_params(path: str) -> dict:
    """Load parameters saved by save_params()."""
    with open(path) as f:
        return json.load(f)


def load_latest_params(save_dir: str = "results") -> dict:
    """
    Return the most recently saved window params from *save_dir*.
    Useful for live trading to always use the latest optimized params.
    """
    files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("window_") and f.endswith(".json")]
    )
    if not files:
        raise FileNotFoundError(f"No saved parameter files found in '{save_dir}'.")
    return load_params(os.path.join(save_dir, files[-1]))


def plot_walk_forward(wf_df: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """
    Plot train vs. test Sharpe ratio and cumulative test P/L across
    walk-forward windows.

    Parameters
    ----------
    wf_df : pd.DataFrame
        Output of walk_forward_optimize().
    ticker : str
    save_path : str, optional
        If provided, saves the figure to this path instead of displaying.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f"Walk-Forward Optimization – {ticker}", fontsize=14, fontweight="bold")

    windows = wf_df["window"]

    # Panel 1: Sharpe ratio (train vs test)
    ax = axes[0]
    ax.plot(windows, wf_df["train_sharpe"], marker="o", label="Train Sharpe", color="#2196F3")
    ax.plot(windows, wf_df["test_sharpe"],  marker="s", label="Test Sharpe",  color="#FF5722")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Train vs. Out-of-Sample Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Cumulative test P/L
    ax = axes[1]
    cum_pnl = wf_df["test_pnl"].cumsum()
    ax.bar(windows, wf_df["test_pnl"],
           color=["green" if v >= 0 else "red" for v in wf_df["test_pnl"]],
           alpha=0.6, label="Window P/L")
    ax.plot(windows, cum_pnl, color="#9C27B0", linewidth=1.5, label="Cumulative P/L")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("P/L ($)")
    ax.set_title("Out-of-Sample P/L per Window")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Indicator weights heatmap
    ax = axes[2]
    ind_names = [fn.__name__.replace("indicator_", "") for fn, _ in INDICATORS]
    weight_matrix = pd.DataFrame(
        wf_df["weights"].tolist(), columns=ind_names, index=windows
    )
    im = ax.imshow(weight_matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=5)
    ax.set_yticks(range(len(ind_names)))
    ax.set_yticklabels(ind_names, fontsize=8)
    ax.set_xlabel("Window")
    ax.set_title("Indicator Weights per Window")
    plt.colorbar(im, ax=ax, orientation="vertical", label="Weight")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Walk-forward chart saved to {save_path}")
    else:
        plt.show()
