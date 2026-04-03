import pandas as pd


def _normalise_yq_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a yahooquery MultiIndex DataFrame into a simple DatetimeIndex DataFrame."""
    if isinstance(df, pd.DataFrame):
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
    return df


def fetch_data(ticker: str, time_frame: str, interval: str) -> dict:
    """
    Fetch all data needed by every indicator in a single batch of API calls.

    Returns a dict with keys:
        price        – Intraday OHLCV for *ticker* at *interval*.
        daily_price  – Daily OHLCV for *ticker*.
        vix          – Daily OHLCV for ^VIX.
        spx_price    – Intraday OHLCV for ^GSPC at *interval*.
        spx_daily    – Daily OHLCV for ^GSPC.
        fundamentals – dict of snapshot values (P/E, debt-to-equity, short ratio, etc.)
    """
    from yahooquery import Ticker

    period_map = {"1w": "5d", "60d": "60d", "1y": "1y"}
    period = period_map.get(time_frame)
    if period is None:
        raise ValueError(
            f"Unsupported time_frame '{time_frame}'. Use '1w', '60d', or '1y'."
        )

    valid_intervals = {"1m", "5m", "15m", "30m", "1h"}
    if interval not in valid_intervals:
        raise ValueError(
            f"Unsupported interval '{interval}'. Use one of {valid_intervals}."
        )

    tk     = Ticker(ticker)
    bench  = Ticker("^GSPC")
    vix_tk = Ticker("^VIX")

    price       = _normalise_yq_df(tk.history(period=period, interval=interval))
    daily_price = _normalise_yq_df(tk.history(period=period, interval="1d"))
    vix         = _normalise_yq_df(vix_tk.history(period=period, interval="1d"))
    spx_price   = _normalise_yq_df(bench.history(period=period, interval=interval))
    spx_daily   = _normalise_yq_df(bench.history(period=period, interval="1d"))

    key_stats_raw  = tk.key_stats
    fin_data_raw   = tk.financial_data
    sum_detail_raw = tk.summary_detail

    def _extract(raw, symbol):
        if isinstance(raw, dict):
            val = raw.get(symbol.upper(), raw.get(symbol.lower(), {}))
            return val if isinstance(val, dict) else {}
        return {}

    ks = _extract(key_stats_raw, ticker)
    fd = _extract(fin_data_raw, ticker)
    sd = _extract(sum_detail_raw, ticker)

    fundamentals = {
        "trailingPE":          sd.get("trailingPE", None),
        "forwardPE":           ks.get("forwardPE", None),
        "debtToEquity":        fd.get("debtToEquity", None),
        "totalDebt":           fd.get("totalDebt", None),
        "totalCash":           fd.get("totalCash", None),
        "shortRatio":          ks.get("shortRatio", None),
        "sharesShort":         ks.get("sharesShort", None),
        "sharesOutstanding":   ks.get("sharesOutstanding",
                                      sd.get("sharesOutstanding", None)),
        "shortPercentOfFloat": ks.get("shortPercentOfFloat", None),
    }

    return {
        "price":        price,
        "daily_price":  daily_price,
        "vix":          vix,
        "spx_price":    spx_price,
        "spx_daily":    spx_daily,
        "fundamentals": fundamentals,
    }
