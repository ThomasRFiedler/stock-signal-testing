"""
Stock Trading Backtest Algorithm
================================
Backtests a signal-based trading strategy using technical indicators.

Dependencies:
    pip install yahooquery pandas numpy matplotlib

Usage:
    from backtest import backtest
    results = backtest("AAPL", n=2, time_frame="60d", interval="5m",
                       take_profit=0.02, stop_loss=0.01)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# 1. DATA FETCHING
# ---------------------------------------------------------------------------

def _normalise_yq_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a yahooquery DataFrame that may have a MultiIndex (symbol, date)
    into a simple DatetimeIndex DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
    return df


def fetch_data(ticker: str, time_frame: str, interval: str) -> dict:
    """
    Fetch all data needed by every indicator in a single batch of API calls.

    Returns a dict with these keys:
        price        – Intraday OHLCV for *ticker* at *interval*.
        daily_price  – Daily OHLCV for *ticker*.
        vix          – Daily OHLCV for ^VIX.
        spx_price    – Intraday OHLCV for ^GSPC at *interval*.
        spx_daily    – Daily OHLCV for ^GSPC.
        fundamentals – dict of snapshot values (P/E, debt-to-equity,
                       short ratio, etc.) from yahooquery modules.
    """
    from yahooquery import Ticker

    # --- Map time_frame to yahooquery period ---
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

    # --- Create Ticker objects ---
    tk     = Ticker(ticker)
    bench  = Ticker("^GSPC")   # S&P 500
    vix_tk = Ticker("^VIX")

    # --- Price histories ---
    price       = _normalise_yq_df(tk.history(period=period, interval=interval))
    daily_price = _normalise_yq_df(tk.history(period=period, interval="1d"))
    vix         = _normalise_yq_df(vix_tk.history(period=period, interval="1d"))
    spx_price   = _normalise_yq_df(bench.history(period=period, interval=interval))
    spx_daily   = _normalise_yq_df(bench.history(period=period, interval="1d"))

    # --- Fundamental snapshot (one API call per module) ---
    # key_stats      → shortRatio, sharesShort, forwardPE, etc.
    # financial_data → totalDebt, debtToEquity, totalCash, etc.
    # summary_detail → trailingPE
    key_stats_raw  = tk.key_stats
    fin_data_raw   = tk.financial_data
    sum_detail_raw = tk.summary_detail

    def _extract(raw, symbol):
        """Pull the dict for *symbol* out of a yahooquery module response."""
        if isinstance(raw, dict):
            val = raw.get(symbol.upper(), raw.get(symbol.lower(), {}))
            return val if isinstance(val, dict) else {}
        return {}

    ks = _extract(key_stats_raw, ticker)
    fd = _extract(fin_data_raw, ticker)
    sd = _extract(sum_detail_raw, ticker)

    fundamentals = {
        # P/E
        "trailingPE":         sd.get("trailingPE", None),
        "forwardPE":          ks.get("forwardPE", None),
        # Debt
        "debtToEquity":       fd.get("debtToEquity", None),
        "totalDebt":          fd.get("totalDebt", None),
        "totalCash":          fd.get("totalCash", None),
        # Short interest
        "shortRatio":         ks.get("shortRatio", None),
        "sharesShort":        ks.get("sharesShort", None),
        "sharesOutstanding":  ks.get("sharesOutstanding",
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


# ---------------------------------------------------------------------------
# 2. INDICATOR FUNCTIONS  (each returns 1, 0, or -1)
#
#    Signature convention:
#        indicator_xxx(signals, i, **ctx) -> int
#
#    *ctx* carries pre-fetched data (daily_price, vix, spx_price, spx_daily,
#    fundamentals) so that indicators never make their own API calls.
# ---------------------------------------------------------------------------

# ===== ORIGINAL TECHNICAL INDICATORS =======================================

def indicator_sma_cross(signals: pd.DataFrame, i: int,
                        fast_period: int = 10, slow_period: int = 30,
                        **ctx) -> int:
    """
    Simple Moving Average crossover.
    Returns  1  if fast SMA > slow SMA  (bullish).
    Returns -1  if fast SMA < slow SMA  (bearish).
    Returns  0  if not enough data.
    """
    if i < slow_period:
        return 0
    window = signals["close"].iloc[max(0, i - slow_period):i + 1]
    fast_sma = window.iloc[-fast_period:].mean()
    slow_sma = window.mean()
    if fast_sma > slow_sma:
        return 1
    elif fast_sma < slow_sma:
        return -1
    return 0


def indicator_rsi(signals: pd.DataFrame, i: int, period: int = 14,
                  overbought: float = 70.0, oversold: float = 30.0,
                  **ctx) -> int:
    """
    Relative Strength Index.
    Returns  1  if RSI < oversold   (buy).
    Returns -1  if RSI > overbought (sell).
    Returns  0  otherwise.
    """
    if i < period:
        return 0
    closes = signals["close"].iloc[max(0, i - period):i + 1]
    deltas = closes.diff().dropna()
    gains  = deltas.clip(lower=0)
    losses = (-deltas.clip(upper=0))
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 1
    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if rsi < oversold:
        return 1
    elif rsi > overbought:
        return -1
    return 0


def indicator_macd(signals: pd.DataFrame, i: int,
                   fast: int = 12, slow: int = 26, signal_period: int = 9,
                   **ctx) -> int:
    """
    MACD crossover.
    Returns  1  if MACD line > signal line (bullish).
    Returns -1  if MACD line < signal line (bearish).
    Returns  0  if not enough data.
    """
    needed = slow + signal_period
    if i < needed:
        return 0
    closes     = signals["close"].iloc[:i + 1]
    ema_fast   = closes.ewm(span=fast, adjust=False).mean()
    ema_slow   = closes.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        return 1
    elif macd_line.iloc[-1] < signal_line.iloc[-1]:
        return -1
    return 0


def indicator_bollinger(signals: pd.DataFrame, i: int,
                        period: int = 20, num_std: float = 2.0,
                        **ctx) -> int:
    """
    Bollinger Bands.
    Returns  1  if price < lower band (oversold).
    Returns -1  if price > upper band (overbought).
    Returns  0  otherwise.
    """
    if i < period:
        return 0
    window = signals["close"].iloc[i - period + 1:i + 1]
    mid   = window.mean()
    std   = window.std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    px    = signals["close"].iloc[i]
    if px < lower:
        return 1
    elif px > upper:
        return -1
    return 0


def indicator_vwap(signals: pd.DataFrame, i: int, **ctx) -> int:
    """
    VWAP (intraday, resets each trading day).
    Returns  1  if price < VWAP (potential long).
    Returns -1  if price > VWAP (potential short).
    Returns  0  if no volume data.
    """
    if i < 1:
        return 0
    current_date = signals.index[i].date()
    day_mask = signals.index[:i + 1].date == current_date
    day_data = signals.iloc[:i + 1][day_mask]
    if day_data.empty or "volume" not in day_data.columns or day_data["volume"].sum() == 0:
        return 0
    typical = (day_data["high"] + day_data["low"] + day_data["close"]) / 3.0
    vwap    = (typical * day_data["volume"]).sum() / day_data["volume"].sum()
    px      = signals["close"].iloc[i]
    if px < vwap:
        return 1
    elif px > vwap:
        return -1
    return 0


# ===== VOLUME-BASED INDICATORS =============================================

def indicator_obv(signals: pd.DataFrame, i: int,
                  sma_period: int = 10, **ctx) -> int:
    """
    On-Balance Volume (OBV).

    Data source: daily_price (passed via ctx).  Daily granularity avoids
    noisy intraday volume spikes.  The daily OBV value is extrapolated to
    the intraday bar by matching on date.

    Algorithm
    ---------
    1. Build a cumulative OBV series from daily close and volume:
       - If today's close > yesterday's close: OBV += today's volume.
       - If today's close < yesterday's close: OBV -= today's volume.
       - If unchanged: OBV unchanged.
    2. Compute a *sma_period*-day SMA of OBV.
    3. Map the current intraday bar to the most recent daily OBV value.
    4. Signal:
       - OBV > OBV_SMA → accumulation (buying pressure) → 1.
       - OBV < OBV_SMA → distribution (selling pressure) → -1.
       - Otherwise → 0.

    Reference: Granville, J. (1963). *Granville's New Key to Stock Market
    Profits*.  OBV theory: volume precedes price; rising OBV with a flat
    price suggests upcoming breakout.
    """
    daily_price = ctx.get("daily_price")
    if daily_price is None or daily_price.empty:
        return 0
    if len(daily_price) < sma_period + 1:
        return 0

    d_close = daily_price["close"]
    d_vol   = daily_price["volume"]

    # Build OBV series
    obv = pd.Series(0.0, index=daily_price.index)
    for j in range(1, len(daily_price)):
        if d_close.iloc[j] > d_close.iloc[j - 1]:
            obv.iloc[j] = obv.iloc[j - 1] + d_vol.iloc[j]
        elif d_close.iloc[j] < d_close.iloc[j - 1]:
            obv.iloc[j] = obv.iloc[j - 1] - d_vol.iloc[j]
        else:
            obv.iloc[j] = obv.iloc[j - 1]

    obv_sma = obv.rolling(window=sma_period).mean()

    # Map intraday bar → most recent daily OBV
    bar_date = signals.index[i].normalize()
    valid = pd.to_datetime(np.array(obv.index)[np.array(obv.index) <= bar_date.tz_localize(None)])

    if valid.empty:
        return 0
    last_date = pd.to_datetime(valid[-1].date())
    obv_val = obv.loc[last_date]
    sma_val = obv_sma.loc[last_date]
    if pd.isna(sma_val):
        return 0

    if obv_val > sma_val:
        return 1
    elif obv_val < sma_val:
        return -1
    return 0


def indicator_volume_surge(signals: pd.DataFrame, i: int,
                           lookback: int = 20, threshold: float = 2.0,
                           **ctx) -> int:
    """
    Volume Surge detector (intraday timeframe).

    Compares the current bar's volume to a rolling average of the previous
    *lookback* bars.

    Algorithm
    ---------
    1. Compute a *lookback*-bar SMA of volume (excluding the current bar).
    2. If current bar volume > *threshold* × SMA:
       - Price rose vs. prior bar → 1 (bullish surge).
       - Price fell vs. prior bar → -1 (bearish surge).
    3. Otherwise → 0.

    Default threshold of 2× is a common institutional benchmark for
    identifying "unusual" volume events.
    """
    if i < lookback:
        return 0
    if "volume" not in signals.columns:
        return 0

    vol_window = signals["volume"].iloc[i - lookback:i]
    avg_vol    = vol_window.mean()
    if avg_vol == 0:
        return 0

    cur_vol = signals["volume"].iloc[i]
    if cur_vol <= threshold * avg_vol:
        return 0

    # Determine direction
    if i < 1:
        return 0
    price_change = signals["close"].iloc[i] - signals["close"].iloc[i - 1]
    if price_change > 0:
        return 1
    elif price_change < 0:
        return -1
    return 0


# ===== SENTIMENT INDICATORS ================================================

def indicator_vix(signals: pd.DataFrame, i: int,
                  high_threshold: float = 25.0, low_threshold: float = 15.0,
                  **ctx) -> int:
    """
    VIX (CBOE Volatility Index) sentiment signal.

    Data source: daily VIX data (passed via ctx["vix"]).

    Algorithm
    ---------
    1. Find the most recent daily VIX close on or before the current bar's
       date.
    2. Signal (contrarian interpretation):
       - VIX > *high_threshold* (default 25) → elevated fear → 1.
         Rationale: extreme fear tends to precede mean-reversion rallies.
       - VIX < *low_threshold* (default 15) → complacency → -1.
         Rationale: low-vol environments often precede corrections.
       - Otherwise → 0.

    Reference: Whaley, R. (2000). "The Investor Fear Gauge."
    *Journal of Portfolio Management*, 26(3), 12-17.
    """
    vix_df = ctx.get("vix")
    if vix_df is None or vix_df.empty:
        return 0

    bar_date = signals.index[i].normalize()
    valid    = pd.to_datetime(np.array(vix_df.index)[np.array(vix_df.index) <= bar_date.tz_localize(None)])
    if valid.empty:
        return 0

    vix_close = vix_df.loc[valid[-1], "close"]
    if pd.isna(vix_close):
        return 0

    if vix_close > high_threshold:
        return 1   # contrarian bullish
    elif vix_close < low_threshold:
        return -1  # contrarian bearish
    return 0


# ===== FUNDAMENTAL / FINANCIAL INDICATORS ==================================

def indicator_pe_ratio(signals: pd.DataFrame, i: int,
                       high_pe: float = 30.0, low_pe: float = 15.0,
                       **ctx) -> int:
    """
    Price-to-Earnings Ratio signal.

    Data source: trailingPE from yahooquery's summary_detail module
    (passed via ctx["fundamentals"]["trailingPE"]).  This is a static
    snapshot — it does not change bar-by-bar, so it provides a persistent
    valuation bias across the entire backtest window.

    Algorithm
    ---------
    - trailingPE < *low_pe* (default 15) → undervalued → 1.
    - trailingPE > *high_pe* (default 30) → overvalued  → -1.
    - Otherwise → 0.

    Note: P/E alone is a coarse valuation measure.  It is most useful when
    combined with momentum and breadth signals to filter false positives
    (e.g., a "cheap" stock in secular decline).
    """
    fundamentals = ctx.get("fundamentals", {})
    pe = fundamentals.get("trailingPE")
    if pe is None or not isinstance(pe, (int, float)):
        return 0
    if np.isnan(pe):
        return 0
    if pe < low_pe:
        return 1
    elif pe > high_pe:
        return -1
    return 0


def indicator_debt_signaling(signals: pd.DataFrame, i: int,
                             high_dte: float = 150.0, low_dte: float = 50.0,
                             **ctx) -> int:
    """
    Debt-to-Equity (D/E) signaling.

    Data source: debtToEquity from yahooquery's financial_data module
    (passed via ctx["fundamentals"]["debtToEquity"]).  Static snapshot.

    Algorithm
    ---------
    - D/E < *low_dte* (default 50)  → conservatively financed → 1.
    - D/E > *high_dte* (default 150) → heavily leveraged → -1.
    - Otherwise → 0.

    Rationale: high leverage amplifies equity volatility and increases
    default risk, while low leverage signals financial stability.

    Reference: Modigliani, F. & Miller, M. (1958). "The Cost of Capital,
    Corporation Finance and the Theory of Investment." *American Economic
    Review*, 48(3), 261-297.
    """
    fundamentals = ctx.get("fundamentals", {})
    dte = fundamentals.get("debtToEquity")
    if dte is None or not isinstance(dte, (int, float)):
        return 0
    if np.isnan(dte):
        return 0
    if dte < low_dte:
        return 1
    elif dte > high_dte:
        return -1
    return 0


def indicator_short_interest(signals: pd.DataFrame, i: int,
                             high_ratio: float = 5.0, low_ratio: float = 1.0,
                             **ctx) -> int:
    """
    Short Interest Ratio (Days-to-Cover) signal.

    Data source: shortRatio from yahooquery's key_stats module
    (passed via ctx["fundamentals"]["shortRatio"]).  Static snapshot.
    The short ratio equals shares shorted ÷ average daily volume,
    representing the number of days needed for shorts to cover.

    Algorithm
    ---------
    - shortRatio > *high_ratio* (default 5) → heavy short interest → 1.
      Rationale: high days-to-cover creates short-squeeze potential, which
      is a contrarian bullish catalyst.
    - shortRatio < *low_ratio* (default 1) → minimal short interest → -1.
      Rationale: very low short interest removes squeeze pressure; it can
      also indicate that informed short sellers see no downside, but here
      we treat it as absence of a bullish catalyst.
    - Otherwise → 0.

    Reference: Desai, H., Ramesh, K., Thiagarajan, S.R. & Balachandran,
    B.V. (2002). "An Investigation of the Informational Role of Short
    Interest in the Nasdaq Market." *Journal of Finance*, 57(5), 2263-2287.
    """
    fundamentals = ctx.get("fundamentals", {})
    sr = fundamentals.get("shortRatio")
    if sr is None or not isinstance(sr, (int, float)):
        return 0
    if np.isnan(sr):
        return 0
    if sr > high_ratio:
        return 1   # contrarian bullish (squeeze potential)
    elif sr < low_ratio:
        return -1
    return 0


# ===== MARKET BREADTH INDICATORS ===========================================

def indicator_advance_decline(signals: pd.DataFrame, i: int,
                              sma_period: int = 10, **ctx) -> int:
    """
    Advance / Decline Line proxy using S&P 500 daily data.

    Yahoo Finance does not expose raw NYSE advancing/declining issue counts
    through yahooquery.  We approximate market breadth by building a
    cumulative A/D line from daily S&P 500 close-to-close changes:
    each up-day scores +1, each down-day scores -1.

    Algorithm
    ---------
    1. Compute daily close-to-close direction from spx_daily:
       +1 if SPX closed up, -1 if down, 0 if flat.
    2. Cumulative sum → A/D line.
    3. Compute *sma_period*-day SMA of the A/D line.
    4. Map the intraday bar to the most recent daily A/D value.
    5. Signal:
       - A/D > A/D_SMA → breadth is improving → 1.
       - A/D < A/D_SMA → breadth is deteriorating → -1.
       - Otherwise → 0.

    Limitation: single-index proxy, not a true multi-stock A/D count.
    For production, consider an external breadth data feed.
    """
    spx_daily = ctx.get("spx_daily")
    if spx_daily is None or spx_daily.empty:
        return 0
    if len(spx_daily) < sma_period + 1:
        return 0

    daily_change = spx_daily["close"].diff().dropna()
    ad_values = daily_change.apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    ad_line = ad_values.cumsum()
    ad_sma  = ad_line.rolling(window=sma_period).mean()

    bar_date = signals.index[i].normalize()
    #valid    = pd.to_datetime(np.array(ad_line.index)[np.array(ad_line.index[0]) <= np.array(bar_date.tz_localize(None))])
    valid = pd.to_datetime(np.array(ad_line.index)[np.array(ad_line.index) <= bar_date.tz_localize(None)])
    if valid.empty:
        return 0
    last_date = valid[-1]
    ad_val  = ad_line.loc[last_date]
    sma_val = ad_sma.loc[last_date]
    if pd.isna(sma_val):
        return 0

    if ad_val > sma_val:
        return 1
    elif ad_val < sma_val:
        return -1
    return 0


def indicator_mcclellan(signals: pd.DataFrame, i: int, **ctx) -> int:
    """
    McClellan Oscillator proxy.

    The canonical McClellan Oscillator = 19-day EMA(A-D) − 39-day EMA(A-D),
    where A-D is the daily count of NYSE advancing issues minus declining
    issues.  Since yahooquery does not provide raw A/D issue counts, we
    approximate using the same daily SPX direction proxy as the A/D line
    indicator (+1 / -1 per day).

    Algorithm
    ---------
    1. Compute daily A-D values (+1 / -1) from spx_daily close changes.
    2. Compute 19-day EMA and 39-day EMA of the A-D series.
    3. McClellan Oscillator = EMA_19 − EMA_39.
    4. Map the intraday bar to the most recent daily oscillator value.
    5. Signal:
       - Oscillator > 0 → positive breadth momentum → 1.
       - Oscillator < 0 → negative breadth momentum → -1.
       - Otherwise → 0.

    Reference: McClellan, S.B. & McClellan, M.B. (2001).
    *Marvels of Wall Street*.
    """
    spx_daily = ctx.get("spx_daily")
    if spx_daily is None or spx_daily.empty:
        return 0
    if len(spx_daily) < 40:
        return 0

    daily_change = spx_daily["close"].diff().dropna()
    ad_values = daily_change.apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    ema_19    = ad_values.ewm(span=19, adjust=False).mean()
    ema_39    = ad_values.ewm(span=39, adjust=False).mean()
    mcclellan = ema_19 - ema_39

    bar_date = signals.index[i].normalize()
    valid    = pd.to_datetime(np.array(mcclellan.index)[np.array(mcclellan.index) <= bar_date.tz_localize(None)])
    if valid.empty:
        return 0
    osc_val = mcclellan.loc[valid[-1]]
    if pd.isna(osc_val):
        return 0

    if osc_val > 0:
        return 1
    elif osc_val < 0:
        return -1
    return 0


# ===== RELATIVE STRENGTH (VS. BENCHMARK) ===================================

def indicator_relative_strength(signals: pd.DataFrame, i: int,
                                lookback: int = 20, **ctx) -> int:
    """
    Relative Strength (RS) vs. the S&P 500.

    Compares the ticker's return over the last *lookback* intraday bars to
    the S&P 500's return over the same bars.

    Data source: spx_price (intraday ^GSPC data at the same interval,
    passed via ctx).

    Algorithm
    ---------
    1. Ticker return = close[i] / close[i − lookback] − 1.
    2. SPX return over the same timestamp range (nearest-match if bars
       do not align exactly).
    3. RS = ticker return − SPX return.
    4. Signal:
       - RS > 0 → ticker outperforming the benchmark → 1.
       - RS < 0 → ticker underperforming → -1.
       - Otherwise → 0.

    Reference: Levy, R.A. (1967). "Relative Strength as a Criterion for
    Investment Selection." *Journal of Finance*, 22(4), 595-610.
    """
    spx_price = ctx.get("spx_price")
    if spx_price is None or spx_price.empty:
        return 0
    if i < lookback:
        return 0

    # Ticker return
    ticker_now  = signals["close"].iloc[i]
    ticker_prev = signals["close"].iloc[i - lookback]
    if ticker_prev == 0:
        return 0
    ticker_ret = (ticker_now / ticker_prev) - 1.0

    # SPX return – nearest-match timestamps
    ts_now  = signals.index[i]
    ts_prev = signals.index[i - lookback]

    spx_idx  = spx_price.index
    if len(spx_idx) == 0:
        return 0
    pos_now  = max(0, min(spx_idx.searchsorted(ts_now, side="right") - 1,
                          len(spx_idx) - 1))
    pos_prev = max(0, min(spx_idx.searchsorted(ts_prev, side="right") - 1,
                          len(spx_idx) - 1))

    spx_now  = spx_price["close"].iloc[pos_now]
    spx_prev = spx_price["close"].iloc[pos_prev]
    if spx_prev == 0:
        return 0
    spx_ret = (spx_now / spx_prev) - 1.0

    rs = ticker_ret - spx_ret
    if rs > 0:
        return 1
    elif rs < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# INDICATOR REGISTRY
# ---------------------------------------------------------------------------

# Each entry is (indicator_function, weight).
# A weight of 0 disables the indicator without removing it from the list.
# Higher weights contribute more to the signal sum compared against n.
INDICATORS = [
    # --- Original technical ---
    (indicator_sma_cross,           1),  # SMA crossover (fast/slow)
    (indicator_rsi,                 1),  # Relative Strength Index
    (indicator_macd,                1),  # MACD crossover
    (indicator_bollinger,           1),  # Bollinger Bands
    (indicator_vwap,                1),  # Intraday VWAP
    # --- Volume-based ---
    (indicator_obv,                 1),  # On-Balance Volume (daily → intraday)
    (indicator_volume_surge,        1),  # Volume surge detector (intraday)
    # --- Sentiment ---
    (indicator_vix,                 1),  # VIX contrarian signal
    # --- Fundamental / Financial ---
    (indicator_pe_ratio,            0),  # Trailing P/E ratio (disabled)
    (indicator_debt_signaling,      1),  # Debt-to-Equity ratio
    (indicator_short_interest,      1),  # Short interest ratio (days-to-cover)
    # --- Market breadth ---
    (indicator_advance_decline,     1),  # A/D line proxy (SPX daily)
    (indicator_mcclellan,           1),  # McClellan Oscillator proxy (SPX daily)
    # --- Relative strength ---
    (indicator_relative_strength,   1),  # RS vs. S&P 500 (intraday)
]


# ---------------------------------------------------------------------------
# 3. BACKTEST ENGINE
# ---------------------------------------------------------------------------

def backtest(
    ticker: str,
    n: int,
    time_frame: str,
    interval: str,
    take_profit: float,
    stop_loss: float,
    position_size: float = 100.0,
    starting_equity: float = 10_000.0,
):
    """
    Run a backtest on *ticker* using a multi-indicator voting strategy.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    n : int
        Number of indicator agreements required to open a position.
        Long when sum(signals) >= n.  Short when sum(signals) <= -n.
    time_frame : str
        Historical window: "1w", "60d", or "1y".
    interval : str
        Bar interval: "1m", "5m", "15m", "30m", or "1h".
    take_profit : float
        Fractional take-profit threshold (e.g. 0.02 = 2 %).
    stop_loss : float
        Fractional stop-loss threshold (e.g. 0.01 = 1 %).
    position_size : float
        Dollar amount per trade (default 100).
    starting_equity : float
        Starting account equity in dollars (default 10,000).

    Returns
    -------
    dict with keys:
        pnl, sharpe_ratio, max_drawdown, total_trades, profit_factor,
        trades_df, equity_series, price_df
    Also prints the trade log and saves charts to disk.
    """

    # ------------------------------------------------------------------
    # Step 1 – Fetch all data (single batch of API calls)
    # ------------------------------------------------------------------
    data = fetch_data(ticker, time_frame, interval)

    price        = data["price"]
    daily_price  = data["daily_price"]
    vix          = data["vix"]
    spx_price    = data["spx_price"]
    spx_daily    = data["spx_daily"]
    fundamentals = data["fundamentals"]

    if price.empty:
        raise RuntimeError(f"No price data returned for {ticker}.")

    # ------------------------------------------------------------------
    # Step 2 – Create signals dataframe (copy of price)
    # ------------------------------------------------------------------
    signals = price.copy()

    # Context dict passed to every indicator (no extra API calls)
    ctx = {
        "daily_price":  daily_price,
        "vix":          vix,
        "spx_price":    spx_price,
        "spx_daily":    spx_daily,
        "fundamentals": fundamentals,
    }

    # ------------------------------------------------------------------
    # Steps 3-5 – Iterate bars, compute signals, manage positions
    # ------------------------------------------------------------------
    trades         = []
    equity_curve   = []
    cumulative_pnl = 0.0

    # Position state
    in_position      = False
    position_type    = None      # "long" or "short"
    entry_price      = None
    entry_time       = None
    entry_signal_sum = None
    trade_high       = None      # for favorable excursion
    trade_low        = None      # for adverse excursion


    num_bars = len(signals)

    for i in range(num_bars):
        bar_time  = signals.index[i]
        bar_close = signals["close"].iloc[i]
        bar_high  = signals["high"].iloc[i]
        bar_low   = signals["low"].iloc[i]

        # --- Step 3: compute indicator signals ---
        ind_values = [w * ind_func(signals, i, **ctx) for ind_func, w in INDICATORS]
        signal_sum = sum(ind_values)

        is_last_bar_of_day = (
            i == num_bars - 1
            or signals.index[i + 1].date() != bar_time.date()
        )

        # --- Step 5: if position is open, check exit conditions ---
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

            # End-of-day exit: last bar of the trading day
            if exit_price is None and is_last_bar_of_day:
                exit_price, exit_reason = bar_close, "eod_exit"

            # Execute exit
            if exit_price is not None:
                if position_type == "long":
                    net_pnl      = (exit_price - entry_price) / entry_price * position_size
                    favorable_exc = (trade_high - entry_price) / entry_price
                    adverse_exc   = (entry_price - trade_low) / entry_price
                else:
                    net_pnl      = (entry_price - exit_price) / entry_price * position_size
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

        # --- Step 4: if no position, check entry conditions ---
        if not in_position:
            if not is_last_bar_of_day:
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

    if total_trades > 0:
        winning      = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"]
        losing       = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"]
        gross_profit = winning.sum() if len(winning) > 0 else 0.0
        gross_loss   = abs(losing.sum()) if len(losing) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss
                         if gross_loss != 0 else float("inf"))

        returns = trades_df["net_pnl"] / position_size
        sharpe_ratio = ((returns.mean() / returns.std()) * np.sqrt(252)
                        if returns.std() != 0 else 0.0)

        eq_series   = pd.Series([e[1] for e in equity_curve],
                                index=[e[0] for e in equity_curve])
        running_max = eq_series.cummax()
        drawdown    = eq_series - running_max
        max_drawdown = drawdown.min()
    else:
        profit_factor = 0.0
        sharpe_ratio  = 0.0
        max_drawdown  = 0.0
        eq_series     = pd.Series([e[1] for e in equity_curve],
                                  index=[e[0] for e in equity_curve])

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 7 – Trade log
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 8 – Charts
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.6], hspace=0.35)

    # 8a. Equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eq_series.index, eq_series.values, color="#2196F3", linewidth=1.2)
    ax1.fill_between(eq_series.index, eq_series.values,
                     alpha=0.15, color="#2196F3")
    ax1.set_title(f"Equity Curve – {ticker}", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.axhline(starting_equity, color="grey", linewidth=0.5, linestyle="--")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30)

    # 8b. Price chart with trade markers
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(signals.index, signals["close"], color="#455A64", linewidth=0.8,
             label="Close")
    ax2.set_title(f"Price Chart with Trades – {ticker}",
                  fontsize=13, fontweight="bold")
    ax2.set_ylabel("Price ($)")
    ax2.grid(True, alpha=0.3)

    if total_trades > 0:
        for _, trade in trades_df.iterrows():
            entry_dt = pd.Timestamp(
                f"{trade['entry_date']} {trade['entry_time']}")
            exit_dt  = pd.Timestamp(
                f"{trade['exit_date']} {trade['exit_time']}")
            if trade["type"] == "long":
                ax2.annotate("▲", xy=(entry_dt, trade["entry_price"]),
                             fontsize=10, color="green",
                             ha="center", va="bottom")
                ax2.annotate("×", xy=(exit_dt, trade["exit_price"]),
                             fontsize=12, color="darkgreen",
                             ha="center", va="center", fontweight="bold")
            else:
                ax2.annotate("▼", xy=(entry_dt, trade["entry_price"]),
                             fontsize=10, color="red",
                             ha="center", va="top")
                ax2.annotate("×", xy=(exit_dt, trade["exit_price"]),
                             fontsize=12, color="darkred",
                             ha="center", va="center", fontweight="bold")

    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # 8c. Per-trade P/L bar chart
    ax3 = fig.add_subplot(gs[2])
    if total_trades > 0:
        colors = ["green" if pnl > 0 else "red"
                  for pnl in trades_df["net_pnl"]]
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

    return {
        "pnl":            round(total_pnl, 4),
        "sharpe_ratio":   round(sharpe_ratio, 4),
        "max_drawdown":   round(max_drawdown, 4),
        "total_trades":   total_trades,
        "profit_factor":  round(profit_factor, 4),
        "trades_df":      trades_df,
        "equity_series":  eq_series,
        "price_df":       signals,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# NOTE: This file is kept for reference. The project has been refactored
# into the signal_testing/ package. Use the following instead:
#
#   from signal_testing import backtest
#   python optimize.py AAPL
#   python run_walk_forward.py AAPL
#
if __name__ == "__main__":
    results = backtest(
        ticker="AAPL",
        n=2,
        time_frame="60d",
        interval="5m",
        take_profit=0.02,
        stop_loss=0.01,
    )