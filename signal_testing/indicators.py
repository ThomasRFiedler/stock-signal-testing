import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Indicator functions
# Each returns 1 (bullish), -1 (bearish), or 0 (neutral).
# Signature: indicator_xxx(signals, i, **ctx) -> int
# ctx carries pre-fetched data so indicators never make API calls.
# ---------------------------------------------------------------------------

def indicator_sma_cross(signals, i, fast_period=10, slow_period=30, **ctx):
    """SMA crossover: 1 if fast > slow, -1 if fast < slow."""
    if i < slow_period:
        return 0
    window   = signals["close"].iloc[max(0, i - slow_period):i + 1]
    fast_sma = window.iloc[-fast_period:].mean()
    slow_sma = window.mean()
    if fast_sma > slow_sma:
        return 1
    elif fast_sma < slow_sma:
        return -1
    return 0


def indicator_rsi(signals, i, period=14, overbought=70.0, oversold=30.0, **ctx):
    """RSI: 1 if oversold, -1 if overbought."""
    if i < period:
        return 0
    closes   = signals["close"].iloc[max(0, i - period):i + 1]
    deltas   = closes.diff().dropna()
    gains    = deltas.clip(lower=0)
    losses   = (-deltas.clip(upper=0))
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 1
    rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    if rsi < oversold:
        return 1
    elif rsi > overbought:
        return -1
    return 0


def indicator_macd(signals, i, fast=12, slow=26, signal_period=9, **ctx):
    """MACD crossover: 1 if MACD > signal line, -1 if below."""
    if i < slow + signal_period:
        return 0
    closes      = signals["close"].iloc[:i + 1]
    macd_line   = closes.ewm(span=fast, adjust=False).mean() - closes.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        return 1
    elif macd_line.iloc[-1] < signal_line.iloc[-1]:
        return -1
    return 0


def indicator_bollinger(signals, i, period=20, num_std=2.0, **ctx):
    """Bollinger Bands: 1 if price < lower band, -1 if above upper band."""
    if i < period:
        return 0
    window = signals["close"].iloc[i - period + 1:i + 1]
    mid    = window.mean()
    std    = window.std()
    px     = signals["close"].iloc[i]
    if px < mid - num_std * std:
        return 1
    elif px > mid + num_std * std:
        return -1
    return 0


def indicator_vwap(signals, i, **ctx):
    """Intraday VWAP: 1 if price < VWAP, -1 if above."""
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


def indicator_obv(signals, i, sma_period=10, **ctx):
    """On-Balance Volume (daily): 1 if OBV > OBV_SMA (accumulation), -1 if below."""
    daily_price = ctx.get("daily_price")
    if daily_price is None or daily_price.empty or len(daily_price) < sma_period + 1:
        return 0

    d_close = daily_price["close"]
    d_vol   = daily_price["volume"]
    obv     = pd.Series(0.0, index=daily_price.index)
    for j in range(1, len(daily_price)):
        if d_close.iloc[j] > d_close.iloc[j - 1]:
            obv.iloc[j] = obv.iloc[j - 1] + d_vol.iloc[j]
        elif d_close.iloc[j] < d_close.iloc[j - 1]:
            obv.iloc[j] = obv.iloc[j - 1] - d_vol.iloc[j]
        else:
            obv.iloc[j] = obv.iloc[j - 1]

    obv_sma  = obv.rolling(window=sma_period).mean()
    bar_date = signals.index[i].normalize()
    valid    = pd.to_datetime(np.array(obv.index)[np.array(obv.index) <= bar_date.tz_localize(None)])
    if valid.empty:
        return 0
    last_date = pd.to_datetime(valid[-1].date())
    obv_val   = obv.loc[last_date]
    sma_val   = obv_sma.loc[last_date]
    if pd.isna(sma_val):
        return 0
    if obv_val > sma_val:
        return 1
    elif obv_val < sma_val:
        return -1
    return 0


def indicator_volume_surge(signals, i, lookback=20, threshold=2.0, **ctx):
    """Volume surge: 1 if bullish surge, -1 if bearish surge."""
    if i < lookback or "volume" not in signals.columns:
        return 0
    avg_vol = signals["volume"].iloc[i - lookback:i].mean()
    if avg_vol == 0 or signals["volume"].iloc[i] <= threshold * avg_vol:
        return 0
    price_change = signals["close"].iloc[i] - signals["close"].iloc[i - 1]
    if price_change > 0:
        return 1
    elif price_change < 0:
        return -1
    return 0


def indicator_vix(signals, i, high_threshold=25.0, low_threshold=15.0, **ctx):
    """VIX contrarian: 1 if VIX > high (fear), -1 if VIX < low (complacency)."""
    vix_df = ctx.get("vix")
    if vix_df is None or vix_df.empty:
        return 0
    bar_date  = signals.index[i].normalize()
    valid     = pd.to_datetime(np.array(vix_df.index)[np.array(vix_df.index) <= bar_date.tz_localize(None)])
    if valid.empty:
        return 0
    vix_close = vix_df.loc[valid[-1], "close"]
    if pd.isna(vix_close):
        return 0
    if vix_close > high_threshold:
        return 1
    elif vix_close < low_threshold:
        return -1
    return 0


def indicator_pe_ratio(signals, i, high_pe=30.0, low_pe=15.0, **ctx):
    """P/E ratio: 1 if undervalued, -1 if overvalued (static snapshot)."""
    pe = ctx.get("fundamentals", {}).get("trailingPE")
    if pe is None or not isinstance(pe, (int, float)) or np.isnan(pe):
        return 0
    if pe < low_pe:
        return 1
    elif pe > high_pe:
        return -1
    return 0


def indicator_debt_signaling(signals, i, high_dte=150.0, low_dte=50.0, **ctx):
    """Debt-to-Equity: 1 if conservatively financed, -1 if heavily leveraged."""
    dte = ctx.get("fundamentals", {}).get("debtToEquity")
    if dte is None or not isinstance(dte, (int, float)) or np.isnan(dte):
        return 0
    if dte < low_dte:
        return 1
    elif dte > high_dte:
        return -1
    return 0


def indicator_short_interest(signals, i, high_ratio=5.0, low_ratio=1.0, **ctx):
    """Short interest ratio: 1 if high (squeeze potential), -1 if very low."""
    sr = ctx.get("fundamentals", {}).get("shortRatio")
    if sr is None or not isinstance(sr, (int, float)) or np.isnan(sr):
        return 0
    if sr > high_ratio:
        return 1
    elif sr < low_ratio:
        return -1
    return 0


def indicator_advance_decline(signals, i, sma_period=10, **ctx):
    """A/D line proxy (SPX daily): 1 if breadth improving, -1 if deteriorating."""
    spx_daily = ctx.get("spx_daily")
    if spx_daily is None or spx_daily.empty or len(spx_daily) < sma_period + 1:
        return 0
    daily_change = spx_daily["close"].diff().dropna()
    ad_line = daily_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).cumsum()
    ad_sma  = ad_line.rolling(window=sma_period).mean()
    bar_date = signals.index[i].normalize()
    valid    = pd.to_datetime(np.array(ad_line.index)[np.array(ad_line.index) <= bar_date.tz_localize(None)])
    if valid.empty:
        return 0
    last_date = valid[-1]
    sma_val   = ad_sma.loc[last_date]
    if pd.isna(sma_val):
        return 0
    if ad_line.loc[last_date] > sma_val:
        return 1
    elif ad_line.loc[last_date] < sma_val:
        return -1
    return 0


def indicator_mcclellan(signals, i, **ctx):
    """McClellan Oscillator proxy: 1 if positive breadth momentum, -1 if negative."""
    spx_daily = ctx.get("spx_daily")
    if spx_daily is None or spx_daily.empty or len(spx_daily) < 40:
        return 0
    ad_values = spx_daily["close"].diff().dropna().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    mcclellan = ad_values.ewm(span=19, adjust=False).mean() - ad_values.ewm(span=39, adjust=False).mean()
    bar_date  = signals.index[i].normalize()
    valid     = pd.to_datetime(np.array(mcclellan.index)[np.array(mcclellan.index) <= bar_date.tz_localize(None)])
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


def indicator_relative_strength(signals, i, lookback=20, **ctx):
    """Relative strength vs. S&P 500: 1 if outperforming, -1 if underperforming."""
    spx_price = ctx.get("spx_price")
    if spx_price is None or spx_price.empty or i < lookback:
        return 0
    ticker_prev = signals["close"].iloc[i - lookback]
    if ticker_prev == 0:
        return 0
    ticker_ret = (signals["close"].iloc[i] / ticker_prev) - 1.0

    spx_idx  = spx_price.index
    pos_now  = max(0, min(spx_idx.searchsorted(signals.index[i], side="right") - 1, len(spx_idx) - 1))
    pos_prev = max(0, min(spx_idx.searchsorted(signals.index[i - lookback], side="right") - 1, len(spx_idx) - 1))
    spx_prev = spx_price["close"].iloc[pos_prev]
    if spx_prev == 0:
        return 0
    spx_ret = (spx_price["close"].iloc[pos_now] / spx_prev) - 1.0

    rs = ticker_ret - spx_ret
    if rs > 0:
        return 1
    elif rs < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Indicator registry
# Each entry: (function, default_weight)
# weight=0 disables without removing the indicator from the list.
# The optimizer overrides weights at runtime — defaults are used for
# standard backtest() calls without explicit weights.
# ---------------------------------------------------------------------------
INDICATORS = [
    (indicator_sma_cross,          1),  # SMA crossover
    (indicator_rsi,                1),  # RSI
    (indicator_macd,               1),  # MACD crossover
    (indicator_bollinger,          1),  # Bollinger Bands
    (indicator_vwap,               1),  # Intraday VWAP
    (indicator_obv,                1),  # On-Balance Volume
    (indicator_volume_surge,       1),  # Volume surge
    (indicator_vix,                1),  # VIX contrarian
    (indicator_pe_ratio,           0),  # P/E ratio (disabled by default)
    (indicator_debt_signaling,     1),  # Debt-to-Equity
    (indicator_short_interest,     1),  # Short interest
    (indicator_advance_decline,    1),  # A/D line proxy
    (indicator_mcclellan,          1),  # McClellan Oscillator proxy
    (indicator_relative_strength,  1),  # RS vs. S&P 500
]


# ---------------------------------------------------------------------------
# Vectorized signal pre-computation
#
# Each _vec_* function takes (signals, ctx) and returns an int8 ndarray of
# shape (n_bars,) with values in {-1, 0, 1} for the full price history.
# compute_signals_matrix() calls all of them once and stacks the result
# into a (n_bars, n_indicators) matrix — eliminating the O(n²) per-bar
# recomputation that was the primary backtest bottleneck.
# ---------------------------------------------------------------------------

def _daily_to_intraday_pos(signals_index, daily_index):
    """
    For each intraday bar, return the integer position of the most recent
    daily bar whose date <= bar's date. Returns -1 where no daily bar exists.
    Uses day-precision comparison to avoid timezone ambiguity.
    """
    bar_d   = pd.DatetimeIndex(signals_index).values.astype("datetime64[D]")
    daily_d = pd.DatetimeIndex(daily_index).values.astype("datetime64[D]")
    pos = np.searchsorted(daily_d, bar_d, side="right") - 1
    return pos


def _vec_sma_cross(signals, ctx):
    close    = signals["close"]
    fast_sma = close.rolling(10).mean()
    slow_sma = close.rolling(30).mean()
    sig = np.where(fast_sma > slow_sma, 1, np.where(fast_sma < slow_sma, -1, 0))
    sig[:30] = 0
    return sig.astype(np.int8)


def _vec_rsi(signals, ctx, period=14, overbought=70.0, oversold=30.0):
    delta    = signals["close"].diff()
    avg_gain = delta.clip(lower=0).rolling(period).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    sig = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
    sig[:period] = 0
    sig = np.where(np.isnan(rsi), 0, sig)
    return sig.astype(np.int8)


def _vec_macd(signals, ctx, fast=12, slow=26, signal_period=9):
    close       = signals["close"]
    macd_line   = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    sig = np.where(macd_line > signal_line, 1, np.where(macd_line < signal_line, -1, 0))
    sig[:slow + signal_period] = 0
    return sig.astype(np.int8)


def _vec_bollinger(signals, ctx, period=20, num_std=2.0):
    close = signals["close"]
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    sig   = np.where(close < mid - num_std * std, 1,
            np.where(close > mid + num_std * std, -1, 0))
    sig[:period] = 0
    sig = np.where(np.isnan(mid), 0, sig)
    return sig.astype(np.int8)


def _vec_vwap(signals, ctx):
    if "volume" not in signals.columns:
        return np.zeros(len(signals), dtype=np.int8)
    typical = (signals["high"] + signals["low"] + signals["close"]) / 3.0
    tv      = typical * signals["volume"]
    dates   = signals.index.date
    cum_tv  = tv.groupby(dates).cumsum()
    cum_vol = signals["volume"].groupby(dates).cumsum()
    vwap    = np.where(cum_vol > 0, cum_tv / cum_vol, np.nan)
    close   = signals["close"].values
    sig     = np.where(np.isnan(vwap), 0,
              np.where(close < vwap, 1,
              np.where(close > vwap, -1, 0)))
    sig[0]  = 0
    return sig.astype(np.int8)


def _vec_obv(signals, ctx, sma_period=10):
    daily = ctx.get("daily_price")
    if daily is None or daily.empty or len(daily) < sma_period + 1:
        return np.zeros(len(signals), dtype=np.int8)
    d_close = daily["close"].values
    d_vol   = daily["volume"].values
    # Vectorised OBV: cumsum of (direction * volume)
    direction  = np.sign(np.diff(d_close))
    increments = np.concatenate([[0.0], direction * d_vol[1:]])
    obv        = np.cumsum(increments)
    obv_sma    = pd.Series(obv).rolling(sma_period).mean().values
    pos        = _daily_to_intraday_pos(signals.index, daily.index)
    valid_pos  = np.clip(pos, 0, len(obv) - 1)
    obv_vals   = obv[valid_pos]
    sma_vals   = obv_sma[valid_pos]
    sig = np.where((pos < 0) | np.isnan(sma_vals), 0,
          np.where(obv_vals > sma_vals, 1,
          np.where(obv_vals < sma_vals, -1, 0)))
    return sig.astype(np.int8)


def _vec_volume_surge(signals, ctx, lookback=20, threshold=2.0):
    if "volume" not in signals.columns:
        return np.zeros(len(signals), dtype=np.int8)
    vol      = signals["volume"]
    avg_vol  = vol.shift(1).rolling(lookback).mean()
    cur_vol  = vol
    surge    = cur_vol > threshold * avg_vol
    price_ch = signals["close"].diff()
    sig      = np.where(surge & (price_ch > 0), 1,
               np.where(surge & (price_ch < 0), -1, 0))
    sig[:lookback] = 0
    return sig.astype(np.int8)


def _vec_vix(signals, ctx, high_threshold=25.0, low_threshold=15.0):
    vix_df = ctx.get("vix")
    if vix_df is None or vix_df.empty:
        return np.zeros(len(signals), dtype=np.int8)
    pos       = _daily_to_intraday_pos(signals.index, vix_df.index)
    valid_pos = np.clip(pos, 0, len(vix_df) - 1)
    vix_vals  = np.where(pos < 0, np.nan, vix_df["close"].values[valid_pos].astype(float))
    sig = np.where(np.isnan(vix_vals), 0,
          np.where(vix_vals > high_threshold, 1,
          np.where(vix_vals < low_threshold, -1, 0)))
    return sig.astype(np.int8)


def _vec_pe_ratio(signals, ctx, high_pe=30.0, low_pe=15.0):
    pe = ctx.get("fundamentals", {}).get("trailingPE")
    if pe is None or not isinstance(pe, (int, float)) or np.isnan(pe):
        return np.zeros(len(signals), dtype=np.int8)
    val = 1 if pe < low_pe else (-1 if pe > high_pe else 0)
    return np.full(len(signals), val, dtype=np.int8)


def _vec_debt_signaling(signals, ctx, high_dte=150.0, low_dte=50.0):
    dte = ctx.get("fundamentals", {}).get("debtToEquity")
    if dte is None or not isinstance(dte, (int, float)) or np.isnan(dte):
        return np.zeros(len(signals), dtype=np.int8)
    val = 1 if dte < low_dte else (-1 if dte > high_dte else 0)
    return np.full(len(signals), val, dtype=np.int8)


def _vec_short_interest(signals, ctx, high_ratio=5.0, low_ratio=1.0):
    sr = ctx.get("fundamentals", {}).get("shortRatio")
    if sr is None or not isinstance(sr, (int, float)) or np.isnan(sr):
        return np.zeros(len(signals), dtype=np.int8)
    val = 1 if sr > high_ratio else (-1 if sr < low_ratio else 0)
    return np.full(len(signals), val, dtype=np.int8)


def _vec_advance_decline(signals, ctx, sma_period=10):
    spx_daily = ctx.get("spx_daily")
    if spx_daily is None or spx_daily.empty or len(spx_daily) < sma_period + 1:
        return np.zeros(len(signals), dtype=np.int8)
    daily_chg = spx_daily["close"].diff().dropna()
    ad_line   = np.cumsum(np.sign(daily_chg.values))
    ad_sma    = pd.Series(ad_line).rolling(sma_period).mean().values
    pos       = _daily_to_intraday_pos(signals.index, daily_chg.index)
    valid_pos = np.clip(pos, 0, len(ad_line) - 1)
    al_vals   = ad_line[valid_pos]
    sma_vals  = ad_sma[valid_pos]
    sig = np.where((pos < 0) | np.isnan(sma_vals), 0,
          np.where(al_vals > sma_vals, 1,
          np.where(al_vals < sma_vals, -1, 0)))
    return sig.astype(np.int8)


def _vec_mcclellan(signals, ctx):
    spx_daily = ctx.get("spx_daily")
    if spx_daily is None or spx_daily.empty or len(spx_daily) < 40:
        return np.zeros(len(signals), dtype=np.int8)
    daily_chg = spx_daily["close"].diff().dropna()
    ad_vals   = pd.Series(np.sign(daily_chg.values), index=daily_chg.index)
    osc       = ad_vals.ewm(span=19, adjust=False).mean() - ad_vals.ewm(span=39, adjust=False).mean()
    pos       = _daily_to_intraday_pos(signals.index, osc.index)
    valid_pos = np.clip(pos, 0, len(osc) - 1)
    osc_vals  = np.where(pos < 0, np.nan, osc.values[valid_pos].astype(float))
    sig = np.where(np.isnan(osc_vals), 0,
          np.where(osc_vals > 0, 1,
          np.where(osc_vals < 0, -1, 0)))
    return sig.astype(np.int8)


def _vec_relative_strength(signals, ctx, lookback=20):
    spx_price = ctx.get("spx_price")
    if spx_price is None or spx_price.empty:
        return np.zeros(len(signals), dtype=np.int8)
    ticker_ret  = signals["close"].pct_change(lookback)
    spx_aligned = spx_price["close"].reindex(signals.index, method="ffill")
    spx_ret     = spx_aligned.pct_change(lookback)
    rs  = ticker_ret.values - spx_ret.values
    sig = np.where(np.isnan(rs), 0, np.where(rs > 0, 1, np.where(rs < 0, -1, 0)))
    sig[:lookback] = 0
    return sig.astype(np.int8)


# Vectorized functions in the same order as INDICATORS
_VEC_FUNCTIONS = [
    _vec_sma_cross,
    _vec_rsi,
    _vec_macd,
    _vec_bollinger,
    _vec_vwap,
    _vec_obv,
    _vec_volume_surge,
    _vec_vix,
    _vec_pe_ratio,
    _vec_debt_signaling,
    _vec_short_interest,
    _vec_advance_decline,
    _vec_mcclellan,
    _vec_relative_strength,
]


def compute_signals_matrix(signals: pd.DataFrame, ctx: dict) -> np.ndarray:
    """
    Pre-compute all indicator signals for the full price history in one pass.

    Returns an int8 array of shape (n_bars, n_indicators). Each column is
    one indicator (same order as INDICATORS), with values -1, 0, or 1.

    Calling this once before the bar loop eliminates the O(n * per_bar_cost)
    indicator overhead, reducing the hot loop to a single dot-product lookup.
    """
    cols = [fn(signals, ctx) for fn in _VEC_FUNCTIONS]
    return np.column_stack(cols).astype(np.int8)
