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
