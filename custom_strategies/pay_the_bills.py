"""custom_strategies/pay_the_bills.py

"Pay the Bills" universal trend-following strategy with an embedded
volatility-shock exit guard.

Asset-agnostic by design (no ticker-specific branching) and timeframe-agnostic
via ``get_bars_for_period``. Designed for steady, cash-beating compounding
across diversified liquid ETFs with shallow drawdowns using the engine's
default equal-weight (``fixed``, 10% per ETF) sizing and realistic costs.

Reproducible official run (10-ETF macro universe, 2008-01-01 to 2026-11-20):
    python main.py \\
        --name ptb-official \\
        --provider yahoo \\
        --symbols SPY QQQ IWM EFA XLP XLU XLV IEF GLD DBC \\
        --start 2008-01-01 --end 2026-11-20 \\
        --strategies "Trend Vol-Guard (EMA20/SMA150)" \\
        --wfa-split 0.80 --wfa-folds 3 \\
        --mc-sampling block --seed 42 --verbose
"""

import numpy as np
import pandas as pd

from config import CONFIG
from helpers.registry import register_strategy
from helpers.timeframe_utils import get_bars_for_period

_TF = CONFIG.get("timeframe", "D")
_MUL = CONFIG.get("timeframe_multiplier", 1)


@register_strategy(
    name="Trend Vol-Guard (EMA20/SMA150)",
    dependencies=[],
    params={
        "fast_ema": get_bars_for_period("20d", _TF, _MUL),
        "slow_sma": get_bars_for_period("150d", _TF, _MUL),
        "vol_baseline": get_bars_for_period("126d", _TF, _MUL),
        "vol_shock_max": 1.50,
    },
)
def trend_vol_guard(df, **kwargs):
    """Hold long when EMA(20) > SMA(150) and 20d realized volatility is below
    1.50x its 126d baseline; exit to cash otherwise.

    Combines a medium-to-long-term moving-average trend filter (smoothing raw
    daily closes with a 20-bar EMA against a 150-bar SMA to suppress short-term
    whipsaws) with a scale-free volatility-shock guard embedded directly in the
    signal. Embedding the shock exit inside the signal rather than using a
    static stop loss avoids blocking re-entry under ``entry_trigger="edge"``:
    once volatility normalizes back below ``vol_shock_max`` and the trend
    remains intact, the signal cleanly flips from -1 back to 1.

    Signal convention: 1 = enter/hold long, -1 = exit/flat, 0 = warmup or
    undefined baseline volatility.

    Parameters
    ----------
    fast_ema : int
        Exponential moving average span and short-term realized volatility
        window in bars (default: 20 trading days).
    slow_sma : int
        Simple moving average anchor window in bars (default: 150 trading days).
    vol_baseline : int
        Long-term realized volatility baseline window in bars (default: 126
        trading days, ~6 months).
    vol_shock_max : float
        Maximum allowed ratio of short-term to baseline realized volatility
        (default: 1.50). Readings at or above this threshold force an exit.
    """
    fast_len = int(kwargs["fast_ema"])
    slow_len = int(kwargs["slow_sma"])
    vol_baseline = int(kwargs["vol_baseline"])
    vol_shock_max = float(kwargs["vol_shock_max"])

    if fast_len < 1 or slow_len < 1 or vol_baseline < 2 or vol_shock_max <= 0:
        raise ValueError(
            f"Invalid parameters: fast_ema={fast_len}, slow_sma={slow_len}, "
            f"vol_baseline={vol_baseline}, vol_shock_max={vol_shock_max}"
        )

    if df.empty:
        df["Signal"] = pd.Series(dtype=int, index=df.index)
        return df

    close = df["Close"].astype(float)
    ema_fast = close.ewm(span=fast_len, adjust=False).mean()
    sma_slow = close.rolling(window=slow_len).mean()

    rets = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    short_vol_w = max(2, fast_len)
    vol_short = rets.rolling(window=short_vol_w).std()
    vol_long = rets.rolling(window=vol_baseline).std().replace(0.0, np.nan)
    vol_ratio = (vol_short / vol_long).replace([np.inf, -np.inf], np.nan)

    valid = sma_slow.notna() & vol_ratio.notna()
    long_cond = valid & (ema_fast > sma_slow) & (vol_ratio < vol_shock_max)

    df["Signal"] = 0
    df.loc[long_cond, "Signal"] = 1
    df.loc[valid & ~long_cond, "Signal"] = -1
    return df
