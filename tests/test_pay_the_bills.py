"""tests/test_pay_the_bills.py

Unit and integration tests for custom_strategies/pay_the_bills.py.

Verifies:
  1. Auto-discovery via get_active_strategies() with clean registry hygiene.
  2. Signal contract: valid integer values in {-1, 0, 1}, no NaNs, neutral warmup.
  3. Core behavior: uptrend long, downtrend flat, and vol-shock exit + auto re-entry.
  4. Strict causality (zero look-ahead bias): mutating future bars never alters past signals.
  5. Edge cases: empty frame, short frame, flat/zero-vol series, zero price, invalid params.
  6. End-to-end execution through helpers.portfolio_simulations.run_portfolio_simulation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers.portfolio_simulations import run_portfolio_simulation  # noqa: E402
from helpers.registry import get_active_strategies  # noqa: E402

STRATEGY_NAME = "Trend Vol-Guard (EMA20/SMA150)"


def _make_ohlcv(closes: np.ndarray, start: str = "2020-01-02") -> pd.DataFrame:
    """Build a well-formed OHLCV DataFrame from a Close price array."""
    dates = pd.bdate_range(start, periods=len(closes), freq="B")
    close = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(len(close), 2_000_000.0),
        },
        index=dates,
    )
    df.index.name = "Datetime"
    return df


def _make_multi_regime_ohlcv(n: int = 360, seed: int = 7) -> pd.DataFrame:
    """Deterministic multi-regime synthetic OHLCV series."""
    rng = np.random.default_rng(seed)
    drift = np.where(np.arange(n) < 220, 0.0012, -0.0006)
    shock = rng.normal(0.0, 0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(drift + shock))
    df = _make_ohlcv(close)
    df["Open"] = close * (1.0 + rng.normal(0.0, 0.003, size=n))
    df["High"] = np.maximum(df["Open"], close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n)))
    df["Low"] = np.minimum(df["Open"], close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n)))
    return df


@pytest.fixture(scope="module")
def strategy_entry():
    active = get_active_strategies()
    assert STRATEGY_NAME in active, f"Missing registered strategy: {STRATEGY_NAME}"
    return active[STRATEGY_NAME]


def test_registration_and_registry_hygiene(strategy_entry):
    """Strategy is auto-discovered cleanly with expected params and no draft clutter."""
    assert callable(strategy_entry["logic"])
    assert strategy_entry["dependencies"] == []
    assert set(strategy_entry["params"].keys()) == {
        "fast_ema",
        "slow_sma",
        "vol_baseline",
        "vol_shock_max",
    }
    assert strategy_entry["params"]["vol_shock_max"] == pytest.approx(1.50)

    # Confirm no leftover scratch/draft PTB-* strategies remain registered
    active = get_active_strategies()
    stray = [name for name in active if name.startswith("PTB-")]
    assert stray == [], f"Unexpected draft strategies still registered: {stray}"


def test_signal_validity_and_warmup(strategy_entry):
    """Signals must be integer {-1, 0, 1}, NaN-free, and neutral during warmup."""
    df = _make_multi_regime_ohlcv()
    out = strategy_entry["logic"](df.copy(), **strategy_entry["params"])

    assert "Signal" in out.columns
    assert not out["Signal"].isna().any()
    assert out["Signal"].isin([-1, 0, 1]).all()
    assert pd.api.types.is_integer_dtype(out["Signal"])

    warmup = max(strategy_entry["params"]["slow_sma"] - 1, strategy_entry["params"]["vol_baseline"])
    assert (out["Signal"].iloc[:warmup] == 0).all()
    assert (out["Signal"].iloc[warmup:] != 0).all()


def test_uptrend_long_and_downtrend_flat(strategy_entry):
    """Steady calm uptrend holds long (1); sustained downtrend stays flat (-1)."""
    rng = np.random.default_rng(123)
    # Smooth uptrend with small steady daily noise so vol_ratio hovers near 1.0
    up_rets = 0.0015 + rng.normal(0.0, 0.003, size=260)
    up_df = _make_ohlcv(100.0 * np.exp(np.cumsum(up_rets)))
    up_out = strategy_entry["logic"](up_df, **strategy_entry["params"])
    assert (up_out["Signal"].iloc[-40:] == 1).all()

    down_rets = -0.0015 + rng.normal(0.0, 0.003, size=260)
    down_df = _make_ohlcv(100.0 * np.exp(np.cumsum(down_rets)))
    down_out = strategy_entry["logic"](down_df, **strategy_entry["params"])
    assert (down_out["Signal"].iloc[-40:] == -1).all()


def test_vol_shock_exits_and_auto_reenters(strategy_entry):
    """A sudden volatility burst forces exit (-1) even in an uptrend, then re-enters (1)."""
    rng = np.random.default_rng(99)
    n_calm1, n_shock, n_calm2 = 220, 15, 60
    r1 = 0.0015 + rng.normal(0.0, 0.0025, size=n_calm1)
    # High-amplitude alternating burst that spikes 20d vol > 1.5x baseline while keeping EMA > SMA
    r_shock = np.array([0.025 if i % 2 == 0 else -0.020 for i in range(n_shock)])
    r2 = 0.0015 + rng.normal(0.0, 0.0025, size=n_calm2)
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([r1, r_shock, r2])))

    df = _make_ohlcv(closes)
    out = strategy_entry["logic"](df, **strategy_entry["params"])
    sig = out["Signal"]

    # Long before the shock
    assert sig.iloc[n_calm1 - 1] == 1
    # Exits during the volatility burst even though price stayed in an uptrend
    assert (sig.iloc[n_calm1 : n_calm1 + n_shock] == -1).any()
    # Automatically re-enters once 20d volatility normalizes
    assert sig.iloc[-1] == 1


def test_strict_causality_no_lookahead(strategy_entry):
    """Mutating future bars [cutoff:] must never change any signal in [:cutoff]."""
    df = _make_multi_regime_ohlcv(n=360, seed=42)
    cutoff = 260
    params = dict(strategy_entry["params"])

    sig_base = strategy_entry["logic"](df.copy(), **params)["Signal"]

    df_mut = df.copy()
    df_mut.iloc[cutoff:, df_mut.columns.get_loc("Open")] *= 0.25
    df_mut.iloc[cutoff:, df_mut.columns.get_loc("High")] *= 0.30
    df_mut.iloc[cutoff:, df_mut.columns.get_loc("Low")] *= 0.20
    df_mut.iloc[cutoff:, df_mut.columns.get_loc("Close")] *= 0.25
    df_mut.iloc[cutoff:, df_mut.columns.get_loc("Volume")] *= 10.0

    sig_mut = strategy_entry["logic"](df_mut, **params)["Signal"]
    pd.testing.assert_series_equal(sig_base.iloc[:cutoff], sig_mut.iloc[:cutoff], check_names=False)


def test_edge_cases_empty_short_flat_and_zero_price(strategy_entry):
    """Strategy handles empty, short, flat/zero-vol, and zero-price frames safely."""
    params = dict(strategy_entry["params"])

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out_empty = strategy_entry["logic"](empty_df.copy(), **params)
    assert "Signal" in out_empty.columns
    assert len(out_empty) == 0

    # Short DataFrame (< slow_sma)
    short_df = _make_ohlcv(np.linspace(100.0, 110.0, 50))
    out_short = strategy_entry["logic"](short_df, **params)
    assert (out_short["Signal"] == 0).all()

    # Constant price series (zero baseline volatility -> stays neutral 0, no NaN/crash)
    flat_df = _make_ohlcv(np.full(220, 100.0))
    out_flat = strategy_entry["logic"](flat_df, **params)
    assert not out_flat["Signal"].isna().any()
    assert (out_flat["Signal"] == 0).all()

    # Zero-price and NaN-close corruption rows -> no inf propagation or crash
    bad_df = _make_multi_regime_ohlcv(n=220, seed=11)
    bad_df.iloc[180, bad_df.columns.get_loc("Close")] = 0.0
    bad_df.iloc[190, bad_df.columns.get_loc("Close")] = np.nan
    out_bad = strategy_entry["logic"](bad_df, **params)
    assert not out_bad["Signal"].isna().any()
    assert out_bad["Signal"].isin([-1, 0, 1]).all()


@pytest.mark.parametrize(
    "bad_override",
    [
        {"fast_ema": 0},
        {"slow_sma": 0},
        {"vol_baseline": 1},
        {"vol_shock_max": 0.0},
        {"vol_shock_max": -1.0},
    ],
)
def test_invalid_parameters_raise_value_error(strategy_entry, bad_override):
    """Non-sensical parameter configurations fail fast with ValueError."""
    df = _make_multi_regime_ohlcv(n=200)
    params = {**strategy_entry["params"], **bad_override}
    with pytest.raises(ValueError):
        strategy_entry["logic"](df, **params)


def test_engine_integration_portfolio_simulation(strategy_entry):
    """Strategy executes cleanly inside run_portfolio_simulation with default fixed sizing."""
    df_a = _make_multi_regime_ohlcv(n=360, seed=7)
    df_b = _make_multi_regime_ohlcv(n=360, seed=19)
    portfolio_data = {"AAA": df_a, "BBB": df_b}

    signals = {
        sym: strategy_entry["logic"](frame.copy(), **strategy_entry["params"])["Signal"]
        for sym, frame in portfolio_data.items()
    }

    result = run_portfolio_simulation(
        portfolio_data=portfolio_data,
        signals=signals,
        initial_capital=100_000.0,
        allocation_pct=0.10,
        spy_df=df_a,
        vix_df=None,
        tnx_df=None,
        stop_config={"type": "none"},
    )
    assert result is not None
    assert "equity_curve" in result or "portfolio_timeline" in result
    trade_log = result.get("trade_log", [])
    assert len(trade_log) > 0
