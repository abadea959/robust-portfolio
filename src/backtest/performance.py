from __future__ import annotations
import numpy as np
import pandas as pd


def as_simple(rets: pd.DataFrame | pd.Series, kind: str = "simple") -> pd.DataFrame | pd.Series:
    if kind not in {"simple", "log"}:
        raise ValueError("kind must be 'simple' or 'log'")
    if kind == "log":
        return np.exp(rets) - 1.0
    return rets


def portfolio_returns(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    returns_kind: str = "simple",
) -> pd.Series:

    asset_returns = asset_returns.loc[:, weights.index]
    simple_asset_rets = as_simple(asset_returns, returns_kind)
    port = (simple_asset_rets * weights).sum(axis=1)
    return port


def cumulative_returns(simple_returns: pd.Series) -> pd.Series:
    return (1.0 + simple_returns).cumprod()


def max_drawdown(cum_curve: pd.Series) -> float:

    running_peak = cum_curve.cummax()
    drawdowns = 1.0 - (cum_curve / running_peak)
    return float(drawdowns.max())


def summary_stats(
    simple_returns: pd.Series,
    rf: float = 0.0,
    periods: int = 252,
) -> dict:

    simple_returns = simple_returns.dropna()
    if simple_returns.empty:
        raise ValueError("No returns provided.")

    growth = cumulative_returns(simple_returns)
    start_val, end_val = float(growth.iloc[0]), float(growth.iloc[-1])
    n_years = len(simple_returns) / periods
    cagr = end_val ** (1.0 / n_years) - 1.0

    vol = simple_returns.std(ddof=1) * np.sqrt(periods)

    ann_ret = simple_returns.mean() * periods
    sharpe = (ann_ret - rf) / vol if vol > 0 else np.nan

    mdd = max_drawdown(growth)

    return {
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": float(mdd),
        "n_periods": int(len(simple_returns)),
        "start": simple_returns.index[0],
        "end": simple_returns.index[-1],
    }
