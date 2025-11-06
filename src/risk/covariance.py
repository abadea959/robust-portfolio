import numpy as np
import pandas as pd


def sample_mean_cov(returns: pd.DataFrame, annualize: bool = True):
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    rets = returns.dropna(how="any").copy()

    mean = rets.mean()
    cov = rets.cov()

    if annualize:
        trading_days = 252
        mean *= trading_days
        cov *= trading_days

    return mean, cov


def summarize_stats(mean: pd.Series, cov: pd.DataFrame):
    """Convenience: print key risk/return metrics."""
    print("Expected annual returns:")
    print(mean.sort_values(ascending=False).round(3))
    print("\nCovariance matrix (annualized):")
    print(cov.round(5))
