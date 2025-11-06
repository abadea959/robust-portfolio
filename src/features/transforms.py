from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    idx = pd.to_datetime(prices.index)
    if not idx.is_monotonic_increasing:
        prices = prices.copy()
        prices.index = idx
        prices = prices.sort_index()

    numeric = prices.select_dtypes(include=[np.number])
    if numeric.shape[1] != prices.shape[1]:
        raise ValueError("Non-numeric columns found in prices; ensure all columns are numeric price series.")

    numeric = numeric.dropna(how="all").ffill().dropna(how="any")

    return numeric


def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:

    px = _validate_prices(prices)
    rets = px.pct_change().dropna(how="any")
    return rets


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:

    px = _validate_prices(prices)
    rets = np.log(px).diff().dropna(how="any")
    return rets
