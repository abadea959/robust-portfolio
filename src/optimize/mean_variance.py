from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    no_short: bool = True,
    w_cap: float | None = None,
) -> pd.Series:
   
    tickers = list(mu.index)
    n = len(tickers)

    # Align shapes
    cov = cov.loc[tickers, tickers]
    Sigma = cov.values
    excess = (mu - rf).values  # (mu - rf*1)

    # Decision variable
    w = cp.Variable(n)

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        cp.quad_form(w, Sigma) <= 1  # variance normalization
    ]
    if no_short:
        constraints.append(w >= 0)
    if w_cap is not None:
        constraints.append(w <= w_cap)

    # Objective: maximize excess return
    objective = cp.Maximize(w @ excess)

    # Solve
    prob = cp.Problem(objective, constraints)
    # Let cvxpy pick a default; SCS/ECOS usually work. You can pass solver=cp.ECOS if needed.
    prob.solve()

    if w.value is None:
        raise RuntimeError("Optimization failed; try relaxing constraints or a different solver.")

    weights = pd.Series(np.array(w.value).ravel(), index=tickers)
    # Numerical cleanup
    weights = weights.clip(lower=-0.0)  # avoid tiny negative rounding
    weights /= weights.sum()            # normalize exactly to 1
    return weights
