# main_baseline.py
from __future__ import annotations
import pandas as pd

from src.config import load_config
from src.data.download import download_adj_close, cache_prices_csv, load_cached_prices
from src.features.transforms import to_log_returns, to_simple_returns
from src.risk.covariance import sample_mean_cov
from src.optimize.mean_variance import max_sharpe
from src.backtest.performance import portfolio_returns, summary_stats, cumulative_returns, max_drawdown



def load_prices_or_download(cfg) -> pd.DataFrame:
    try:
        prices = load_cached_prices("data/prices.csv")
        # ensure we have all requested tickers, in the requested order
        missing = [t for t in cfg["tickers"] if t not in prices.columns]
        if missing:
            raise KeyError(f"Missing tickers in cache: {missing}")
        return prices.loc[:, cfg["tickers"]]
    except Exception:
        prices = download_adj_close(cfg["tickers"], cfg["start"], cfg["end"])
        cache_prices_csv(prices, "data/prices.csv")
        return prices


def pretty_weights(w) -> str:
    return w.sort_values(ascending=False).to_string(float_format=lambda x: f"{x:0.2%}")


def main():
    # 1) config
    cfg = load_config()
    tickers = cfg["tickers"]
    rf = cfg.get("risk_free_rate_annual", 0.0)
    cap = cfg.get("max_alloc_per_asset", None)
    no_short = cfg.get("no_shorting", True)

    # 2) prices (cached or fresh)
    prices = load_prices_or_download(cfg).dropna(how="any")
    print(f"Data range: {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} rows)")
    print(f"Tickers: {', '.join(tickers)}")

    # 3) returns
    #    - use LOG returns for estimating mean/cov
    #    - use SIMPLE returns for realized compounding/backtest
    log_rets = to_log_returns(prices)
    simple_rets = to_simple_returns(prices)

    # 4) estimates (annualized)
    mu, cov = sample_mean_cov(log_rets, annualize=True)

    # 5) optimize: max-sharpe with simple constraints
    w = max_sharpe(mu=mu, cov=cov, rf=rf, no_short=no_short, w_cap=cap)

    # 6) realized portfolio performance (static weights over full period)
    port_rets = portfolio_returns(weights=w, asset_returns=simple_rets, returns_kind="simple")
    stats = summary_stats(simple_returns=port_rets, rf=rf, periods=252)

    # (optional) extra: cumulative curve + max drawdown check
    curve = cumulative_returns(port_rets)
    mdd = max_drawdown(curve)

    # 7) report
    print("\n=== Max-Sharpe Weights ===")
    print(pretty_weights(w))
    print(f"\nSum of weights: {w.sum():.6f}")

    print("\n=== Performance (realized) ===")
    print(f"Period: {stats['start']} → {stats['end']}  ({stats['n_periods']} obs)")
    print(f"CAGR:        {stats['cagr']:.2%}")
    print(f"Ann. Vol:    {stats['vol']:.2%}")
    print(f"Sharpe:      {stats['sharpe']:.4f}")
    print(f"Max Drawdown:{mdd:.2%}")


if __name__ == "__main__":
    main()
